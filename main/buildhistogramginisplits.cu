// Online C++ compiler to run C++ program online
#include <iostream>
#include <vector>
#include <cstdlib>   // for rand()
#include <ctime>     // for time()
#include <random>
#include <cmath>
#include <cfloat>
#include <numeric>
#include <algorithm>
#include <iomanip>
#include <vector>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cublas_v2.h>
#include <cstdint>
#include <chrono>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "util.hpp"

__global__ void warmup1() {}

__global__ void ComputeMinMaxKernel(
    const float* __restrict__ d_col_add_projected,
    float* __restrict__ d_block_min,
    float* __restrict__ d_block_max,
    int num_rows,
    int num_proj
){
    extern __shared__ float shared[];
    float* shared_min = shared;
    float* shared_max = shared + blockDim.x;

    int proj_id = blockIdx.y;
    int tid     = threadIdx.x;
    if (proj_id >= num_proj) return;

    int base_idx = proj_id * num_rows;

    float local_min = FLT_MAX;
    float local_max = -FLT_MAX;

    // strided loop over rows
    for (int i = blockIdx.x * blockDim.x + tid; i < num_rows; i += gridDim.x * blockDim.x) {
        float val = d_col_add_projected[base_idx + i];
        local_min = fminf(local_min, val);
        local_max = fmaxf(local_max, val);
    }

    // store per-thread local min/max in shared
    shared_min[tid] = local_min;
    shared_max[tid] = local_max;
    __syncthreads();

    // block-wide reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_min[tid] = fminf(shared_min[tid], shared_min[tid + stride]);
            shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + stride]);
        }
        __syncthreads();
    }

    // write block result
    if (tid == 0) {
        int idx = proj_id * gridDim.x + blockIdx.x;
        d_block_min[idx] = shared_min[0];
        d_block_max[idx] = shared_max[0];
    }
}


__global__ void FinalMinMaxKernel(
    const float* __restrict__ d_block_min,
    const float* __restrict__ d_block_max,
    float* __restrict__ d_min_vals,
    float* __restrict__ d_max_vals,
    float* __restrict__ d_bin_widths,
    int num_blocks,
    int num_proj,
    int num_bins
) {
    int proj_id = blockIdx.x;
    if (proj_id >= num_proj) return;

    float min_val = FLT_MAX;
    float max_val = -FLT_MAX;

    // sequential reduction across all blocks of this projection
    for (int i = 0; i < num_blocks; ++i) {
        int idx = proj_id * num_blocks + i;
        min_val = fminf(min_val, d_block_min[idx]);
        max_val = fmaxf(max_val, d_block_max[idx]);
    }

    d_min_vals[proj_id] = min_val;
    d_max_vals[proj_id] = max_val;
    d_bin_widths[proj_id] = (max_val > min_val)
                            ? (max_val - min_val) / (float)num_bins
                            : 1.0f;
}


__global__ void BuildHistogramExactWidthKernel(
    const float* __restrict__ d_col_add_projected, //input
    const int* __restrict__ d_labels,
    int* __restrict__ d_hist_class0,          // [num_samples], values 0 or 1
    int* __restrict__ d_hist_class1,           // [NUM_BINS], count of
    float* __restrict__ d_min_vals, //output
    float* __restrict__ d_max_vals, //output
    float* __restrict__ d_bin_widths, //output
    int num_rows,
    int num_proj,
    int num_bins
)
{
    extern __shared__ int shared_mem[];
    int proj_id = blockIdx.y;
    if (proj_id >= num_proj) return;

    // 1. zero shared histogram
    for (int i = threadIdx.x; i < 2 * num_bins; i += blockDim.x)
        shared_mem[i] = 0;
    __syncthreads();


     // Each block handles one projection
    int base_idx = proj_id * num_rows; // Base index for the current projection


    float min_val = d_min_vals[proj_id];
    float max_val = d_max_vals[proj_id];
    float inv_bin_width  = 1.f / d_bin_widths[proj_id]; 

  
    const std::size_t col_offset = std::size_t(proj_id) * num_rows;
    const int stride = blockDim.x * gridDim.x;
    const int tid    = threadIdx.x + blockIdx.x * blockDim.x;

    for (int i = tid; i < num_rows; i += stride) {
        float val   = __ldg(&d_col_add_projected[col_offset + i]);      // <<< changed >>>
        int   label = __ldg(&d_labels[i]);  
     
        int bin = (val >= max_val)
                  ? (num_bins - 1)
                  : max(0, min(int((val - min_val) * inv_bin_width),     // <<< changed >>>
                               num_bins - 1));
        
        if (label == 0) {
            atomicAdd(&shared_mem[bin], 1);
        } 
        else {
            atomicAdd(&shared_mem[num_bins + bin], 1);
        }
    }
    __syncthreads();

    int offset =  proj_id * num_bins;
    for (int i = threadIdx.x; i < num_bins; i += blockDim.x) {
         atomicAdd(&d_hist_class0[offset + i], shared_mem[i]);
         atomicAdd(&d_hist_class1[offset + i], shared_mem[num_bins + i]);
    }    
}

__global__ void FindBestGiniSplitKernel(
    const int* hist_class0,
    const int* hist_class1,
    const float* min_vals,
    const float* bin_widths,
    int num_proj,
    int num_bins,
    float* gini_out_per_bin_per_proj,
    int* best_bin_out, // per proj
    float* best_gini_gain_out, // per proj
    float* best_threshold_out)  // per proj
{
    int proj_id = blockIdx.y;   
    int bin_id  = threadIdx.x + blockIdx.x * blockDim.x;

    // Only evaluate splits between bins, not after the last bin
    if (proj_id >= num_proj) return;
    if (bin_id >= num_bins - 1) return;

    int base_idx = proj_id * num_bins;

    // Compute total class counts (redundantly across threads)
    // Total class is same for all projections
    int total_class0 = 0;
    int total_class1 = 0;
    for (int i = 0; i < num_bins; ++i) {
        total_class0 += hist_class0[i];
        total_class1 += hist_class1[i];
    }

    // Compute left class counts for this split point
    int left_class0 = 0;
    int left_class1 = 0;
    for (int i = 0; i <= bin_id; ++i) {
        left_class0 += hist_class0[base_idx + i];
        left_class1 += hist_class1[base_idx + i];
    }

    int right_class0 = total_class0 - left_class0;
    int right_class1 = total_class1 - left_class1;

    int left_total  = left_class0 + left_class1;
    int right_total = right_class0 + right_class1;

    float gini_left = 1.0f - powf((float)left_class0 / max(left_total, 1), 2.0f)
                           - powf((float)left_class1 / max(left_total, 1), 2.0f);
    float gini_right = 1.0f - powf((float)right_class0 / max(right_total, 1), 2.0f)
                            - powf((float)right_class1 / max(right_total, 1), 2.0f);

    float total = total_class0 + total_class1;
    float p0 = (float)total_class0 / max(total, 1.0f);
    float p1 = (float)total_class1 / max(total, 1.0f);
    float gini_before = 1.0f - p0 * p0 - p1 * p1;

    float weighted_gini = (left_total * gini_left + right_total * gini_right) / max(total, 1.0f);
    float gini_gain = gini_before - weighted_gini;

    // Store per-thread result in global memory
    // just index it right so it can store the result for each bin per projection
    gini_out_per_bin_per_proj[base_idx + bin_id] = gini_gain;

    __syncthreads();

    // Thread 0 finds best split for this projection
    // TO DO : Optimize this reduction, potentially use parallel reduction
    if (bin_id == 0) {
        float best_gain = -1.0f;
        int best_bin = -1;
        for (int i = 0; i < num_bins - 1; ++i) {
            float gain = gini_out_per_bin_per_proj[base_idx + i];
            if (gain > best_gain) {
                best_gain = gain;
                best_bin = i; // Change here to match ydf convention
            }
        }


    best_bin_out[proj_id] = best_bin;
    best_gini_gain_out[proj_id] = best_gain;
    best_threshold_out[proj_id] = (best_bin + 1) * bin_widths[proj_id] + min_vals[proj_id]; // If needed, change here to match ydf convention

    // Compute threshold assuming uniform binning in [0, 1]
    //float bin_width = 1.0f / num_bins;
    //threshold_out[proj_id] = (best_bin + 1) * bin_width;
    }
}


void BuildExactWidthHistogram (float* d_col_add_projected,
                               int** d_hist_class0_out,
                               int** d_hist_class1_out,
                               float** d_min_vals_out,
                               float** d_max_vals_out,
                               float** d_bin_widths_out, 
                               const std::vector<int>& labels,
                               const int num_rows,
                               const int num_proj,
                               const int num_bins,
                               double& elapsed_ms,
                               bool verbose
                              )
{
    warmup1<<<1, 1>>>();
    cudaDeviceSynchronize();
    
    //////////////////////////ComputeMinMax///////////////////////////////////////////////////////
    float* d_min_vals;
    float* d_max_vals;
    float* d_bin_widths;
    cudaMalloc(&d_min_vals, num_proj * sizeof(float));
    cudaMalloc(&d_max_vals, num_proj * sizeof(float));
    cudaMalloc(&d_bin_widths, num_proj * sizeof(float));
    
    *d_min_vals_out = d_min_vals;
    *d_max_vals_out = d_max_vals;
    *d_bin_widths_out = d_bin_widths;

    int threads_per_block_minmax = 256; //using 256 threads per block for min/max computation and use this for buffer size
    int blocks_per_projection = 256;
    
    // safety: must be >= 1
    blocks_per_projection = std::max(1, blocks_per_projection);
    const int total_blocks = num_proj * blocks_per_projection;

    // Allocate intermediate buffers
    float* d_block_min;
    float* d_block_max;
    cudaMalloc(&d_block_min, total_blocks * sizeof(float));
    cudaMalloc(&d_block_max, total_blocks * sizeof(float));

    dim3 grid(blocks_per_projection, num_proj);
    dim3 block(threads_per_block_minmax);

    size_t shmem = 2 * threads_per_block_minmax * sizeof(float);
    
    // Launch Pass 1: local min/max per block

    // std::cout << "Launching ComputeMinMaxKernel with grid=("
    //       << grid.x << "," << grid.y << "), block=("
    //       << block.x << "), shared=" << shmem << " bytes\n";
    // std::cout << "num_proj = " << num_proj << ", num_rows = " << num_rows << "\n";
    auto startA = std::chrono::high_resolution_clock::now();
    ComputeMinMaxKernel<<<grid, block, shmem>>>(d_col_add_projected,
                                                d_block_min,
                                                d_block_max,
                                                num_rows,
                                                num_proj);
    cudaError_t err0 = cudaGetLastError();
    cudaDeviceSynchronize();
    auto endA = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> durationA = endA - startA;
    if (err0 != cudaSuccess) {
        printf("Kernel launch failed min/max: %s\n", cudaGetErrorString(err0));
    }
    printf("Elapsed time for ComputeMinMaxKernel: %f ms\n", durationA.count());
    
    ////ComputeMinMaxKernel done//////////////////////////////////////////////////////////////
    // Launch Pass 2: final reduction per projection
    auto startB = std::chrono::high_resolution_clock::now();
    FinalMinMaxKernel<<<num_proj, 1>>>(
        d_block_min,
        d_block_max,
        d_min_vals,
        d_max_vals,
        d_bin_widths,
        blocks_per_projection,
        num_proj,
        num_bins
    );
    cudaError_t err1 = cudaGetLastError();
    cudaDeviceSynchronize();
    auto endB = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> durationB = endB - startB;
    if (err1 != cudaSuccess) {
        printf("Kernel launch failed min/max: %s\n", cudaGetErrorString(err0));
    }
    printf("Elapsed time for FinalMinMaxKernel: %f ms\n", durationB.count());


    // Cleanup
    cudaFree(d_block_min);
    cudaFree(d_block_max);


    //////////////////////////BuildHistogramExactWidth///////////////////////////////////////////////////////
    int* d_labels;
    cudaMalloc(&d_labels, num_rows * sizeof(int));
    cudaMemcpy(d_labels, labels.data(), num_rows * sizeof(int), cudaMemcpyHostToDevice);
    int* d_hist_class0;
    int* d_hist_class1;
    cudaMalloc(&d_hist_class0, num_proj * num_bins * sizeof(int));
    cudaMalloc(&d_hist_class1, num_proj * num_bins * sizeof(int));
    cudaMemset(d_hist_class0, 0, num_proj * num_bins * sizeof(int));
    cudaMemset(d_hist_class1, 0, num_proj * num_bins * sizeof(int));
    *d_hist_class0_out = d_hist_class0;
    *d_hist_class1_out = d_hist_class1;
    
    int threads_per_block_hist = 256;
    int blocks_per_grid_hist = (num_rows + threads_per_block_hist - 1) / threads_per_block_hist;
    //int blocks_per_grid_hist = 256;
    dim3 grid_hist(blocks_per_grid_hist, num_proj);
    dim3 block_hist(threads_per_block_hist);    

    
    int sharedMemSize = 2 * num_bins * sizeof(int); // For Hist 0 and Hist 1
    auto startC = std::chrono::high_resolution_clock::now();
    BuildHistogramExactWidthKernel<<<grid_hist, block_hist, sharedMemSize>>>(d_col_add_projected, d_labels,
                                                                             d_hist_class0, d_hist_class1,
                                                                             d_min_vals, d_max_vals, d_bin_widths,
                                                                             num_rows, num_proj, num_bins);        
    cudaError_t err2 = cudaGetLastError();
    cudaDeviceSynchronize(); 
    auto endC = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> durationC = endC - startC;
    elapsed_ms = durationA.count() + durationB.count() + durationC.count();
   

    
    if (err2 != cudaSuccess) {
        printf("Kernel launch failed hist: %s\n", cudaGetErrorString(err1));
    }
 
    // BuildExactWidthHistogramKernel done//////////////////////////////////////////

    if (verbose){
            // Copy min, max, bin_widths to host and print
            std::vector<float> h_min_vals(num_proj);
            std::vector<float> h_max_vals(num_proj);
            std::vector<float> h_bin_widths(num_proj);
            cudaMemcpy(h_min_vals.data(), d_min_vals, num_proj * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_max_vals.data(), d_max_vals, num_proj * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_bin_widths.data(), d_bin_widths, num_proj * sizeof(float), cudaMemcpyDeviceToHost);
            printf("h_min_vals: \n");
            for (float val : h_min_vals){
                printf("%f  ", val);
            }
            printf("\n");
            printf("h_max_vals: \n");
            for (float val : h_max_vals){
                printf("%f  ", val);
            }
            printf("\n");
            printf("h_bin_widths: \n");
            for (float val : h_bin_widths){
                printf("%f  ", val);    
            }
            printf("\n");

            // Copy histograms to host and print
            std::vector<int> h_hist_class0(num_bins * num_proj);
            std::vector<int> h_hist_class1(num_bins * num_proj);
            cudaMemcpy(h_hist_class0.data(), d_hist_class0, num_bins * num_proj * sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_hist_class1.data(), d_hist_class1, num_bins * num_proj * sizeof(int), cudaMemcpyDeviceToHost);
        
            printf("Histogram Class 0:\n");
            for (int proj = 0; proj < num_proj; ++proj) {
                printf("Projection %d:\n", proj);
                for (int bin = 0; bin < num_bins; ++bin) {
                    int idx = proj * num_bins + bin;
                    float bin_start = h_min_vals[proj] + bin * h_bin_widths[proj];
                    float bin_end   = bin_start + h_bin_widths[proj];
                    printf("  Bin [%f, %f): %d\n", bin_start, bin_end, h_hist_class0[idx]);
                }
            }
            printf("Histogram Class 1:\n");
            for (int proj = 0; proj < num_proj; ++proj) {
                printf("Projection %d:\n", proj);
                for (int bin = 0; bin < num_bins; ++bin) {
                    int idx = proj * num_bins + bin;                            
                    float bin_start = h_min_vals[proj] + bin * h_bin_widths[proj];
                    float bin_end   = bin_start + h_bin_widths[proj];
                    printf("  Bin [%f, %f): %d\n", bin_start, bin_end, h_hist_class1[idx]);
                }
            }
        }
    
    cudaFree(d_labels);
    cudaFree(d_max_vals);
    cudaFree(d_col_add_projected); // Free device memory passed from randomprojection
}

void FindBestGiniSplit (int* d_hist_class0,
                        int* d_hist_class1,
                        float* d_min_vals,
                        float* d_bin_widths,
                        int num_proj,
                        int num_bins,
                        int* best_proj,
                        int* best_bin_out,
                        float* best_gini_gain_out,
                        float* best_threshold_out,
                        double& elapsed_ms,
                        bool verbose)
{
    int* d_best_bin_out;
    float* d_best_gini_gain_out;
    float* d_best_threshold_out;
    float* d_gini_out_per_bin_per_proj;
    cudaMalloc(&d_best_bin_out, num_proj * sizeof(int));
    cudaMalloc(&d_best_gini_gain_out, num_proj * sizeof(float));
    cudaMalloc(&d_best_threshold_out, num_proj * sizeof(float));
    cudaMalloc(&d_gini_out_per_bin_per_proj, num_proj * (num_bins - 1) * sizeof(float)); // Store Gini gain for each bin (except last)

    int threads_per_block_split = 256;
    int blocks_per_grid_split = (num_bins + threads_per_block_split - 1) / threads_per_block_split; // +1 to ensure we have enough blocks to cover all bins
    dim3 grid_split(blocks_per_grid_split, num_proj);
    dim3 block_split(threads_per_block_split);

    auto start = std::chrono::high_resolution_clock::now();
    FindBestGiniSplitKernel<<<grid_split, block_split>>>(
        d_hist_class0, d_hist_class1, d_min_vals, d_bin_widths, num_proj, num_bins,
        d_gini_out_per_bin_per_proj,
        d_best_bin_out, d_best_gini_gain_out,
        d_best_threshold_out);
    cudaError_t err3 = cudaGetLastError();
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    elapsed_ms += duration.count();

    if (err3 != cudaSuccess) {
    printf("Kernel launch failed gini: %s\n", cudaGetErrorString(err3));
    }


    // Copy results from device to host
    cudaMemcpy(best_bin_out, d_best_bin_out,
                num_proj * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(best_threshold_out, d_best_threshold_out,
                num_proj * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(best_gini_gain_out, d_best_gini_gain_out,
                num_proj * sizeof(float), cudaMemcpyDeviceToHost);

    float best_gain = -FLT_MAX;
    for (int i = 0; i < num_proj; ++i) {
        if (verbose){
            printf("Gain[%d] = %f\n", i, best_gini_gain_out[i]);
        }
        if (best_gini_gain_out[i] > best_gain) {
            best_gain = best_gini_gain_out[i];
            *best_proj = i;
        }
    }

    cudaFree(d_best_bin_out);
    cudaFree(d_hist_class0);
    cudaFree(d_hist_class1);
    cudaFree(d_min_vals);
    cudaFree(d_bin_widths);
    cudaFree(d_best_gini_gain_out);
    cudaFree(d_best_threshold_out);
    cudaFree(d_gini_out_per_bin_per_proj);
}