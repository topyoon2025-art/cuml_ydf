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

__global__ void BuildExactWidthHistogramKernel(
    const float* __restrict__ d_col_add_projected,  // [num_samples]
    const int* __restrict__ labels,
    int* __restrict__ hist_class0,          // [num_samples], values 0 or 1
    int* __restrict__ hist_class1,           // [NUM_BINS], count of class 1 per bin
    float* __restrict__ d_min_vals,
    float* __restrict__ d_max_vals,
    float* __restrict__ d_bin_widths,
    int num_rows,
    int num_proj,
    int num_bins
    ){
    extern __shared__ float shared_minmax[];  // [2 × blockDim.x]
    float* shared_min = shared_minmax;
    float* shared_max = shared_minmax + blockDim.x;

    int col = blockIdx.x;
    int tid = threadIdx.x;
    int idx_base = col * num_rows;

    if (col >= num_proj) return;

    // Step 1: grid-stride loop to compute local min/max
    float local_min = FLT_MAX;
    float local_max = -FLT_MAX;

    for (int row = tid; row < num_rows; row += blockDim.x) {
        float val = d_col_add_projected[idx_base + row];
        local_min = fminf(local_min, val);
        local_max = fmaxf(local_max, val);
    }

    shared_min[tid] = local_min;
    shared_max[tid] = local_max;
    __syncthreads();

    // Step 2: parallel reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_min[tid] = fminf(shared_min[tid], shared_min[tid + stride]);
            shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + stride]);
        }
        __syncthreads();
    }

    float min_val = shared_min[0];
    float max_val = shared_max[0];
    float bin_width = (max_val - min_val) / static_cast<float>(num_bins);

    if (tid == 0) {
        d_min_vals[col] = min_val;
        d_max_vals[col] = max_val;
        d_bin_widths[col] = bin_width;
    }
    __syncthreads();

    // Step 3: grid-stride loop to bin values
    for (int row = tid; row < num_rows; row += blockDim.x) {
        int idx = idx_base + row;
        float val = d_col_add_projected[idx];
        int cls = labels[row];

        int bin = __float2int_rd((val - min_val) / bin_width);
        bin = min(bin, num_bins - 1);

        int offset = col * num_bins + bin;
        if (cls == 0)
            atomicAdd(&hist_class0[offset], 1);
        else
            atomicAdd(&hist_class1[offset], 1);
    }
}

__global__ void FindBestGiniSplitKernel(
    const int* hist_class0,
    const int* hist_class1,
    const float* min_vals,
    const float* bin_widths,
    int num_proj,
    int num_bins,
    int* best_bin_out,
    float* best_gini_out,
    int* partial_class0,
    int* partial_class1,
    float* threshold_out)  // New output
{
    int proj_id = blockIdx.x;
    int bin_id  = threadIdx.x;

    // Only evaluate splits between bins, not after the last bin
    if (bin_id >= num_bins - 1) return;

    int base_idx = proj_id * num_bins;

    // Compute total class counts (redundantly across threads)
    int total_class0 = 0;
    int total_class1 = 0;
    for (int b = 0; b < num_bins; ++b) {
        int idx = base_idx + b;
        total_class0 += hist_class0[idx];
        total_class1 += hist_class1[idx];
    }

    // Compute left class counts for this split point
    int left_class0 = 0;
    int left_class1 = 0;
    for (int i = 0; i <= bin_id; ++i) {
        int sub_idx = base_idx + i;
        left_class0 += hist_class0[sub_idx];
        left_class1 += hist_class1[sub_idx];
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
    int out_idx = base_idx + bin_id;
    best_gini_out[out_idx] = gini_gain;

    // Thread 0 finds best split for this projection
    if (bin_id == 0) {
        float best_gain = -1.0f;
        int best_bin = -1;
        for (int i = 0; i < num_bins - 1; ++i) {
            int idx = base_idx + i;
            float gain = best_gini_out[idx];
            if (gain > best_gain) {
                best_gain = gain;
                best_bin = i + 1;
            }
        }

        best_bin_out[proj_id] = best_bin;
        best_gini_out[proj_id] = best_gain;
        partial_class0[proj_id] = total_class0;
        partial_class1[proj_id] = total_class1;

        // Compute threshold assuming uniform binning in [0, 1]
        //float bin_width = 1.0f / num_bins;
        //threshold_out[proj_id] = (best_bin + 1) * bin_width;
        threshold_out[proj_id] = best_bin * bin_widths[proj_id] + min_vals[proj_id];
    }
}


void BuildExactWidthHistogram (float* d_col_add_projected,
                               int** d_hist_class0_out,
                               int** d_hist_class1_out,
                               float** d_min_vals_out,
                               float** d_bin_widths_out, 
                               const std::vector<int> labels,
                               const int num_rows,
                               const int num_proj,
                               const int num_bins,
                               double& elapsed_ms,
                               bool verbose
                              )
{

  if (verbose) {
        printf("Label is: ");
        for (int value : labels){
            
            printf("%d  ", value);
        }
        printf("\n");

        printf("d_col_add_projected address: %p \n", (void*)d_col_add_projected);
        int N = /* number of elements in d_col_add_projected */ num_rows * num_proj;

        // Allocate host buffer
        float* h_buffer = new float[N];

        // Copy from device to host
        cudaMemcpy(h_buffer, d_col_add_projected, N * sizeof(float), cudaMemcpyDeviceToHost);

        // Print values
        for (int i = 0; i < N; ++i) {
            printf("d_col_add_projected[%d] = %f\n", i, h_buffer[i]);
        }

    // Clean up
    delete[] h_buffer;
  }

  int *d_labels, *d_hist_class0, *d_hist_class1;
  cudaMalloc(&d_labels, num_rows * sizeof(int));
  cudaMalloc(&d_hist_class0, num_bins * num_proj * sizeof(int));
  cudaMalloc(&d_hist_class1, num_bins * num_proj * sizeof(int));
  cudaMemset(d_hist_class0, 0, num_bins * num_proj * sizeof(int));
  cudaMemset(d_hist_class1, 0, num_bins * num_proj * sizeof(int));

  cudaMemcpy(d_labels, labels.data(), num_rows * sizeof(int), cudaMemcpyHostToDevice);

  int threads = 256;
  //int threads = num_bins;
  dim3 grid_Hist(num_proj);
  //dim3 blockDim1(2 * num_rows);  // 256 threads per block
  size_t shared_mem_size = 2 * threads * sizeof(float);

  float* d_min_vals;
  float* d_max_vals;
  float* d_bin_widths;

  cudaMalloc(&d_min_vals, num_proj * sizeof(float));
  cudaMalloc(&d_max_vals, num_proj * sizeof(float));
  cudaMalloc(&d_bin_widths, num_proj * sizeof(float));


  BuildExactWidthHistogramKernel<<<grid_Hist, threads, shared_mem_size>>>(
    d_col_add_projected, d_labels, d_hist_class0, d_hist_class1, d_min_vals, d_max_vals, d_bin_widths, num_rows, num_proj, num_bins);
  cudaDeviceSynchronize();

  *d_hist_class0_out = d_hist_class0;
  *d_hist_class1_out = d_hist_class1;
  *d_min_vals_out = d_min_vals;
  *d_bin_widths_out = d_bin_widths;

  if (verbose) {
    std::vector<int> h_hist_class0(num_bins * num_proj);
    std::vector<int> h_hist_class1(num_bins * num_proj);
    std::vector<float> h_min_vals(num_proj);
    std::vector<float> h_max_vals(num_proj);
    std::vector<float> h_bin_widths(num_proj);
    cudaMemcpy(h_hist_class0.data(), d_hist_class0, num_bins * num_proj * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_hist_class1.data(), d_hist_class1, num_bins * num_proj * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_min_vals.data(), d_min_vals, num_proj * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_max_vals.data(), d_max_vals, num_proj * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_bin_widths.data(), d_bin_widths, num_proj * sizeof(float), cudaMemcpyDeviceToHost);

    PrintGroupedBinaryHistogram2D(h_hist_class0.data(), h_hist_class1.data(), h_min_vals.data(), h_max_vals.data(), h_bin_widths.data(), num_proj, num_bins);


  }

  cudaFree(d_labels);
  cudaFree(d_max_vals);
  cudaFree(d_col_add_projected);//Free here since gini splits need to use the projected values, not right after apply projection


}

void FindBestGiniSplit (int* d_hist_class0,
                        int* d_hist_class1,
                        float* d_min_vals,
                        float* d_bin_widths,
                        int num_proj,
                        int num_bins,
                        int* best_proj,
                        int* best_bin_out,
                        float* best_gini_out,
                        float* threshold_out)
{
    int* d_best_bin_out;
    float* d_best_gini_gains_out;
    cudaMalloc(&d_best_bin_out, num_proj * sizeof(int));
    cudaMalloc(&d_best_gini_gains_out, num_proj * sizeof(float));

    assert(d_hist_class0 != nullptr);
    assert(d_hist_class1 != nullptr);
    assert(d_best_bin_out != nullptr);
    assert(d_best_gini_gains_out != nullptr);
    assert(num_proj > 0 && num_bins > 1);

    int threads_per_block_partial = 128;
    int* d_partial_class0;
    int* d_partial_class1;
    cudaMalloc(&d_partial_class0, num_proj * threads_per_block_partial * sizeof(int));
    cudaMalloc(&d_partial_class1, num_proj * threads_per_block_partial * sizeof(int));
    float* d_best_threshold_out;
    cudaMalloc(&d_best_threshold_out, num_proj * sizeof(float));

    dim3 grid_Splits(num_proj);
    dim3 block_Splits(num_bins);
    FindBestGiniSplitKernel<<<grid_Splits, block_Splits>>>(
        d_hist_class0, d_hist_class1, d_min_vals, d_bin_widths, num_proj, num_bins,
        d_best_bin_out, d_best_gini_gains_out,
        d_partial_class0, d_partial_class1,
        d_best_threshold_out);
    cudaDeviceSynchronize();
    cudaError_t err1 = cudaGetLastError();
    if (err1 != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err1));
    }

     // Copy results from device to host
    cudaMemcpy(best_bin_out, d_best_bin_out,
                num_proj * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(threshold_out, d_best_threshold_out,
                num_proj * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(best_gini_out, d_best_gini_gains_out,
                num_proj * sizeof(float), cudaMemcpyDeviceToHost);

 
    float best_gain = -FLT_MAX;

    for (int i = 0; i < num_proj; ++i) {
        if (best_gini_out[i] > best_gain) {
            best_gain = best_gini_out[i];
            *best_proj = i;
        }
    }

    cudaFree(d_best_bin_out);
    cudaFree(d_hist_class0);
    cudaFree(d_hist_class1);
    cudaFree(d_min_vals);
    cudaFree(d_bin_widths);
    cudaFree(d_best_gini_gains_out);
    cudaFree(d_partial_class0);
    cudaFree(d_partial_class1);
    cudaFree(d_best_threshold_out);

}