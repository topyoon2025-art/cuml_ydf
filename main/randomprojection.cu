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


absl::Status CudaStatus(cudaError_t code) {
  if (code != cudaSuccess) {
    const char *error = cudaGetErrorString(code);
    return absl::InvalidArgumentError(absl::StrCat("Cuda error: ", error));
  }
  return absl::OkStatus();
}
#define RETURN_IF_ERROR(expr) do {absl::Status _status = (expr);if (!_status.ok()) return _status;} while (0)

#define RET_CUDA(x) RETURN_IF_ERROR(CudaStatus(x))

#define RET_CUBLAS(stat) do { \
    if ((stat) != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__ << std::endl; \
        return absl::InternalError("cuBLAS call failed."); \
    } \
} while (0)

absl::Status CheckHasGPU(bool print_info) {
  static absl::Status status = [&]() -> absl::Status {
    int driver_version = 0;
    RET_CUDA(cudaDriverGetVersion(&driver_version));
    if (driver_version == 0) {
      return absl::InvalidArgumentError("No matching cuda driver found");
    }
    cudaDeviceProp prop;
    RET_CUDA(cudaGetDeviceProperties(&prop, 0));
    if (print_info) {
      LOG(INFO) << "Using CUDA device: " << prop.name
                << " (driver:" << driver_version << ")";
    }
    return absl::OkStatus();
  }();
  return status;
}

//warm-up kernel
__global__ void warmup() {}

__global__ void RandomColumnGenerationKernel(int* total_col_indices,
                                             int* shuffle_buffer, 
                                             int num_cols, 
                                             int selected_features_count, 
                                             int num_proj, 
                                             unsigned long long seed) {

    int proj_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (proj_id >= num_proj) return;

    // Initialize RNG
    curandState state;
    curand_init(seed, proj_id, 0, &state);

    // Each thread gets its own slice of shared memory
    int* local_indices = shuffle_buffer + proj_id * num_cols;

    // Fill array 0..(num_cols - 1)
    for (int i = 0; i < num_cols; ++i) {
        local_indices[i] = i;
    }

    __syncthreads(); // sync before shuffle if threads cooperate

    // Shuffle the last selected_features_count elements
    for (int i = num_cols - 1; i >= num_cols - selected_features_count; --i) {
        int j = curand(&state) % (i + 1);
        int tmp = local_indices[i];
        local_indices[i] = local_indices[j];
        local_indices[j] = tmp;
    }
    __syncthreads(); // sync before writing results if threads cooperate
    // Write results to global memory
    int offset = proj_id * selected_features_count;
    for (int i = 0; i < selected_features_count; ++i) {
        total_col_indices[offset + i] = local_indices[num_cols - selected_features_count + i];
    }
}


__global__ void ColumnAddProjectionKernel(
  // [num_total_rows * num_cols]
  // [num_rows * num_cols]
  // [num_cols * num_proj]
  // [num_rows * num_proj]
    const float* __restrict__ dataset,            
    const int* __restrict__ flat_col_data,      
    float* projected,               
    int num_rows,
    int num_cols,
    int num_proj, 
    int selected_features_count) 

{
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // projection index
    int row_start = blockIdx.y * blockDim.y + threadIdx.y;

    // Stride over rows to avoid grid.y overflow
    for (int row = row_start; row < num_rows; row += gridDim.y * blockDim.y) {
        if (col < num_proj) {
            float sum = 0.0f;
            for (int i = 0; i < selected_features_count; ++i) {
                int feature_idx = flat_col_data[col * selected_features_count + i];
                if (feature_idx < num_cols) {
                    sum += dataset[feature_idx * num_rows + row];  // column-major access
                }
            }
            projected[col * num_rows + row] = sum;  // column-major output
        }
    }
}



void ApplyProjectionColumnADD (const float* flat_data,
                               float** d_col_add_projected_out,
                               float* GPU_Col_Add_values,
                               int* total_col_indices,
                               const int num_rows,
                               const int num_cols,
                               const int num_proj,
                               const int selected_features_count,
                               double& elapsed_ms,
                               bool verbose
                              ){

// Warm-up launch, The first kernel won’t be artificially inflated by setup costs
  warmup<<<1, 1>>>();
  cudaDeviceSynchronize();

  int total_dataset_size = num_rows * num_cols;
  int total_col_dataset_size = selected_features_count * num_proj;
  int result_size = num_rows * num_proj;
  std::cout << std::endl;

///////////////////////////////////////////////////Debug/////////////////////////////////////
  if (verbose) {
    std::cout << "Passed col add data from main to GPU function: " << std::endl;
    std::cout << "rows: " << num_rows << std::endl;
    std::cout << "cols: " << num_cols << std::endl;
    std::cout << "proj: " << num_proj << std::endl;
  }
  ///////////////////////////////////////////////////Debug/////////////////////////////////////

  //Allocate device memory
  float *d_flat_data = nullptr;
  float *d_col_add_projected = nullptr;
  int *d_flat_col_data = nullptr;
  int* d_shuffle_buffer = nullptr;

  cudaMalloc((void **)&d_flat_data, total_dataset_size * sizeof(float));
  cudaMalloc((void **)&d_flat_col_data, total_col_dataset_size * sizeof(int));                              
  cudaMalloc((void **)&d_col_add_projected, result_size * sizeof(float));
  cudaMalloc((void **)&d_shuffle_buffer, num_proj * num_cols * sizeof(int));

  //Copy dataset to device
  cudaMemcpy(d_flat_data, flat_data, total_dataset_size * sizeof(float), cudaMemcpyHostToDevice);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
      std::cerr << "CUDA ERROR: " << cudaGetErrorString(err) << std::endl;
  }

  int threads_per_block = 256;
  int num_blocks = (num_proj + threads_per_block - 1) / threads_per_block;

  unsigned long long seed = static_cast<unsigned long long>(time(NULL));
  RandomColumnGenerationKernel<<<num_blocks, threads_per_block>>>(d_flat_col_data,
                                                d_shuffle_buffer,
                                                num_cols, 
                                                selected_features_count,
                                                num_proj, 
                                                seed);
  cudaDeviceSynchronize();  // Ensure kernel finishes

  //Copy the generated column data from device to host
  cudaMemcpy(total_col_indices, d_flat_col_data, num_proj * selected_features_count * sizeof(int), cudaMemcpyDeviceToHost);
  //cudaMemcpy(d_flat_col_data, total_col_indices, num_proj * selected_features_count * sizeof(int), cudaMemcpyHostToDevice);  

    // Launch CUDA kernel
  int block_size_x = 16;
  int block_size_y = 16;
  // //int threads_per_block = block_size_x * block_size_y;
  dim3 block_size(block_size_x, block_size_y); 
  dim3 grid_size((num_proj + block_size_x - 1) / block_size_x,
          min((num_rows + block_size_y - 1) / block_size_y, 32768)); 

  auto startA = std::chrono::high_resolution_clock::now();
  ColumnAddProjectionKernel<<<grid_size, block_size>>>(d_flat_data,
                                                       d_flat_col_data,   
                                                       d_col_add_projected,
                                                       num_rows, 
                                                       num_cols,
                                                       num_proj,
                                                       selected_features_count);
  cudaDeviceSynchronize();  // Ensure kernel finishes
  auto endA = std::chrono::high_resolution_clock::now();
  elapsed_ms = std::chrono::duration<double, std::milli>(endA - startA).count();

  *d_col_add_projected_out = d_col_add_projected;  // <-- pass it back

  cudaMemcpy(GPU_Col_Add_values, d_col_add_projected, num_rows * num_proj * sizeof(float), cudaMemcpyDeviceToHost);

//For debugging

  cudaPeekAtLastError();
  // Free device memory
  cudaFree(d_flat_data);
  cudaFree(d_flat_col_data);
  //cudaFree(d_col_add_projected); Don't free here if this is not the end.
  cudaFree(d_shuffle_buffer);
}
