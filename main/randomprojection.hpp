// If num_cols is large, use global memory instead of shared memory for base_indices.
// Each thread performs its own shuffle.
// You can store the output total_col_indices on the device, then copy back to host as needed.
// If this needs to run repeatedly, set up curandState once per thread for reuse.
#pragma once
#include <cuda_runtime.h>
#include <vector>
#include "absl/status/status.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset_io.h"
#include "yggdrasil_decision_forests/dataset/data_spec.h"
#include "yggdrasil_decision_forests/dataset/types.h"
#include "yggdrasil_decision_forests/dataset/data_spec_inference.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"

absl::Status CheckHasGPU(bool print_info);

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
                              );


//extern float* d_col_add_projected;