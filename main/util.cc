#include <iostream>
#include <iomanip>
#include <cassert>
#include <algorithm>
#include <chrono>
#include <optional>
#include <random>
#include <set>
#include <stdexcept>
#include <vector>


//- input[r][c] → output[c][r]
void Transpose1DMatrixInt(const int* input, int* output, int rows, int cols) {
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            output[c * rows + r] = input[r * cols + c];
        }
    }
}

//- input[r][c] → output[c][r]
void Transpose1DMatrixFloat(const float* input, float* output, int rows, int cols) {
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            output[c * rows + r] = input[r * cols + c];
        }
    }
}

void generate_column_matrix(
    const int* column_generated,   // [num_proj × selected_features_count]
    int num_proj,
    int selected_features_count,
    int num_cols,
    int* col_mat                   // [num_proj × num_cols], output matrix
) {
    // Initialize all to zero
    std::fill(col_mat, col_mat + num_proj * num_cols, 0);

    // Fill in 1's where features are selected
    for (int i = 0; i < num_proj; ++i) {  // projection index (row)
        for (int j = 0; j < selected_features_count; ++j) {
            int val = column_generated[i * selected_features_count + j];
            if (val >= 0 && val < num_cols) {
                // Column-major: column index first, then row index
                col_mat[val * num_proj + i] = 1;
            }
        }
    }
}

void transpose_column_matrix(
    const int* col_mat,             // [num_proj × num_cols], input matrix
    int num_proj,
    int num_cols,
    int* transposed_col_mat         // [num_cols × num_proj], output matrix
) {
    for (int i = 0; i < num_proj; ++i) {
        for (int j = 0; j < num_cols; ++j) {
            transposed_col_mat[j * num_proj + i] = col_mat[i * num_cols + j];
        }
    }
}


#include <vector>
#include <cfloat>


void PrintGroupedBinaryHistogram2D(
    const int* h_hist_class0,       // [num_proj × num_bins]
    const int* h_hist_class1,       // [num_proj × num_bins]
    const float* h_min_vals,        // [num_proj]
    const float* h_max_vals,        // [num_proj]
    const float* h_bin_widths,      // [num_proj]
    int num_proj,
    int num_bins
) {
    std::cout << "📊 Grouped Binary Histogram (per column):\n";

    for (int col = 0; col < num_proj; ++col) {
        std::cout << "Column " << col << ":\n";

        float min_val   = h_min_vals[col];
        float max_val   = h_max_vals[col];
        float bin_width = h_bin_widths[col];

        for (int bin = 0; bin < num_bins; ++bin) {
            int idx = col * num_bins + bin;
            assert(idx < num_proj * num_bins);  // Defensive check
            float bin_start = min_val + bin * bin_width;
            float bin_end   = bin_start + bin_width;

            std::cout << "  Bin [" << std::fixed << std::setprecision(6)
                      << bin_start << ", " << bin_end << "): "
                      << "Class 0 = " << h_hist_class0[idx]
                      << ", Class 1 = " << h_hist_class1[idx] << "\n";
        }

        std::cout << std::endl;
    }
}


void print_gini_values(const float* h_gini_values, int num_proj, int num_bins) {
    // Print with formatting

    for (int col = 0; col < num_proj; ++col) {
        std::cout << "Column " << col << " Gini values:\n";
        for (int bin = 0; bin < num_bins; ++bin) {
            int idx = col * num_bins + bin;
            std::cout << "  Bin " << std::setw(2) << bin
                      << ": Gini = " << std::fixed << std::setprecision(4)
                      << h_gini_values[idx] << "\n";
        }
        std::cout << std::endl;
    }
}

