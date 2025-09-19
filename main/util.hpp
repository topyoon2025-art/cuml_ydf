void Transpose1DMatrixInt(const int* input, int* output, int rows, int cols);
void Transpose1DMatrixFloat(const float* input, float* output, int rows, int cols);

void generate_column_matrix(
    const int* column_generated,   // [num_proj × selected_features_count]
    int num_proj,
    int selected_features_count,
    int num_cols,
    int* col_mat);                   // [num_proj × num_cols], output matrix
void transpose_column_matrix(
    const int* col_mat,             // [num_proj × num_cols], input matrix
    int num_proj,
    int num_cols,
    int* transposed_col_mat         // [num_cols × num_proj], output matrix
);
void PrintGroupedBinaryHistogram2D(
    const int* h_hist_class0,       // [num_proj × num_bins]
    const int* h_hist_class1,       // [num_proj × num_bins]
    const float* h_min_vals,        // [num_proj]
    const float* h_max_vals,        // [num_proj]
    const float* h_bin_widths,      // [num_proj]
    int num_proj,
    int num_bins
);
void print_gini_values(const float* h_gini_values, int num_proj, int num_bins);