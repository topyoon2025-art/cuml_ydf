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
                              );

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
                        bool verbose
                       );