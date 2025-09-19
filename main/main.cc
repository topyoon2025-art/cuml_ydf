#include "randomprojection.hpp"
#include "buildhistogramginisplits.hpp"
#include "util.hpp"
#include "utils.hpp"
#include "absl/status/status.h"
#include "absl/log/log.h"  // If you're using LOG(INFO)
#include "absl/types/span.h"
#include "yggdrasil_decision_forests/learner/decision_tree/oblique.h"
#include <google/protobuf/repeated_field.h>
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset_io.h"
#include "yggdrasil_decision_forests/dataset/data_spec.h"
#include "yggdrasil_decision_forests/dataset/types.h"
#include "yggdrasil_decision_forests/dataset/data_spec_inference.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/utils/histogram.h"//histogram
#include "yggdrasil_decision_forests/learner/decision_tree/training.h"

#include <stdio.h>
#include <cstdint>
#include <iostream>
#include <vector>
#include <cstdlib>   
#include <ctime>    
#include <random>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <iomanip>
#include <chrono>
#include <fstream>
#include <cstdint>
#include <cassert>
#include <cfloat>

int main (int argc, char* argv[]) {

    //Check GPU status
    absl::Status gpu_status = CheckHasGPU(true);
    if (!gpu_status.ok()) {
        std::cerr << "GPU Check failed: " << gpu_status.message() << std::endl;
        return 1;
    }
    std::cout << "GPU check passed.\n";
    //Get k, number of randomly chosen column indices
    if (argc != 7) {
        std::cerr << "Usage: " << argv[0] << " <path> <selected_features_count> <num_proj> <max_n_bins> <col_gen_seed> <verbose>" << std::endl;
        return 1;
    }

    //First argument: File Path
    const std::string path = argv[1]; 

    //Second argument: Selected Features Count
    const int selected_features_count = std::atoi(argv[2]);// K number, selected number of features.

    //Third argument: Number of Projections
    const int num_proj = std::atoi(argv[3]);

    //Fourth argument: Verbose for debug information
    const int num_bins = std::atoi(argv[4]);

    const int col_gen_seed = std::atoi(argv[5]);

    //Fifth argument: Verbose for debug information
    bool verbose = false;
    if (std::string(argv[6]) == "true"){
        verbose = true;
    }

    //Load dataset
    auto dataset = Utils::flattenCSVColumnMajorWithLabels<float, int>(path);

    int num_rows = dataset.num_rows;
    int num_cols = dataset.num_cols;
    std::cout << std::endl;
    std::cout << "Dataset number of rows: " << num_rows << std::endl;
    std::cout << "Dataset number of columns: " << num_cols << std::endl;

    //Create flat dataset for CUDA, column major flattened 
    std::vector<float> flat_data = dataset.flattened;
    
    //Create labels for samples to build histograms
    std::vector<int> h_labels = dataset.labels;

    Utils::RandomConfig cfg;
    cfg.minValue = 0;
    cfg.maxValue = num_cols - 1;
    cfg.count    = selected_features_count * num_proj; // total number of indices needed
    cfg.unique   = true;
    cfg.seed     = col_gen_seed;        // comment this line → fresh set every run

    auto numbers = Utils::generateRandom(cfg);

    std::cout << "Random picks:";
    for (int n : numbers) std::cout << ' ' << n;
    std::cout << '\n';
    
   

////////////////////////////////////////////////////////////ColAdd//////////////////////////////////////////
    double elapsed_projection_ms = 0.0;
    int total_col_size = num_proj * selected_features_count;
    std::vector<float> GPU_Col_Add_values(num_rows * num_proj);
    std::vector<int> total_col_indices(total_col_size);

    //////////////////////////////////////////////////////debug
    // Copy generated random indices to total_col_indices
    std::copy(numbers.begin(), numbers.end(), total_col_indices.begin());
    //////////////////////////////////////////////////////debug

    float* d_col_add_projected = nullptr;
    ApplyProjectionColumnADD (flat_data.data(),
                              &d_col_add_projected,
                              GPU_Col_Add_values.data(),
                              total_col_indices.data(),
                              num_rows,
                              num_cols,
                              num_proj,
                              selected_features_count,
                              elapsed_projection_ms,
                              verbose);
    if (d_col_add_projected == nullptr) {
        std::cerr << "Error: d_col_add_projected is null." << std::endl;
        return 1;
    }

    float* h_buffer = new float[5];
    cudaMemcpy(h_buffer, d_col_add_projected, 5 * sizeof(float), cudaMemcpyDeviceToHost);
    // printf("d_col_add_projected values: ");
    // for (int i = 0; i < 5; ++i) {
    //     printf("%f, ", h_buffer[i]);
    // }

    std::cout << "Generated column indices: " << std::endl; 
        for (float value : total_col_indices) {
        std::cout << value << " "; 
        }
    ////////////////////////////////////////////////////////////ColAdddata////////////////////////////////////////
    if (verbose) {
        // printf("Label is: ");
        // for (int value : h_labels){  
        //     printf("%d, ", value);
        // }
        // printf("\n");

        printf("d_col_add_projected address: %p \n", (void*)d_col_add_projected);
        int N = /* number of elements in d_col_add_projected */ num_rows * num_proj;

        // Allocate host buffer
        float* h_buffer = new float[N];

        // Copy from device to host
        cudaMemcpy(h_buffer, d_col_add_projected, N * sizeof(float), cudaMemcpyDeviceToHost);

        // // Print values
        // printf("d_col_add_projected values: ");
        // for (int i = 0; i < N; ++i) {
        //     printf("%f, ", h_buffer[i]);
        // }
        // std::cout << "Generated column indices: " << std::endl; 
        // for (float value : total_col_indices) {
        // std::cout << value << " "; 
        // }

        // for (int i = 0; i < N; ++i) {
        //     printf("d_col_add_projected[%d] = %f\n", i, h_buffer[i]);
        // }

        // Clean up
        delete[] h_buffer;
        // std::cout << "GPU Col Add Values: " << std::endl; 
        // for (float value : GPU_Col_Add_values) {
        // std::cout << value << " "; 
        // }
        std::cout << std::endl;
    }
 
    std::cout << std::endl;
    std::cout << "Kernel Apply Projection time elapsed: " << elapsed_projection_ms << " ms " << std::endl;
    ////////////////////////////////////////////////////////////ColAdddata////////////////////////////////////////
    
    

    int* d_hist_class0 = nullptr;
    int* d_hist_class1 = nullptr;
    float* d_min_vals = nullptr;
    float* d_max_vals = nullptr;   
    float* d_bin_widths = nullptr;
    double elapsed_histogram_ms = 0.0;
    BuildExactWidthHistogram (d_col_add_projected,
                              &d_hist_class0,
                              &d_hist_class1,
                              &d_min_vals,
                              &d_max_vals,
                              &d_bin_widths,
                              h_labels,
                              num_rows,
                              num_proj,
                              num_bins,
                              elapsed_histogram_ms,
                              verbose);
    std::cout << "Kernel Build histogram time elapsed: " << elapsed_histogram_ms << " ms " << std::endl;

    std::vector<int> h_best_bin_out(num_proj);
    std::vector<float> h_best_threshold_out(num_proj);
    std::vector<float> h_best_gini_gain_out(num_proj);
    int best_proj = -1;
    double elapsed_gini_ms = 0.0;
    FindBestGiniSplit (d_hist_class0,
                       d_hist_class1,
                       d_min_vals,
                       d_bin_widths,
                       num_proj,
                       num_bins,
                       &best_proj,
                       h_best_bin_out.data(),
                       h_best_gini_gain_out.data(),
                       h_best_threshold_out.data(),
                       elapsed_gini_ms,
                       verbose);

    std::cout << "Kernel Best Gini split time elapsed: " << elapsed_gini_ms << " ms " << std::endl;
    std::cout << "Gini + Histogram time elapsed: " << (elapsed_histogram_ms + elapsed_gini_ms) << " ms " << std::endl;
    std::cout << "CUDAfied Total time elapsed: " << (elapsed_projection_ms + elapsed_histogram_ms + elapsed_gini_ms) << " ms " << std::endl;
    
    if (best_proj != -1) {
        printf("Kernel Best Projection: %d\n", best_proj);
        printf("Kernel Best Gain: %f\n", h_best_gini_gain_out[best_proj]);
        printf("Kernel Best Threshold: %f\n", h_best_threshold_out[best_proj]);
        printf("Kernel Best Bin index: %d\n", h_best_bin_out[best_proj]);    
    } else {
        printf("No valid projection found.\n");
    }



    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////CPU Orginal Dataset load////////////////
    std::string prefix = "csv:../../nl_";

    std::string ydf_path = prefix + path; //Path to load dataset
    //../../dataset/50000000x22.csv
    ydf_path = "csv:../../dataset/nl_50000000x22.csv";


    



    // Load dataset to test, do this only once to get the experiment time
    yggdrasil_decision_forests::dataset::VerticalDataset train_dataset;
    yggdrasil_decision_forests::dataset::proto::DataSpecification data_spec;
    yggdrasil_decision_forests::dataset::LoadConfig config;
    yggdrasil_decision_forests::dataset::proto::DataSpecificationGuide guide;
    const bool use_flume = false;
    //Create data_spec
    absl::Status spec_status = yggdrasil_decision_forests::dataset::CreateDataSpecWithStatus(ydf_path, use_flume, guide, &data_spec);
    absl::Status load_status = yggdrasil_decision_forests::dataset::LoadVerticalDataset(ydf_path, data_spec, &train_dataset, {}, config);
    if (!load_status.ok()) {
    std::cerr << "Failed to load dataset: " << load_status.message() << std::endl;
    return 1;
    }

    ///////////CPU//////////////////////
    //For CPU Start
    std::vector<uint32_t> selected_indices;
    for (uint32_t i = 0; i < num_rows; ++i) {
        selected_indices.push_back(i);
    }
    absl::Span<const uint32_t> selected_examples(selected_indices);
    //For CPU End

    std::vector<float> CPU_values(num_rows);//Define here to store results from CPU
    std::vector<int> col_indices(selected_features_count); //col_indices for GPU
    float best_gain_overall = 0.0f;
    int best_bin_overall  = 0;
    int best_proj_overall = -1;
    float threshold_overall = 0.0f;
    double ydf_projection_time_elapsed = 0;
    double ydf_build_histogram_time_elapsed = 0;
    double ydf_split_time_elapsed = 0;



    for (int i = 0; i < num_proj; ++i) //Iteration to get them emperiment result, same as project number for each projection
    {
        for (int j = 0; j < selected_features_count; ++j){
            col_indices[j] = total_col_indices[i*selected_features_count + j];
            
        }

        if (verbose) {
            std::cout << "Projection " << i << ": ";
            
            std::cout << "Selected features indices: ";
            for (int idx : col_indices) {
                std::cout << idx << " ";
            }
            std::cout << std::endl;
        }
        
        google::protobuf::RepeatedField<int32_t> numerical_features;//for CPU version to run it using YDF library
        // std::cout << "Selected features indices: ";
        for (int idx : col_indices){
            numerical_features.Add(idx);//numerical_features for CPU
        }
        // Create ProjectionEvaluator for CPU
        yggdrasil_decision_forests::model::decision_tree::internal::ProjectionEvaluator evaluator(train_dataset, numerical_features);

        // Create projection class
        yggdrasil_decision_forests::model::decision_tree::internal::Projection projection;
        for (int n = 0; n < selected_features_count; ++n) {
            projection.push_back({.attribute_idx = numerical_features[n], .weight = 1.0f});
        }
 
        // Step 6: Evaluate projection for CPU
        ////////////////////////////////////////////////////////////CPU////////////////////////////////////////////////////
        auto startA = std::chrono::high_resolution_clock::now();
        absl::Status eval_cpu_status = evaluator.Evaluate(projection, absl::MakeSpan(selected_examples), &CPU_values);
        auto endA = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> durationA = endA - startA;
        ydf_projection_time_elapsed += durationA.count(); 

        if (!eval_cpu_status.ok()) {
            std::cerr << "CPU Evaluation failed: " << eval_cpu_status.message() << std::endl;
            return 1;
        }   
        
        ////////////////////////////////////////////////////////////CPU_data////////////////////////////////////////////////
        
        if (verbose) {
            std::cout << "CPU Projection[" << i << "] values: ";
            for (float value : CPU_values) {
                std::cout << value << ", ";
            }
            std::cout << std::endl;
            printf("labels: ");
            for (int value : h_labels){  
                printf("%d, ", value);
            }
            printf("\n");
        }

        
        
        ////////////////////////////////////////////////////////////CPU_data/////////////////////////////////////////////////
        

        absl::Span<const float> attribute_values = absl::MakeConstSpan(CPU_values);
        // 2. selected_examples <- good to use this variable already
        // 3. Create weights for the samples
        std::vector<float> weights = {};  // Leave empty for unweighted
        // 4. Create labels for targets or take labels previously generated

        /////////////////////////////////////////////////////////////////////////////////
        // 5. num_bins, it gets passed in as an argument
        // 6. num_label_classes
        int32_t num_label_classes = 2;
        // 7. na_replacement 
        const double na_replacement = 0;
        // 8. min_num_obs
        int32_t min_num_obs = 1;
        // 9. dt_config
        yggdrasil_decision_forests::model::decision_tree::proto::DecisionTreeTrainingConfig dt_config;
        dt_config.mutable_numerical_split()->set_num_candidates(num_bins);
        dt_config.mutable_numerical_split()->set_type(yggdrasil_decision_forests::model::decision_tree::proto::NumericalSplit::HISTOGRAM_EQUAL_WIDTH);
        //dt_config.mutable_numerical_split()->set_type(yggdrasil_decision_forests::model::decision_tree::proto::NumericalSplit::UNIFORM);
        


   
        // 10. label_distribution
        yggdrasil_decision_forests::utils::IntegerDistributionDouble label_distribution;
        label_distribution.SetNumClasses(num_label_classes);
        for (int label : h_labels) {
            label_distribution.Add(label);
        }

        // 11. attribute_idx
        int32_t attribute_idx = 0;
        // 12. Allocate a condition
        

 
        
        
        // printf("Attribute values size: %zu\n", attribute_values.size());
        // printf("Selected examples size: %zu\n", selected_examples.size());
        // printf("Labels size: %zu\n", h_labels.size());
        
        yggdrasil_decision_forests::model::decision_tree::proto::NodeCondition best_condition;
        yggdrasil_decision_forests::utils::RandomEngine random(42);
        auto startC = std::chrono::high_resolution_clock::now();
        yggdrasil_decision_forests::model::decision_tree::FindSplitLabelClassificationFeatureNumericalHistogram(
        absl::MakeConstSpan(selected_examples),
        weights,
        attribute_values,
        h_labels,
        num_label_classes,
        na_replacement,
        min_num_obs,
        dt_config,
        label_distribution,
        attribute_idx,
        &random,
        &best_condition);
        auto endC = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> durationC = endC - startC;
        ydf_split_time_elapsed += durationC.count(); 

        



        // std::cout << best_condition.DebugString() << std::endl;
        if (best_condition.split_score() > best_gain_overall) {
            best_gain_overall = best_condition.split_score();
            best_bin_overall = best_condition.na_value(); // Using na_value to store best bin index
            best_proj_overall = i;            
        }   

    }
    std::cout << std::endl;
    if (best_proj_overall != -1) {
        printf("CPU Best Projection: %d\n", best_proj_overall);
        printf("CPU Best Gain: %f\n", best_gain_overall);
        printf("CPU Best Bin index: %d\n", best_bin_overall);    
    } else {
        printf("No valid projection found.\n");
    }
    std::cout << std::endl;
    std::cout << "YDF Apply Projection time elapsed: " << ydf_projection_time_elapsed << " ms " << std::endl;
    std::cout << "YDF Gini + Histogram time elapsed: " << ydf_split_time_elapsed << " ms " << std::endl;
    std::cout << "YDF Total time elapsed: " << ydf_projection_time_elapsed + ydf_split_time_elapsed << " ms " << std::endl;
             
    std::cout << std::endl;



    return 0;
}