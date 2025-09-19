#pragma once
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <type_traits>

namespace Utils {

struct RandomConfig
{
    int              minValue     = 0;                // lower bound (inclusive)
    int              maxValue     = 100;              // upper bound (inclusive)
    std::size_t      count        = 10;               // how many numbers to pick
    bool             unique       = false;            // require uniqueness?
    std::optional<unsigned> seed;                     // fixed seed for consistency
};

/**
 * Returns a vector of random integers according to cfg.
 * Throws std::invalid_argument if 'unique' is requested but impossible
 * (e.g., want 10 unique numbers from the range [0,5]).
 */
std::vector<int> generateRandom(const RandomConfig& cfg)
{
    if (cfg.minValue > cfg.maxValue)
        throw std::invalid_argument("minValue must not exceed maxValue");

    const std::size_t rangeSize =
        static_cast<std::size_t>(cfg.maxValue) - cfg.minValue + 1;

    if (cfg.unique && cfg.count > rangeSize)
        throw std::invalid_argument(
            "Cannot pick the requested number of UNIQUE values from given range");

    /*--------------------------------------------------------------
     * 1. Create and seed engine
     *------------------------------------------------------------*/
    std::mt19937 engine;

    if (cfg.seed.has_value())
    {
        engine.seed(cfg.seed.value());            // deterministic
    }
    else
    {
        // Use high-resolution clock as entropy source
        engine.seed(static_cast<unsigned>(
            std::chrono::high_resolution_clock::now().time_since_epoch().count()));
    }

    /*--------------------------------------------------------------
     * 2. Pick numbers
     *------------------------------------------------------------*/
    std::vector<int> result;
    result.reserve(cfg.count);

    std::uniform_int_distribution<int> dist(cfg.minValue, cfg.maxValue);

    if (!cfg.unique)
    {
        // Non-unique: simple loop
        for (std::size_t i = 0; i < cfg.count; ++i)
            result.push_back(dist(engine));
    }
    else
    {
        // Unique: easiest is shuffle-and-pick
        std::vector<int> pool(rangeSize);
        std::iota(pool.begin(), pool.end(), cfg.minValue); // fill with min..max
        std::shuffle(pool.begin(), pool.end(), engine);

        result.assign(pool.begin(), pool.begin() + static_cast<std::ptrdiff_t>(cfg.count));
    }

    return result;
}

    template <typename DataT, typename LabelT>
    struct CSVData {
        std::vector<DataT> flattened;
        std::vector<LabelT> labels;
        size_t num_rows;
        size_t num_cols;
    };

    template <typename DataT, typename LabelT>
    CSVData<DataT, LabelT> flattenCSVColumnMajorWithLabels(const std::string& filename) {
        static_assert(std::is_arithmetic<DataT>::value, "CSVData type must be numeric");

        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Failed to open file: " << filename << std::endl;
            return {{}, {}, 0, 0};
        }

        std::vector<std::vector<DataT>> rows;
        std::string line;

        // Skip header
        std::getline(file, line);

        while (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string cell;
            std::vector<DataT> row;

            while (std::getline(ss, cell, ',')) {
                try {
                    if constexpr (std::is_same<DataT, int>::value) {
                        row.push_back(std::stoi(cell));
                    } else if constexpr (std::is_same<DataT, float>::value) {
                        row.push_back(std::stof(cell));
                    } else if constexpr (std::is_same<DataT, double>::value) {
                        row.push_back(std::stod(cell));
                    } else {
                        row.push_back(static_cast<DataT>(std::stod(cell))); // fallback
                    }
                } catch (const std::invalid_argument& e) {
                    std::cerr << "Invalid value: " << cell << std::endl;
                    row.push_back(static_cast<DataT>(0)); // fallback
                }
            }

            rows.push_back(row);
        }

        file.close();

        if (rows.empty()) {
            std::cerr << "CSV is empty.\n";
            return {{}, {}, 0, 0};
        }

        size_t num_rows = rows.size();
        size_t num_cols = rows[0].size();

        std::vector<DataT> flattened;
        flattened.reserve(num_rows * (num_cols - 1));

        std::vector<LabelT> labels;
        labels.reserve(num_rows);

        for (size_t col = 0; col < num_cols - 1; ++col) {
            for (size_t row = 0; row < num_rows; ++row) {
                flattened.push_back(rows[row][col]);
            }
        }


        for (size_t row = 0; row < num_rows; ++row) {
            labels.push_back(rows[row][num_cols - 1]); // last column
        }


        return {flattened, labels, num_rows, num_cols - 1};
    }

} // namespace Utils
