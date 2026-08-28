#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <omp.h>

#if defined(_WIN32)
#define XRDMC_EXPORT __declspec(dllexport)
#else
#define XRDMC_EXPORT __attribute__((visibility("default")))
#endif

namespace {

constexpr int kAbiVersion = 1;

void write_error(char* buffer, std::size_t buffer_size, const std::string& message) {
    if (buffer == nullptr || buffer_size == 0) {
        return;
    }
    const std::size_t count = std::min(buffer_size - 1, message.size());
    std::memcpy(buffer, message.data(), count);
    buffer[count] = '\0';
}

std::size_t checked_product(std::size_t left, std::size_t right, const char* name) {
    if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
        throw std::invalid_argument(std::string(name) + " size overflow");
    }
    return left * right;
}

std::uint64_t splitmix64(std::uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31U);
}

std::uint64_t profile_seed(
    std::uint64_t seed,
    std::size_t scale_index,
    std::size_t draw_index,
    std::size_t measurement_index
) {
    std::uint64_t value = splitmix64(seed);
    value ^= splitmix64(static_cast<std::uint64_t>(scale_index));
    value ^= splitmix64(static_cast<std::uint64_t>(draw_index) + 0x100000001b3ULL);
    value ^= splitmix64(static_cast<std::uint64_t>(measurement_index) + 0x517cc1b727220a95ULL);
    return splitmix64(value);
}

double median_in_place(double* values, std::size_t count) {
    const std::size_t middle = count / 2;
    std::nth_element(values, values + middle, values + count);
    const double upper = values[middle];
    if ((count & 1U) != 0U) {
        return upper;
    }
    const double lower = *std::max_element(values, values + middle);
    return 0.5 * (lower + upper);
}

void validate_inputs(
    const double* images,
    std::size_t measurements,
    std::size_t pixels,
    const double* scales,
    std::size_t scale_count,
    std::size_t draws,
    const std::int64_t* csc_indptr,
    const std::int32_t* csc_indices,
    const double* csc_weights,
    std::size_t nonzero_count,
    const double* denominators,
    const double* q_grid,
    std::size_t bins,
    double q_normalization_min,
    double q_normalization_max,
    int threads,
    double* output
) {
    if (images == nullptr || scales == nullptr || csc_indptr == nullptr ||
        csc_indices == nullptr || csc_weights == nullptr || denominators == nullptr ||
        q_grid == nullptr || output == nullptr) {
        throw std::invalid_argument("null input pointer");
    }
    if (measurements == 0 || pixels == 0 || scale_count == 0 || draws == 0 || bins == 0) {
        throw std::invalid_argument("all dimensions must be positive");
    }
    if (bins > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument("bin count exceeds int32 CSC index range");
    }
    if (threads <= 0 || threads > 1024) {
        throw std::invalid_argument("threads must be in [1, 1024]");
    }
    if (!std::isfinite(q_normalization_min) || !std::isfinite(q_normalization_max) ||
        q_normalization_min > q_normalization_max) {
        throw std::invalid_argument("invalid q-normalization band");
    }

    const std::size_t image_values = checked_product(measurements, pixels, "images");
    double maximum_positive = 0.0;
    for (std::size_t index = 0; index < image_values; ++index) {
        if (!std::isfinite(images[index])) {
            throw std::invalid_argument("images must contain only finite values");
        }
        maximum_positive = std::max(maximum_positive, images[index]);
    }
    double minimum_scale_squared = std::numeric_limits<double>::infinity();
    for (std::size_t index = 0; index < scale_count; ++index) {
        if (!std::isfinite(scales[index]) || scales[index] <= 0.0) {
            throw std::invalid_argument("noise scales must be finite and positive");
        }
        const double scale_squared = scales[index] * scales[index];
        if (!std::isfinite(scale_squared) || scale_squared == 0.0) {
            throw std::invalid_argument("squared noise scale must be finite and positive");
        }
        minimum_scale_squared = std::min(minimum_scale_squared, scale_squared);
    }
    const double maximum_lambda = maximum_positive / minimum_scale_squared;
    const double poisson_limit =
        static_cast<double>(std::numeric_limits<std::uint64_t>::max() / 4U);
    if (!std::isfinite(maximum_lambda) || maximum_lambda > poisson_limit) {
        throw std::invalid_argument("Poisson rate exceeds native uint64 range");
    }
    if (csc_indptr[0] != 0 || csc_indptr[pixels] < 0 ||
        static_cast<std::size_t>(csc_indptr[pixels]) != nonzero_count) {
        throw std::invalid_argument("CSC indptr endpoints do not match nonzero count");
    }
    for (std::size_t pixel = 0; pixel < pixels; ++pixel) {
        if (csc_indptr[pixel] < 0 || csc_indptr[pixel + 1] < csc_indptr[pixel] ||
            static_cast<std::size_t>(csc_indptr[pixel + 1]) > nonzero_count) {
            throw std::invalid_argument("CSC indptr must be monotonic and in range");
        }
    }
    for (std::size_t index = 0; index < nonzero_count; ++index) {
        if (csc_indices[index] < 0 || static_cast<std::size_t>(csc_indices[index]) >= bins) {
            throw std::invalid_argument("CSC bin index out of range");
        }
        if (!std::isfinite(csc_weights[index])) {
            throw std::invalid_argument("CSC weights must be finite");
        }
    }
    bool normalization_bin_found = false;
    for (std::size_t bin = 0; bin < bins; ++bin) {
        if (!std::isfinite(denominators[bin]) || denominators[bin] <= 0.0) {
            throw std::invalid_argument("normalization denominators must be finite and positive");
        }
        if (!std::isfinite(q_grid[bin])) {
            throw std::invalid_argument("q grid must be finite");
        }
        if (bin > 0 && q_grid[bin] <= q_grid[bin - 1]) {
            throw std::invalid_argument("q grid must be strictly increasing");
        }
        normalization_bin_found = normalization_bin_found ||
            (q_grid[bin] >= q_normalization_min && q_grid[bin] <= q_normalization_max);
    }
    if (!normalization_bin_found) {
        throw std::invalid_argument("q-normalization band contains no q-grid bins");
    }

    std::size_t profile_count = checked_product(scale_count, draws, "profiles");
    profile_count = checked_product(profile_count, measurements, "profiles");
    checked_product(profile_count, bins, "output");
    if (profile_count > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
        throw std::invalid_argument("profile count exceeds OpenMP loop range");
    }
}

void run_direct_monte_carlo(
    const double* images,
    std::size_t measurements,
    std::size_t pixels,
    const double* scales,
    std::size_t scale_count,
    std::size_t draws,
    const std::int64_t* csc_indptr,
    const std::int32_t* csc_indices,
    const double* csc_weights,
    std::size_t nonzero_count,
    const double* denominators,
    const double* q_grid,
    std::size_t bins,
    double q_normalization_min,
    double q_normalization_max,
    std::uint64_t seed,
    int threads,
    double* output
) {
    validate_inputs(
        images,
        measurements,
        pixels,
        scales,
        scale_count,
        draws,
        csc_indptr,
        csc_indices,
        csc_weights,
        nonzero_count,
        denominators,
        q_grid,
        bins,
        q_normalization_min,
        q_normalization_max,
        threads,
        output
    );

    std::vector<std::size_t> normalization_bins;
    normalization_bins.reserve(bins);
    for (std::size_t bin = 0; bin < bins; ++bin) {
        if (q_grid[bin] >= q_normalization_min && q_grid[bin] <= q_normalization_max) {
            normalization_bins.push_back(bin);
        }
    }

    const std::size_t profile_count = scale_count * draws * measurements;
    const std::size_t workspace_per_thread = bins + normalization_bins.size();
    std::vector<double> workspace(
        checked_product(static_cast<std::size_t>(threads), workspace_per_thread, "workspace"),
        0.0
    );
    std::atomic<std::int64_t> failed_profile{-1};

#pragma omp parallel num_threads(threads)
    {
        const int thread_index = omp_get_thread_num();
        double* sums = workspace.data() +
            static_cast<std::size_t>(thread_index) * workspace_per_thread;
        double* normalization_values = sums + bins;
        std::poisson_distribution<std::uint64_t> poisson;

#pragma omp for schedule(static)
        for (std::int64_t profile = 0;
             profile < static_cast<std::int64_t>(profile_count);
             ++profile) {
            if (failed_profile.load(std::memory_order_relaxed) >= 0) {
                continue;
            }
            std::fill(sums, sums + bins, 0.0);

            const std::size_t profile_index = static_cast<std::size_t>(profile);
            const std::size_t measurement_index = profile_index % measurements;
            const std::size_t draw_index = (profile_index / measurements) % draws;
            const std::size_t scale_index = profile_index / (measurements * draws);
            const double scale = scales[scale_index];
            const double scale_squared = scale * scale;
            const double* image = images + measurement_index * pixels;
            std::mt19937_64 rng(profile_seed(seed, scale_index, draw_index, measurement_index));

            for (std::size_t pixel = 0; pixel < pixels; ++pixel) {
                const std::int64_t begin = csc_indptr[pixel];
                const std::int64_t end = csc_indptr[pixel + 1];
                if (begin == end) {
                    continue;
                }

                const double value = image[pixel];
                const double positive = std::max(value, 0.0);
                const double negative_component = value - positive;
                double sampled = negative_component;
                if (positive > 0.0) {
                    const double lambda = positive / scale_squared;
                    const auto count = poisson(
                        rng,
                        std::poisson_distribution<std::uint64_t>::param_type(lambda)
                    );
                    sampled += scale_squared * static_cast<double>(count);
                }

                for (std::int64_t offset = begin; offset < end; ++offset) {
                    const std::size_t bin = static_cast<std::size_t>(csc_indices[offset]);
                    sums[bin] += csc_weights[offset] * sampled;
                }
            }

            double* profile_output = output + profile_index * bins;
            bool valid = true;
            for (std::size_t bin = 0; bin < bins; ++bin) {
                profile_output[bin] = sums[bin] / denominators[bin];
                valid = valid && std::isfinite(profile_output[bin]);
            }
            for (std::size_t index = 0; index < normalization_bins.size(); ++index) {
                normalization_values[index] = profile_output[normalization_bins[index]];
            }
            const double normalization = median_in_place(
                normalization_values,
                normalization_bins.size()
            );
            valid = valid && std::isfinite(normalization) && normalization > 1e-12;
            if (!valid) {
                std::int64_t expected = -1;
                failed_profile.compare_exchange_strong(expected, profile);
                continue;
            }
            for (std::size_t bin = 0; bin < bins; ++bin) {
                profile_output[bin] /= normalization;
            }
        }
    }

    const std::int64_t failure = failed_profile.load();
    if (failure >= 0) {
        throw std::runtime_error(
            "non-finite or zero profile normalization at flattened profile index " +
            std::to_string(failure)
        );
    }
}

void integrate_detector_frames(
    const double* images,
    std::size_t measurements,
    std::size_t pixels,
    const std::int64_t* csc_indptr,
    const std::int32_t* csc_indices,
    const double* csc_weights,
    std::size_t nonzero_count,
    const double* denominators,
    const double* q_grid,
    std::size_t bins,
    double q_normalization_min,
    double q_normalization_max,
    int threads,
    double* output
) {
    const double unit_scale = 1.0;
    validate_inputs(
        images,
        measurements,
        pixels,
        &unit_scale,
        1,
        1,
        csc_indptr,
        csc_indices,
        csc_weights,
        nonzero_count,
        denominators,
        q_grid,
        bins,
        q_normalization_min,
        q_normalization_max,
        threads,
        output
    );

    std::vector<std::size_t> normalization_bins;
    normalization_bins.reserve(bins);
    for (std::size_t bin = 0; bin < bins; ++bin) {
        if (q_grid[bin] >= q_normalization_min && q_grid[bin] <= q_normalization_max) {
            normalization_bins.push_back(bin);
        }
    }

    const std::size_t workspace_per_thread = bins + normalization_bins.size();
    std::vector<double> workspace(
        checked_product(static_cast<std::size_t>(threads), workspace_per_thread, "workspace"),
        0.0
    );
    std::atomic<std::int64_t> failed_measurement{-1};

#pragma omp parallel num_threads(threads)
    {
        const int thread_index = omp_get_thread_num();
        double* sums = workspace.data() +
            static_cast<std::size_t>(thread_index) * workspace_per_thread;
        double* normalization_values = sums + bins;

#pragma omp for schedule(static)
        for (std::int64_t measurement = 0;
             measurement < static_cast<std::int64_t>(measurements);
             ++measurement) {
            std::fill(sums, sums + bins, 0.0);
            const double* image = images + static_cast<std::size_t>(measurement) * pixels;
            for (std::size_t pixel = 0; pixel < pixels; ++pixel) {
                const double value = image[pixel];
                for (std::int64_t offset = csc_indptr[pixel];
                     offset < csc_indptr[pixel + 1];
                     ++offset) {
                    const std::size_t bin = static_cast<std::size_t>(csc_indices[offset]);
                    sums[bin] += csc_weights[offset] * value;
                }
            }

            double* profile_output = output + static_cast<std::size_t>(measurement) * bins;
            bool valid = true;
            for (std::size_t bin = 0; bin < bins; ++bin) {
                profile_output[bin] = sums[bin] / denominators[bin];
                valid = valid && std::isfinite(profile_output[bin]);
            }
            for (std::size_t index = 0; index < normalization_bins.size(); ++index) {
                normalization_values[index] = profile_output[normalization_bins[index]];
            }
            const double normalization = median_in_place(
                normalization_values,
                normalization_bins.size()
            );
            valid = valid && std::isfinite(normalization) && normalization > 1e-12;
            if (!valid) {
                std::int64_t expected = -1;
                failed_measurement.compare_exchange_strong(expected, measurement);
                continue;
            }
            for (std::size_t bin = 0; bin < bins; ++bin) {
                profile_output[bin] /= normalization;
            }
        }
    }

    const std::int64_t failure = failed_measurement.load();
    if (failure >= 0) {
        throw std::runtime_error(
            "non-finite or non-positive profile normalization at measurement index " +
            std::to_string(failure)
        );
    }
}

}  // namespace

extern "C" {

XRDMC_EXPORT int xrdmc_abi_version() noexcept {
    return kAbiVersion;
}

XRDMC_EXPORT int xrdmc_run(
    const double* images,
    std::size_t measurements,
    std::size_t pixels,
    const double* scales,
    std::size_t scale_count,
    std::size_t draws,
    const std::int64_t* csc_indptr,
    const std::int32_t* csc_indices,
    const double* csc_weights,
    std::size_t nonzero_count,
    const double* denominators,
    const double* q_grid,
    std::size_t bins,
    double q_normalization_min,
    double q_normalization_max,
    std::uint64_t seed,
    int threads,
    double* output,
    char* error_buffer,
    std::size_t error_buffer_size
) noexcept {
    try {
        write_error(error_buffer, error_buffer_size, "");
        run_direct_monte_carlo(
            images,
            measurements,
            pixels,
            scales,
            scale_count,
            draws,
            csc_indptr,
            csc_indices,
            csc_weights,
            nonzero_count,
            denominators,
            q_grid,
            bins,
            q_normalization_min,
            q_normalization_max,
            seed,
            threads,
            output
        );
        return 0;
    } catch (const std::exception& error) {
        write_error(error_buffer, error_buffer_size, error.what());
        return 1;
    } catch (...) {
        write_error(error_buffer, error_buffer_size, "unknown native exception");
        return 2;
    }
}

XRDMC_EXPORT int xrdmc_integrate(
    const double* images,
    std::size_t measurements,
    std::size_t pixels,
    const std::int64_t* csc_indptr,
    const std::int32_t* csc_indices,
    const double* csc_weights,
    std::size_t nonzero_count,
    const double* denominators,
    const double* q_grid,
    std::size_t bins,
    double q_normalization_min,
    double q_normalization_max,
    int threads,
    double* output,
    char* error_buffer,
    std::size_t error_buffer_size
) noexcept {
    try {
        write_error(error_buffer, error_buffer_size, "");
        integrate_detector_frames(
            images,
            measurements,
            pixels,
            csc_indptr,
            csc_indices,
            csc_weights,
            nonzero_count,
            denominators,
            q_grid,
            bins,
            q_normalization_min,
            q_normalization_max,
            threads,
            output
        );
        return 0;
    } catch (const std::exception& error) {
        write_error(error_buffer, error_buffer_size, error.what());
        return 1;
    } catch (...) {
        write_error(error_buffer, error_buffer_size, "unknown native exception");
        return 2;
    }
}

}  // extern "C"
