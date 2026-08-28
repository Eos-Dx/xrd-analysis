#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(_WIN32)
#define XRDMC_CUDA_EXPORT __declspec(dllexport)
#else
#define XRDMC_CUDA_EXPORT __attribute__((visibility("default")))
#endif

namespace {

constexpr int kAbiVersion = 1;
constexpr int kKernelSuccess = 0;
constexpr int kKernelInvalidNormalization = 1;
constexpr int kKernelNonFiniteProfile = 2;

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

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + " failed: " + cudaGetErrorString(status)
        );
    }
}

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t count) : count_(count) {
        if (count_ > 0) {
            const std::size_t bytes = checked_product(count_, sizeof(T), "device buffer");
            check_cuda(
                cudaMalloc(reinterpret_cast<void**>(&data_), bytes),
                "cudaMalloc"
            );
        }
    }

    ~DeviceBuffer() {
        if (data_ != nullptr) {
            cudaFree(data_);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T* data() {
        return data_;
    }

    const T* data() const {
        return data_;
    }

    std::size_t size() const {
        return count_;
    }

private:
    T* data_ = nullptr;
    std::size_t count_ = 0;
};

template <typename T>
void copy_to_device(DeviceBuffer<T>& destination, const T* source, const char* name) {
    if (destination.size() == 0) {
        return;
    }
    check_cuda(
        cudaMemcpy(
            destination.data(),
            source,
            destination.size() * sizeof(T),
            cudaMemcpyHostToDevice
        ),
        name
    );
}

__host__ __device__ std::uint64_t splitmix64(std::uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31U);
}

__device__ std::uint64_t pixel_seed(
    std::uint64_t base_seed,
    std::uint64_t measurement_seed,
    std::size_t scale_index,
    std::size_t draw_index,
    std::size_t measurement_index,
    std::size_t pixel_index
) {
    std::uint64_t value = splitmix64(base_seed);
    value ^= splitmix64(measurement_seed + 0xd6e8feb86659fd93ULL);
    value ^= splitmix64(static_cast<std::uint64_t>(scale_index));
    value ^= splitmix64(static_cast<std::uint64_t>(draw_index) + 0x100000001b3ULL);
    value ^= splitmix64(
        static_cast<std::uint64_t>(measurement_index) + 0x517cc1b727220a95ULL
    );
    value ^= splitmix64(
        static_cast<std::uint64_t>(pixel_index) + 0x94d049bb133111ebULL
    );
    return splitmix64(value);
}

__device__ void record_kernel_failure(
    int failure_code,
    unsigned long long profile_index,
    int* status,
    unsigned long long* failed_profile
) {
    if (atomicCAS(status, kKernelSuccess, failure_code) == kKernelSuccess) {
        *failed_profile = profile_index;
    }
}

template <bool SamplePoisson>
__global__ void direct_monte_carlo_kernel(
    const double* images,
    std::size_t measurements,
    std::size_t pixels,
    const std::uint64_t* measurement_seeds,
    const double* scales,
    std::size_t draws,
    const std::int64_t* csc_indptr,
    const std::int32_t* csc_indices,
    const double* csc_weights,
    const double* denominators,
    std::size_t bins,
    const std::int32_t* normalization_indices,
    std::size_t normalization_count,
    std::uint64_t base_seed,
    std::size_t profile_offset,
    double* output,
    int* status,
    unsigned long long* failed_profile
) {
    extern __shared__ double workspace[];
    double* sums = workspace;
    double* normalization_values = sums + bins;

    for (std::size_t bin = threadIdx.x; bin < bins; bin += blockDim.x) {
        sums[bin] = 0.0;
    }
    __syncthreads();

    const std::size_t profile_index = profile_offset + blockIdx.x;
    const std::size_t measurement_index = profile_index % measurements;
    const std::size_t draw_index = (profile_index / measurements) % draws;
    const std::size_t scale_index = profile_index / (measurements * draws);
    const double scale_squared = scales[scale_index] * scales[scale_index];
    const double* image = images + measurement_index * pixels;

    for (std::size_t pixel = threadIdx.x; pixel < pixels; pixel += blockDim.x) {
        const std::int64_t begin = csc_indptr[pixel];
        const std::int64_t end = csc_indptr[pixel + 1];
        if (begin == end) {
            continue;
        }

        const double value = image[pixel];
        double sampled = value;
        if constexpr (SamplePoisson) {
            const double positive = fmax(value, 0.0);
            const double negative = fmin(value, 0.0);
            sampled = negative;
            if (positive > 0.0) {
                curandStatePhilox4_32_10_t state;
                curand_init(
                    pixel_seed(
                        base_seed,
                        measurement_seeds[measurement_index],
                        scale_index,
                        draw_index,
                        measurement_index,
                        pixel
                    ),
                    0,
                    0,
                    &state
                );
                const double lambda = positive / scale_squared;
                sampled += scale_squared * static_cast<double>(
                    curand_poisson(&state, lambda)
                );
            }
        }

        for (std::int64_t offset = begin; offset < end; ++offset) {
            const std::size_t bin = static_cast<std::size_t>(csc_indices[offset]);
            atomicAdd(sums + bin, csc_weights[offset] * sampled);
        }
    }
    __syncthreads();

    double* profile_output = output + static_cast<std::size_t>(blockIdx.x) * bins;
    for (std::size_t bin = threadIdx.x; bin < bins; bin += blockDim.x) {
        profile_output[bin] = sums[bin] / denominators[bin];
        if (!isfinite(profile_output[bin])) {
            record_kernel_failure(
                kKernelNonFiniteProfile,
                static_cast<unsigned long long>(profile_index),
                status,
                failed_profile
            );
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        for (std::size_t index = 0; index < normalization_count; ++index) {
            normalization_values[index] = profile_output[normalization_indices[index]];
        }
        for (std::size_t index = 1; index < normalization_count; ++index) {
            const double value = normalization_values[index];
            std::size_t position = index;
            while (position > 0 && normalization_values[position - 1] > value) {
                normalization_values[position] = normalization_values[position - 1];
                --position;
            }
            normalization_values[position] = value;
        }

        const std::size_t middle = normalization_count / 2;
        double normalization = normalization_values[middle];
        if ((normalization_count & 1U) == 0U) {
            normalization = 0.5 * (
                normalization_values[middle - 1] + normalization_values[middle]
            );
        }
        if (!isfinite(normalization) || normalization <= 1e-12) {
            record_kernel_failure(
                kKernelInvalidNormalization,
                static_cast<unsigned long long>(profile_index),
                status,
                failed_profile
            );
            normalization_values[0] = 1.0;
        } else {
            normalization_values[0] = normalization;
        }
    }
    __syncthreads();

    const double normalization = normalization_values[0];
    for (std::size_t bin = threadIdx.x; bin < bins; bin += blockDim.x) {
        profile_output[bin] /= normalization;
    }
}

void validate_inputs(
    const double* images,
    std::size_t measurements,
    std::size_t pixels,
    const std::uint64_t* measurement_seeds,
    const double* scales,
    std::size_t scale_count,
    std::size_t draws,
    const std::int64_t* csc_indptr,
    const std::int32_t* csc_indices,
    const double* csc_weights,
    std::size_t nonzero_count,
    const double* denominators,
    std::size_t bins,
    const std::int32_t* normalization_indices,
    std::size_t normalization_count,
    std::size_t profile_batch_size,
    bool sample_poisson,
    double* output
) {
    if (images == nullptr || measurement_seeds == nullptr || scales == nullptr ||
        csc_indptr == nullptr || csc_indices == nullptr || csc_weights == nullptr ||
        denominators == nullptr || normalization_indices == nullptr || output == nullptr) {
        throw std::invalid_argument("null input pointer");
    }
    if (measurements == 0 || pixels == 0 || scale_count == 0 || draws == 0 || bins == 0 ||
        nonzero_count == 0 || normalization_count == 0 || profile_batch_size == 0) {
        throw std::invalid_argument("all dimensions and batch size must be positive");
    }
    if (bins > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument("bin count exceeds int32 CSC index range");
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
        const double squared = scales[index] * scales[index];
        if (!std::isfinite(squared) || squared == 0.0) {
            throw std::invalid_argument("squared noise scale must be finite and positive");
        }
        minimum_scale_squared = std::min(minimum_scale_squared, squared);
    }
    if (sample_poisson) {
        const double maximum_lambda = maximum_positive / minimum_scale_squared;
        if (!std::isfinite(maximum_lambda) ||
            maximum_lambda >
                static_cast<double>(std::numeric_limits<unsigned int>::max())) {
            throw std::invalid_argument("Poisson rate exceeds cuRAND uint32 range");
        }
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
    for (std::size_t bin = 0; bin < bins; ++bin) {
        if (!std::isfinite(denominators[bin]) || denominators[bin] <= 0.0) {
            throw std::invalid_argument(
                "normalization denominators must be finite and positive"
            );
        }
    }
    for (std::size_t index = 0; index < normalization_count; ++index) {
        if (normalization_indices[index] < 0 ||
            static_cast<std::size_t>(normalization_indices[index]) >= bins) {
            throw std::invalid_argument("q-normalization bin index out of range");
        }
    }

    std::size_t profile_count = checked_product(scale_count, draws, "profiles");
    profile_count = checked_product(profile_count, measurements, "profiles");
    checked_product(profile_count, bins, "output");
}

void run_direct_monte_carlo(
    const double* images,
    std::size_t measurements,
    std::size_t pixels,
    const std::uint64_t* measurement_seeds,
    const double* scales,
    std::size_t scale_count,
    std::size_t draws,
    const std::int64_t* csc_indptr,
    const std::int32_t* csc_indices,
    const double* csc_weights,
    std::size_t nonzero_count,
    const double* denominators,
    std::size_t bins,
    const std::int32_t* normalization_indices,
    std::size_t normalization_count,
    std::uint64_t base_seed,
    int device_index,
    int threads_per_block,
    std::size_t profile_batch_size,
    bool sample_poisson,
    double* output
) {
    validate_inputs(
        images,
        measurements,
        pixels,
        measurement_seeds,
        scales,
        scale_count,
        draws,
        csc_indptr,
        csc_indices,
        csc_weights,
        nonzero_count,
        denominators,
        bins,
        normalization_indices,
        normalization_count,
        profile_batch_size,
        sample_poisson,
        output
    );

    int device_count = 0;
    check_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count == 0) {
        throw std::runtime_error("no CUDA device is available");
    }
    if (device_index < 0 || device_index >= device_count) {
        throw std::invalid_argument("CUDA device index is out of range");
    }
    check_cuda(cudaSetDevice(device_index), "cudaSetDevice");

    cudaDeviceProp properties{};
    check_cuda(cudaGetDeviceProperties(&properties, device_index), "cudaGetDeviceProperties");
    if (properties.major < 6) {
        throw std::runtime_error(
            "CUDA compute capability 6.0 or newer is required for double atomicAdd"
        );
    }
    if (threads_per_block <= 0 || threads_per_block > properties.maxThreadsPerBlock) {
        throw std::invalid_argument(
            "threads per block must be positive and not exceed the CUDA device limit"
        );
    }

    if (normalization_count > std::numeric_limits<std::size_t>::max() - bins) {
        throw std::invalid_argument("shared workspace size overflow");
    }
    const std::size_t shared_values = bins + normalization_count;
    const std::size_t shared_bytes = checked_product(shared_values, sizeof(double), "shared");
    if (shared_bytes > properties.sharedMemPerBlock) {
        throw std::runtime_error(
            "bin and normalization workspaces exceed CUDA shared memory per block"
        );
    }

    const std::size_t image_values = checked_product(measurements, pixels, "images");
    std::size_t profile_count = checked_product(scale_count, draws, "profiles");
    profile_count = checked_product(profile_count, measurements, "profiles");
    const std::size_t maximum_grid = static_cast<std::size_t>(properties.maxGridSize[0]);
    const std::size_t batch_capacity = std::min(profile_batch_size, maximum_grid);
    if (batch_capacity == 0) {
        throw std::runtime_error("CUDA device reports zero one-dimensional grid capacity");
    }

    DeviceBuffer<double> device_images(image_values);
    DeviceBuffer<std::uint64_t> device_measurement_seeds(measurements);
    DeviceBuffer<double> device_scales(scale_count);
    DeviceBuffer<std::int64_t> device_csc_indptr(pixels + 1);
    DeviceBuffer<std::int32_t> device_csc_indices(nonzero_count);
    DeviceBuffer<double> device_csc_weights(nonzero_count);
    DeviceBuffer<double> device_denominators(bins);
    DeviceBuffer<std::int32_t> device_normalization_indices(normalization_count);
    DeviceBuffer<double> device_output(checked_product(batch_capacity, bins, "batch output"));
    DeviceBuffer<int> device_status(1);
    DeviceBuffer<unsigned long long> device_failed_profile(1);

    copy_to_device(device_images, images, "copy images to CUDA device");
    copy_to_device(
        device_measurement_seeds,
        measurement_seeds,
        "copy measurement seeds to CUDA device"
    );
    copy_to_device(device_scales, scales, "copy scales to CUDA device");
    copy_to_device(device_csc_indptr, csc_indptr, "copy CSC indptr to CUDA device");
    copy_to_device(device_csc_indices, csc_indices, "copy CSC indices to CUDA device");
    copy_to_device(device_csc_weights, csc_weights, "copy CSC weights to CUDA device");
    copy_to_device(
        device_denominators,
        denominators,
        "copy denominators to CUDA device"
    );
    copy_to_device(
        device_normalization_indices,
        normalization_indices,
        "copy normalization indices to CUDA device"
    );

    for (std::size_t profile_offset = 0; profile_offset < profile_count;) {
        const std::size_t batch_profiles = std::min(
            batch_capacity,
            profile_count - profile_offset
        );
        int kernel_status = kKernelSuccess;
        unsigned long long failed_profile = std::numeric_limits<unsigned long long>::max();
        check_cuda(
            cudaMemcpy(
                device_status.data(),
                &kernel_status,
                sizeof(kernel_status),
                cudaMemcpyHostToDevice
            ),
            "reset CUDA kernel status"
        );
        check_cuda(
            cudaMemcpy(
                device_failed_profile.data(),
                &failed_profile,
                sizeof(failed_profile),
                cudaMemcpyHostToDevice
            ),
            "reset CUDA failed-profile index"
        );

        if (sample_poisson) {
            direct_monte_carlo_kernel<true><<<
                static_cast<unsigned int>(batch_profiles),
                threads_per_block,
                shared_bytes
            >>>(
                device_images.data(), measurements, pixels,
                device_measurement_seeds.data(), device_scales.data(), draws,
                device_csc_indptr.data(), device_csc_indices.data(),
                device_csc_weights.data(), device_denominators.data(), bins,
                device_normalization_indices.data(), normalization_count,
                base_seed, profile_offset, device_output.data(),
                device_status.data(), device_failed_profile.data()
            );
        } else {
            direct_monte_carlo_kernel<false><<<
                static_cast<unsigned int>(batch_profiles),
                threads_per_block,
                shared_bytes
            >>>(
                device_images.data(), measurements, pixels,
                device_measurement_seeds.data(), device_scales.data(), draws,
                device_csc_indptr.data(), device_csc_indices.data(),
                device_csc_weights.data(), device_denominators.data(), bins,
                device_normalization_indices.data(), normalization_count,
                base_seed, profile_offset, device_output.data(),
                device_status.data(), device_failed_profile.data()
            );
        }
        check_cuda(cudaGetLastError(), "launch direct Monte Carlo CUDA kernel");
        check_cuda(cudaDeviceSynchronize(), "execute direct Monte Carlo CUDA kernel");
        check_cuda(
            cudaMemcpy(
                &kernel_status,
                device_status.data(),
                sizeof(kernel_status),
                cudaMemcpyDeviceToHost
            ),
            "copy CUDA kernel status to host"
        );
        if (kernel_status != kKernelSuccess) {
            check_cuda(
                cudaMemcpy(
                    &failed_profile,
                    device_failed_profile.data(),
                    sizeof(failed_profile),
                    cudaMemcpyDeviceToHost
                ),
                "copy failed CUDA profile index to host"
            );
            if (kernel_status == kKernelInvalidNormalization) {
                throw std::runtime_error(
                    "non-finite or non-positive profile normalization at flattened "
                    "profile index " + std::to_string(failed_profile)
                );
            }
            if (kernel_status == kKernelNonFiniteProfile) {
                throw std::runtime_error(
                    "non-finite integrated profile at flattened profile index " +
                    std::to_string(failed_profile)
                );
            }
            throw std::runtime_error("unknown direct Monte Carlo CUDA kernel failure");
        }

        const std::size_t batch_values = batch_profiles * bins;
        check_cuda(
            cudaMemcpy(
                output + profile_offset * bins,
                device_output.data(),
                batch_values * sizeof(double),
                cudaMemcpyDeviceToHost
            ),
            "copy CUDA profiles to host"
        );
        profile_offset += batch_profiles;
    }
}

}  // namespace

extern "C" {

XRDMC_CUDA_EXPORT int xrdmc_cuda_abi_version() noexcept {
    return kAbiVersion;
}

XRDMC_CUDA_EXPORT int xrdmc_cuda_device_count(
    char* error_buffer,
    std::size_t error_buffer_size
) noexcept {
    try {
        write_error(error_buffer, error_buffer_size, "");
        int count = 0;
        check_cuda(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
        return count;
    } catch (const std::exception& error) {
        write_error(error_buffer, error_buffer_size, error.what());
        return -1;
    } catch (...) {
        write_error(error_buffer, error_buffer_size, "unknown native CUDA exception");
        return -2;
    }
}

XRDMC_CUDA_EXPORT int xrdmc_cuda_run(
    const double* images,
    std::size_t measurements,
    std::size_t pixels,
    const std::uint64_t* measurement_seeds,
    const double* scales,
    std::size_t scale_count,
    std::size_t draws,
    const std::int64_t* csc_indptr,
    const std::int32_t* csc_indices,
    const double* csc_weights,
    std::size_t nonzero_count,
    const double* denominators,
    std::size_t bins,
    const std::int32_t* normalization_indices,
    std::size_t normalization_count,
    std::uint64_t base_seed,
    int device_index,
    int threads_per_block,
    std::size_t profile_batch_size,
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
            measurement_seeds,
            scales,
            scale_count,
            draws,
            csc_indptr,
            csc_indices,
            csc_weights,
            nonzero_count,
            denominators,
            bins,
            normalization_indices,
            normalization_count,
            base_seed,
            device_index,
            threads_per_block,
            profile_batch_size,
            true,
            output
        );
        return 0;
    } catch (const std::exception& error) {
        write_error(error_buffer, error_buffer_size, error.what());
        return 1;
    } catch (...) {
        write_error(error_buffer, error_buffer_size, "unknown native CUDA exception");
        return 2;
    }
}

XRDMC_CUDA_EXPORT int xrdmc_cuda_integrate(
    const double* images,
    std::size_t measurements,
    std::size_t pixels,
    const std::int64_t* csc_indptr,
    const std::int32_t* csc_indices,
    const double* csc_weights,
    std::size_t nonzero_count,
    const double* denominators,
    std::size_t bins,
    const std::int32_t* normalization_indices,
    std::size_t normalization_count,
    int device_index,
    int threads_per_block,
    std::size_t profile_batch_size,
    double* output,
    char* error_buffer,
    std::size_t error_buffer_size
) noexcept {
    try {
        write_error(error_buffer, error_buffer_size, "");
        const double unit_scale = 1.0;
        const std::vector<std::uint64_t> measurement_seeds(measurements, 0);
        run_direct_monte_carlo(
            images,
            measurements,
            pixels,
            measurement_seeds.data(),
            &unit_scale,
            1,
            1,
            csc_indptr,
            csc_indices,
            csc_weights,
            nonzero_count,
            denominators,
            bins,
            normalization_indices,
            normalization_count,
            0,
            device_index,
            threads_per_block,
            profile_batch_size,
            false,
            output
        );
        return 0;
    } catch (const std::exception& error) {
        write_error(error_buffer, error_buffer_size, error.what());
        return 1;
    } catch (...) {
        write_error(error_buffer, error_buffer_size, "unknown native CUDA exception");
        return 2;
    }
}

}  // extern "C"
