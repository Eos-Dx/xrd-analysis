#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#define XRDMC_METAL_EXPORT __attribute__((visibility("default")))

namespace {

constexpr int kAbiVersion = 2;
constexpr double kMaximumExactFloatPoissonRate = 16777216.0;

struct KernelParams {
    std::uint64_t measurements;
    std::uint64_t pixels;
    std::uint64_t scale_count;
    std::uint64_t draws;
    std::uint64_t bins;
    std::uint64_t normalization_count;
    std::uint64_t profile_offset;
    std::uint64_t batch_profiles;
    std::uint64_t base_seed;
};

struct PipelineBundle {
    __strong id<MTLDevice> device;
    __strong id<MTLComputePipelineState> run_pipeline;
    __strong id<MTLComputePipelineState> integrate_pipeline;
};

struct PersistentMetalSession {
    std::shared_ptr<PipelineBundle> pipelines;
    __strong id<MTLCommandQueue> queue;
    __strong id<MTLBuffer> image_buffer;
    __strong id<MTLBuffer> seed_buffer;
    __strong id<MTLBuffer> scale_buffer;
    __strong id<MTLBuffer> indptr_buffer;
    __strong id<MTLBuffer> index_buffer;
    __strong id<MTLBuffer> weight_buffer;
    __strong id<MTLBuffer> denominator_buffer;
    __strong id<MTLBuffer> normalization_buffer;
    __strong id<MTLBuffer> measurement_plan_buffer;
    __strong id<MTLBuffer> output_buffer;
    __strong id<MTLBuffer> status_buffer;
    std::size_t measurements;
    std::size_t pixels;
    std::size_t bins;
    std::size_t plan_count;
    std::size_t normalization_count;
    std::size_t scale_capacity;
    std::size_t profile_batch_size;
    float maximum_positive;
};

std::mutex g_pipeline_mutex;
std::unordered_map<std::string, std::shared_ptr<PipelineBundle>> g_pipeline_cache;

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

std::size_t checked_bytes(std::size_t count, std::size_t item_size, const char* name) {
    return checked_product(count, item_size, name);
}

std::string ns_error(NSError* error) {
    return error == nil ? "unknown Metal error" :
        std::string([[error localizedDescription] UTF8String]);
}

std::string read_source(const char* source_path) {
    if (source_path == nullptr || source_path[0] == '\0') {
        throw std::invalid_argument("Metal source path is empty");
    }
    std::ifstream stream(source_path, std::ios::binary);
    if (!stream) {
        throw std::invalid_argument(std::string("Metal source is unavailable: ") + source_path);
    }
    std::ostringstream contents;
    contents << stream.rdbuf();
    if (!stream.good() && !stream.eof()) {
        throw std::runtime_error(std::string("failed to read Metal source: ") + source_path);
    }
    const std::string source = contents.str();
    if (source.empty()) {
        throw std::invalid_argument("Metal source is empty");
    }
    return source;
}

NSArray<id<MTLDevice>>* metal_devices() {
    NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
    if (devices.count == 0) {
        id<MTLDevice> default_device = MTLCreateSystemDefaultDevice();
        if (default_device != nil) {
            devices = @[default_device];
        }
    }
    return devices;
}

id<MTLDevice> select_device(int device_index) {
    NSArray<id<MTLDevice>>* devices = metal_devices();
    if (device_index < 0 || static_cast<NSUInteger>(device_index) >= devices.count) {
        throw std::invalid_argument(
            "Metal device index is out of range; available device count is " +
            std::to_string(devices.count)
        );
    }
    return devices[static_cast<NSUInteger>(device_index)];
}

std::shared_ptr<PipelineBundle> load_pipelines(const char* source_path, int device_index) {
    const std::string source = read_source(source_path);
    id<MTLDevice> device = select_device(device_index);
    const std::string key = std::to_string(device.registryID) + ":" + source_path + ":" +
        std::to_string(std::hash<std::string>{}(source));

    std::lock_guard<std::mutex> lock(g_pipeline_mutex);
    const auto cached = g_pipeline_cache.find(key);
    if (cached != g_pipeline_cache.end()) {
        return cached->second;
    }

    NSString* shader_source = [[NSString alloc]
        initWithBytes:source.data()
        length:source.size()
        encoding:NSUTF8StringEncoding];
    if (shader_source == nil) {
        throw std::invalid_argument("Metal source is not valid UTF-8");
    }
    MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
    options.mathMode = MTLMathModeSafe;
    NSError* error = nil;
    id<MTLLibrary> library = [device
        newLibraryWithSource:shader_source
        options:options
        error:&error];
    if (library == nil) {
        throw std::runtime_error("Metal source compilation failed: " + ns_error(error));
    }

    id<MTLFunction> run_function = [library newFunctionWithName:@"xrdmc_metal_run_kernel"];
    id<MTLFunction> integrate_function =
        [library newFunctionWithName:@"xrdmc_metal_integrate_kernel"];
    if (run_function == nil || integrate_function == nil) {
        throw std::runtime_error("required Metal kernel function is missing");
    }
    error = nil;
    id<MTLComputePipelineState> run_pipeline =
        [device newComputePipelineStateWithFunction:run_function error:&error];
    if (run_pipeline == nil) {
        throw std::runtime_error("Metal run-pipeline creation failed: " + ns_error(error));
    }
    error = nil;
    id<MTLComputePipelineState> integrate_pipeline =
        [device newComputePipelineStateWithFunction:integrate_function error:&error];
    if (integrate_pipeline == nil) {
        throw std::runtime_error(
            "Metal integrate-pipeline creation failed: " + ns_error(error)
        );
    }

    auto result = std::make_shared<PipelineBundle>();
    result->device = device;
    result->run_pipeline = run_pipeline;
    result->integrate_pipeline = integrate_pipeline;
    g_pipeline_cache.emplace(key, result);
    return result;
}

std::vector<float> to_float_vector(
    const double* values,
    std::size_t count,
    const char* name
) {
    std::vector<float> result(count);
    for (std::size_t index = 0; index < count; ++index) {
        if (!std::isfinite(values[index])) {
            throw std::invalid_argument(std::string(name) + " must contain only finite values");
        }
        const float converted = static_cast<float>(values[index]);
        if (!std::isfinite(converted)) {
            throw std::invalid_argument(std::string(name) + " exceeds float32 range");
        }
        result[index] = converted;
    }
    return result;
}

std::vector<float> prepare_noise_scales(
    const double* scales,
    std::size_t scale_count,
    float maximum_positive
) {
    if (scales == nullptr || scale_count == 0) {
        throw std::invalid_argument("noise scales must be present");
    }
    std::vector<float> result = to_float_vector(scales, scale_count, "noise scales");
    float minimum_scale_squared = std::numeric_limits<float>::infinity();
    for (float scale : result) {
        if (scale <= 0.0f || !std::isfinite(scale * scale)) {
            throw std::invalid_argument("noise scales must be finite and positive");
        }
        minimum_scale_squared = std::min(minimum_scale_squared, scale * scale);
    }
    const double maximum_rate =
        static_cast<double>(maximum_positive) / minimum_scale_squared;
    if (!std::isfinite(maximum_rate) || maximum_rate > kMaximumExactFloatPoissonRate) {
        throw std::invalid_argument(
            "Poisson rate exceeds exact float32 integer range of 16777216"
        );
    }
    return result;
}

void validate_common(
    const double* images,
    std::size_t measurements,
    std::size_t pixels,
    const std::int64_t* csr_indptr,
    const std::int32_t* csr_indices,
    const double* csr_weights,
    std::size_t nonzero_count,
    const double* denominators,
    std::size_t bins,
    std::size_t plan_count,
    const std::int32_t* measurement_plan_indices,
    const std::int32_t* normalization_indices,
    std::size_t normalization_count,
    std::size_t profile_batch_size
) {
    if (images == nullptr || csr_indptr == nullptr || csr_indices == nullptr ||
        csr_weights == nullptr || denominators == nullptr ||
        measurement_plan_indices == nullptr ||
        normalization_indices == nullptr) {
        throw std::invalid_argument("null input pointer");
    }
    if (measurements == 0 || pixels == 0 || bins == 0 || plan_count == 0 ||
        normalization_count == 0 || profile_batch_size == 0) {
        throw std::invalid_argument("dimensions and profile batch size must be positive");
    }
    if (bins > 1024) {
        throw std::invalid_argument("bin count exceeds Metal threadgroup limit of 1024");
    }
    if (normalization_count > bins) {
        throw std::invalid_argument("normalization count exceeds bin count");
    }

    const std::size_t image_count = checked_product(measurements, pixels, "images");
    for (std::size_t index = 0; index < image_count; ++index) {
        if (!std::isfinite(images[index]) ||
            !std::isfinite(static_cast<float>(images[index]))) {
            throw std::invalid_argument("images must contain finite float32 values");
        }
    }
    std::size_t previous_end = 0;
    for (std::size_t plan = 0; plan < plan_count; ++plan) {
        const std::size_t base = checked_product(plan, bins + 1, "CSR indptr");
        if (csr_indptr[base] < 0 ||
            static_cast<std::size_t>(csr_indptr[base]) != previous_end) {
            throw std::invalid_argument("CSR plan offsets are not contiguous");
        }
        for (std::size_t bin = 0; bin < bins; ++bin) {
            if (csr_indptr[base + bin] < 0 ||
                csr_indptr[base + bin + 1] < csr_indptr[base + bin] ||
                static_cast<std::size_t>(csr_indptr[base + bin + 1]) > nonzero_count) {
                throw std::invalid_argument("CSR indptr must be monotonic and in range");
            }
        }
        previous_end = static_cast<std::size_t>(csr_indptr[base + bins]);
    }
    if (previous_end != nonzero_count) {
        throw std::invalid_argument("CSR indptr endpoints do not match nonzero count");
    }
    for (std::size_t index = 0; index < nonzero_count; ++index) {
        if (csr_indices[index] < 0 ||
            static_cast<std::size_t>(csr_indices[index]) >= pixels) {
            throw std::invalid_argument("CSR pixel index is out of range");
        }
        if (!std::isfinite(csr_weights[index]) ||
            !std::isfinite(static_cast<float>(csr_weights[index]))) {
            throw std::invalid_argument("CSR weights must contain finite float32 values");
        }
    }
    const std::size_t denominator_count =
        checked_product(plan_count, bins, "normalization denominators");
    for (std::size_t index = 0; index < denominator_count; ++index) {
        if (!std::isfinite(denominators[index]) || denominators[index] <= 0.0 ||
            !std::isfinite(static_cast<float>(denominators[index]))) {
            throw std::invalid_argument(
                "normalization denominators must be finite, positive float32 values"
            );
        }
    }
    for (std::size_t measurement = 0; measurement < measurements; ++measurement) {
        const std::int32_t plan = measurement_plan_indices[measurement];
        if (plan < 0 || static_cast<std::size_t>(plan) >= plan_count) {
            throw std::invalid_argument("measurement plan index is out of range");
        }
    }
    std::unordered_set<std::int32_t> normalization_set;
    for (std::size_t index = 0; index < normalization_count; ++index) {
        const std::int32_t bin = normalization_indices[index];
        if (bin < 0 || static_cast<std::size_t>(bin) >= bins) {
            throw std::invalid_argument("normalization bin index is out of range");
        }
        if (!normalization_set.insert(bin).second) {
            throw std::invalid_argument("normalization bin indices must be unique");
        }
    }
    checked_bytes(image_count, sizeof(float), "Metal image buffer");
    checked_bytes(nonzero_count, sizeof(float), "Metal CSR weight buffer");
    checked_bytes(
        checked_product(plan_count, bins + 1, "Metal CSR indptr"),
        sizeof(std::int64_t), "Metal CSR indptr"
    );
    checked_bytes(denominator_count, sizeof(float), "Metal denominator buffer");
}

void validate_device_limits(
    const PipelineBundle& pipelines,
    std::size_t bins,
    std::size_t normalization_count,
    std::size_t maximum_buffer_bytes,
    bool sample_poisson
) {
    id<MTLComputePipelineState> pipeline =
        sample_poisson ? pipelines.run_pipeline : pipelines.integrate_pipeline;
    if (bins > pipeline.maxTotalThreadsPerThreadgroup) {
        throw std::invalid_argument(
            "bin count exceeds Metal pipeline threadgroup capacity of " +
            std::to_string(pipeline.maxTotalThreadsPerThreadgroup)
        );
    }
    const std::size_t threadgroup_bytes =
        checked_bytes(bins + normalization_count, sizeof(float), "threadgroup memory");
    if (threadgroup_bytes > pipelines.device.maxThreadgroupMemoryLength) {
        throw std::invalid_argument(
            "profile and normalization workspace exceed Metal threadgroup memory"
        );
    }
    if (maximum_buffer_bytes > pipelines.device.maxBufferLength) {
        throw std::invalid_argument("requested buffer exceeds Metal device allocation limit");
    }
}

id<MTLBuffer> make_buffer(
    id<MTLDevice> device,
    const void* values,
    std::size_t bytes,
    const char* name
) {
    if (bytes == 0) {
        throw std::invalid_argument(std::string(name) + " buffer is empty");
    }
    id<MTLBuffer> buffer = [device
        newBufferWithBytes:values
        length:bytes
        options:MTLResourceStorageModeShared];
    if (buffer == nil) {
        throw std::runtime_error(std::string("failed to allocate Metal ") + name + " buffer");
    }
    return buffer;
}

id<MTLBuffer> make_empty_buffer(id<MTLDevice> device, std::size_t bytes, const char* name) {
    if (bytes == 0) {
        throw std::invalid_argument(std::string(name) + " buffer is empty");
    }
    id<MTLBuffer> buffer = [device
        newBufferWithLength:bytes
        options:MTLResourceStorageModeShared];
    if (buffer == nil) {
        throw std::runtime_error(std::string("failed to allocate Metal ") + name + " buffer");
    }
    return buffer;
}

void wait_for_command(id<MTLCommandBuffer> command) {
    [command commit];
    [command waitUntilCompleted];
    if (command.status == MTLCommandBufferStatusError) {
        throw std::runtime_error("Metal command failed: " + ns_error(command.error));
    }
}

std::string profile_failure(int status, std::size_t profile) {
    if (status == 1) {
        return "non-finite or non-positive profile normalization at profile " +
            std::to_string(profile);
    }
    if (status == 2) {
        return "non-finite integrated profile at profile " + std::to_string(profile);
    }
    if (status == 3) {
        return "Poisson sampler did not converge at profile " + std::to_string(profile);
    }
    return "unknown Metal kernel failure at profile " + std::to_string(profile);
}

void run_metal(
    const char* metal_source_path,
    const double* images,
    std::size_t measurements,
    std::size_t pixels,
    const std::uint64_t* measurement_seeds,
    const double* scales,
    std::size_t scale_count,
    std::size_t draws,
    const std::int64_t* csr_indptr,
    const std::int32_t* csr_indices,
    const double* csr_weights,
    std::size_t nonzero_count,
    const double* denominators,
    std::size_t bins,
    const std::int32_t* normalization_indices,
    std::size_t normalization_count,
    std::uint64_t base_seed,
    int device_index,
    std::size_t profile_batch_size,
    bool sample_poisson,
    double* output
) {
    if (output == nullptr) {
        throw std::invalid_argument("output pointer is null");
    }
    const std::vector<std::int32_t> measurement_plan_indices(measurements, 0);
    validate_common(
        images, measurements, pixels, csr_indptr, csr_indices, csr_weights,
        nonzero_count, denominators, bins, 1, measurement_plan_indices.data(),
        normalization_indices,
        normalization_count, profile_batch_size
    );
    if (sample_poisson &&
        (measurement_seeds == nullptr || scales == nullptr || scale_count == 0 || draws == 0)) {
        throw std::invalid_argument("sampling inputs and dimensions must be present");
    }

    const std::size_t image_count = checked_product(measurements, pixels, "images");
    std::vector<float> float_images = to_float_vector(images, image_count, "images");
    std::vector<float> float_weights =
        to_float_vector(csr_weights, nonzero_count, "CSR weights");
    std::vector<float> float_denominators =
        to_float_vector(denominators, bins, "normalization denominators");
    std::vector<float> float_scales;
    if (sample_poisson) {
        float maximum_positive = 0.0f;
        for (float value : float_images) {
            maximum_positive = std::max(maximum_positive, value);
        }
        float_scales = prepare_noise_scales(scales, scale_count, maximum_positive);
    }

    std::size_t profile_count = measurements;
    if (sample_poisson) {
        profile_count = checked_product(scale_count, draws, "profiles");
        profile_count = checked_product(profile_count, measurements, "profiles");
    }
    checked_product(profile_count, bins, "output");
    const std::size_t maximum_batch = std::min(profile_batch_size, profile_count);
    const std::size_t maximum_output_bytes = checked_bytes(
        checked_product(maximum_batch, bins, "batch output"), sizeof(float), "batch output"
    );

    const std::shared_ptr<PipelineBundle> pipelines =
        load_pipelines(metal_source_path, device_index);
    validate_device_limits(
        *pipelines,
        bins,
        normalization_count,
        std::max({
            checked_bytes(image_count, sizeof(float), "image buffer"),
            maximum_output_bytes,
            checked_bytes(nonzero_count, sizeof(float), "CSR weight buffer"),
        }),
        sample_poisson
    );

    id<MTLDevice> device = pipelines->device;
    id<MTLCommandQueue> queue = [device newCommandQueue];
    if (queue == nil) {
        throw std::runtime_error("failed to create Metal command queue");
    }
    id<MTLBuffer> image_buffer = make_buffer(
        device, float_images.data(), checked_bytes(image_count, sizeof(float), "images"),
        "image"
    );
    id<MTLBuffer> indptr_buffer = make_buffer(
        device, csr_indptr, checked_bytes(bins + 1, sizeof(std::int64_t), "CSR indptr"),
        "CSR indptr"
    );
    id<MTLBuffer> index_buffer = make_buffer(
        device, csr_indices,
        checked_bytes(nonzero_count, sizeof(std::int32_t), "CSR indices"), "CSR index"
    );
    id<MTLBuffer> weight_buffer = make_buffer(
        device, float_weights.data(),
        checked_bytes(nonzero_count, sizeof(float), "CSR weights"), "CSR weight"
    );
    id<MTLBuffer> denominator_buffer = make_buffer(
        device, float_denominators.data(), checked_bytes(bins, sizeof(float), "denominators"),
        "denominator"
    );
    id<MTLBuffer> normalization_buffer = make_buffer(
        device, normalization_indices,
        checked_bytes(normalization_count, sizeof(std::int32_t), "normalization indices"),
        "normalization index"
    );
    id<MTLBuffer> measurement_plan_buffer = make_buffer(
        device, measurement_plan_indices.data(),
        checked_bytes(measurements, sizeof(std::int32_t), "measurement plans"),
        "measurement plan"
    );
    id<MTLBuffer> seed_buffer = nil;
    id<MTLBuffer> scale_buffer = nil;
    if (sample_poisson) {
        seed_buffer = make_buffer(
            device, measurement_seeds,
            checked_bytes(measurements, sizeof(std::uint64_t), "measurement seeds"),
            "measurement seed"
        );
        scale_buffer = make_buffer(
            device, float_scales.data(),
            checked_bytes(scale_count, sizeof(float), "noise scales"), "noise scale"
        );
    }

    std::size_t profile_offset = 0;
    while (profile_offset < profile_count) {
        const std::size_t current_batch =
            std::min(profile_batch_size, profile_count - profile_offset);
        const std::size_t batch_value_count =
            checked_product(current_batch, bins, "batch output");
        id<MTLBuffer> output_buffer = make_empty_buffer(
            device, checked_bytes(batch_value_count, sizeof(float), "batch output"), "output"
        );
        id<MTLBuffer> status_buffer = make_empty_buffer(
            device, checked_bytes(current_batch, sizeof(std::int32_t), "profile status"),
            "profile status"
        );
        std::memset(status_buffer.contents, 0, current_batch * sizeof(std::int32_t));

        KernelParams params{
            static_cast<std::uint64_t>(measurements),
            static_cast<std::uint64_t>(pixels),
            static_cast<std::uint64_t>(sample_poisson ? scale_count : 1),
            static_cast<std::uint64_t>(sample_poisson ? draws : 1),
            static_cast<std::uint64_t>(bins),
            static_cast<std::uint64_t>(normalization_count),
            static_cast<std::uint64_t>(profile_offset),
            static_cast<std::uint64_t>(current_batch),
            base_seed,
        };
        id<MTLCommandBuffer> command = [queue commandBuffer];
        if (command == nil) {
            throw std::runtime_error("failed to create Metal command buffer");
        }
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        if (encoder == nil) {
            throw std::runtime_error("failed to create Metal command encoder");
        }

        if (sample_poisson) {
            [encoder setComputePipelineState:pipelines->run_pipeline];
            [encoder setBuffer:image_buffer offset:0 atIndex:0];
            [encoder setBuffer:seed_buffer offset:0 atIndex:1];
            [encoder setBuffer:scale_buffer offset:0 atIndex:2];
            [encoder setBuffer:indptr_buffer offset:0 atIndex:3];
            [encoder setBuffer:index_buffer offset:0 atIndex:4];
            [encoder setBuffer:weight_buffer offset:0 atIndex:5];
            [encoder setBuffer:denominator_buffer offset:0 atIndex:6];
            [encoder setBuffer:normalization_buffer offset:0 atIndex:7];
            [encoder setBuffer:output_buffer offset:0 atIndex:8];
            [encoder setBuffer:status_buffer offset:0 atIndex:9];
            [encoder setBytes:&params length:sizeof(params) atIndex:10];
            [encoder setBuffer:measurement_plan_buffer offset:0 atIndex:11];
        } else {
            [encoder setComputePipelineState:pipelines->integrate_pipeline];
            [encoder setBuffer:image_buffer offset:0 atIndex:0];
            [encoder setBuffer:indptr_buffer offset:0 atIndex:1];
            [encoder setBuffer:index_buffer offset:0 atIndex:2];
            [encoder setBuffer:weight_buffer offset:0 atIndex:3];
            [encoder setBuffer:denominator_buffer offset:0 atIndex:4];
            [encoder setBuffer:normalization_buffer offset:0 atIndex:5];
            [encoder setBuffer:output_buffer offset:0 atIndex:6];
            [encoder setBuffer:status_buffer offset:0 atIndex:7];
            [encoder setBytes:&params length:sizeof(params) atIndex:8];
            [encoder setBuffer:measurement_plan_buffer offset:0 atIndex:9];
        }
        [encoder setThreadgroupMemoryLength:bins * sizeof(float) atIndex:0];
        [encoder
            setThreadgroupMemoryLength:normalization_count * sizeof(float)
            atIndex:1];
        [encoder
            dispatchThreadgroups:MTLSizeMake(current_batch, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(bins, 1, 1)];
        [encoder endEncoding];
        wait_for_command(command);

        const auto* statuses = static_cast<const std::int32_t*>(status_buffer.contents);
        for (std::size_t local_profile = 0; local_profile < current_batch; ++local_profile) {
            if (statuses[local_profile] != 0) {
                throw std::runtime_error(
                    profile_failure(statuses[local_profile], profile_offset + local_profile)
                );
            }
        }
        const auto* float_output = static_cast<const float*>(output_buffer.contents);
        const std::size_t output_offset = checked_product(profile_offset, bins, "output offset");
        for (std::size_t index = 0; index < batch_value_count; ++index) {
            output[output_offset + index] = static_cast<double>(float_output[index]);
        }
        profile_offset += current_batch;
    }
}

std::unique_ptr<PersistentMetalSession> create_persistent_session(
    const char* metal_source_path,
    const double* images,
    std::size_t measurements,
    std::size_t pixels,
    const std::uint64_t* measurement_seeds,
    const std::int64_t* csr_indptr,
    const std::int32_t* csr_indices,
    const double* csr_weights,
    std::size_t nonzero_count,
    const double* denominators,
    std::size_t bins,
    std::size_t plan_count,
    const std::int32_t* measurement_plan_indices,
    const std::int32_t* normalization_indices,
    std::size_t normalization_count,
    int device_index,
    std::size_t scale_capacity,
    std::size_t profile_batch_size
) {
    if (measurement_seeds == nullptr || scale_capacity == 0) {
        throw std::invalid_argument("measurement seeds and scale capacity must be present");
    }
    validate_common(
        images, measurements, pixels, csr_indptr, csr_indices, csr_weights,
        nonzero_count, denominators, bins, plan_count, measurement_plan_indices,
        normalization_indices,
        normalization_count, profile_batch_size
    );

    const std::size_t image_count = checked_product(measurements, pixels, "images");
    std::vector<float> float_images = to_float_vector(images, image_count, "images");
    std::vector<float> float_weights =
        to_float_vector(csr_weights, nonzero_count, "CSR weights");
    std::vector<float> float_denominators =
        to_float_vector(
            denominators,
            checked_product(plan_count, bins, "normalization denominators"),
            "normalization denominators"
        );
    float maximum_positive = 0.0f;
    for (float value : float_images) {
        maximum_positive = std::max(maximum_positive, value);
    }

    const std::size_t output_value_capacity =
        checked_product(profile_batch_size, bins, "persistent output");
    const std::shared_ptr<PipelineBundle> pipelines =
        load_pipelines(metal_source_path, device_index);
    validate_device_limits(
        *pipelines,
        bins,
        normalization_count,
        std::max({
            checked_bytes(image_count, sizeof(float), "image buffer"),
            checked_bytes(output_value_capacity, sizeof(float), "output buffer"),
            checked_bytes(nonzero_count, sizeof(float), "CSR weight buffer"),
            checked_bytes(scale_capacity, sizeof(float), "scale buffer"),
        }),
        true
    );

    auto session = std::make_unique<PersistentMetalSession>();
    session->pipelines = pipelines;
    session->queue = [pipelines->device newCommandQueue];
    if (session->queue == nil) {
        throw std::runtime_error("failed to create persistent Metal command queue");
    }
    session->image_buffer = make_buffer(
        pipelines->device, float_images.data(),
        checked_bytes(image_count, sizeof(float), "images"), "image"
    );
    session->seed_buffer = make_buffer(
        pipelines->device, measurement_seeds,
        checked_bytes(measurements, sizeof(std::uint64_t), "measurement seeds"),
        "measurement seed"
    );
    session->scale_buffer = make_empty_buffer(
        pipelines->device, checked_bytes(scale_capacity, sizeof(float), "noise scales"),
        "noise scale"
    );
    session->indptr_buffer = make_buffer(
        pipelines->device, csr_indptr,
        checked_bytes(
            checked_product(plan_count, bins + 1, "CSR indptr"),
            sizeof(std::int64_t), "CSR indptr"
        ),
        "CSR indptr"
    );
    session->index_buffer = make_buffer(
        pipelines->device, csr_indices,
        checked_bytes(nonzero_count, sizeof(std::int32_t), "CSR indices"), "CSR index"
    );
    session->weight_buffer = make_buffer(
        pipelines->device, float_weights.data(),
        checked_bytes(nonzero_count, sizeof(float), "CSR weights"), "CSR weight"
    );
    session->denominator_buffer = make_buffer(
        pipelines->device, float_denominators.data(),
        checked_bytes(
            checked_product(plan_count, bins, "denominators"),
            sizeof(float), "denominators"
        ),
        "denominator"
    );
    session->normalization_buffer = make_buffer(
        pipelines->device, normalization_indices,
        checked_bytes(normalization_count, sizeof(std::int32_t), "normalization indices"),
        "normalization index"
    );
    session->measurement_plan_buffer = make_buffer(
        pipelines->device, measurement_plan_indices,
        checked_bytes(measurements, sizeof(std::int32_t), "measurement plans"),
        "measurement plan"
    );
    session->output_buffer = make_empty_buffer(
        pipelines->device,
        checked_bytes(output_value_capacity, sizeof(float), "persistent output"), "output"
    );
    session->status_buffer = make_empty_buffer(
        pipelines->device,
        checked_bytes(profile_batch_size, sizeof(std::int32_t), "profile status"),
        "profile status"
    );
    session->measurements = measurements;
    session->pixels = pixels;
    session->bins = bins;
    session->plan_count = plan_count;
    session->normalization_count = normalization_count;
    session->scale_capacity = scale_capacity;
    session->profile_batch_size = profile_batch_size;
    session->maximum_positive = maximum_positive;
    return session;
}

void execute_persistent_session(
    PersistentMetalSession& session,
    std::size_t scale_count,
    std::size_t draws,
    std::uint64_t base_seed,
    bool sample_poisson,
    double* output
) {
    if (output == nullptr) {
        throw std::invalid_argument("output pointer is null");
    }
    std::size_t profile_count = session.measurements;
    if (sample_poisson) {
        if (scale_count == 0 || draws == 0) {
            throw std::invalid_argument("scale count and draws must be positive");
        }
        profile_count = checked_product(scale_count, draws, "profiles");
        profile_count = checked_product(profile_count, session.measurements, "profiles");
    }

    std::size_t profile_offset = 0;
    while (profile_offset < profile_count) {
        const std::size_t current_batch = std::min(
            session.profile_batch_size, profile_count - profile_offset
        );
        const std::size_t batch_value_count =
            checked_product(current_batch, session.bins, "batch output");
        std::memset(
            session.status_buffer.contents, 0, current_batch * sizeof(std::int32_t)
        );

        KernelParams params{
            static_cast<std::uint64_t>(session.measurements),
            static_cast<std::uint64_t>(session.pixels),
            static_cast<std::uint64_t>(sample_poisson ? scale_count : 1),
            static_cast<std::uint64_t>(sample_poisson ? draws : 1),
            static_cast<std::uint64_t>(session.bins),
            static_cast<std::uint64_t>(session.normalization_count),
            static_cast<std::uint64_t>(profile_offset),
            static_cast<std::uint64_t>(current_batch),
            base_seed,
        };
        id<MTLCommandBuffer> command = [session.queue commandBuffer];
        if (command == nil) {
            throw std::runtime_error("failed to create persistent Metal command buffer");
        }
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        if (encoder == nil) {
            throw std::runtime_error("failed to create persistent Metal command encoder");
        }

        if (sample_poisson) {
            [encoder setComputePipelineState:session.pipelines->run_pipeline];
            [encoder setBuffer:session.image_buffer offset:0 atIndex:0];
            [encoder setBuffer:session.seed_buffer offset:0 atIndex:1];
            [encoder setBuffer:session.scale_buffer offset:0 atIndex:2];
            [encoder setBuffer:session.indptr_buffer offset:0 atIndex:3];
            [encoder setBuffer:session.index_buffer offset:0 atIndex:4];
            [encoder setBuffer:session.weight_buffer offset:0 atIndex:5];
            [encoder setBuffer:session.denominator_buffer offset:0 atIndex:6];
            [encoder setBuffer:session.normalization_buffer offset:0 atIndex:7];
            [encoder setBuffer:session.output_buffer offset:0 atIndex:8];
            [encoder setBuffer:session.status_buffer offset:0 atIndex:9];
            [encoder setBytes:&params length:sizeof(params) atIndex:10];
            [encoder setBuffer:session.measurement_plan_buffer offset:0 atIndex:11];
        } else {
            [encoder setComputePipelineState:session.pipelines->integrate_pipeline];
            [encoder setBuffer:session.image_buffer offset:0 atIndex:0];
            [encoder setBuffer:session.indptr_buffer offset:0 atIndex:1];
            [encoder setBuffer:session.index_buffer offset:0 atIndex:2];
            [encoder setBuffer:session.weight_buffer offset:0 atIndex:3];
            [encoder setBuffer:session.denominator_buffer offset:0 atIndex:4];
            [encoder setBuffer:session.normalization_buffer offset:0 atIndex:5];
            [encoder setBuffer:session.output_buffer offset:0 atIndex:6];
            [encoder setBuffer:session.status_buffer offset:0 atIndex:7];
            [encoder setBytes:&params length:sizeof(params) atIndex:8];
            [encoder setBuffer:session.measurement_plan_buffer offset:0 atIndex:9];
        }
        [encoder setThreadgroupMemoryLength:session.bins * sizeof(float) atIndex:0];
        [encoder
            setThreadgroupMemoryLength:session.normalization_count * sizeof(float)
            atIndex:1];
        [encoder
            dispatchThreadgroups:MTLSizeMake(current_batch, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(session.bins, 1, 1)];
        [encoder endEncoding];
        wait_for_command(command);

        const auto* statuses =
            static_cast<const std::int32_t*>(session.status_buffer.contents);
        for (std::size_t local_profile = 0; local_profile < current_batch; ++local_profile) {
            if (statuses[local_profile] != 0) {
                throw std::runtime_error(
                    profile_failure(statuses[local_profile], profile_offset + local_profile)
                );
            }
        }
        const auto* float_output = static_cast<const float*>(session.output_buffer.contents);
        const std::size_t output_offset =
            checked_product(profile_offset, session.bins, "output offset");
        for (std::size_t index = 0; index < batch_value_count; ++index) {
            output[output_offset + index] = static_cast<double>(float_output[index]);
        }
        profile_offset += current_batch;
    }
}

void run_persistent_session(
    PersistentMetalSession& session,
    const double* scales,
    std::size_t scale_count,
    std::size_t draws,
    std::uint64_t base_seed,
    double* output
) {
    if (scale_count > session.scale_capacity) {
        throw std::invalid_argument("noise scale count exceeds persistent capacity");
    }
    const std::vector<float> float_scales =
        prepare_noise_scales(scales, scale_count, session.maximum_positive);
    std::memcpy(
        session.scale_buffer.contents, float_scales.data(),
        checked_bytes(scale_count, sizeof(float), "noise scales")
    );
    execute_persistent_session(session, scale_count, draws, base_seed, true, output);
}

}  // namespace

extern "C" {

XRDMC_METAL_EXPORT int xrdmc_metal_abi_version() noexcept {
    return kAbiVersion;
}

XRDMC_METAL_EXPORT int xrdmc_metal_device_count(
    char* error_buffer,
    std::size_t error_buffer_size
) noexcept {
    @autoreleasepool {
        try {
            write_error(error_buffer, error_buffer_size, "");
            return static_cast<int>(metal_devices().count);
        } catch (const std::exception& error) {
            write_error(error_buffer, error_buffer_size, error.what());
            return -1;
        } catch (...) {
            write_error(error_buffer, error_buffer_size, "unknown native Metal exception");
            return -2;
        }
    }
}

XRDMC_METAL_EXPORT void* xrdmc_metal_session_create(
    const char* metal_source_path,
    const double* images,
    std::size_t measurements,
    std::size_t pixels,
    const std::uint64_t* measurement_seeds,
    const std::int64_t* csr_indptr,
    const std::int32_t* csr_indices,
    const double* csr_weights,
    std::size_t nonzero_count,
    const double* denominators,
    std::size_t bins,
    const std::int32_t* normalization_indices,
    std::size_t normalization_count,
    int device_index,
    std::size_t scale_capacity,
    std::size_t profile_batch_size,
    char* error_buffer,
    std::size_t error_buffer_size
) noexcept {
    @autoreleasepool {
        try {
            write_error(error_buffer, error_buffer_size, "");
            const std::vector<std::int32_t> measurement_plan_indices(measurements, 0);
            return create_persistent_session(
                metal_source_path, images, measurements, pixels, measurement_seeds,
                csr_indptr, csr_indices, csr_weights, nonzero_count, denominators,
                bins, 1, measurement_plan_indices.data(), normalization_indices,
                normalization_count, device_index,
                scale_capacity, profile_batch_size
            ).release();
        } catch (const std::exception& error) {
            write_error(error_buffer, error_buffer_size, error.what());
            return nullptr;
        } catch (...) {
            write_error(error_buffer, error_buffer_size, "unknown native Metal exception");
            return nullptr;
        }
    }
}

XRDMC_METAL_EXPORT void* xrdmc_metal_multi_session_create(
    const char* metal_source_path,
    const double* images,
    std::size_t measurements,
    std::size_t pixels,
    const std::uint64_t* measurement_seeds,
    std::size_t plan_count,
    const std::int32_t* measurement_plan_indices,
    const std::int64_t* csr_indptr,
    const std::int32_t* csr_indices,
    const double* csr_weights,
    std::size_t nonzero_count,
    const double* denominators,
    std::size_t bins,
    const std::int32_t* normalization_indices,
    std::size_t normalization_count,
    int device_index,
    std::size_t scale_capacity,
    std::size_t profile_batch_size,
    char* error_buffer,
    std::size_t error_buffer_size
) noexcept {
    @autoreleasepool {
        try {
            write_error(error_buffer, error_buffer_size, "");
            return create_persistent_session(
                metal_source_path, images, measurements, pixels, measurement_seeds,
                csr_indptr, csr_indices, csr_weights, nonzero_count, denominators,
                bins, plan_count, measurement_plan_indices, normalization_indices,
                normalization_count, device_index, scale_capacity, profile_batch_size
            ).release();
        } catch (const std::exception& error) {
            write_error(error_buffer, error_buffer_size, error.what());
            return nullptr;
        } catch (...) {
            write_error(error_buffer, error_buffer_size, "unknown native Metal exception");
            return nullptr;
        }
    }
}

XRDMC_METAL_EXPORT void xrdmc_metal_session_destroy(void* session) noexcept {
    @autoreleasepool {
        delete static_cast<PersistentMetalSession*>(session);
    }
}

XRDMC_METAL_EXPORT int xrdmc_metal_session_run(
    void* session,
    const double* scales,
    std::size_t scale_count,
    std::size_t draws,
    std::uint64_t base_seed,
    double* output,
    char* error_buffer,
    std::size_t error_buffer_size
) noexcept {
    @autoreleasepool {
        try {
            write_error(error_buffer, error_buffer_size, "");
            if (session == nullptr) {
                throw std::invalid_argument("persistent Metal session is null");
            }
            run_persistent_session(
                *static_cast<PersistentMetalSession*>(session), scales, scale_count,
                draws, base_seed, output
            );
            return 0;
        } catch (const std::exception& error) {
            write_error(error_buffer, error_buffer_size, error.what());
            return 1;
        } catch (...) {
            write_error(error_buffer, error_buffer_size, "unknown native Metal exception");
            return 2;
        }
    }
}

XRDMC_METAL_EXPORT int xrdmc_metal_session_integrate(
    void* session,
    double* output,
    char* error_buffer,
    std::size_t error_buffer_size
) noexcept {
    @autoreleasepool {
        try {
            write_error(error_buffer, error_buffer_size, "");
            if (session == nullptr) {
                throw std::invalid_argument("persistent Metal session is null");
            }
            execute_persistent_session(
                *static_cast<PersistentMetalSession*>(session), 1, 1, 0, false, output
            );
            return 0;
        } catch (const std::exception& error) {
            write_error(error_buffer, error_buffer_size, error.what());
            return 1;
        } catch (...) {
            write_error(error_buffer, error_buffer_size, "unknown native Metal exception");
            return 2;
        }
    }
}

XRDMC_METAL_EXPORT int xrdmc_metal_run(
    const char* metal_source_path,
    const double* images,
    std::size_t measurements,
    std::size_t pixels,
    const std::uint64_t* measurement_seeds,
    const double* scales,
    std::size_t scale_count,
    std::size_t draws,
    const std::int64_t* csr_indptr,
    const std::int32_t* csr_indices,
    const double* csr_weights,
    std::size_t nonzero_count,
    const double* denominators,
    std::size_t bins,
    const std::int32_t* normalization_indices,
    std::size_t normalization_count,
    std::uint64_t base_seed,
    int device_index,
    std::size_t profile_batch_size,
    double* output,
    char* error_buffer,
    std::size_t error_buffer_size
) noexcept {
    @autoreleasepool {
        try {
            write_error(error_buffer, error_buffer_size, "");
            run_metal(
                metal_source_path, images, measurements, pixels, measurement_seeds,
                scales, scale_count, draws, csr_indptr, csr_indices, csr_weights,
                nonzero_count, denominators, bins, normalization_indices,
                normalization_count, base_seed, device_index, profile_batch_size, true,
                output
            );
            return 0;
        } catch (const std::exception& error) {
            write_error(error_buffer, error_buffer_size, error.what());
            return 1;
        } catch (...) {
            write_error(error_buffer, error_buffer_size, "unknown native Metal exception");
            return 2;
        }
    }
}

XRDMC_METAL_EXPORT int xrdmc_metal_integrate(
    const char* metal_source_path,
    const double* images,
    std::size_t measurements,
    std::size_t pixels,
    const std::int64_t* csr_indptr,
    const std::int32_t* csr_indices,
    const double* csr_weights,
    std::size_t nonzero_count,
    const double* denominators,
    std::size_t bins,
    const std::int32_t* normalization_indices,
    std::size_t normalization_count,
    int device_index,
    std::size_t profile_batch_size,
    double* output,
    char* error_buffer,
    std::size_t error_buffer_size
) noexcept {
    @autoreleasepool {
        try {
            write_error(error_buffer, error_buffer_size, "");
            run_metal(
                metal_source_path, images, measurements, pixels, nullptr, nullptr, 1, 1,
                csr_indptr, csr_indices, csr_weights, nonzero_count, denominators, bins,
                normalization_indices, normalization_count, 0, device_index,
                profile_batch_size, false, output
            );
            return 0;
        } catch (const std::exception& error) {
            write_error(error_buffer, error_buffer_size, error.what());
            return 1;
        } catch (...) {
            write_error(error_buffer, error_buffer_size, "unknown native Metal exception");
            return 2;
        }
    }
}

}  // extern "C"
