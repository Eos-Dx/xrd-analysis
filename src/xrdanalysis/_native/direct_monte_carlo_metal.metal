#include <metal_stdlib>

using namespace metal;

constant ulong kGoldenRatio = 0x9e3779b97f4a7c15UL;
constant ulong kSeedOffset = 0xd1b54a32d192ed03UL;
constant uint kPoissonIterationLimit = 1000000U;
constant int kStatusInvalidNormalization = 1;
constant int kStatusNonFiniteProfile = 2;
constant int kStatusPoissonFailure = 3;
constant float kLanczosCoefficients[9] = {
    0.99999999999980993f,
    676.5203681218851f,
    -1259.1392167224028f,
    771.32342877765313f,
    -176.61502916214059f,
    12.507343278686905f,
    -0.13857109526572012f,
    9.9843695780195716e-6f,
    1.5056327351493116e-7f,
};

struct KernelParams {
    ulong measurements;
    ulong pixels;
    ulong scale_count;
    ulong draws;
    ulong bins;
    ulong normalization_count;
    ulong profile_offset;
    ulong batch_profiles;
    ulong base_seed;
};

struct CounterRng {
    ulong key;
    ulong counter;
};

struct PoissonResult {
    uint value;
    bool valid;
};

inline ulong splitmix64(ulong value) {
    value += kGoldenRatio;
    value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9UL;
    value = (value ^ (value >> 27U)) * 0x94d049bb133111ebUL;
    return value ^ (value >> 31U);
}

inline ulong sample_key(
    ulong base_seed,
    ulong measurement_seed,
    ulong scale_index,
    ulong draw_index,
    ulong pixel_index
) {
    ulong value = splitmix64(base_seed);
    value ^= splitmix64(measurement_seed + kSeedOffset);
    value ^= splitmix64(scale_index + 0x100000001b3UL);
    value ^= splitmix64(draw_index + 0x517cc1b727220a95UL);
    value ^= splitmix64(pixel_index + 0x94d049bb133111ebUL);
    return splitmix64(value);
}

inline ulong next_u64(thread CounterRng& rng) {
    const ulong value = splitmix64(rng.key + rng.counter * kGoldenRatio);
    rng.counter += 1UL;
    return value;
}

inline float uniform_open(thread CounterRng& rng) {
    const uint mantissa = static_cast<uint>(next_u64(rng) >> 40U);
    return (static_cast<float>(mantissa) + 0.5f) * 0x1.0p-24f;
}

inline PoissonResult poisson_small(float lambda, thread CounterRng& rng) {
    const float limit = exp(-lambda);
    float product = 1.0f;
    uint count = 0U;
    for (uint iteration = 0U; iteration < kPoissonIterationLimit; ++iteration) {
        product *= uniform_open(rng);
        if (product <= limit) {
            return {count, true};
        }
        count += 1U;
    }
    return {0U, false};
}

inline float log_gamma_positive(float value) {
    const float shifted = value - 1.0f;
    float series = kLanczosCoefficients[0];
    for (uint index = 1U; index < 9U; ++index) {
        series += kLanczosCoefficients[index] / (shifted + static_cast<float>(index));
    }
    const float t = shifted + 7.5f;
    return 0.91893853320467274f + (shifted + 0.5f) * log(t) - t + log(series);
}

inline PoissonResult poisson_ptrs(float lambda, thread CounterRng& rng) {
    const float root = sqrt(lambda);
    const float b = 0.931f + 2.53f * root;
    const float a = -0.059f + 0.02483f * b;
    const float inverse_alpha = 1.1239f + 1.1328f / (b - 3.4f);
    const float squeeze = 0.9277f - 3.6224f / (b - 2.0f);

    for (uint iteration = 0U; iteration < kPoissonIterationLimit; ++iteration) {
        const float u = uniform_open(rng) - 0.5f;
        const float v = uniform_open(rng);
        const float us = 0.5f - abs(u);
        const float candidate_value = floor((2.0f * a / us + b) * u + lambda + 0.43f);
        if (candidate_value < 0.0f) {
            continue;
        }
        const uint candidate = static_cast<uint>(candidate_value);
        if (us >= 0.07f && v <= squeeze) {
            return {candidate, true};
        }
        if (us < 0.013f && v > us) {
            continue;
        }
        const float lhs = log(v * inverse_alpha / (a / (us * us) + b));
        const float rhs = -lambda + candidate_value * log(lambda) -
            log_gamma_positive(candidate_value + 1.0f);
        if (lhs <= rhs) {
            return {candidate, true};
        }
    }
    return {0U, false};
}

inline PoissonResult poisson_sample(float lambda, thread CounterRng& rng) {
    if (lambda == 0.0f) {
        return {0U, true};
    }
    return lambda < 30.0f ? poisson_small(lambda, rng) : poisson_ptrs(lambda, rng);
}

inline float centered_poisson_sample(
    float value,
    float scale_squared,
    ulong base_seed,
    ulong measurement_seed,
    ulong scale_index,
    ulong draw_index,
    ulong pixel_index,
    thread bool& valid
) {
    const float positive = max(value, 0.0f);
    const float negative = min(value, 0.0f);
    if (positive == 0.0f) {
        valid = true;
        return negative;
    }
    CounterRng rng{
        sample_key(
            base_seed, measurement_seed, scale_index, draw_index,
            pixel_index
        ),
        0UL,
    };
    const PoissonResult sampled = poisson_sample(positive / scale_squared, rng);
    valid = sampled.valid;
    return negative + scale_squared * static_cast<float>(sampled.value);
}

inline void set_status(device atomic_int* status, ulong profile, int value) {
    atomic_fetch_max_explicit(status + profile, value, memory_order_relaxed);
}

inline void normalize_profile(
    device float* output,
    device atomic_int* status,
    constant KernelParams& params,
    device const int* normalization_indices,
    threadgroup float* profile_values,
    threadgroup float* normalization_values,
    uint bin,
    uint local_profile
) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (bin == 0U) {
        for (ulong index = 0UL; index < params.normalization_count; ++index) {
            normalization_values[index] =
                profile_values[static_cast<uint>(normalization_indices[index])];
        }
        for (ulong index = 1UL; index < params.normalization_count; ++index) {
            const float value = normalization_values[index];
            ulong position = index;
            while (position > 0UL && normalization_values[position - 1UL] > value) {
                normalization_values[position] = normalization_values[position - 1UL];
                position -= 1UL;
            }
            normalization_values[position] = value;
        }
        const ulong middle = params.normalization_count / 2UL;
        float median = normalization_values[middle];
        if ((params.normalization_count & 1UL) == 0UL) {
            median = 0.5f * (normalization_values[middle - 1UL] + median);
        }
        if (!isfinite(median) || median <= 1.0e-12f) {
            set_status(status, local_profile, kStatusInvalidNormalization);
            normalization_values[0] = 1.0f;
        } else {
            normalization_values[0] = median;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (static_cast<ulong>(bin) < params.bins) {
        const float normalized = profile_values[bin] / normalization_values[0];
        if (!isfinite(normalized)) {
            set_status(status, local_profile, kStatusNonFiniteProfile);
        }
        output[static_cast<ulong>(local_profile) * params.bins + bin] = normalized;
    }
}

kernel void xrdmc_metal_run_kernel(
    device const float* images [[buffer(0)]],
    device const ulong* measurement_seeds [[buffer(1)]],
    device const float* scales [[buffer(2)]],
    device const long* csr_indptr [[buffer(3)]],
    device const int* csr_indices [[buffer(4)]],
    device const float* csr_weights [[buffer(5)]],
    device const float* denominators [[buffer(6)]],
    device const int* normalization_indices [[buffer(7)]],
    device float* output [[buffer(8)]],
    device atomic_int* status [[buffer(9)]],
    constant KernelParams& params [[buffer(10)]],
    device const int* measurement_plan_indices [[buffer(11)]],
    threadgroup float* profile_values [[threadgroup(0)]],
    threadgroup float* normalization_values [[threadgroup(1)]],
    uint bin [[thread_position_in_threadgroup]],
    uint local_profile [[threadgroup_position_in_grid]]
) {
    const ulong global_profile = params.profile_offset + local_profile;
    const ulong measurement = global_profile % params.measurements;
    const ulong plan = static_cast<ulong>(measurement_plan_indices[measurement]);
    const ulong indptr_base = plan * (params.bins + 1UL);
    const ulong draw = (global_profile / params.measurements) % params.draws;
    const ulong scale_index = global_profile / (params.measurements * params.draws);
    const float scale_squared = scales[scale_index] * scales[scale_index];
    if (static_cast<ulong>(bin) < params.bins) {
        float sum = 0.0f;
        bool poisson_valid = true;
        for (
            long offset = csr_indptr[indptr_base + bin];
            offset < csr_indptr[indptr_base + bin + 1U];
            ++offset
        ) {
            const ulong pixel = static_cast<ulong>(csr_indices[offset]);
            const float sampled = centered_poisson_sample(
                images[measurement * params.pixels + pixel], scale_squared,
                params.base_seed, measurement_seeds[measurement], scale_index, draw,
                pixel, poisson_valid
            );
            if (!poisson_valid) {
                set_status(status, local_profile, kStatusPoissonFailure);
            }
            sum += csr_weights[offset] * sampled;
        }
        const float integrated = sum / denominators[plan * params.bins + bin];
        profile_values[bin] = integrated;
        if (!isfinite(integrated)) {
            set_status(status, local_profile, kStatusNonFiniteProfile);
        }
    }
    normalize_profile(
        output, status, params, normalization_indices, profile_values,
        normalization_values, bin, local_profile
    );
}

kernel void xrdmc_metal_integrate_kernel(
    device const float* images [[buffer(0)]],
    device const long* csr_indptr [[buffer(1)]],
    device const int* csr_indices [[buffer(2)]],
    device const float* csr_weights [[buffer(3)]],
    device const float* denominators [[buffer(4)]],
    device const int* normalization_indices [[buffer(5)]],
    device float* output [[buffer(6)]],
    device atomic_int* status [[buffer(7)]],
    constant KernelParams& params [[buffer(8)]],
    device const int* measurement_plan_indices [[buffer(9)]],
    threadgroup float* profile_values [[threadgroup(0)]],
    threadgroup float* normalization_values [[threadgroup(1)]],
    uint bin [[thread_position_in_threadgroup]],
    uint local_profile [[threadgroup_position_in_grid]]
) {
    const ulong measurement = params.profile_offset + local_profile;
    const ulong plan = static_cast<ulong>(measurement_plan_indices[measurement]);
    const ulong indptr_base = plan * (params.bins + 1UL);
    if (static_cast<ulong>(bin) < params.bins) {
        float sum = 0.0f;
        for (
            long offset = csr_indptr[indptr_base + bin];
            offset < csr_indptr[indptr_base + bin + 1U];
            ++offset
        ) {
            const ulong pixel = static_cast<ulong>(csr_indices[offset]);
            sum += csr_weights[offset] * images[measurement * params.pixels + pixel];
        }
        const float integrated = sum / denominators[plan * params.bins + bin];
        profile_values[bin] = integrated;
        if (!isfinite(integrated)) {
            set_status(status, local_profile, kStatusNonFiniteProfile);
        }
    }
    normalize_profile(
        output, status, params, normalization_indices, profile_values,
        normalization_values, bin, local_profile
    );
}
