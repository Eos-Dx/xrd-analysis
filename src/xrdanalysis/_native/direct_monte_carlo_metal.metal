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

constant float kFourPi = 12.56637061435917295385f;
constant float kInverseMetresToInverseNanometres = 1.0e-9f;
constant int kStatusInvalidGeometry = 4;

struct DetectorGeometry {
    float distance;
    float poni1;
    float poni2;
    float pixel1;
    float pixel2;
    float wavelength;
    uint orientation;
    uint reserved;
};

struct GeometryKernelParams {
    ulong measurements;
    ulong pixels;
    ulong rows;
    ulong columns;
    ulong scale_count;
    ulong draws;
    ulong bins;
    ulong normalization_count;
    ulong profile_offset;
    ulong batch_profiles;
    ulong draw_offset;
    ulong base_seed;
    float q_min;
    float q_delta;
    uint sample_poisson;
    uint reserved;
};

inline void atomic_add_float(device atomic_uint* address, float value) {
    uint observed = atomic_load_explicit(address, memory_order_relaxed);
    while (true) {
        const float current = as_type<float>(observed);
        const uint desired = as_type<uint>(current + value);
        if (atomic_compare_exchange_weak_explicit(
                address,
                &observed,
                desired,
                memory_order_relaxed,
                memory_order_relaxed
            )) {
            return;
        }
    }
}

inline float geometry_q(float p1, float p2, float distance, float wavelength) {
    const float radius = sqrt(p1 * p1 + p2 * p2);
    const float two_theta = atan2(radius, distance);
    return kFourPi * sin(0.5f * two_theta) / wavelength *
        kInverseMetresToInverseNanometres;
}

inline void physical_pixel_center(
    ulong pixel,
    constant GeometryKernelParams& params,
    thread const DetectorGeometry& geometry,
    thread float& center1,
    thread float& center2
) {
    const ulong image_row = pixel / params.columns;
    const ulong image_column = pixel - image_row * params.columns;
    float detector_row = static_cast<float>(image_row);
    float detector_column = static_cast<float>(image_column);
    if (geometry.orientation == 1U) {
        detector_row = static_cast<float>(params.rows - image_row - 1UL);
        detector_column = static_cast<float>(params.columns - image_column - 1UL);
    } else if (geometry.orientation == 2U) {
        detector_row = static_cast<float>(params.rows - image_row - 1UL);
    } else if (geometry.orientation == 4U) {
        detector_column = static_cast<float>(params.columns - image_column - 1UL);
    }
    center1 = (detector_row + 0.5f) * geometry.pixel1;
    center2 = (detector_column + 0.5f) * geometry.pixel2;
}

inline void accumulate_bbox_pixel(
    device atomic_uint* signal,
    device atomic_uint* denominator,
    ulong profile,
    ulong bins,
    float q_min,
    float q_delta,
    float q_center,
    float q_half_width,
    float value,
    float solid_angle
) {
    const float fractional_min = (q_center - q_half_width - q_min) / q_delta;
    const float fractional_max = (q_center + q_half_width - q_min) / q_delta;
    int bin_min = static_cast<int>(fractional_min);
    int bin_max = static_cast<int>(fractional_max);
    if (bin_max < 0 || bin_min >= static_cast<int>(bins)) {
        return;
    }
    bin_min = max(bin_min, 0);
    bin_max = min(bin_max, static_cast<int>(bins) - 1);
    const ulong profile_base = profile * bins;
    if (bin_min == bin_max) {
        const ulong index = profile_base + static_cast<ulong>(bin_min);
        atomic_add_float(signal + index, value);
        atomic_add_float(denominator + index, solid_angle);
        return;
    }
    const float inverse_span = 1.0f / (fractional_max - fractional_min);
    const float left = (static_cast<float>(bin_min) + 1.0f - fractional_min) *
        inverse_span;
    const float right = (fractional_max - static_cast<float>(bin_max)) *
        inverse_span;
    ulong index = profile_base + static_cast<ulong>(bin_min);
    atomic_add_float(signal + index, left * value);
    atomic_add_float(denominator + index, left * solid_angle);
    index = profile_base + static_cast<ulong>(bin_max);
    atomic_add_float(signal + index, right * value);
    atomic_add_float(denominator + index, right * solid_angle);
    for (int bin = bin_min + 1; bin < bin_max; ++bin) {
        index = profile_base + static_cast<ulong>(bin);
        atomic_add_float(signal + index, inverse_span * value);
        atomic_add_float(denominator + index, inverse_span * solid_angle);
    }
}

kernel void xrdmc_metal_geometry_clear_kernel(
    device atomic_uint* signal [[buffer(0)]],
    device atomic_uint* denominator [[buffer(1)]],
    device atomic_int* status [[buffer(2)]],
    constant GeometryKernelParams& params [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    const ulong values = params.batch_profiles * params.bins;
    if (index < values) {
        atomic_store_explicit(signal + index, 0U, memory_order_relaxed);
        atomic_store_explicit(denominator + index, 0U, memory_order_relaxed);
    }
    if (index < params.batch_profiles) {
        atomic_store_explicit(status + index, 0, memory_order_relaxed);
    }
}

kernel void xrdmc_metal_geometry_accumulate_kernel(
    device const float* images [[buffer(0)]],
    device const uchar* masks [[buffer(1)]],
    device const ulong* measurement_seeds [[buffer(2)]],
    device const float* scales [[buffer(3)]],
    device const DetectorGeometry* nominal_geometry [[buffer(4)]],
    device const float* effective_distance [[buffer(5)]],
    device const float* poni1 [[buffer(6)]],
    device const float* poni2 [[buffer(7)]],
    device atomic_uint* signal [[buffer(8)]],
    device atomic_uint* denominator [[buffer(9)]],
    device atomic_int* status [[buffer(10)]],
    constant GeometryKernelParams& params [[buffer(11)]],
    uint index [[thread_position_in_grid]]
) {
    const ulong wide_index = static_cast<ulong>(index);
    const ulong local_profile = wide_index / params.pixels;
    if (local_profile >= params.batch_profiles) {
        return;
    }
    const ulong pixel = wide_index - local_profile * params.pixels;
    const ulong global_profile = params.profile_offset + local_profile;
    const ulong measurement = global_profile % params.measurements;
    const ulong draw = (global_profile / params.measurements) % params.draws;
    const ulong scale_index = global_profile /
        (params.measurements * params.draws);
    const ulong image_index = measurement * params.pixels + pixel;
    if (masks[image_index] != 0U) {
        return;
    }

    DetectorGeometry geometry = nominal_geometry[measurement];
    const ulong geometry_index = draw * params.measurements + measurement;
    geometry.distance = effective_distance[geometry_index];
    geometry.poni1 = poni1[geometry_index];
    geometry.poni2 = poni2[geometry_index];
    if (
        !isfinite(geometry.distance) || geometry.distance <= 0.0f ||
        !isfinite(geometry.poni1) || !isfinite(geometry.poni2)
    ) {
        atomic_fetch_max_explicit(
            status + local_profile,
            kStatusInvalidGeometry,
            memory_order_relaxed
        );
        return;
    }

    float detector_center1 = 0.0f;
    float detector_center2 = 0.0f;
    physical_pixel_center(
        pixel,
        params,
        geometry,
        detector_center1,
        detector_center2
    );
    const float center1 = detector_center1 - geometry.poni1;
    const float center2 = detector_center2 - geometry.poni2;
    const float q_center = geometry_q(
        center1,
        center2,
        geometry.distance,
        geometry.wavelength
    );
    const float half1 = 0.5f * geometry.pixel1;
    const float half2 = 0.5f * geometry.pixel2;
    float q_half_width = 0.0f;
    q_half_width = max(
        q_half_width,
        abs(geometry_q(
            center1 - half1,
            center2 - half2,
            geometry.distance,
            geometry.wavelength
        ) - q_center)
    );
    q_half_width = max(
        q_half_width,
        abs(geometry_q(
            center1 + half1,
            center2 - half2,
            geometry.distance,
            geometry.wavelength
        ) - q_center)
    );
    q_half_width = max(
        q_half_width,
        abs(geometry_q(
            center1 + half1,
            center2 + half2,
            geometry.distance,
            geometry.wavelength
        ) - q_center)
    );
    q_half_width = max(
        q_half_width,
        abs(geometry_q(
            center1 - half1,
            center2 + half2,
            geometry.distance,
            geometry.wavelength
        ) - q_center)
    );
    if (!isfinite(q_center) || !isfinite(q_half_width)) {
        atomic_fetch_max_explicit(
            status + local_profile,
            kStatusInvalidGeometry,
            memory_order_relaxed
        );
        return;
    }

    float value = images[image_index];
    if (params.sample_poisson != 0U) {
        bool poisson_valid = true;
        const float scale = scales[scale_index];
        value = centered_poisson_sample(
            value,
            scale * scale,
            params.base_seed,
            measurement_seeds[measurement],
            scale_index,
            params.draw_offset + draw,
            pixel,
            poisson_valid
        );
        if (!poisson_valid) {
            atomic_fetch_max_explicit(
                status + local_profile,
                kStatusPoissonFailure,
                memory_order_relaxed
            );
            return;
        }
    }
    const float radius_squared = center1 * center1 + center2 * center2;
    const float distance_squared = geometry.distance * geometry.distance;
    const float cosine = geometry.distance /
        sqrt(distance_squared + radius_squared);
    const float solid_angle = cosine * cosine * cosine;
    accumulate_bbox_pixel(
        signal,
        denominator,
        local_profile,
        params.bins,
        params.q_min,
        params.q_delta,
        q_center,
        q_half_width,
        value,
        solid_angle
    );
}

kernel void xrdmc_metal_geometry_normalize_kernel(
    device const float* signal [[buffer(0)]],
    device const float* denominator [[buffer(1)]],
    device const int* normalization_indices [[buffer(2)]],
    device float* output [[buffer(3)]],
    device atomic_int* status [[buffer(4)]],
    constant GeometryKernelParams& params [[buffer(5)]],
    threadgroup float* profile_values [[threadgroup(0)]],
    threadgroup float* normalization_values [[threadgroup(1)]],
    uint bin [[thread_position_in_threadgroup]],
    uint local_profile [[threadgroup_position_in_grid]]
) {
    if (local_profile >= params.batch_profiles) {
        return;
    }
    if (static_cast<ulong>(bin) < params.bins) {
        const ulong index = static_cast<ulong>(local_profile) * params.bins + bin;
        const float norm = denominator[index];
        if (!isfinite(norm) || norm <= 0.0f) {
            profile_values[bin] = 0.0f;
        } else {
            const float integrated = signal[index] / norm;
            profile_values[bin] = integrated;
            if (!isfinite(integrated)) {
                set_status(status, local_profile, kStatusNonFiniteProfile);
            }
        }
    }
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
        const ulong index = static_cast<ulong>(local_profile) * params.bins + bin;
        const float normalized = profile_values[bin] / normalization_values[0];
        output[index] = normalized;
        if (!isfinite(normalized)) {
            set_status(status, local_profile, kStatusNonFiniteProfile);
        }
    }
}
