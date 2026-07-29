# SNRTransformer contract

`SNRTransformer` adds scalar signal-to-noise metrics to rows containing a
1D XRD profile. It preserves the input DataFrame and writes:

- `noise_std`, `snr_linear`, and `snr_db`;
- `snr`, a legacy alias for `snr_db`;
- `snr_method_used`;
- optionally, `radial_profile_data_snr` and `radial_profile_residual`.

## Methods

`snr_method="poisson"` uses aligned `radial_profile_data` and
`radial_profile_sigma` from azimuthal integration. For finite aligned
intensity samples with positive finite sigma it calculates:

```text
SNR(q)      = abs(I(q)) / sigma(q)
snr_linear  = RMS(SNR(q))
snr_db      = 20 * log10(snr_linear)
noise_std   = RMS(sigma(q))
```

For Poisson mode, new transformer objects use native aligned samples by default.
`regrid_poisson=True` explicitly requests the historical regridded Poisson
path. Restored legacy pickles retain their historical regridded Poisson scalar
calculation.
Native scalar metrics ignore q spacing, ordering, and non-finite q values;
q remains relevant to the legacy regridded path and to profile-array output.

`snr_method="residual"` is retained for legacy analyses: it normalizes,
smooths, and estimates SNR from the residual. `snr_method="auto"` uses valid
Poisson sigma when available, otherwise falls back to the residual method.
These modes, plus the alias and optional profile arrays, are intentionally
specific to `xrd-analysis`.

## XRD-preprocessing reference

`XRD-preprocessing` is a read-only local reference for the common native
Poisson scalar contract only. Normal `xrd-analysis` CI neither imports it nor
requires a sibling checkout. Maintainers may run the opt-in cross-repository
gate with:

```bash
XRD_PREPROCESSING_ROOT=/absolute/path/to/XRD-preprocessing pytest -m cross_repo
```

The gate compares valid uniform and nonuniform aligned profiles only. It does
not compare residual or auto modes, batch-tolerance behavior, profile arrays,
or FaultyPixel behavior. A parity result is compatibility evidence; it does
not authorize changes to `XRD-preprocessing` from this repository.

## Deferred work

`FaultyPixelDetector` refactoring is explicitly deferred. Do not move, split,
or change its public imports, serialization, or pipeline behavior in the
current refactoring stages.
