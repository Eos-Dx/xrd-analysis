# SK-Ana Parity Fixtures

Optional parity fixtures for comparing Python outputs against SK-Ana (R) reference outputs.

Expected file:
- `parity_fixture.npz`
- `I_vs_q_100samples.dat` (Keele standard data fixture used by spectrokinetic tests)
- `xrd_component_fat.txt` (Keele fixed spectral profile for fat)
- `xrd_component_water.txt` (Keele fixed spectral profile for water)

Generate with:
- `Rscript /Users/sad/dev/xrd-analysis/src/tests/data_processing/build_skana_reference_fixture.R`

When the fixture file is absent, parity tests are skipped.
