"""Example usage for SpectroSVDTransformer and MCRALSTransformer."""

from __future__ import annotations

import numpy as np
import pandas as pd

from xrdanalysis.data_processing.pipeline import MLPipeline
from xrdanalysis.data_processing.spectrokinetic_transformers import (
    MCRALSTransformer,
    SpectroSVDTransformer,
)


def _build_example_df(n_rows: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(123)
    delay = np.linspace(0, 5, 40)
    wav = np.linspace(300, 700, 120)

    rows = []
    for i in range(n_rows):
        c = np.abs(rng.normal(size=(delay.size, 2)))
        s = np.zeros((wav.size, 2), dtype=float)
        s[:, 0] = np.exp(-((wav - 420) ** 2) / (2 * 25**2))
        s[:, 1] = np.exp(-((wav - 580) ** 2) / (2 * 35**2))
        mat = c @ s.T + 0.001 * rng.normal(size=(delay.size, wav.size))

        rows.append(
            {
                "specimen_id": f"spec_{i}",
                "group_id": "gA" if i < 2 else "gB",
                "spectro_matrix": mat,
                "delay_axis": delay,
                "wavelength_axis": wav,
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    df = _build_example_df()

    # Step 1: SVD diagnostics
    svd = SpectroSVDTransformer(max_rank=10)
    df_svd = svd.fit_transform(df)

    # Step 2: ALS decomposition (row-wise)
    als_row = MCRALSTransformer(
        n_components=2,
        init_method="svd",
        maxiter=50,
        thresh=1e-5,
        decomposition_mode="row",
    )
    df_als_row = als_row.fit_transform(df_svd)

    print("Row-wise ALS LOF:")
    print(df_als_row[["specimen_id", "als_lof_pct"]])

    # Step 3: ALS decomposition (grouped tile-delay mode)
    als_group = MCRALSTransformer(
        n_components=2,
        init_method="svd",
        maxiter=50,
        thresh=1e-5,
        decomposition_mode="group",
        group_col="group_id",
        group_strategy="tile_delay",
    )
    df_als_group = als_group.fit_transform(df_svd)

    print("\nGrouped ALS LOF:")
    print(df_als_group[["specimen_id", "group_id", "als_lof_pct"]])

    # Optional integration with existing MLPipeline wrangle path.
    pipeline = MLPipeline(
        data_wrangling_steps=[
            ("svd", SpectroSVDTransformer(max_rank=10)),
            (
                "als",
                MCRALSTransformer(
                    n_components=2,
                    init_method="svd",
                    maxiter=30,
                    thresh=1e-4,
                ),
            ),
        ]
    )
    df_pipe = pipeline.transform(df)
    print("\nPipeline transformed columns include:")
    print([c for c in ["svd_lof_curve", "als_lof_pct", "als_C", "als_S"] if c in df_pipe.columns])


if __name__ == "__main__":
    main()
