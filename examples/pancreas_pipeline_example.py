"""
Example: XRD Data Processing Pipeline
======================================

This example demonstrates how to use MLPipeline with three transformers:
1. FaultyPixelDetector - detects faulty pixels in detector images
2. DetectorJoiner - joins primary and secondary detector measurements
3. MeasurementTypeClassifier - classifies measurements as WAXS, SAXS, or MIXED

The pipeline.transform() method returns:
- If stat=False: just the transformed dataframe
- If stat=True: tuple of (transformed_dataframe, statistics_dict)
"""

import pandas as pd
import numpy as np
from pathlib import Path

from xrdanalysis.data_processing.pipeline import MLPipeline
from xrdanalysis.data_processing import (
    FaultyPixelDetector,
    MeasurementTypeClassifier,
)
from xrdanalysis.data_processing.transformers import DetectorJoiner
from xrdanalysis.data_processing.utility_functions import h5_to_df


def main():
    # Load your data
    folder = Path("/Users/sad/dev/Data")
    h5_path = folder / 'AUTO_PROJECT_Project_2_Grant_2_Pancreatic_Cancer.h5'
    
    df_calib, df = h5_to_df(h5_path)
    print(f"Loaded dataframe shape: {df.shape}")
    
    # Create pipeline with three transformers
    pipeline = MLPipeline(
        data_wrangling_steps=[
            ('faulty_pixel_detection', FaultyPixelDetector(
                region_size=2,
                outlier_n_std=3.0,
                zero_frac_threshold=0.6,
                temporal_consistency=0.7,
                debug=False
            )),
            ('detector_joining', DetectorJoiner(
                name_field='meas_name',
                calibration_mode='poni',
                interpolation_q_range={
                    'SAXS': (1, 2),
                    'WAXS': (3, 21.0),
                },
                debug=False,
                npt=100,
                angles=90
            )),
            ('type_classification', MeasurementTypeClassifier(
                distance_col='calculated_distance',
                output_col='type_measurement',
                waxs_threshold=0.05,
                saxs_threshold=0.1,
            )),
        ]
    )
    
    # Transform WITHOUT statistics
    print("\n" + "="*60)
    print("Transform without statistics (stat=False)")
    print("="*60)
    df_transformed = pipeline.transform(df, stat=False)
    print(f"Transformed dataframe shape: {df_transformed.shape}")
    print(f"Columns: {list(df_transformed.columns)}")
    
    # Transform WITH statistics
    print("\n" + "="*60)
    print("Transform with statistics (stat=True)")
    print("="*60)
    df_transformed, stats = pipeline.transform(df, stat=True)
    print(f"Transformed dataframe shape: {df_transformed.shape}")
    
    # Display statistics from each transformer
    print("\n" + "="*60)
    print("STATISTICS FROM TRANSFORMERS")
    print("="*60)
    
    for step_name, step_stats in stats.items():
        print(f"\n{step_name.upper()}:")
        if isinstance(step_stats, dict):
            for key, value in step_stats.items():
                print(f"  {key}: {value}")
        else:
            print(f"  {step_stats}")
    
    # You can also access individual transformer stats via pipeline steps
    print("\n" + "="*60)
    print("Access transformer stats directly from pipeline")
    print("="*60)
    
    for step_name, transformer in pipeline.data_wrangling_steps:
        print(f"\n{step_name}:")
        if hasattr(transformer, 'stats_'):
            print(f"  stats_: {transformer.stats_}")
        elif hasattr(transformer, 'get_stats'):
            try:
                print(f"  get_stats(): {transformer.get_stats()}")
            except:
                print("  get_stats() not available (need to call transform first)")
    
    return df_transformed, stats


if __name__ == "__main__":
    df_result, stats = main()
