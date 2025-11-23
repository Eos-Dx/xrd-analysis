# SHAP Analysis Package for Cancer Detection

Comprehensive SHAP-based interpretability framework for tree-based ML models predicting cancer from XRD-SKana features.

## Features

- **Main Effects Analysis**: Compute SHAP values for individual SKana component contributions
- **Interaction Analysis**: Calculate pairwise SHAP interactions to identify synergies
- **Diagnostic Rules**: Automatically generate human-readable diagnostic rules
- **Visualization Suite**: Heatmaps, bar plots, summary plots, and force plots
- **Validation Tools**: SHAP additivity checks and cross-validation stability analysis

## Installation

```python
import sys
sys.path.insert(0, r"E:\dev\xrd-analysis")
```

## Quick Start

```python
from shap_analysis import (
    calculate_shap_values_with_original,
    calculate_shap_interactions,
    plot_shap_interaction_heatmap,
    generate_diagnostic_rules
)
from shap_cancer_decision_analysis import analyze_cancer_decision

# 1. Calculate main effects
shap_results = calculate_shap_values_with_original(
    pipeline=trained_pipeline,
    df_original=df,
    y_column='isCancerDiagnosed',
    group_col='specimenId',
    feature_names=['S_1', 'S_2', 'S_3', 'S_4', 'S_5']
)

# 2. Calculate interactions
interaction_results = calculate_shap_interactions(
    pipeline=trained_pipeline,
    df_original=df,
    feature_names=['S_1', 'S_2', 'S_3', 'S_4', 'S_5']
)

# 3. Analyze decisions
df_analysis = analyze_cancer_decision(shap_results)

# 4. Generate diagnostic rules
rules = generate_diagnostic_rules(shap_results, interaction_results, df_analysis)

# 5. Visualize interactions
fig = plot_shap_interaction_heatmap(interaction_results)
fig.savefig('interaction_heatmap.png')
```

## Module Structure

- **core.py**: SHAP value computation (main effects and interactions)
- **visualization.py**: Plotting functions
- **interpretation.py**: Diagnostic rule extraction
- **validation.py**: SHAP validation and stability analysis

## Pipeline Integration

Compatible with `xrdanalysis.data_processing.pipeline.MLPipeline`.

Supports:
- LightGBM (`LGBMClassifier`)
- RandomForest (`RandomForestClassifier`)
- XGBoost (`XGBClassifier`)
- Any scikit-learn compatible tree ensemble

## Output Structure

```
results/
├── shap_values/           # Pickled SHAP results
├── figures/               # PNG/PDF visualizations
├── summaries/             # CSV tables
└── reports/               # Markdown reports

notebooks/                 # Jupyter/Marimo notebooks
```

## References

- Lundberg, S. M., & Lee, S. I. (2017). "A unified approach to interpreting model predictions." NeurIPS.
- TreeSHAP: https://arxiv.org/abs/1802.03888

## Author

Development team @ XRD Analysis Project
