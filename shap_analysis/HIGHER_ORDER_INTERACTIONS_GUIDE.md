# Higher-Order SHAP Interactions Guide

## Overview

This guide shows how to analyze **3-way, 4-way, and higher-order interactions** (e.g., S1×S2×S3) using the new `shap_analysis.higher_order` module.

## Three Methods Available

| Method | Function | Speed | Accuracy | Use Case |
|--------|----------|-------|----------|----------|
| **Method 2** | `calculate_shapley_taylor_interactions` | ⚡ Fast | ~Approximate | Quick exploration |
| **Method 3** | `calculate_treeshap_iq` | 🐌 Slow | ✅ Exact | Validation & publication |
| **Method 4** | `analyze_conditional_interactions` | ⚡ Fast | 📊 Statistical | Interpretation |

## Installation

### For Method 2 (Shapley-Taylor)
Already included! No extra dependencies.

### For Method 3 (TreeSHAP-IQ)
```bash
pip install shapiq
```

## Quick Start

### Method 2: Shapley-Taylor (Fast Approximation)

```python
from shap_analysis import calculate_shapley_taylor_interactions, visualize_higher_order_interactions

# Get trained model
trained_pipeline = pipeline.trained_estimator
model = trained_pipeline.steps[-1][1]

# Compute 3-way interactions
interactions_3way = calculate_shapley_taylor_interactions(
    model=model,
    X=shap_results['X_test_scaled'],
    order=3,  # 3-way: S1×S2×S3
    max_features=5,  # Limit to first 5 features to avoid explosion
    n_samples=100,
    top_k=20
)

print(interactions_3way)

# Visualize
fig = visualize_higher_order_interactions(interactions_3way, order=3, top_k=15)
fig.savefig('3way_interactions.png', dpi=300, bbox_inches='tight')
```

### Method 3: TreeSHAP-IQ (Exact, Slow)

```python
from shap_analysis import calculate_treeshap_iq

# Get trained model
trained_pipeline = pipeline.trained_estimator
model = trained_pipeline.steps[-1][1]

# Compute exact 3-way interactions
result_iq = calculate_treeshap_iq(
    model=model,
    X=shap_results['X_test_scaled'],
    max_order=3,
    n_samples=20,  # Keep small - very expensive!
    interaction_type='shapley_taylor'
)

# View top 3-way interactions
print("Top 10 3-way interactions:")
for features, value in result_iq['top_interactions_per_order'][3][:10]:
    features_str = " × ".join(features)
    print(f"  {features_str}: {value:+.4f}")
```

### Method 4: Conditional Analysis (Interpretable)

```python
from shap_analysis import analyze_conditional_interactions

# Analyze how S1 effect changes when S2 and S3 are both high
conditional_int = analyze_conditional_interactions(
    df_analysis=df_analysis,
    shap_results=shap_results,
    feature_names=['S_1', 'S_2', 'S_3', 'S_4', 'S_5'],
    min_samples=10
)

# Save results
conditional_int.to_csv('conditional_3way_interactions.csv', index=False)
```

## Complete Example for Keele Notebook

Add these cells to `Keele_reconstruction2.py`:

### Cell: Higher-Order Method 2 (Shapley-Taylor)

```python
@app.cell
def _(pipeline, shap_results, Path, plt):
    """Method 2: Shapley-Taylor 3-Way Interactions (Fast)"""
    from shap_analysis import (
        calculate_shapley_taylor_interactions,
        visualize_higher_order_interactions
    )

    # Get trained model
    trained_pipeline = pipeline.trained_estimator
    model = trained_pipeline.steps[-1][1]

    print("="*80)
    print("METHOD 2: SHAPLEY-TAYLOR 3-WAY INTERACTIONS (APPROXIMATE)")
    print("="*80)

    # Compute 3-way interactions
    interactions_3way = calculate_shapley_taylor_interactions(
        model=model,
        X=shap_results['X_test_scaled'],
        order=3,
        max_features=None,  # Use all features
        n_samples=100,
        top_k=20
    )

    print("\nTop 10 3-way interactions:")
    for idx, row in interactions_3way.head(10).iterrows():
        features_str = " × ".join(row['features'])
        print(f"  {features_str}: {row['interaction_strength']:+.4f}")

    # Visualize
    output_path = Path(r'E:\dev\eos_play\jupyter_notebooks\Keele')
    fig_3way = visualize_higher_order_interactions(interactions_3way, order=3, top_k=15)
    fig_3way.savefig(output_path / 'shap_3way_interactions.png',
                     dpi=300, bbox_inches='tight')
    plt.show()

    # Save to CSV
    interactions_3way.to_csv(output_path / 'shap_3way_interactions.csv', index=False)

    print(f"\n✓ Results saved to {output_path}")

    return interactions_3way, fig_3way
```

### Cell: Higher-Order Method 3 (TreeSHAP-IQ)

```python
@app.cell
def _(pipeline, shap_results, Path):
    """Method 3: TreeSHAP-IQ Exact 3-Way Interactions (Slow but Exact)"""
    from shap_analysis import calculate_treeshap_iq

    try:
        # Get trained model
        trained_pipeline = pipeline.trained_estimator
        model = trained_pipeline.steps[-1][1]

        print("="*80)
        print("METHOD 3: TREESHAP-IQ EXACT 3-WAY INTERACTIONS")
        print("="*80)
        print("⚠ This may take several minutes for 20 samples...")

        # Compute exact interactions
        result_iq = calculate_treeshap_iq(
            model=model,
            X=shap_results['X_test_scaled'],
            max_order=3,
            n_samples=20,  # Start small!
            interaction_type='shapley_taylor'
        )

        # Extract and save results
        output_path = Path(r'E:\dev\eos_play\jupyter_notebooks\Keele')

        # Save all orders
        for order in range(1, 4):
            if order in result_iq['top_interactions_per_order']:
                interactions_list = result_iq['top_interactions_per_order'][order]

                df_order = pd.DataFrame([
                    {
                        'features': ' × '.join(features),
                        'interaction_value': value
                    }
                    for features, value in interactions_list
                ])

                df_order.to_csv(
                    output_path / f'treeshap_iq_{order}way_interactions.csv',
                    index=False
                )

                print(f"\n✓ Saved {len(df_order)} {order}-way interactions")

        return result_iq

    except ImportError:
        print("TreeSHAP-IQ not installed.")
        print("To use this method, run: pip install shapiq")
        return None
```

### Cell: Higher-Order Method 4 (Conditional Analysis)

```python
@app.cell
def _(df_analysis, shap_results, feature_names, Path):
    """Method 4: Conditional 3-Way Interaction Analysis"""
    from shap_analysis import analyze_conditional_interactions

    print("="*80)
    print("METHOD 4: CONDITIONAL 3-WAY INTERACTION ANALYSIS")
    print("="*80)

    conditional_int = analyze_conditional_interactions(
        df_analysis=df_analysis,
        shap_results=shap_results,
        feature_names=feature_names,
        min_samples=10
    )

    # Save results
    output_path = Path(r'E:\dev\eos_play\jupyter_notebooks\Keele')
    conditional_int.to_csv(
        output_path / 'conditional_3way_interactions.csv',
        index=False
    )

    print(f"\n✓ Saved {len(conditional_int)} conditional interactions")

    # Show strongest interaction example
    if len(conditional_int) > 0:
        top = conditional_int.iloc[0]
        print(f"\n📌 Strongest conditional interaction:")
        print(f"   {top['feature']}'s effect changes by {top['effect_change']:+.3f}")
        print(f"   when {top['condition_1']} AND {top['condition_2']} are both HIGH")
        print(f"   - Effect overall: {top['effect_overall']:+.3f}")
        print(f"   - Effect conditional: {top['effect_conditional']:+.3f}")

    return conditional_int
```

## Interpretation Guide

### Understanding 3-Way Interactions

A **3-way interaction** S1×S2×S3 means:
> "The combined effect of S1, S2, and S3 together is different from the sum of their pairwise interactions"

### Example Interpretation

```
Top 3-way interaction: S_1 × S_3 × S_5: +0.234
```

**Physical interpretation** (adjust to your SKana components):
- S_1 = Collagen
- S_3 = Mineralization
- S_5 = Disorder marker

**Interpretation**:
> "When collagen, mineralization, and disorder are all present together, they create a strong synergistic cancer signal (+0.234) that goes beyond their pairwise interactions. This three-way combination is a unique cancer signature."

### Conditional Interaction Example

```
Feature: S_1
Condition: S_2 HIGH & S_3 HIGH
Effect change: -0.45
```

**Interpretation**:
> "S_1's cancer effect is strongly suppressed (-0.45) when both S_2 and S_3 are elevated. This suggests S_2 and S_3 together counteract S_1's influence."

## Performance Notes

### Computational Complexity

For **n** features and order **k**:
- Combinations: C(n, k) = n! / (k! × (n-k)!)
- Examples:
  - n=5, k=3: 10 combinations ✅ Fast
  - n=10, k=3: 120 combinations ✅ OK
  - n=10, k=4: 210 combinations ⚠️ Getting slow
  - n=20, k=3: 1140 combinations ⚠️ Slow

### Runtime Estimates

| Method | n=5, order=3 | n=10, order=3 | n=10, order=4 |
|--------|--------------|---------------|---------------|
| Shapley-Taylor | ~5 seconds | ~30 seconds | ~60 seconds |
| TreeSHAP-IQ | ~2 minutes | ~10 minutes | ~30 minutes |
| Conditional | ~1 second | ~3 seconds | ~5 seconds |

## Best Practices

1. **Start with Method 2** (Shapley-Taylor) for exploration
2. **Use Method 4** (Conditional) for interpretation
3. **Validate with Method 3** (TreeSHAP-IQ) only if needed for publication

4. **Keep order ≤ 3** for most applications
5. **Use max_features** to limit combinatorial explosion
6. **Start with small n_samples** for TreeSHAP-IQ

## Troubleshooting

### TreeSHAP-IQ Not Installed
```bash
pip install shapiq
```

### Out of Memory
Reduce `n_samples` or `max_features`:
```python
interactions_3way = calculate_shapley_taylor_interactions(
    model=model,
    X=X,
    order=3,
    max_features=5,  # Only first 5 features
    n_samples=50     # Fewer samples
)
```

### Takes Too Long
Use Shapley-Taylor (Method 2) instead of TreeSHAP-IQ (Method 3):
```python
# Fast approximation
interactions_3way = calculate_shapley_taylor_interactions(...)  # Seconds

# Exact (slow)
# result_iq = calculate_treeshap_iq(...)  # Minutes
```

## References

1. Shapley-Taylor Index: Sundararajan & Najmi (2020)
2. TreeSHAP-IQ: Muschalik et al. (2023) - https://arxiv.org/abs/2310.12082
3. Original SHAP: Lundberg & Lee (2017)

---

**Added to**: `shap_analysis/higher_order.py`
**Exports**: 4 new functions
**Status**: ✅ Ready to use
