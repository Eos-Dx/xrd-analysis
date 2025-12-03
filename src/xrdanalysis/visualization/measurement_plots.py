"""
Visualization utilities for XRD measurements grouped by type.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import Optional, Dict, Tuple


def plot_measurements_by_type(
    df: pd.DataFrame,
    type_col: str = "type_measurement",
    q_col: str = "q_range",
    y_col: str = "radial_profile_data",
    label_col: Optional[str] = None,
    color_map: Optional[Dict] = None,
    title_prefix: str = "Measurements",
    xlim_map: Optional[Dict[str, Tuple[float, float]]] = None,
    logy: bool = True,
    figsize_per_type: Tuple[float, float] = (5, 4),
) -> plt.Figure:
    """
    Plot measurements grouped by type, with optional label-based coloring.
    
    Creates one subplot per measurement type (WAXS, SAXS, MIXED, etc.) with
    individual measurement curves plotted together. Curves can be colored by
    a label column (e.g., species, diagnosis).
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with measurement data
    type_col : str, default="type_measurement"
        Column name containing measurement type (WAXS, SAXS, etc.)
    q_col : str, default="q_range"
        Column name for q_range (x-axis)
    y_col : str, default="radial_profile_data"
        Column name for radial profile data (y-axis)
    label_col : str, optional
        Column name for labels to colorize curves (e.g., 'species', 'diagnosis')
    color_map : dict, optional
        Mapping of label values to colors. If None, auto-generated from matplotlib colormap
    title_prefix : str, default="Measurements"
        Prefix for plot titles. Full titles will be "prefix — TYPE"
    xlim_map : dict, optional
        Mapping of measurement type to (xmin, xmax) limits.
        Example: {'WAXS': (0, 20), 'SAXS': (0.5, 2.0)}
    logy : bool, default=True
        Use logarithmic scale for y-axis
    figsize_per_type : tuple, default=(5, 4)
        Figure size (width, height) per measurement type subplot
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure
    
    Examples
    --------
    >>> # Plot with automatic coloring by species
    >>> fig = plot_measurements_by_type(
    ...     df,
    ...     label_col='species',
    ...     xlim_map={'WAXS': (0, 20), 'SAXS': (0.5, 2.0)},
    ... )
    
    >>> # Plot with custom color mapping
    >>> color_map = {'HUMAN': 'red', 'MOUSE': 'blue'}
    >>> fig = plot_measurements_by_type(
    ...     df,
    ...     label_col='species',
    ...     color_map=color_map,
    ... )
    """
    _df = df.copy()
    
    # Get unique measurement types
    types = sorted(_df[type_col].dropna().unique())
    if not types:
        print("No measurement types found.")
        return None
    
    # Auto-generate color map if not provided
    if label_col is not None:
        unique_labels = sorted(_df[label_col].dropna().unique())
        if color_map is None:
            cmap = cm.get_cmap('tab10', max(len(unique_labels), 1))
            color_map = {label: cmap(i % cmap.N) for i, label in enumerate(unique_labels)}
    
    # Create subplots
    n_types = len(types)
    fig, axes = plt.subplots(
        1, n_types,
        figsize=(figsize_per_type[0] * n_types, figsize_per_type[1]),
        squeeze=False
    )
    axes = axes.flatten()
    
    for ax, mtype in zip(axes[:n_types], types):
        df_type = _df[_df[type_col] == mtype]
        
        # Plot curves
        for _, row in df_type.iterrows():
            q = np.asarray(row.get(q_col, []), dtype=float)
            y = np.asarray(row.get(y_col, []), dtype=float)
            
            if q.size == 0 or y.size == 0:
                continue
            
            n = min(q.size, y.size)
            q, y = q[:n], y[:n]
            
            # Determine color
            color = 'black'
            label = None
            
            if label_col is not None and label_col in row.index:
                lab_val = row[label_col]
                if color_map is not None and lab_val in color_map:
                    color = color_map[lab_val]
                label = str(lab_val)
            
            ax.plot(q, y, color=color, alpha=0.5, lw=1)
        
        # Styling
        ax.set_title(f"{title_prefix} — {mtype}")
        ax.set_xlabel("q range (nm⁻¹)")
        ax.set_ylabel("Intensity (a.u.)")
        ax.grid(which="major", linestyle="--", alpha=0.3)
        ax.minorticks_on()
        ax.grid(which="minor", linestyle=":", alpha=0.2)
        
        if logy:
            ax.set_yscale("log")
        
        # Set x-limits if provided
        if xlim_map is not None and mtype in xlim_map:
            ax.set_xlim(*xlim_map[mtype])
    
    # Hide unused subplots
    for ax in axes[n_types:]:
        ax.set_visible(False)
    
    # Add legend if labels were used
    if label_col is not None and color_map is not None:
        handles = [
            plt.Line2D([0], [0], color=color_map[label], lw=2, label=str(label))
            for label in sorted(color_map.keys())
        ]
        fig.legend(
            handles=handles,
            loc='upper center',
            bbox_to_anchor=(0.5, -0.02),
            ncol=len(handles),
            frameon=False,
        )
    
    plt.tight_layout()
    return fig
