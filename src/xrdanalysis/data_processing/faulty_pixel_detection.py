"""
Faulty Pixel Detection Transformer

Detects faulty pixels in XRD detector images using multiple strategies:
- Dead/zero pixels (consistently zero across frames)
- Local region deviation (pixel vs neighbor blocks)
- Global statistical outliers (z-score based)
- Temporal consistency filtering
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Set, Dict, Optional


class FaultyPixelDetector:
    """
    Detect faulty pixels independently for primary and secondary detectors.
    
    Uses multiple detection strategies:
    1. Dead/zero pixels: pixels that are near-zero in many frames
    2. Local region deviation: pixels that deviate from 2x2/3x3 neighborhood
    3. Global statistical outliers: z-score based outlier detection
    4. Temporal consistency: filters to pixels abnormal in >N% of frames
    
    Parameters
    ----------
    region_size : int, default=2
        Neighborhood block size (2 or 3) for local consistency checks
    outlier_n_std : float, default=3.0
        Global z-score threshold for per-pixel mean/std outliers
    zero_frac_threshold : float, default=0.6
        Fraction of frames with (near-)zero to call a pixel 'dead' (0.0-1.0)
    temporal_consistency : float, default=0.7
        Fraction of frames a pixel must be abnormal to keep it (0.0-1.0).
        Set to 0 to keep ALL detected faulty pixels (i.e., faulty in any single frame)
    debug : bool, default=False
        Print extra diagnostics
    """
    
    def __init__(
        self,
        region_size: int = 2,
        outlier_n_std: float = 3.0,
        zero_frac_threshold: float = 0.6,
        temporal_consistency: float = 0.7,
        debug: bool = False,
    ):
        self.region_size = int(region_size)
        self.outlier_n_std = float(outlier_n_std)
        self.zero_frac_threshold = float(zero_frac_threshold)
        self.temporal_consistency = float(temporal_consistency)
        self.debug = bool(debug)

    def _find_image_column(self, df: pd.DataFrame) -> Optional[str]:
        """Heuristic: find first column with 2D ndarray values"""
        for col in df.columns:
            s = df[col].dropna()
            if not len(s):
                continue
            v = s.iloc[0]
            try:
                arr = np.asarray(v)
                if arr.ndim == 2:
                    return col
            except Exception:
                continue
        return None

    def _split_primary_secondary(
        self, df: pd.DataFrame, name_field: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split dataframe into primary and secondary measurements"""
        mask_secondary = df[name_field].astype(str).str.contains(
            'SECONDARY', na=False, case=False
        )
        return df[~mask_secondary].copy(), df[mask_secondary].copy()

    def _stack_images(self, df: pd.DataFrame, image_col: str) -> Optional[np.ndarray]:
        """Stack 2D images into 3D array (N, H, W)"""
        imgs = []
        for v in df[image_col].dropna():
            try:
                a = np.asarray(v, dtype=float)
                if a.ndim == 2:
                    imgs.append(a)
            except Exception:
                continue
        if not imgs:
            return None
        
        # Align to smallest H×W
        h = min(im.shape[0] for im in imgs)
        w = min(im.shape[1] for im in imgs)
        if h <= 1 or w <= 1:
            return None
        
        imgs_cropped = [im[:h, :w] for im in imgs]
        return np.stack(imgs_cropped, axis=0)  # (N, H, W)

    def _dead_zero_pixels(self, images: np.ndarray) -> Set[Tuple[int, int]]:
        """Find pixels that are near-zero in many frames"""
        eps = 1e-8
        frac_zero = np.mean((images <= eps) | ~np.isfinite(images), axis=0)
        return set(zip(*np.where(frac_zero >= self.zero_frac_threshold)))

    def _region_deviation(self, images: np.ndarray) -> Set[Tuple[int, int]]:
        """Compare pixel means to block means (2×2 or 3×3)"""
        n, h, w = images.shape
        k = self.region_size
        means = np.nanmean(images, axis=0)
        faulty = set()
        
        for i in range(0, h - k + 1, k):
            for j in range(0, w - k + 1, k):
                blk = means[i:i+k, j:j+k]
                m = np.nanmean(blk)
                if not np.isfinite(m):
                    continue
                
                low_thr = m * 0.3   # very dim vs neighbors
                high_thr = m * 3.5  # very bright vs neighbors
                
                for di in range(k):
                    for dj in range(k):
                        val = blk[di, dj]
                        if not np.isfinite(val):
                            faulty.add((i+di, j+dj))
                        elif val < low_thr or val > high_thr:
                            faulty.add((i+di, j+dj))
        
        return faulty

    def _global_outliers(self, images: np.ndarray) -> Set[Tuple[int, int]]:
        """Detect per-pixel outliers using z-scores"""
        mu = np.nanmean(images)
        sd = np.nanstd(images)
        if not np.isfinite(sd) or sd <= 0:
            return set()
        
        pmu = np.nanmean(images, axis=0)
        psd = np.nanstd(images, axis=0)
        z_mu = (pmu - mu) / (sd + 1e-12)
        z_sd = (psd - np.nanmean(psd)) / (np.nanstd(psd) + 1e-12)
        mask = (np.abs(z_mu) > self.outlier_n_std) | (z_sd > self.outlier_n_std)
        return set(zip(*np.where(mask)))

    def _temporal_keep(
        self, images: np.ndarray, candidates: Set[Tuple[int, int]]
    ) -> Set[Tuple[int, int]]:
        """Filter candidates: keep only if abnormal in >N% of frames.
        
        Note: If temporal_consistency=0, returns all candidates (i.e., faulty in ANY frame).
        This is appropriate when a single faulty pixel in any frame should be masked.
        """
        if images is None or not candidates:
            return set()
        
        # If temporal_consistency is 0, keep all candidates
        if self.temporal_consistency <= 0:
            return candidates
        
        n = images.shape[0]
        need = max(1, int(np.ceil(n * self.temporal_consistency)))
        kept = set()
        
        for (i, j) in candidates:
            pix = images[:, i, j]
            cnt = np.sum((~np.isfinite(pix)) | (pix <= 1e-8))
            if cnt >= need:
                kept.add((i, j))
        
        return kept

    def detect(
        self, df: pd.DataFrame, name_field: str = 'meas_name'
    ) -> Tuple[Set[Tuple[int, int]], Set[Tuple[int, int]], Dict]:
        """
        Detect faulty pixels for primary and secondary detectors.
        
        Returns
        -------
        faulty_prim : set of (i, j) tuples
        faulty_sec : set of (i, j) tuples
        stats : dict with detection statistics
        """
        img_col = self._find_image_column(df)
        if img_col is None:
            if self.debug:
                print("No 2D image-like column found.")
            return set(), set(), {'image_column': None}
        
        df_prim, df_sec = self._split_primary_secondary(df, name_field)
        images_prim = self._stack_images(df_prim, img_col)
        images_sec = self._stack_images(df_sec, img_col)

        def one_side(images):
            if images is None:
                return set()
            cand = set()
            cand |= self._dead_zero_pixels(images)
            cand |= self._region_deviation(images)
            cand |= self._global_outliers(images)
            return self._temporal_keep(images, cand)

        f_prim = one_side(images_prim)
        f_sec = one_side(images_sec)
        
        stats = {
            'image_column': img_col,
            'n_frames_prim': 0 if images_prim is None else int(images_prim.shape[0]),
            'n_frames_sec': 0 if images_sec is None else int(images_sec.shape[0]),
            'faulty_prim': len(f_prim),
            'faulty_sec': len(f_sec),
        }
        
        return f_prim, f_sec, stats

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform: detect faulty pixels and add masks to dataframe.
        
        Adds columns:
        - 'detector': 'PRIM', 'SEC', etc. based on meas_name
        - 'faulty_pixel_mask': numpy array of faulty pixel coordinates for that detector
        
        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe with measurements
        
        Returns
        -------
        df : pd.DataFrame
            DataFrame with detector and faulty_pixel_mask columns added
        """
        df = df.copy()
        
        # Detect faulty pixels
        f_prim, f_sec, stats = self.detect(df)
        
        # Convert to numpy arrays
        fp_prim = np.array(sorted(list(f_prim)), dtype=int) if len(f_prim) else np.empty((0, 2), dtype=int)
        fp_sec = np.array(sorted(list(f_sec)), dtype=int) if len(f_sec) else np.empty((0, 2), dtype=int)
        
        # Create detector column based on meas_name
        df['detector'] = 'PRIM'
        mask_secondary = df['meas_name'].astype(str).str.contains('SECONDARY', na=False, case=False)
        df.loc[mask_secondary, 'detector'] = 'SEC'
        
        # Add faulty_pixel_mask column
        def get_mask_for_row(row):
            if row['detector'] == 'PRIM':
                return fp_prim
            elif row['detector'] == 'SEC':
                return fp_sec
            else:
                return np.empty((0, 2), dtype=int)
        
        df['faulty_pixel_mask'] = df.apply(get_mask_for_row, axis=1)
        
        # Store stats for reference
        self.stats_ = stats
        
        return df
    
    def fit_transform(self, df: pd.DataFrame, y=None) -> pd.DataFrame:
        """Fit and transform (no-op fit, just transform)."""
        return self.transform(df)
