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
    5. Beam center exclusion: optionally excludes intense beam center region
    
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
    exclude_beam_center_radius : float or None, default=None
        If set, excludes pixels within this radius fraction from beam center.
        For example, 0.15 excludes pixels within 15% of detector size from center.
        Beam center coordinates are read from PONI files (Poni1, Poni2).
        If None, uses full image for faulty pixel detection.
    poni_column : str, default='ponifile'
        Column name containing PONI file content (as string)
    debug : bool, default=False
        Print extra diagnostics
    """
    
    def __init__(
        self,
        region_size: int = 2,
        outlier_n_std: float = 3.0,
        zero_frac_threshold: float = 0.6,
        temporal_consistency: float = 0.7,
        exclude_beam_center_radius: Optional[float] = None,
        poni_column: str = 'ponifile',
        debug: bool = False,
    ):
        self.region_size = int(region_size)
        self.outlier_n_std = float(outlier_n_std)
        self.zero_frac_threshold = float(zero_frac_threshold)
        self.temporal_consistency = float(temporal_consistency)
        self.exclude_beam_center_radius = exclude_beam_center_radius
        self.poni_column = poni_column
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
    
    def _read_beam_center_from_poni(self, poni_content: str) -> Optional[Tuple[float, float]]:
        """Extract beam center (poni1, poni2) from PONI file content.
        
        Returns
        -------
        (poni1, poni2) : tuple of float or None
            Beam center coordinates in meters, or None if not found
        """
        if pd.isna(poni_content) or not isinstance(poni_content, str):
            return None
        
        import re
        poni1, poni2 = None, None
        
        # Match lines like "Poni1: 0.123456"
        match1 = re.search(r'^Poni1:\s*([0-9.eE+-]+)', poni_content, re.MULTILINE)
        match2 = re.search(r'^Poni2:\s*([0-9.eE+-]+)', poni_content, re.MULTILINE)
        
        if match1:
            poni1 = float(match1.group(1))
        if match2:
            poni2 = float(match2.group(1))
        
        if poni1 is not None and poni2 is not None:
            return (poni1, poni2)
        return None
    
    def _get_beam_center_pixels(self, df: pd.DataFrame, detector_type: str = 'PRIM') -> Optional[Tuple[int, int]]:
        """Get average beam center in pixel coordinates for a detector.
        
        Parameters
        ----------
        df : pd.DataFrame
            Dataframe with PONI information
        detector_type : str
            'PRIM' or 'SEC'
        
        Returns
        -------
        (center_y, center_x) : tuple of int or None
            Beam center in pixel coordinates (row, col)
        """
        if self.poni_column not in df.columns:
            if self.debug:
                print(f"PONI column '{self.poni_column}' not found")
            return None
        
        # Get first valid PONI content
        poni_content = None
        for v in df[self.poni_column].dropna():
            if isinstance(v, str) and len(v) > 0:
                poni_content = v
                break
        
        if poni_content is None:
            return None
        
        # Read beam center in meters and pixel size
        beam_center_m = self._read_beam_center_from_poni(poni_content)
        if beam_center_m is None:
            return None
        
        poni1, poni2 = beam_center_m
        
        # Read pixel size - try both PONI v2.1 (Detector_config) and older format (PixelSize1/2)
        import re
        import json
        
        pixel_size1, pixel_size2 = None, None
        
        # Try PONI v2.1 format: Detector_config JSON with "pixel1" and "pixel2"
        match_config = re.search(r'^Detector_config:\s*(.+)$', poni_content, re.MULTILINE)
        if match_config:
            try:
                config_str = match_config.group(1)
                # Replace single quotes with double quotes for valid JSON
                config_str = config_str.replace("'", '"')
                config = json.loads(config_str)
                pixel_size1 = float(config.get('pixel1', 0))
                pixel_size2 = float(config.get('pixel2', 0))
                if pixel_size1 > 0 and pixel_size2 > 0:
                    # Successfully read from Detector_config
                    pass
                else:
                    pixel_size1, pixel_size2 = None, None
            except (json.JSONDecodeError, ValueError, KeyError):
                pixel_size1, pixel_size2 = None, None
        
        # Fall back to older format: PixelSize1, PixelSize2
        if pixel_size1 is None or pixel_size2 is None:
            match_ps1 = re.search(r'^PixelSize1:\s*([0-9.eE+-]+)', poni_content, re.MULTILINE)
            match_ps2 = re.search(r'^PixelSize2:\s*([0-9.eE+-]+)', poni_content, re.MULTILINE)
            
            if match_ps1 and match_ps2:
                pixel_size1 = float(match_ps1.group(1))
                pixel_size2 = float(match_ps2.group(1))
        
        # If still not found, return None
        if pixel_size1 is None or pixel_size2 is None:
            if self.debug:
                print("Could not read pixel sizes from PONI (tried both v2.1 and legacy formats)")
            return None
        
        # Convert to pixel coordinates
        center_y = int(round(poni1 / pixel_size1))
        center_x = int(round(poni2 / pixel_size2))
        
        return (center_y, center_x)
    
    def _create_beam_exclusion_mask(self, shape: Tuple[int, int], center: Tuple[int, int], radius_frac: float) -> np.ndarray:
        """Create boolean mask for beam center exclusion region.
        
        Parameters
        ----------
        shape : tuple of (height, width)
            Image shape
        center : tuple of (center_y, center_x)
            Beam center in pixel coordinates
        radius_frac : float
            Radius as fraction of image size (e.g., 0.15 for 15%)
        
        Returns
        -------
        mask : np.ndarray of bool
            True for pixels to EXCLUDE from faulty pixel detection
        """
        h, w = shape
        center_y, center_x = center
        
        # Calculate radius in pixels (use max dimension)
        max_dim = max(h, w)
        radius_pixels = radius_frac * max_dim
        
        # Create coordinate grids
        y, x = np.ogrid[:h, :w]
        
        # Calculate distance from center
        dist = np.sqrt((y - center_y)**2 + (x - center_x)**2)
        
        # True where pixels should be excluded
        return dist <= radius_pixels

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

        # Get beam center exclusion masks if enabled
        exclude_mask_prim = None
        exclude_mask_sec = None
        
        if self.exclude_beam_center_radius is not None and self.exclude_beam_center_radius > 0:
            if images_prim is not None:
                center_prim = self._get_beam_center_pixels(df_prim, 'PRIM')
                if center_prim is not None:
                    exclude_mask_prim = self._create_beam_exclusion_mask(
                        images_prim.shape[1:], center_prim, self.exclude_beam_center_radius
                    )
                    if self.debug:
                        n_excluded = np.sum(exclude_mask_prim)
                        print(f"PRIMARY: Excluding {n_excluded} pixels ({100*n_excluded/exclude_mask_prim.size:.2f}%) around beam center at {center_prim}")
            
            if images_sec is not None:
                center_sec = self._get_beam_center_pixels(df_sec, 'SEC')
                if center_sec is not None:
                    exclude_mask_sec = self._create_beam_exclusion_mask(
                        images_sec.shape[1:], center_sec, self.exclude_beam_center_radius
                    )
                    if self.debug:
                        n_excluded = np.sum(exclude_mask_sec)
                        print(f"SECONDARY: Excluding {n_excluded} pixels ({100*n_excluded/exclude_mask_sec.size:.2f}%) around beam center at {center_sec}")

        def one_side(images, exclude_mask=None):
            if images is None:
                return set()
            cand = set()
            cand |= self._dead_zero_pixels(images)
            cand |= self._region_deviation(images)
            cand |= self._global_outliers(images)
            
            # Apply temporal consistency filter
            cand = self._temporal_keep(images, cand)
            
            # Remove pixels in beam center exclusion region if mask provided
            if exclude_mask is not None:
                cand = {(i, j) for (i, j) in cand if not exclude_mask[i, j]}
            
            return cand

        f_prim = one_side(images_prim, exclude_mask_prim)
        f_sec = one_side(images_sec, exclude_mask_sec)
        
        stats = {
            'image_column': img_col,
            'n_frames_prim': 0 if images_prim is None else int(images_prim.shape[0]),
            'n_frames_sec': 0 if images_sec is None else int(images_sec.shape[0]),
            'faulty_prim': len(f_prim),
            'faulty_sec': len(f_sec),
            'beam_center_excluded': self.exclude_beam_center_radius is not None,
            'beam_center_radius_frac': self.exclude_beam_center_radius if self.exclude_beam_center_radius is not None else 0,
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
