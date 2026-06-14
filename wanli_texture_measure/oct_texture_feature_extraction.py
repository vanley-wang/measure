"""3D OCT organoid texture feature extraction.

This module provides practical, memory-conscious implementations for:
- 3D masked GLCM features via averaged orthogonal 2D planes
- 3D masked GLRLM features via principal-axis run-length counting
- 3D wavelet features via db4 decomposition
- batch extraction from numpy arrays or NIfTI files
- heatmap visualization of the resulting feature table

Notes
-----
- The GLCM implementation is a masked custom implementation because classic
  gray-level co-occurrence matrix helpers do not directly support excluding
  background voxels in irregular 3D ROIs.
- The GLRLM implementation is a practical 3D approximation based on the
  three principal axes. It is suitable for organoid-scale analysis, but if
  you need strict IBSI-compliant 3D GLRLM in many directions, you may want
  to extend the direction set.
- For very large volumes such as (400, 800, 800), cropping to the mask
  bounding box is strongly recommended before feature extraction.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import pywt
except ImportError:  # pragma: no cover
    pywt = None

try:
    import nibabel as nib
except ImportError:  # pragma: no cover
    nib = None

ArrayLike3D = Union[np.ndarray, Sequence[Sequence[Sequence[float]]]]
Sample = Dict[str, object]


# -----------------------------------------------------------------------------
# Basic utilities
# -----------------------------------------------------------------------------

def _ensure_numpy_volume(volume: ArrayLike3D) -> np.ndarray:
    """Convert an input volume to a 3D numpy array."""

    arr = np.asarray(volume)
    if arr.ndim != 3:
        raise ValueError(f"Expected a 3D array, got shape {arr.shape}")
    return arr


def _ensure_mask(mask: ArrayLike3D) -> np.ndarray:
    """Convert mask to a boolean 3D numpy array."""

    arr = np.asarray(mask)
    if arr.ndim != 3:
        raise ValueError(f"Expected a 3D mask, got shape {arr.shape}")
    return arr.astype(bool)


def _crop_to_mask_bbox(volume: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Tuple[slice, slice, slice]]:
    """Crop volume and mask to the bounding box of the mask."""

    if not np.any(mask):
        raise ValueError("Mask is empty; cannot compute features.")

    coords = np.where(mask)
    z_min, z_max = int(coords[0].min()), int(coords[0].max()) + 1
    y_min, y_max = int(coords[1].min()), int(coords[1].max()) + 1
    x_min, x_max = int(coords[2].min()), int(coords[2].max()) + 1
    slc = (slice(z_min, z_max), slice(y_min, y_max), slice(x_min, x_max))
    return volume[slc], mask[slc], slc


def _quantize_volume(volume: np.ndarray, levels: int = 32, value_range: Tuple[float, float] = (0.0, 255.0)) -> np.ndarray:
    """Quantize a volume into discrete gray levels."""

    lo, hi = value_range
    if hi <= lo:
        raise ValueError("Invalid value_range; hi must be larger than lo.")

    vol = np.asarray(volume, dtype=np.float32)
    vol = np.clip(vol, lo, hi)
    scaled = (vol - lo) / (hi - lo)
    quantized = np.floor(scaled * levels).astype(np.int32)
    quantized = np.clip(quantized, 0, levels - 1)
    return quantized


def _safe_entropy(probabilities: np.ndarray) -> float:
    """Compute Shannon entropy in base 2, ignoring zero probabilities."""

    p = probabilities[probabilities > 0]
    if p.size == 0:
        return 0.0
    return float(-np.sum(p * np.log2(p)))


# -----------------------------------------------------------------------------
# NIfTI helpers
# -----------------------------------------------------------------------------

def load_nifti_volume(path: Union[str, Path]) -> np.ndarray:
    """Load a 3D volume from a NIfTI file."""

    if nib is None:
        raise ImportError("nibabel is required to read NIfTI files. Install it with `pip install nibabel`.")
    img = nib.load(str(path))
    data = img.get_fdata()
    if data.ndim != 3:
        raise ValueError(f"Expected a 3D NIfTI volume, got shape {data.shape} from {path}")
    return data


# -----------------------------------------------------------------------------
# GLCM features
# -----------------------------------------------------------------------------

def _accumulate_glcm_counts_for_2d_slice(
    image2d: np.ndarray,
    mask2d: np.ndarray,
    offset: Tuple[int, int],
    levels: int,
) -> np.ndarray:
    """Accumulate masked gray-level co-occurrence counts for a single 2D slice."""

    counts = np.zeros((levels, levels), dtype=np.float64)
    dr, dc = offset

    if dr >= 0:
        r_src = slice(0, image2d.shape[0] - dr)
        r_dst = slice(dr, image2d.shape[0])
    else:
        r_src = slice(-dr, image2d.shape[0])
        r_dst = slice(0, image2d.shape[0] + dr)

    if dc >= 0:
        c_src = slice(0, image2d.shape[1] - dc)
        c_dst = slice(dc, image2d.shape[1])
    else:
        c_src = slice(-dc, image2d.shape[1])
        c_dst = slice(0, image2d.shape[1] + dc)

    src_vals = image2d[r_src, c_src]
    dst_vals = image2d[r_dst, c_dst]
    src_mask = mask2d[r_src, c_src]
    dst_mask = mask2d[r_dst, c_dst]
    valid = src_mask & dst_mask
    if not np.any(valid):
        return counts

    a = src_vals[valid].astype(np.int32)
    b = dst_vals[valid].astype(np.int32)
    flat_idx = a * levels + b
    hist = np.bincount(flat_idx, minlength=levels * levels)
    counts += hist.reshape((levels, levels))
    return counts


def _glcm_features_from_matrix(matrix: np.ndarray) -> Dict[str, float]:
    """Compute standard GLCM features from a co-occurrence matrix."""

    mat = np.asarray(matrix, dtype=np.float64)
    total = mat.sum()
    if total <= 0:
        return {
            "glcm_contrast": 0.0,
            "glcm_energy": 0.0,
            "glcm_entropy": 0.0,
            "glcm_homogeneity": 0.0,
            "glcm_correlation": 0.0,
        }

    p = mat / total
    levels = p.shape[0]
    i, j = np.ogrid[:levels, :levels]
    mean_i = float(np.sum(i * p))
    mean_j = float(np.sum(j * p))
    std_i = float(np.sqrt(np.sum(((i - mean_i) ** 2) * p)))
    std_j = float(np.sqrt(np.sum(((j - mean_j) ** 2) * p)))

    contrast = float(np.sum(((i - j) ** 2) * p))
    energy = float(np.sum(p ** 2))
    entropy = _safe_entropy(p)
    homogeneity = float(np.sum(p / (1.0 + np.abs(i - j))))

    if std_i > 0 and std_j > 0:
        correlation = float(np.sum(((i - mean_i) * (j - mean_j) * p)) / (std_i * std_j))
    else:
        correlation = 1.0

    return {
        "glcm_contrast": contrast,
        "glcm_energy": energy,
        "glcm_entropy": entropy,
        "glcm_homogeneity": homogeneity,
        "glcm_correlation": correlation,
    }


def extract_glcm_features(
    volume: ArrayLike3D,
    mask: ArrayLike3D,
    levels: int = 32,
    value_range: Tuple[float, float] = (0.0, 255.0),
    use_bbox_crop: bool = True,
    symmetrize: bool = True,
) -> Dict[str, float]:
    """Extract masked 3D GLCM features by averaging over XY, XZ, and YZ planes.

    Parameters
    ----------
    volume:
        3D intensity array in (Z, Y, X) order.
    mask:
        3D boolean mask with the same shape as volume.
    levels:
        Number of gray levels after quantization.
    value_range:
        Input intensity range used for quantization.
    use_bbox_crop:
        Crop to the mask bounding box before feature calculation.
    symmetrize:
        If True, symmetrize the GLCM matrix by adding its transpose.
    """

    vol = _ensure_numpy_volume(volume)
    msk = _ensure_mask(mask)
    if vol.shape != msk.shape:
        raise ValueError(f"Volume shape {vol.shape} and mask shape {msk.shape} do not match.")

    if use_bbox_crop:
        vol, msk, _ = _crop_to_mask_bbox(vol, msk)

    qvol = _quantize_volume(vol, levels=levels, value_range=value_range)

    # Plane definitions:
    # XY plane: slices along Z, offsets along Y/X
    # XZ plane: slices along Y, offsets along Z/X
    # YZ plane: slices along X, offsets along Z/Y
    plane_specs = {
        "xy": (0, (1, 2)),
        "xz": (1, (0, 2)),
        "yz": (2, (0, 1)),
    }
    offsets_2d = [(0, 1), (1, 0)]

    features_accumulator: List[Dict[str, float]] = []
    for plane_name, (axis, _) in plane_specs.items():
        plane_counts = np.zeros((levels, levels), dtype=np.float64)
        for idx in range(qvol.shape[axis]):
            if axis == 0:
                image2d = qvol[idx, :, :]
                mask2d = msk[idx, :, :]
            elif axis == 1:
                image2d = qvol[:, idx, :]
                mask2d = msk[:, idx, :]
            else:
                image2d = qvol[:, :, idx]
                mask2d = msk[:, :, idx]

            for offset in offsets_2d:
                plane_counts += _accumulate_glcm_counts_for_2d_slice(image2d, mask2d, offset, levels)

        if symmetrize:
            plane_counts = plane_counts + plane_counts.T

        plane_features = _glcm_features_from_matrix(plane_counts)
        features_accumulator.append(plane_features)

    # Average the features across the three orthogonal planes.
    feature_names = list(features_accumulator[0].keys())
    averaged = {}
    for name in feature_names:
        averaged[name] = float(np.mean([feat[name] for feat in features_accumulator]))
    return averaged


# -----------------------------------------------------------------------------
# GLRLM features
# -----------------------------------------------------------------------------

def _update_run_counts_from_line(
    values: np.ndarray,
    mask_line: np.ndarray,
    levels: int,
    run_counts: Dict[Tuple[int, int], int],
) -> None:
    """Update run counts for a single 1D line within the masked ROI."""

    current_gray = None
    current_len = 0

    for v, valid in zip(values, mask_line):
        if not valid:
            if current_gray is not None and current_len > 0:
                run_counts[(current_gray, current_len)] = run_counts.get((current_gray, current_len), 0) + 1
            current_gray = None
            current_len = 0
            continue

        gray = int(v)
        if gray < 0:
            gray = 0
        elif gray >= levels:
            gray = levels - 1

        if current_gray is None:
            current_gray = gray
            current_len = 1
        elif gray == current_gray:
            current_len += 1
        else:
            run_counts[(current_gray, current_len)] = run_counts.get((current_gray, current_len), 0) + 1
            current_gray = gray
            current_len = 1

    if current_gray is not None and current_len > 0:
        run_counts[(current_gray, current_len)] = run_counts.get((current_gray, current_len), 0) + 1


def extract_glrlm_features(
    volume: ArrayLike3D,
    mask: ArrayLike3D,
    levels: int = 32,
    value_range: Tuple[float, float] = (0.0, 255.0),
    use_bbox_crop: bool = True,
) -> Dict[str, float]:
    """Extract practical 3D GLRLM features along the three principal axes.

    This is a computationally efficient approximation suitable for organoid
    analysis. It counts runs along Z, Y, and X directions within the masked ROI.
    """

    vol = _ensure_numpy_volume(volume)
    msk = _ensure_mask(mask)
    if vol.shape != msk.shape:
        raise ValueError(f"Volume shape {vol.shape} and mask shape {msk.shape} do not match.")

    if use_bbox_crop:
        vol, msk, _ = _crop_to_mask_bbox(vol, msk)

    qvol = _quantize_volume(vol, levels=levels, value_range=value_range)
    run_counts: Dict[Tuple[int, int], int] = {}

    # Z-axis lines (iterate over Y,X)
    for y in range(qvol.shape[1]):
        for x in range(qvol.shape[2]):
            _update_run_counts_from_line(qvol[:, y, x], msk[:, y, x], levels, run_counts)

    # Y-axis lines (iterate over Z,X)
    for z in range(qvol.shape[0]):
        for x in range(qvol.shape[2]):
            _update_run_counts_from_line(qvol[z, :, x], msk[z, :, x], levels, run_counts)

    # X-axis lines (iterate over Z,Y)
    for z in range(qvol.shape[0]):
        for y in range(qvol.shape[1]):
            _update_run_counts_from_line(qvol[z, y, :], msk[z, y, :], levels, run_counts)

    if not run_counts:
        return {
            "glrlm_sre": 0.0,
            "glrlm_lre": 0.0,
            "glrlm_rln": 0.0,
        }

    max_run_len = max(length for (_, length) in run_counts.keys())
    rlm = np.zeros((levels, max_run_len), dtype=np.float64)
    for (gray, run_len), count in run_counts.items():
        rlm[gray, run_len - 1] += count

    total_runs = rlm.sum()
    if total_runs <= 0:
        return {
            "glrlm_sre": 0.0,
            "glrlm_lre": 0.0,
            "glrlm_rln": 0.0,
        }

    p = rlm / total_runs
    run_lengths = np.arange(1, max_run_len + 1, dtype=np.float64)
    sre = float(np.sum(p / (run_lengths ** 2)))
    lre = float(np.sum(p * (run_lengths ** 2)))
    gray_marginal = np.sum(p, axis=1)
    rln = float(np.sum(gray_marginal ** 2))

    return {
        "glrlm_sre": sre,
        "glrlm_lre": lre,
        "glrlm_rln": rln,
    }


# -----------------------------------------------------------------------------
# Wavelet features
# -----------------------------------------------------------------------------

def extract_wavelet_features(
    volume: ArrayLike3D,
    mask: ArrayLike3D,
    wavelet: str = "db4",
    level: int = 2,
    use_bbox_crop: bool = True,
) -> Dict[str, float]:
    """Extract 3D wavelet features from the masked ROI.

    The masked voxels are preserved and the background is set to zero before
    decomposition. For 3D data, PyWavelets produces multiple subbands per level.
    We report the energy and standard deviation for each subband.
    """

    if pywt is None:
        raise ImportError("PyWavelets is required for wavelet features. Install it with `pip install PyWavelets`.")

    vol = _ensure_numpy_volume(volume)
    msk = _ensure_mask(mask)
    if vol.shape != msk.shape:
        raise ValueError(f"Volume shape {vol.shape} and mask shape {msk.shape} do not match.")

    if use_bbox_crop:
        vol, msk, _ = _crop_to_mask_bbox(vol, msk)

    roi = np.asarray(vol, dtype=np.float32) * msk.astype(np.float32)

    max_level = pywt.dwtn_max_level(roi.shape, pywt.Wavelet(wavelet))
    actual_level = int(max(1, min(level, max_level)))
    coeffs = pywt.wavedecn(roi, wavelet=wavelet, mode="periodization", level=actual_level)

    features: Dict[str, float] = {}

    approx = coeffs[0]
    features[f"wavelet_L{actual_level}_approx_energy"] = float(np.sum(approx ** 2))
    features[f"wavelet_L{actual_level}_approx_std"] = float(np.std(approx))

    for lvl_idx, detail_dict in enumerate(coeffs[1:], start=1):
        # lvl_idx is counted from the coarsest representation in PyWavelets output.
        for band_name, arr in detail_dict.items():
            features[f"wavelet_L{actual_level - lvl_idx + 1}_{band_name}_energy"] = float(np.sum(arr ** 2))
            features[f"wavelet_L{actual_level - lvl_idx + 1}_{band_name}_std"] = float(np.std(arr))

    return features


# -----------------------------------------------------------------------------
# Batch processing
# -----------------------------------------------------------------------------

def _iter_samples(
    images: Union[Sequence[ArrayLike3D], Dict[str, ArrayLike3D]],
    masks: Union[Sequence[ArrayLike3D], Dict[str, ArrayLike3D]],
) -> Iterator[Tuple[str, ArrayLike3D, ArrayLike3D]]:
    """Yield (sample_id, image, mask) tuples from lists or dictionaries."""

    if isinstance(images, dict) and isinstance(masks, dict):
        common_keys = list(images.keys())
        for key in common_keys:
            if key not in masks:
                raise KeyError(f"Mask for sample '{key}' not found.")
            yield str(key), images[key], masks[key]
        return

    if isinstance(images, Sequence) and isinstance(masks, Sequence):
        if len(images) != len(masks):
            raise ValueError(f"Images and masks must have the same length: {len(images)} vs {len(masks)}")
        for idx, (img, msk) in enumerate(zip(images, masks)):
            yield f"sample_{idx}", img, msk
        return

    raise TypeError("Images and masks must both be lists/tuples or both be dictionaries.")


def extract_texture_feature_dataframe(
    images: Union[Sequence[ArrayLike3D], Dict[str, ArrayLike3D]],
    masks: Union[Sequence[ArrayLike3D], Dict[str, ArrayLike3D]],
    levels: int = 32,
    value_range: Tuple[float, float] = (0.0, 255.0),
    include_glcm: bool = True,
    include_glrlm: bool = True,
    include_wavelet: bool = True,
) -> pd.DataFrame:
    """Batch extract texture features and return a pandas DataFrame."""

    rows: List[Dict[str, object]] = []
    for sample_id, image, mask in _iter_samples(images, masks):
        row: Dict[str, object] = {"sample_id": sample_id}
        if include_glcm:
            row.update(extract_glcm_features(image, mask, levels=levels, value_range=value_range))
        if include_glrlm:
            row.update(extract_glrlm_features(image, mask, levels=levels, value_range=value_range))
        if include_wavelet:
            try:
                row.update(extract_wavelet_features(image, mask))
            except ImportError:
                row["wavelet_available"] = False
        rows.append(row)

    df = pd.DataFrame(rows)
    if "sample_id" in df.columns:
        ordered_cols = ["sample_id"] + [c for c in df.columns if c != "sample_id"]
        df = df[ordered_cols]
    return df


# -----------------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------------

def plot_feature_heatmap(
    feature_df: pd.DataFrame,
    figsize: Tuple[int, int] = (14, 8),
    cmap: str = "viridis",
    normalize_per_column: bool = False,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> None:
    """Plot a heatmap for the extracted feature table."""

    if feature_df.empty:
        raise ValueError("Feature DataFrame is empty.")

    data = feature_df.copy()
    if "sample_id" in data.columns:
        data = data.set_index("sample_id")
    numeric = data.select_dtypes(include=[np.number]).copy()
    if numeric.empty:
        raise ValueError("No numeric columns found for heatmap visualization.")

    if normalize_per_column:
        numeric = (numeric - numeric.min(axis=0)) / (numeric.max(axis=0) - numeric.min(axis=0) + 1e-12)

    plt.figure(figsize=figsize)
    im = plt.imshow(numeric.values, aspect="auto", cmap=cmap)
    plt.colorbar(im, fraction=0.03, pad=0.02)
    plt.xticks(np.arange(numeric.shape[1]), numeric.columns, rotation=90)
    plt.yticks(np.arange(numeric.shape[0]), numeric.index)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(str(save_path), dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()


# -----------------------------------------------------------------------------
# Directory batch processing and summary analysis
# -----------------------------------------------------------------------------

def _strip_nii_suffix(filename: str) -> str:
    """Remove common NIfTI suffixes and nnUNet image channel suffixes."""

    name = filename
    if name.endswith(".nii.gz"):
        name = name[:-7]
    elif name.endswith(".nii"):
        name = name[:-4]
    if name.endswith("_0000"):
        name = name[:-5]
    return name


def match_nifti_image_mask_pairs(
    images_dir: Union[str, Path],
    masks_dir: Union[str, Path],
) -> List[Tuple[str, Path, Path]]:
    """Match nnUNet image files and predicted mask files by case name.

    Parameters
    ----------
    images_dir:
        Directory containing `imagesTr` style files, typically `*_0000.nii.gz`.
    masks_dir:
        Directory containing predicted mask files, typically `*.nii.gz`.

    Returns
    -------
    list of tuples
        Each tuple is `(case_id, image_path, mask_path)`.
    """

    images_path = Path(images_dir)
    masks_path = Path(masks_dir)
    if not images_path.exists():
        raise FileNotFoundError(f"Image directory not found: {images_path}")
    if not masks_path.exists():
        raise FileNotFoundError(f"Mask directory not found: {masks_path}")

    image_files = sorted([p for p in images_path.iterdir() if p.is_file() and (p.name.endswith(".nii.gz") or p.name.endswith(".nii"))])
    mask_lookup = {_strip_nii_suffix(p.name): p for p in masks_path.iterdir() if p.is_file() and (p.name.endswith(".nii.gz") or p.name.endswith(".nii"))}

    pairs: List[Tuple[str, Path, Path]] = []
    for image_path in image_files:
        case_id = _strip_nii_suffix(image_path.name)
        if case_id not in mask_lookup:
            raise KeyError(f"No matching mask found for image '{image_path.name}' in {masks_path}")
        pairs.append((case_id, image_path, mask_lookup[case_id]))
    return pairs


def extract_texture_features_from_nifti_directories(
    images_dir: Union[str, Path],
    masks_dir: Union[str, Path],
    output_csv: Union[str, Path],
    output_summary_csv: Optional[Union[str, Path]] = None,
    output_heatmap: Optional[Union[str, Path]] = None,
    levels: int = 32,
    value_range: Tuple[float, float] = (0.0, 255.0),
    include_glcm: bool = True,
    include_glrlm: bool = True,
    include_wavelet: bool = True,
) -> pd.DataFrame:
    """Batch extract features from NIfTI directories and save the results.
    Processes one pair at a time to prevent Out Of Memory (OOM) errors.
    """

    pairs = match_nifti_image_mask_pairs(images_dir, masks_dir)
    
    rows = []
    print(f"Found {len(pairs)} image-mask pairs to process.")
    for i, (case_id, image_path, mask_path) in enumerate(pairs, start=1):
        print(f"[{i}/{len(pairs)}] Processing volume {case_id}...")
        
        # Load one pair at a time, extract, then discard to save memory
        vol = load_nifti_volume(image_path)
        msk = load_nifti_volume(mask_path) > 0
        
        # Use existing single-item pipeline directly to build the row
        df_single = extract_texture_feature_dataframe(
            {case_id: vol},
            {case_id: msk},
            levels=levels,
            value_range=value_range,
            include_glcm=include_glcm,
            include_glrlm=include_glrlm,
            include_wavelet=include_wavelet,
        )
        
        # Save memory instantly
        del vol, msk
        
        rows.append(df_single)
        
        # Optional: Save intermediate CSV after every patient, to avoid lost data on crash
        temp_df = pd.concat(rows, ignore_index=True)
        save_feature_table(temp_df, output_csv)

    # Final feature table built from all single extractions
    feature_df = pd.concat(rows, ignore_index=True)

    summary_df = summarize_feature_dataframe(feature_df)
    if output_summary_csv is not None:
        Path(output_summary_csv).parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(output_summary_csv, index=False)

    if output_heatmap is not None:
        plot_feature_heatmap(feature_df, save_path=output_heatmap, show=False, normalize_per_column=True)

    return feature_df


def save_feature_table(feature_df: pd.DataFrame, output_csv: Union[str, Path]) -> None:
    """Save the feature table to CSV and ensure the parent folder exists."""

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_csv(output_path, index=False)


def summarize_feature_dataframe(feature_df: pd.DataFrame) -> pd.DataFrame:
    """Compute dataset-level summary statistics for all numeric features."""

    if feature_df.empty:
        raise ValueError("Feature DataFrame is empty.")

    numeric = feature_df.select_dtypes(include=[np.number]).copy()
    if numeric.empty:
        raise ValueError("No numeric columns found for summary statistics.")

    summary = pd.DataFrame(
        {
            "feature": numeric.columns,
            "mean": numeric.mean(axis=0).values,
            "std": numeric.std(axis=0, ddof=0).values,
            "min": numeric.min(axis=0).values,
            "max": numeric.max(axis=0).values,
            "median": numeric.median(axis=0).values,
            "missing_ratio": numeric.isna().mean(axis=0).values,
        }
    )
    return summary.sort_values("feature").reset_index(drop=True)


def plot_feature_correlation_heatmap(
    feature_df: pd.DataFrame,
    figsize: Tuple[int, int] = (12, 10),
    cmap: str = "coolwarm",
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> None:
    """Plot a correlation heatmap of numeric features across the dataset."""

    numeric = feature_df.select_dtypes(include=[np.number]).copy()
    if numeric.empty:
        raise ValueError("No numeric columns found for correlation heatmap.")

    corr = numeric.corr(method="pearson")
    plt.figure(figsize=figsize)
    im = plt.imshow(corr.values, aspect="auto", cmap=cmap, vmin=-1.0, vmax=1.0)
    plt.colorbar(im, fraction=0.03, pad=0.02)
    plt.xticks(np.arange(corr.shape[1]), corr.columns, rotation=90)
    plt.yticks(np.arange(corr.shape[0]), corr.index)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(str(save_path), dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()


# -----------------------------------------------------------------------------
# Example usage / simple validation test
# -----------------------------------------------------------------------------

def _create_random_test_sample(shape: Tuple[int, int, int] = (24, 24, 24), seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Create a random synthetic volume and a spherical mask for validation."""

    rng = np.random.default_rng(seed)
    volume = rng.integers(0, 256, size=shape, dtype=np.uint8)

    zz, yy, xx = np.indices(shape)
    center = np.array(shape) / 2.0
    radius = min(shape) * 0.28
    dist2 = (zz - center[0]) ** 2 + (yy - center[1]) ** 2 + (xx - center[2]) ** 2
    mask = dist2 <= radius ** 2
    return volume, mask


if __name__ == "__main__":
    # Simple validation test with synthetic data.
    vol1, mask1 = _create_random_test_sample(shape=(24, 28, 30), seed=1)
    vol2, mask2 = _create_random_test_sample(shape=(24, 28, 30), seed=2)

    samples = [vol1, vol2]
    masks = [mask1, mask2]

    df = extract_texture_feature_dataframe(
        samples,
        masks,
        levels=32,
        value_range=(0, 255),
        include_glcm=True,
        include_glrlm=True,
        include_wavelet=True,
    )
    print(df.head())

    save_feature_table(df, "texture_features_demo.csv")
    summary_df = summarize_feature_dataframe(df)
    summary_df.to_csv("texture_features_demo_summary.csv", index=False)
    plot_feature_heatmap(df, figsize=(16, 6), normalize_per_column=True, show=False, save_path="texture_features_demo_heatmap.png")
    plot_feature_correlation_heatmap(df, show=False, save_path="texture_features_demo_corr.png")

"""
Usage
-----
1) Numpy arrays:

    from oct_texture_feature_extraction import extract_texture_feature_dataframe
    df = extract_texture_feature_dataframe(images_list, masks_list)

2) Dictionary input:

    images = {"organoid_001": vol1, "organoid_002": vol2}
    masks = {"organoid_001": mask1, "organoid_002": mask2}
    df = extract_texture_feature_dataframe(images, masks)

3) NIfTI files:

    volume = load_nifti_volume("path/to/image.nii.gz")
    mask = load_nifti_volume("path/to/mask.nii.gz") > 0
    features = extract_glcm_features(volume, mask)

4) Directory workflow for your data:

    feature_df = extract_texture_features_from_nifti_directories(
        images_dir="nnUNet_data/nnUNet_raw/Dataset507_organoid/imagesTr",
        masks_dir="nnUNet_results/Dataset507_organoid/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres/fold_0/predictions_imageTr",
        output_csv="OCT_texture_outputs/texture_features.csv",
        output_summary_csv="OCT_texture_outputs/texture_features_summary.csv",
        output_heatmap="OCT_texture_outputs/texture_heatmap.png",
    )

Parameter suggestions
---------------------
- `levels=32` matches your request and is a good balance between detail and speed.
- Always crop by mask bounding box for large 3D volumes to save memory and time.
- If wavelet features are not needed, set `include_wavelet=False` in batch extraction.
- The GLRLM implementation here is a practical principal-axis approximation. If you later need a stricter 3D directional GLRLM, I can extend it to more directions.
- For very large datasets, consider extracting features in chunks or parallelizing over organoids.
- After extraction, analyze the whole cohort with summary statistics, feature correlation, PCA, clustering, and group comparison.
"""
