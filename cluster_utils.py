"""
Shared utilities for organoid clustering pipelines.

Provides:
  - Feature engineering (log1p transform, Cavity_Ratio)
  - Preprocessing pipeline (drop constants, log transform, StandardScaler)
  - Phenotype mapping logic
  - Model I/O helpers with unified package format
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ============================================================================
# 1. Feature definitions
# ============================================================================

# Raw features present in measure_excel files
RAW_FEATURES = [
    'Organoids_Volume',
    'Organoids_Volume_Fill',
    'Organoids_Surface',
    'Cavity_Volume',
    'CavityNum',
    'LongAxis',
    'ShortAxis',
    'Wall_Thickness',
    'Sphericity',
    'Scatt_Mean',
    'Scatt_STD',
]

# Features to drop (zero information or redundant)
DROP_FEATURES = ['Wall_Thickness', 'Organoids_Volume']

# Features that need log1p transform (right-skewed, wide dynamic range)
LOG_FEATURES = [
    'Organoids_Volume_Fill',
    'Organoids_Surface',
    'LongAxis',
    'ShortAxis',
    'Cavity_Volume',
]

# Features kept as-is (already well-behaved)
KEEP_FEATURES = ['Sphericity', 'Scatt_Mean', 'Scatt_STD', 'CavityNum']

# Engineered features
ENGINEERED_FEATURES = ['Cavity_Ratio']

# Final processed feature list (order matters for model consistency)
PROCESSED_FEATURES = [f for f in LOG_FEATURES + KEEP_FEATURES + ENGINEERED_FEATURES
                      if f not in DROP_FEATURES]

# ============================================================================
# 1b. Reduced feature definitions (5-dim, collinearity-free)
# ============================================================================

REDUCED_RAW_FEATURES = [
    'Organoids_Volume_Fill',
    'Sphericity',
    'Scatt_Mean',
    'Scatt_STD',
    'Cavity_Volume',
]

# Only Volume_Fill needs log1p in reduced set
REDUCED_LOG_FEATURES = ['Organoids_Volume_Fill']
REDUCED_KEEP_FEATURES = ['Sphericity', 'Scatt_Mean', 'Scatt_STD']
REDUCED_ENGINEERED_FEATURES = ['Cavity_Ratio']

REDUCED_PROCESSED_FEATURES = REDUCED_LOG_FEATURES + REDUCED_KEEP_FEATURES + REDUCED_ENGINEERED_FEATURES

# ============================================================================
# 1c. Reconstructed feature definitions (6-dim, PCA-synthesized size)
# ============================================================================

# Group A: Size features (log1p → PCA → Size_PC1)
SIZE_FEATURES = [
    'Organoids_Volume_Fill',
    'Organoids_Surface',
    'LongAxis',
    'ShortAxis',
]

# Group B: Scattering features (keep as-is, no transform)
SCATT_FEATURES = ['Scatt_Mean', 'Scatt_STD']

# Group C: Morphology features (keep as-is, no transform)
MORPH_FEATURES = ['Sphericity', 'CavityNum']

# Engineered features (computed from raw columns)
RECONSTRUCTED_ENGINEERED = ['Cavity_Ratio']

# Final processed 6-dim feature list
RECONSTRUCTED_PROCESSED_FEATURES = (
    ['Size_PC1'] + SCATT_FEATURES + MORPH_FEATURES + RECONSTRUCTED_ENGINEERED
)

# ============================================================================
# 1d. Feature short-name mapping (for compact, readable feature names)
# ============================================================================

FEATURE_SHORT_NAMES = {
    'Organoids_Volume_Fill': 'FillVol',
    'Organoids_Surface': 'SurfArea',
    'Cavity_Volume': 'CavVol',
    'CavityNum': 'CavNum',
    'LongAxis': 'LongAxis',
    'ShortAxis': 'ShortAxis',
    'Sphericity': 'Spher',
    'Scatt_Mean': 'OACm',
    'Scatt_STD': 'OACs',
    'Cavity_Ratio': 'CavRatio',
    'Roughness': 'Rough',
}

SHORT_TO_LONG = {v: k for k, v in FEATURE_SHORT_NAMES.items()}

# Features where IQR provides additional heterogeneity signal beyond CV
IQR_FEATURES = []

# Features where CV is most informative (subset to avoid overfitting)
CV_FEATURES = ['FillVol', 'OACm']

# Features where Growth Rate is most informative
GR_FEATURES = []

# ============================================================================
# 2. Phenotype definitions
# ============================================================================

PHENOTYPE_NAMES = {
    0: '大囊状健康类器官 (Cluster 1)',
    1: '大实心健康类器官 (Cluster 2)',
    2: '小实心休眠/幼类器官 (Cluster 3)',
    3: '极小高致密受损类器官 (Cluster 4)',
}

PHENOTYPE_COLORS = {
    0: '#FF0000',   # 红
    1: '#FFFF00',   # 黄
    2: '#00FF00',   # 绿
    3: '#0000FF',   # 蓝
}

# ============================================================================
# 3. Preprocessing pipeline
# ============================================================================

class Preprocessor:
    """
    Organoid feature preprocessor.

    Supports three modes:
      - 'full'         (default): 10-dim feature set (backward compatible)
      - 'reduced':      5-dim collinearity-free feature set
      - 'reconstructed': 6-dim PCA-synthesized feature set
          Group A (Size):  Volume_Fill/Surface/LongAxis/ShortAxis
                           → log1p → StandardScaler → PCA(1) → Size_PC1
          Group B (Scatt): Scatt_Mean, Scatt_STD → keep as-is
          Group C (Morph): Cavity_Ratio, CavityNum, Sphericity → keep as-is

    Steps:
      1. Compute Cavity_Ratio = Cavity_Volume / (Volume_Fill + 1)
      2. Drop constant/redundant features (full mode only)
      3. Log1p-transform skewed volume features
      4. (reconstructed) PCA-synthesize size features into Size_PC1
      5. Fit / transform with StandardScaler
    """

    def __init__(self, mode='full'):
        if mode not in ('full', 'reduced', 'reconstructed'):
            raise ValueError("mode must be 'full', 'reduced', or 'reconstructed'")
        self.mode = mode
        self.scaler = StandardScaler()
        self._fitted = False

        if mode == 'reduced':
            self._log_features = REDUCED_LOG_FEATURES
            self._processed_features = REDUCED_PROCESSED_FEATURES
        elif mode == 'reconstructed':
            self._log_features = SIZE_FEATURES
            self._processed_features = RECONSTRUCTED_PROCESSED_FEATURES
            self._size_pca = PCA(n_components=1, random_state=42)
            self._size_scaler = StandardScaler()
        else:
            self._log_features = LOG_FEATURES
            self._processed_features = PROCESSED_FEATURES

    def _engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add engineered features without modifying original df."""
        df = df.copy()
        df['Cavity_Ratio'] = df['Cavity_Volume'] / (df['Organoids_Volume_Fill'].clip(lower=1))
        return df

    def _select_and_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop, log-transform, PCA-synthesize, and select final feature columns."""
        df = df.copy()

        # Drop (only in full mode)
        if self.mode == 'full':
            for col in DROP_FEATURES:
                if col in df.columns:
                    df.drop(columns=[col], inplace=True)

        # Log1p transform
        for col in self._log_features:
            if col in df.columns:
                df[col] = np.log1p(df[col])

        # Reconstructed mode: PCA-synthesize size features into Size_PC1
        if self.mode == 'reconstructed':
            missing_size = [c for c in SIZE_FEATURES if c not in df.columns]
            if missing_size:
                raise ValueError(f"Missing size features for PCA: {missing_size}")

            if self._fitted:
                X_size = df[SIZE_FEATURES].values
                X_size_std = self._size_scaler.transform(X_size)
                df['Size_PC1'] = self._size_pca.transform(X_size_std).flatten()
            else:
                X_size = df[SIZE_FEATURES].values
                X_size_std = self._size_scaler.fit_transform(X_size)
                df['Size_PC1'] = self._size_pca.fit_transform(X_size_std).flatten()

        # Select final columns
        missing = [c for c in self._processed_features if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        return df[self._processed_features]

    def fit(self, df: pd.DataFrame):
        """Fit the scaler on raw feature DataFrame."""
        df_eng = self._engineer(df)
        X = self._select_and_transform(df_eng)
        self.scaler.fit(X)
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform raw feature DataFrame to standardized numpy array."""
        if not self._fitted:
            raise RuntimeError("Preprocessor must be fit() before transform().")
        df_eng = self._engineer(df)
        X = self._select_and_transform(df_eng)
        return self.scaler.transform(X)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(df)
        return self.transform(df)

    def get_feature_names(self) -> list:
        return list(self._processed_features)

    def get_size_pca_info(self) -> dict:
        """Return PCA info for size features (reconstructed mode only)."""
        if self.mode != 'reconstructed':
            raise RuntimeError("get_size_pca_info() only available in 'reconstructed' mode")
        if not self._fitted:
            raise RuntimeError("Preprocessor must be fit() first")
        return {
            'explained_variance_ratio': self._size_pca.explained_variance_ratio_[0],
            'components': dict(zip(SIZE_FEATURES, self._size_pca.components_[0].round(4))),
        }


# ============================================================================
# 3b. Simple log-volume preprocessor (for raw 11-feature KMeans models)
# ============================================================================

class LogVolumeScaler:
    """
    Minimal preprocessor: log1p-transform skewed size features, then StandardScaler.

    Works with raw 11-feature measure_excel tables. Unlike the full Preprocessor
    class, this one does NOT drop features, engineer Cavity_Ratio, or run PCA.
    It's used by MedNext-style KMeans models trained directly on 11 raw features
    with log1p applied to size/volume features.

    Pickle-safe when this module (cluster_utils) is importable.
    """
    def __init__(self, feature_names, log_features):
        self.feature_names = list(feature_names)
        self.log_features = list(log_features)
        self.log_mask = np.array(
            [f in log_features for f in feature_names], dtype=bool
        )
        self.scaler = StandardScaler()
        self._fitted = False

    def _apply_log(self, X):
        X_out = np.array(X, dtype=np.float64).copy()
        X_out[:, self.log_mask] = np.maximum(X_out[:, self.log_mask], 0)
        X_out[:, self.log_mask] = np.log1p(X_out[:, self.log_mask])
        return X_out

    def fit(self, X, y=None):
        X_log = self._apply_log(X)
        self.scaler.fit(X_log)
        self._fitted = True
        return self

    def transform(self, X):
        X_log = self._apply_log(X)
        return self.scaler.transform(X_log)

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)


class CavityBoostedPreprocessor:
    """
    Preprocessing pipeline with enhanced cavity feature weight.

    Input: 11 raw features from measure_excel (feature_names)
    Output: 12-dimensional standardized array for KMeans

    Steps:
      1. Compute Cavity_Ratio = Cavity_Volume / Volume_Fill  (engineered)
      2. Build 12-dim array: [11 raw features, Cavity_Ratio appended]
      3. Boost Cavity_Ratio by a multiplier (gives it more distance weight)
      4. log1p transform size/cavity features (long-tail distributions)
      5. StandardScaler on all 12 features

    feature_names lists only the 11 raw features so cluster-merge.py can
    validate columns against measure_excel tables.
    """
    def __init__(self, feature_names, log_features, cavity_ratio_boost=20.0):
        # Raw feature names (11, matching measure_excel columns)
        self.feature_names = list(feature_names)
        self.log_features = list(log_features)
        self.cavity_ratio_boost = cavity_ratio_boost

        # Build internal 12-dim feature list: raw 11 + Cavity_Ratio at end
        self._internal_features = list(feature_names) + ['Cavity_Ratio']
        self._ratio_idx = len(self._internal_features) - 1

        # log_mask for 12-dim internal array
        self._log_mask = np.array(
            [f in log_features for f in self._internal_features], dtype=bool
        )
        # Cavity_Ratio is NOT log1p'd (it's already a 0-1 ratio)
        self._log_mask[self._ratio_idx] = False

        self.scaler = StandardScaler()
        self._fitted = False

    def _build_internal(self, X):
        """Convert input (11 raw features) to 12-dim internal array."""
        # Try DataFrame first
        if isinstance(X, pd.DataFrame):
            df = X.copy()
            if 'Cavity_Ratio' not in df.columns:
                vol = df['Organoids_Volume_Fill'].clip(lower=1)
                df['Cavity_Ratio'] = df['Cavity_Volume'] / vol
            # Reorder to internal feature order
            arr = df[self._internal_features].values.astype(np.float64)
        else:
            arr = np.array(X, dtype=np.float64).copy()
            # If input is already 12-dim, pass through
            if arr.shape[1] == len(self._internal_features):
                pass
            elif arr.shape[1] == len(self.feature_names):
                # Assume columns are in feature_names order; append Cavity_Ratio
                vol_col = self.feature_names.index('Organoids_Volume_Fill')
                cav_col = self.feature_names.index('Cavity_Volume')
                vol = np.maximum(arr[:, vol_col], 1.0)
                cav_ratio = arr[:, cav_col] / vol
                arr = np.column_stack([arr, cav_ratio])
            else:
                raise ValueError(f"Expected {len(self.feature_names)} or "
                                 f"{len(self._internal_features)} cols, got {arr.shape[1]}")
        return arr

    def _apply_transforms(self, arr):
        """Apply boost + log1p to internal 12-dim array."""
        arr = arr.copy()
        # Boost Cavity_Ratio
        arr[:, self._ratio_idx] = arr[:, self._ratio_idx] * self.cavity_ratio_boost
        # log1p on log features
        arr[:, self._log_mask] = np.maximum(arr[:, self._log_mask], 0)
        arr[:, self._log_mask] = np.log1p(arr[:, self._log_mask])
        return arr

    def fit(self, X, y=None):
        arr = self._build_internal(X)
        arr_trans = self._apply_transforms(arr)
        self.scaler.fit(arr_trans)
        self._fitted = True
        return self

    def transform(self, X):
        arr = self._build_internal(X)
        arr_trans = self._apply_transforms(arr)
        return self.scaler.transform(arr_trans)

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)


# ============================================================================
# 4. Phenotype mapping (from raw cluster IDs to biological phenotypes)
# ============================================================================

def map_phenotypes_by_centroids(df: pd.DataFrame, raw_labels: np.ndarray) -> dict:
    """
    Given a DataFrame with original features and raw cluster labels,
    determine the biological phenotype mapping based on cluster centroid stats.

    Returns:
        dict: {raw_id: final_id} where final_id follows 0=R,1=Y,2=G,3=B
    """
    df = df.copy()
    df['RawCluster'] = raw_labels

    # Compute per-cluster core stats in original feature space
    stats = df.groupby('RawCluster')[['Organoids_Volume_Fill', 'Cavity_Volume', 'Scatt_Mean']].mean()
    raw_ids = sorted(stats.index.tolist())

    if len(raw_ids) < 4:
        raise ValueError(f"Expected 4 clusters, got {len(raw_ids)}: {raw_ids}")

    # Blue: highest OAC
    raw_blue = int(stats['Scatt_Mean'].idxmax())

    # Remaining sorted by volume descending
    remaining = [int(c) for c in raw_ids if int(c) != raw_blue]
    vol_sorted = stats.loc[remaining, 'Organoids_Volume_Fill'].sort_values(ascending=False)

    if len(vol_sorted) < 3:
        raise ValueError(f"Not enough remaining clusters for volume sorting: {remaining}")

    raw_red = int(vol_sorted.index[0])      # largest volume
    raw_yellow = int(vol_sorted.index[1])   # 2nd largest
    raw_green = int(vol_sorted.index[2])    # smallest

    return {
        raw_red: 0,
        raw_yellow: 1,
        raw_green: 2,
        raw_blue: 3,
    }


def compute_phenotype_prototypes(df: pd.DataFrame) -> dict:
    """
    Compute phenotype initialization prototypes from data percentiles.
    Returns dict of prototype vectors in *original* feature space.
    """
    # Cavity ratio in original space
    cavity_ratio = df['Cavity_Volume'] / df['Organoids_Volume_Fill'].clip(lower=1)

    prototypes = {
        0: {  # Red / Large Cystic
            'Organoids_Volume_Fill': np.percentile(df['Organoids_Volume_Fill'], 95),
            'Cavity_Ratio': np.percentile(cavity_ratio, 90),
            'Scatt_Mean': np.percentile(df['Scatt_Mean'], 25),
        },
        1: {  # Yellow / Large Solid
            'Organoids_Volume_Fill': np.percentile(df['Organoids_Volume_Fill'], 75),
            'Cavity_Ratio': 0.0,
            'Scatt_Mean': np.percentile(df['Scatt_Mean'], 35),
        },
        2: {  # Green / Small Solid
            'Organoids_Volume_Fill': np.percentile(df['Organoids_Volume_Fill'], 25),
            'Cavity_Ratio': 0.0,
            'Scatt_Mean': np.percentile(df['Scatt_Mean'], 50),
        },
        3: {  # Blue / Minimal High-Density
            'Organoids_Volume_Fill': np.percentile(df['Organoids_Volume_Fill'], 10),
            'Cavity_Ratio': 0.0,
            'Scatt_Mean': np.percentile(df['Scatt_Mean'], 90),
        },
    }
    return prototypes


def build_means_init(prototypes: dict, preprocessor: Preprocessor) -> np.ndarray:
    """
    Convert original-space prototypes to standardized-space mean vectors
    for GMM means_init.
    """
    rows = []
    processed_features = preprocessor.get_feature_names()
    for cid in sorted(prototypes.keys()):
        proto = prototypes[cid]
        # Build a single-row DataFrame with the prototype values
        row = {k: v for k, v in proto.items()}
        # Fill missing columns with 0 (they'll be standardized anyway)
        for f in processed_features:
            if f not in row:
                row[f] = 0.0
        rows.append(row)

    proto_df = pd.DataFrame(rows)[processed_features]
    return preprocessor.scaler.transform(proto_df)


# ============================================================================
# 5. Model I/O helpers
# ============================================================================

def save_model_package(path: str, model, model_type: str, preprocessor: Preprocessor,
                       feature_names: list, extra: dict = None):
    """Save a unified model package."""
    pkg = {
        'model': model,
        'model_type': model_type,
        'scaler': preprocessor.scaler if preprocessor is not None else None,
        'preprocessor': preprocessor,
        'feature_names': feature_names,
        'phenotype_names': PHENOTYPE_NAMES,
    }
    if extra:
        pkg.update(extra)
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(pkg, f)
    print(f"Model package saved -> {path}")


def load_model_package(path: str) -> dict:
    """Load a unified model package."""
    with open(path, 'rb') as f:
        return pickle.load(f)


# ============================================================================
# 6. Rule-based classifier utilities
# ============================================================================

def compute_well_stats(df: pd.DataFrame) -> dict:
    """Compute per-well percentile statistics for rule-based classification."""
    vol = df['Organoids_Volume_Fill']
    oac = df['Scatt_Mean']
    return {
        'vol_median': vol.median(),
        'vol_75': vol.quantile(0.75),
        'vol_60': vol.quantile(0.60),
        'vol_90': vol.quantile(0.90),
        'oac_85': oac.quantile(0.85),
        'oac_90': oac.quantile(0.90),
    }


def rule_classify_row(row: pd.Series, well_stats: dict) -> int:
    """
    Classify a single organoid using biological rules.

    Returns: 0=Red, 1=Yellow, 2=Green, 3=Blue
    """
    vol = row['Organoids_Volume_Fill']
    oac = row['Scatt_Mean']
    cavity_ratio = row['Cavity_Volume'] / max(row['Organoids_Volume_Fill'], 1)

    # 1. Blue: highest OAC + relatively small volume (apoptotic/damaged)
    if oac > well_stats['oac_85'] and vol < well_stats['vol_60']:
        return 3

    # 2. Red: large + cystic (healthy active)
    if vol > well_stats['vol_75'] and cavity_ratio > 0.12:
        return 0

    # 3. Yellow vs Green: split by median volume (solid organoids)
    if vol > well_stats['vol_median']:
        return 1  # Large solid
    else:
        return 2  # Small solid / dormant


def rule_classify_df(df: pd.DataFrame) -> pd.Series:
    """Apply rule-based classification to an entire well DataFrame."""
    well_stats = compute_well_stats(df)
    return df.apply(lambda row: rule_classify_row(row, well_stats), axis=1)