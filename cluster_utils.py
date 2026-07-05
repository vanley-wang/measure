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

    Supports two modes:
      - 'full'  (default): 10-dim feature set (backward compatible)
      - 'reduced': 5-dim collinearity-free feature set

    Steps:
      1. Compute Cavity_Ratio = Cavity_Volume / (Volume_Fill + 1)
      2. Drop constant/redundant features (full mode only)
      3. Log1p-transform skewed volume features
      4. Fit / transform with StandardScaler
    """

    def __init__(self, mode='full'):
        if mode not in ('full', 'reduced'):
            raise ValueError("mode must be 'full' or 'reduced'")
        self.mode = mode
        self.scaler = StandardScaler()
        self._fitted = False

        if mode == 'reduced':
            self._log_features = REDUCED_LOG_FEATURES
            self._processed_features = REDUCED_PROCESSED_FEATURES
        else:
            self._log_features = LOG_FEATURES
            self._processed_features = PROCESSED_FEATURES

    def _engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add engineered features without modifying original df."""
        df = df.copy()
        # Cavity ratio: 0 means solid, >0 means cystic
        df['Cavity_Ratio'] = df['Cavity_Volume'] / (df['Organoids_Volume_Fill'].clip(lower=1))
        return df

    def _select_and_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop, log-transform, and select final feature columns."""
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
        # Select
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
