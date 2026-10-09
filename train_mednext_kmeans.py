"""
Train KMeans++ for MedNext_GC032 — with enhanced cavity feature weight.

Problem: In the previous model, ~80% of "red" (Type 1 / cystic) organoids
were actually solid (no cavity), because Cavity_Volume was just one of 11
equally-weighted features and got drowned out by size.

Fix:
  1. Add Cavity_Ratio = Cavity_Volume / Volume_Fill  (what fraction is lumen)
  2. Apply non-linear boosting to cavity features so "cystic vs solid"
     becomes a primary axis of variation
  3. Keep log-volume transforms for size features
  4. K=6 → K=4 merging, with cystic clusters properly separated

Features used (12 total):
  - Size (log1p): Volume, Volume_Fill, Surface, LongAxis, ShortAxis
  - Cavity (sqrt-boosted): Cavity_Volume, Cavity_Ratio
  - Other (raw): CavityNum, Wall_Thickness, Sphericity
  - Scatter (raw): Scatt_Mean, Scatt_STD
"""
import os
import sys
import glob
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

warnings.filterwarnings('ignore')

from cluster_utils import CavityBoostedPreprocessor

# ================ Config ================
BASE_DIR = "MedNext_GC032"
BATCHES = ["20230701", "20230703"]
MIN_VOLUME = 50
K = 6
N_INIT = 10
MAX_ITER = 300
RANDOM_STATE = 42

# 11 raw features (matching measure_excel columns)
# Cavity_Ratio is engineered internally by the preprocessor
FEATURE_NAMES = [
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

# Internal 12-dim features (raw 11 + Cavity_Ratio appended)
_INTERNAL_FEATURES = FEATURE_NAMES + ['Cavity_Ratio']

# Features to log1p (size / long-tail)
LOG_FEATURES = [
    'Organoids_Volume',
    'Organoids_Volume_Fill',
    'Organoids_Surface',
    'Cavity_Volume',    # log cavity too so huge cysts don't dominate
    'LongAxis',
    'ShortAxis',
]

# Cavity_Ratio is 0-1 but most are 0; we BOOST it so cystic vs solid
# becomes a strong separating axis. Multiply by a boost factor before scaling.
CAVITY_RATIO_BOOST = 20.0  # 0-1 range → 0-20 range, comparable to other log features

OUTPUT_PATH = "model/Kmeans-scatt-mednext.pickle"


def load_all_organoids(base_dir, batches):
    dfs = []
    for batch in batches:
        excel_dir = os.path.join(base_dir, batch, 'measure_excel')
        files = sorted(glob.glob(os.path.join(excel_dir, '*.xlsx')))
        print(f"  {batch}: {len(files)} files")
        for fp in files:
            df = pd.read_excel(fp)
            df['_batch'] = batch
            df['_well'] = os.path.basename(fp).replace('.xlsx', '')
            dfs.append(df)

    full = pd.concat(dfs, ignore_index=True)
    print(f"\n  Total before filtering: {len(full)}")
    full = full[full['Organoids_Volume_Fill'] >= MIN_VOLUME].copy()
    print(f"  After MIN_VOLUME={MIN_VOLUME}: {len(full)}")

    # Check we have all raw features
    missing = [f for f in FEATURE_NAMES if f not in full.columns]
    if missing:
        raise ValueError(f"Missing features: {missing}")

    full = full.dropna(subset=FEATURE_NAMES).copy()
    print(f"  After dropping NaN features: {len(full)}")

    return full


def train_kmeans_k6(df):
    preprocessor = CavityBoostedPreprocessor(
        FEATURE_NAMES, LOG_FEATURES, CAVITY_RATIO_BOOST
    )
    X_std = preprocessor.fit_transform(df[FEATURE_NAMES])

    kmeans = KMeans(
        n_clusters=K,
        init='k-means++',
        n_init=N_INIT,
        max_iter=MAX_ITER,
        random_state=RANDOM_STATE,
    )
    labels = kmeans.fit_predict(X_std)

    # Centers in internal 12-dim feature space (raw 11 + Cavity_Ratio)
    # Compute means from the actual data
    df_eng = df.copy()
    df_eng['Cavity_Ratio'] = df_eng['Cavity_Volume'] / df_eng['Organoids_Volume_Fill'].clip(lower=1)

    centers = np.zeros((K, len(_INTERNAL_FEATURES)))
    for cid in range(K):
        mask = labels == cid
        centers[cid] = df_eng.loc[mask, _INTERNAL_FEATURES].mean().values

    return kmeans, preprocessor, labels, centers


def build_merge_map(centers):
    """
    K=6 → K=4 merge, with cavity as primary separator.

    centers are in the 12-dim internal feature space (11 raw + Cavity_Ratio at end).

    Strategy:
      - Top 3 by volume:
        - Highest cavity → Type 0 (red, giant cystic)
        - Other two → Type 1 (yellow, large solid / transitional)
      - Bottom 3 by volume:
        - Highest scatter → Type 3 (blue, dense damaged)
        - Other two → Type 2 (green, small baseline)
    """
    vol_idx = _INTERNAL_FEATURES.index('Organoids_Volume_Fill')
    cav_idx = _INTERNAL_FEATURES.index('Cavity_Ratio')
    scatt_idx = _INTERNAL_FEATURES.index('Scatt_Mean')

    volumes = centers[:, vol_idx]
    cavity_ratios = centers[:, cav_idx]
    scatts = centers[:, scatt_idx]

    # Rank by volume descending
    vol_order = np.argsort(-volumes)

    raw_to_final = {}

    # --- Top 3 clusters (large organoids) ---
    top3 = vol_order[:3]
    # Sort top3 by cavity_ratio descending
    top3_by_cav = sorted(top3, key=lambda i: -cavity_ratios[i])
    # Highest cavity → Type 0 (red, cystic)
    raw_to_final[int(top3_by_cav[0])] = 0
    # Next two → Type 1 (yellow, large solid / transitional)
    raw_to_final[int(top3_by_cav[1])] = 1
    raw_to_final[int(top3_by_cav[2])] = 1

    # --- Bottom 3 clusters (small organoids) ---
    bot3 = vol_order[3:]
    # Sort bot3 by scatter descending
    bot3_by_scatt = sorted(bot3, key=lambda i: -scatts[i])
    # Highest scatter → Type 3 (blue, dense damaged)
    raw_to_final[int(bot3_by_scatt[0])] = 3
    # Next two → Type 2 (green, small baseline)
    raw_to_final[int(bot3_by_scatt[1])] = 2
    raw_to_final[int(bot3_by_scatt[2])] = 2

    return raw_to_final


def print_centers(centers, raw_to_final):
    vol_idx = _INTERNAL_FEATURES.index('Organoids_Volume_Fill')
    cav_idx = _INTERNAL_FEATURES.index('Cavity_Ratio')
    scatt_idx = _INTERNAL_FEATURES.index('Scatt_Mean')
    long_idx = _INTERNAL_FEATURES.index('LongAxis')
    sph_idx = _INTERNAL_FEATURES.index('Sphericity')

    order = np.argsort(-centers[:, vol_idx])
    names = {0: '巨大囊泡型(红)', 1: '大实心/过渡型(黄)',
             2: '小体积基准型(绿)', 3: '高致密受损型(蓝)'}

    print("\n  K=6 聚类中心 (按体积排序):")
    for rank, raw_id in enumerate(order):
        final_id = raw_to_final.get(int(raw_id), -1)
        c = centers[raw_id]
        cav_pct = c[cav_idx] * 100
        print(f"    Rank {rank+1} | Raw {raw_id} → Final {final_id} [{names.get(final_id, '?')}]")
        print(f"      Vol={c[vol_idx]:.0f}  Cavity%={cav_pct:.1f}%  "
              f"Scatt={c[scatt_idx]:.1f}  LongAxis={c[long_idx]:.1f}  "
              f"Sphericity={c[sph_idx]:.2f}")


def main():
    print("=" * 60)
    print("MedNext_GC032 KMeans 训练 — Cavity-Boosted, K=6→K=4")
    print("=" * 60)

    # 1. Load
    print(f"\n[1/4] 加载数据 (Day3 + Day5)...")
    df = load_all_organoids(BASE_DIR, BATCHES)

    if len(df) < 100:
        print("ERROR: 样本太少")
        sys.exit(1)

    # Cavity stats
    cav_pct = 100 * (df['Cavity_Volume'] > 0).sum() / len(df)
    print(f"  有空腔的类器官: {(df['Cavity_Volume'] > 0).sum()} ({cav_pct:.2f}%)")

    # 2. Train K=6
    print(f"\n[2/4] 训练 KMeans++ (K={K}, cavity-boosted)...")
    kmeans, preprocessor, labels, centers = train_kmeans_k6(df)

    # 3. Merge map
    print(f"\n[3/4] 构建 K=6 → K=4 合并映射...")
    raw_to_final = build_merge_map(centers)
    print(f"  raw_to_final = {raw_to_final}")
    print_centers(centers, raw_to_final)

    merged_labels = np.array([raw_to_final[int(l)] for l in labels])
    print(f"\n  合并后分布 (n={len(df)}):")
    names = {0: '红 巨大囊泡', 1: '黄 大实心/过渡', 2: '绿 小体积基准', 3: '蓝 高致密受损'}
    for t in range(4):
        count = int(np.sum(merged_labels == t))
        print(f"    Type {t} {names[t]}: {count:>6} ({100*count/len(df):5.1f}%)")

    # Verify red cluster is mostly cystic
    red_mask = merged_labels == 0
    red_cav = df.loc[red_mask, 'Cavity_Volume'].values
    red_cystic_pct = 100 * (red_cav > 0).sum() / len(red_cav) if len(red_cav) else 0
    print(f"\n  验证: 红色类中 '有空腔' 比例 = {red_cystic_pct:.1f}% (目标: >50%)")

    # 4. Save
    print(f"\n[4/4] 保存模型 → {OUTPUT_PATH}")
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    model_pkg = {
        'kmeans': kmeans,
        'model': kmeans,              # cluster-merge.py compatibility
        'scaler': preprocessor.scaler, # legacy
        'preprocessor': preprocessor,  # preferred: full pipeline
        'raw_to_final': raw_to_final,
        'feature_names': FEATURE_NAMES,
        'model_type': 'kmeans',
        'n_clusters_raw': K,
        'n_clusters_final': 4,
        'n_organoids_trained': len(df),
        'training_data': f'{BASE_DIR} ({", ".join(BATCHES)})',
        'cavity_ratio_boost': CAVITY_RATIO_BOOST,
        'log_features': LOG_FEATURES,
        'merge_strategy': 'K=6→K=4, cavity-boosted, top=cystic/solid bot=baseline/dense',
    }

    with open(OUTPUT_PATH, 'wb') as f:
        pickle.dump(model_pkg, f)

    print(f"\n  Done. saved: {OUTPUT_PATH}")
    print(f"  samples: {len(df)}, features: {len(FEATURE_NAMES)}, K={K}→4")
    print(f"  cavity_boost: {CAVITY_RATIO_BOOST}x")


if __name__ == '__main__':
    main()
