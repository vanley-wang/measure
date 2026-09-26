# ============================================================
# GC 数据聚类结构分析 — 确定最优K
# ============================================================
import os, sys, pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

CF = ['Organoids_Volume','Organoids_Volume_Fill','Organoids_Surface',
      'Cavity_Volume','CavityNum','LongAxis','ShortAxis',
      'Wall_Thickness','Sphericity','Scatt_Mean','Scatt_STD']

GC_D3_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          'Data','FXN_2023_new（GC）','FXN_20230701','measure_excel')

# 加载所有 GC Day3 类器官
print('Loading all GC Day3 organoids...')
dfs = []
for f in sorted(os.listdir(GC_D3_DIR)):
    if not f.endswith('.xlsx'): continue
    df = pd.read_excel(os.path.join(GC_D3_DIR, f))
    dfs.append(df[CF].fillna(0))
all_gc = pd.concat(dfs, ignore_index=True)
print(f'  Total organoids: {len(all_gc)}')

X = StandardScaler().fit_transform(all_gc.values)
rng = np.random.RandomState(42)

# 肘部曲线 + 轮廓系数
print(f'\n{"K":<4s} {"Inertia":>12s} {"Silhouette":>12s} {"Elbow Δ":>10s}')
print('-' * 42)

results = {}
prev_inertia = None
for k in range(2, 9):
    km = KMeans(n_clusters=k, random_state=rng, n_init=10).fit(X)
    inertia = km.inertia_
    sil = silhouette_score(X, km.labels_, sample_size=5000)
    elbow_delta = (prev_inertia - inertia) / prev_inertia * 100 if prev_inertia else float('nan')
    prev_inertia = inertia
    results[k] = {'inertia': inertia, 'silhouette': sil, 'labels': km.labels_, 'centers': km.cluster_centers_}
    print(f'{k:<4d} {inertia:>12.1f} {sil:>12.4f} {elbow_delta:>9.1f}%')

# 推荐K
best_sil_k = max(results, key=lambda k: results[k]['silhouette'])
print(f'\n  Silhouette best at K={best_sil_k} ({results[best_sil_k]["silhouette"]:.4f})')

# 对于每个K, 展示聚类中心特征
print(f'\n{"="*60}')
print(f'  各 K 值下的聚类中心特征 (标准化值, 正=高于均值, 负=低于均值)')
print(f'{"="*60}')

feature_short = ['Vol','VolFill','Surf','CavVol','CavNum','LongAx','ShortAx','WallTh','Spher','ScattM','ScattS']

for k in [3, 4, 5, 6]:
    if k not in results: continue
    r = results[k]
    print(f'\n  --- K={k} (Silhouette={r["silhouette"]:.4f}) ---')
    # 每个簇的特征
    header = f'  {"Cluster":>8s} {"%":>6s}'
    for s in feature_short:
        header += f'{s:>8s}'
    print(header)
    for c in range(k):
        pct = (r['labels'] == c).sum() / len(r['labels']) * 100
        row = f'  C{c:<7d} {pct:>5.1f}%'
        for j, s in enumerate(feature_short):
            row += f'{r["centers"][c][j]:>8.2f}'
        print(row)

print()