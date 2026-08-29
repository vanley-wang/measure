import os as _os, sys as _sys

_project_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', '..'))
_sys.path.insert(0, _project_root)
_sys.path.insert(0, _os.path.join(_project_root, 'step7_pca_atp_v2_stratified'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
import pickle
import os
import glob

from cluster_utils import RAW_FEATURES

data_folders = [
    os.path.join(_project_root, 'Data', 'nnUNet_FXN_2023', 'FXN_0701', 'measure_excel'),
    os.path.join(_project_root, 'Data', 'nnUNet_FXN_2023', 'FXN_0703', 'measure_excel'),
]

model_dir = os.path.join(_project_root, 'model')
os.makedirs(model_dir, exist_ok=True)
features_list = list(RAW_FEATURES)

print('=' * 70)
print('  Training KMeans on nnUNet Segmentation Data')
print('=' * 70)

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

print('--- Reading nnUNet data ---')
df_list = []
for folder in data_folders:
    if not os.path.exists(folder):
        print(f'  SKIP (not found): {folder}')
        continue
    files = glob.glob(os.path.join(folder, '*.xlsx'))
    print(f'  {os.path.basename(folder)}: {len(files)} files')
    for file in files:
        try:
            temp_df = pd.read_excel(file)
            if 'Index' in temp_df.columns and 'Object_Id' not in temp_df.columns:
                temp_df = temp_df.rename(columns={'Index': 'Object_Id'})
            df_list.append(temp_df)
        except Exception as e:
            print(f'    ERROR: {os.path.basename(file)}: {e}')

if not df_list:
    raise ValueError('No nnUNet data files found!')

Data_All = pd.concat(df_list, ignore_index=True)
before = len(Data_All)
Data_All = Data_All.dropna(subset=features_list)
after = len(Data_All)
print(f'  Objects: {after} (dropped {before - after} NaN)')

Data_Features = Data_All[features_list].copy()

print('\n--- Standardizing ---')
scaler = StandardScaler()
Data_Std = scaler.fit_transform(Data_Features)

print('\n--- Elbow Method (K=1..8) ---')
meanDispersions = []
K_range = range(1, 9)
for k in K_range:
    km = KMeans(n_clusters=k, init='k-means++', n_init='auto', random_state=42)
    km.fit(Data_Std)
    m_Disp = sum(np.min(cdist(Data_Std, km.cluster_centers_, 'euclidean'), axis=1)) / Data_Std.shape[0]
    meanDispersions.append(m_Disp)

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(7, 5))
plt.plot(K_range, meanDispersions, marker='o', linestyle='-', color='black',
         linewidth=1.5, markersize=7, markerfacecolor='white', markeredgewidth=1.5)
plt.xlabel('Number of Clusters (k)', fontsize=12, fontweight='bold', labelpad=10)
plt.ylabel('Mean Dispersion (Euclidean)', fontsize=12, fontweight='bold', labelpad=10)
plt.title('Elbow Method for Optimal k (nnUNet)', fontsize=14, fontweight='bold', pad=15)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.2)
ax.spines['bottom'].set_linewidth(1.2)
ax.tick_params(width=1.2, labelsize=10)
plt.tight_layout()
elbow_path = os.path.join(model_dir, 'Elbow_Method_Plot_nnUNet.png')
plt.savefig(elbow_path, dpi=600, bbox_inches='tight')
print(f'  Elbow plot saved -> {elbow_path}')
plt.show()

# ============================================================
# K=4 clustering
# ============================================================
optimal_k = 4
print(f'\n--- Training KMeans (K={optimal_k}) ---')
final_model = KMeans(n_clusters=optimal_k, init='k-means++', max_iter=300, n_init=10, random_state=42)
cluster_labels = final_model.fit_predict(Data_Std)

Data_All['RawCluster'] = cluster_labels
cluster_stats = Data_All.groupby('RawCluster')[['Organoids_Volume_Fill', 'Cavity_Volume', 'Scatt_Mean']].mean()

print('\nRaw Cluster centroid stats:')
print(cluster_stats)

raw_ids = list(range(optimal_k))
raw_c4 = cluster_stats['Scatt_Mean'].idxmax()
remaining = [c for c in raw_ids if c != raw_c4]
remaining_sorted = cluster_stats.loc[remaining, 'Organoids_Volume_Fill'].sort_values(ascending=False)
raw_c1 = remaining_sorted.index[0]
raw_c2 = remaining_sorted.index[1]
raw_c3 = remaining_sorted.index[2]

raw_to_final = {raw_c1: 0, raw_c2: 1, raw_c3: 2, raw_c4: 3}
Data_All['Cluster'] = Data_All['RawCluster'].map(raw_to_final)

print(f'\nMapping:')
print(f'  Raw {raw_c1} -> Red(0)    [Vol max = {cluster_stats.loc[raw_c1, "Organoids_Volume_Fill"]:.0f}]')
print(f'  Raw {raw_c2} -> Yellow(1) [Vol 2nd = {cluster_stats.loc[raw_c2, "Organoids_Volume_Fill"]:.0f}]')
print(f'  Raw {raw_c3} -> Green(2)  [Vol min = {cluster_stats.loc[raw_c3, "Organoids_Volume_Fill"]:.0f}]')
print(f'  Raw {raw_c4} -> Blue(3)   [Scatt max = {cluster_stats.loc[raw_c4, "Scatt_Mean"]:.0f}]')

print('\nFinal cluster distribution:')
names = {0: 'Red(H)', 1: 'Yellow(H)', 2: 'Green(I)', 3: 'Blue(D)'}
for c in range(4):
    cnt = (Data_All['Cluster'] == c).sum()
    print(f'  {names[c]:12s}: {cnt:>8,} ({cnt/len(Data_All)*100:>5.1f}%)')

# ============================================================
# Save model
# ============================================================
model_package = {
    'kmeans': final_model,
    'model': final_model,
    'scaler': scaler,
    'raw_to_final': raw_to_final,
    'feature_names': features_list,
}
model_path = os.path.join(model_dir, 'Kmeans-scatt-nnUNet.pickle')
with open(model_path, 'wb') as f:
    pickle.dump(model_package, f)
print(f'\nModel saved -> {model_path}')

summary_mean = Data_All.groupby('Cluster')[features_list].mean()
print('\n' + '=' * 50)
print(f'Cluster Summary (K={optimal_k})')
print('=' * 50)
print(summary_mean)

summary_path = os.path.join(model_dir, 'Cluster_scatt_nnUNet.xlsx')
summary_mean.to_excel(summary_path)
print(f'\nSummary saved -> {summary_path}')
print('DONE')