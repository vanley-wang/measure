# ============================================================
# GC 全孔 Delta PCA — 不聚类，每个孔 11 个特征的 D5−D3 均值
# ============================================================
import os, sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import pearsonr

GC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'Data', 'FXN_2023_new（GC）')
D3_DIR = os.path.join(GC_DIR, 'FXN_20230701', 'measure_excel')
D5_DIR = os.path.join(GC_DIR, 'FXN_20230703', 'measure_excel')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

CF = ['Organoids_Volume', 'Organoids_Volume_Fill', 'Organoids_Surface',
      'Cavity_Volume', 'CavityNum', 'LongAxis', 'ShortAxis',
      'Wall_Thickness', 'Sphericity', 'Scatt_Mean', 'Scatt_STD']

CF_SHORT = ['Vol', 'FillVol', 'Surf', 'CavVol', 'CavNum',
            'LongAx', 'ShortAx', 'WallTh', 'Spher', 'ScattM', 'ScattS']

# 1. 加载 ATP
atp_df = pd.read_excel(os.path.join(GC_DIR, 'ATP.xlsx'))
gc_atp = {}
for _, row in atp_df.iterrows():
    wid = str(row['Name']).split('_')[0]
    if pd.notna(row['ATP']): gc_atp[wid] = float(row['ATP'])

# 2. 每孔 Delta = D5均值 − D3均值
print('Computing per-well Delta...')
d3_files = sorted([f for f in os.listdir(D3_DIR) if f.endswith('.xlsx')])
d5_files = sorted([f for f in os.listdir(D5_DIR) if f.endswith('.xlsx')])
d3_map = {f.replace('.xlsx','').split('_')[0]: f for f in d3_files}
d5_map = {f.replace('.xlsx','').split('_')[0]: f for f in d5_files}

common = sorted(set(d3_map) & set(d5_map))
print(f'  Wells: {len(common)}')

delta_rows, well_ids, atp_vals = [], [], []
for wid in common:
    d3 = pd.read_excel(os.path.join(D3_DIR, d3_map[wid]))
    d5 = pd.read_excel(os.path.join(D5_DIR, d5_map[wid]))
    row = []
    for feat in CF:
        row.append(d5[feat].mean() - d3[feat].mean())
    delta_rows.append(row)
    well_ids.append(wid)
    atp_vals.append(gc_atp.get(wid, np.nan))

X_delta = np.array(delta_rows)  # 36 × 11
atp = np.array(atp_vals)

# 3. PCA
scaler = StandardScaler()
X_std = scaler.fit_transform(X_delta)
pca = PCA(n_components=min(len(common), len(CF)))
X_pca = pca.fit_transform(X_std)

print(f'\n{"="*60}')
print(f'  PCA Variance Explained')
print(f'{"="*60}')
cumsum = 0
for i, v in enumerate(pca.explained_variance_ratio_):
    cumsum += v
    print(f'  PC{i+1}: {v*100:>6.1f}%   (cumulative {cumsum*100:.1f}%)')

# 4. 每个 PC 与 ATP 的 r
print(f'\n{"="*60}')
print(f'  Each PC vs ATP (Pearson r)')
print(f'{"="*60}')
for i in range(X_pca.shape[1]):
    r, p = pearsonr(X_pca[:, i], atp)
    sig = '**' if p<0.01 else ('*' if p<0.05 else '')
    print(f'  PC{i+1}: r={r:+.4f}  p={p:.6f}  {sig}')

# 5. 综合得分 — 用方差比加权
weights = pca.explained_variance_ratio_ / pca.explained_variance_ratio_.sum()
pc_scores_weighted = X_pca @ weights
r_weighted, p_weighted = pearsonr(pc_scores_weighted, atp)
print(f'\n  Weighted Score (all PCs): r={r_weighted:+.4f}  p={p_weighted:.6f}')

# 只用前 3 个 PC
weights3 = pca.explained_variance_ratio_[:3] / pca.explained_variance_ratio_[:3].sum()
score3 = X_pca[:, :3] @ weights3
r3, p3 = pearsonr(score3, atp)
print(f'  Weighted Score (PC1-3):  r={r3:+.4f}  p={p3:.6f}')

# 6. PC 成分公式
print(f'\n{"="*60}')
print(f'  PC Loading Matrix (LCᵢⱼ = 特征 j 对 PCᵢ 的贡献)')
print(f'{"="*60}')

# 表头
header = '  ' + ' '.join(f'{s:>8s}' for s in CF_SHORT)
print(header)
for i in range(min(5, len(CF))):
    row_str = f'  PC{i+1}'
    for j in range(len(CF)):
        row_str += f'{pca.components_[i][j]:>+8.3f}'
    print(row_str)

# 7. PC 公式展开
print(f'\n{"="*60}')
print(f'  PC Formula (原始 Delta 值的组合)')
print(f'  Format: PCᵢ = Σ(loadingᵢⱼ × Z_score(ΔFeatureⱼ))')
print(f'{"="*60}')

for i in range(min(4, len(CF))):
    print(f'\n  PC{i+1} = ')
    terms = []
    for j in range(len(CF)):
        c = pca.components_[i][j]
        if abs(c) > 0.15:
            terms.append(f'{c:+.3f} × Z(Δ{CF_SHORT[j]})')
    print('    ' + '\n    '.join(terms))
    # also show in raw units
    print(f'  In raw units (ΔFeatures):')
    raw_terms = []
    for j in range(len(CF)):
        c = pca.components_[i][j]
        if abs(c) > 0.15:
            raw_terms.append(f'{c/scaler.scale_[j]:.4e} × Δ{CF_SHORT[j]}')
    print('    ' + '\n    '.join(raw_terms))

# 8. 综合得分公式
beta = np.zeros(len(CF))
for i in range(len(CF)):
    beta += weights[i] * pca.components_[i]
beta /= scaler.scale_

print(f'\n{"="*60}')
print(f'  GC Delta Score Formula')
print(f'{"="*60}')
print(f'  Score = Σ(βⱼ × ΔFeatureⱼ)')
print(f'  where ΔFeature = D5_mean − D3_mean for each well')
print()
top_features = sorted(zip(CF_SHORT, beta, abs(beta)), key=lambda x: -x[2])
for name, b, _ in top_features:
    print(f'  β_Δ{name:<8s} = {b:+.6e}')

# 9. 用这个公式算分
scores_beta = X_delta @ beta
r_beta, p_beta = pearsonr(scores_beta, atp)
print(f'\n  Beta Score vs ATP: r = {r_beta:.4f}  p = {p_beta:.6f}')

# 10. 保存
df_out = pd.DataFrame({'Well': well_ids, 'ATP': atp})
for i in range(X_pca.shape[1]):
    df_out[f'PC{i+1}'] = X_pca[:, i]
df_out['Score_All'] = pc_scores_weighted
df_out['Score_PC1_3'] = score3
df_out['Score_Beta'] = scores_beta
df_out.to_excel(os.path.join(OUTPUT_DIR, 'GC_delta_PCA.xlsx'), index=False)
print(f'\nSaved: output/GC_delta_PCA.xlsx')

# 11. 对比: 仅用 ΔVolume
r_vol, p_vol = pearsonr(X_delta[:, 0], atp)
print(f'\n  Comparison:')
print(f'    ΔVolume alone:         r = +{r_vol:.4f}')
print(f'    All Delta → PCA Score: r = +{r_beta:.4f}')