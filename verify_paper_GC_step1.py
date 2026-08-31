"""
Step 1 快速验证
A: GC上重新拟合 w1-w4 权重
B: 各分量单独/组合预测ATP
C: LOO交叉验证
"""
import os, sys, glob, warnings
import numpy as np, pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut
from itertools import combinations
warnings.filterwarnings('ignore')

print('=' * 80)
print('  Step 1: GC数据重新拟合权重 + 分量分析')
print('=' * 80)

# ============================================================
# 1. 加载ATP
# ============================================================
atp_file = r'D:\Desktop\music\measure\Data\FXN_2023_new（GC）\ATP.xlsx'
atp_df = pd.read_excel(atp_file)
atp_dict = {}
for _, row in atp_df.iterrows():
    name = str(row['Name']).strip() if pd.notna(row['Name']) else ''
    if not name: continue
    wid = name.split('_')[0]
    if pd.notna(row['ATP']) and float(row['ATP']) > 0:
        atp_dict[wid] = float(row['ATP'])
print(f'ATP: {len(atp_dict)} wells')

# ============================================================
# 2. 加载GC形态学 + Scatt数据
# ============================================================
GC_BASE = r'D:\Desktop\music\measure\Data\FXN_2023_new（GC）'

def load_gc(day_folder):
    measure_dir = os.path.join(GC_BASE, day_folder, 'measure_excel')
    scatt_dir   = os.path.join(GC_BASE, day_folder, 'scatt')
    if not os.path.exists(measure_dir): return None
    dfs = []
    for fp in sorted(glob.glob(os.path.join(measure_dir, '*.xlsx'))):
        fn = os.path.basename(fp)
        parts = fn.replace('.xlsx', '').split('_')
        if len(parts) < 2: continue
        wid, ds = parts[0], parts[1]
        tmp = pd.read_excel(fp)
        if 'Index' in tmp.columns:
            tmp = tmp.rename(columns={'Index': 'Object_Id'})
        sf = os.path.join(scatt_dir, f'{wid}_{ds}_scatt.xlsx')
        if os.path.exists(sf):
            st = pd.read_excel(sf)
            if len(st) == len(tmp):
                for col in st.columns:
                    if col not in tmp.columns and col not in ['Index', 'Object_Id']:
                        tmp[col] = st[col].values
        tmp['_well_id'] = wid
        tmp['_day'] = ds
        dfs.append(tmp)
    return pd.concat(dfs, ignore_index=True) if dfs else None

df_d1 = load_gc('FXN_20230701')
df_d2 = load_gc('FXN_20230703')
print(f'Day1: {len(df_d1)} objects, Day2: {len(df_d2)} objects')

# ============================================================
# 3. GC专用KMeans (4类, 按体积降序重排)
# ============================================================
cfeats = ['Organoids_Volume', 'Organoids_Volume_Fill', 'Organoids_Surface',
          'Cavity_Volume', 'CavityNum', 'LongAxis', 'ShortAxis',
          'Wall_Thickness', 'Sphericity', 'Scatt_Mean', 'Scatt_STD']

df_comb = pd.concat([df_d1[cfeats], df_d2[cfeats]], ignore_index=True)
scaler_k = StandardScaler().fit(df_comb)
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10).fit(scaler_k.transform(df_comb))
labels = kmeans.labels_

df_d1 = df_d1.copy(); df_d2 = df_d2.copy()
df_d1['Cluster'] = labels[:len(df_d1)]
df_d2['Cluster'] = labels[len(df_d1):]

cv_list = [(c, df_comb.iloc[labels == c]['Organoids_Volume'].mean()) for c in range(4)]
cv_list.sort(key=lambda x: x[1], reverse=True)
cmap = {old: new for new, (old, _) in enumerate(cv_list)}
df_d1['Cluster'] = df_d1['Cluster'].map(cmap)
df_d2['Cluster'] = df_d2['Cluster'].map(cmap)

for c, nm in enumerate(['大囊状(H)', '大实心(H)', '小实心(I)', '极小(D)']):
    n1 = (df_d1['Cluster'] == c).sum()
    n2 = (df_d2['Cluster'] == c).sum()
    print(f'  Cluster {c} ({nm}): Day1={n1}, Day2={n2}')

df_all = pd.concat([df_d1, df_d2], ignore_index=True)

# ============================================================
# 4. 提取论文15维特征
# ============================================================
def extract_features(df, wid, dv):
    w = df[(df['_well_id'] == wid) & (df['_day'] == dv)]
    f = {}
    c1 = w[w['Cluster'] == 0]; n1 = len(c1)
    f['X11'] = c1['Cavity_Volume'].mean() if n1 > 0 else 0
    f['X12'] = c1['Organoids_Surface'].mean() if n1 > 0 else 0
    f['X13'] = c1['Sphericity'].mean() if n1 > 0 else 0
    c2 = w[w['Cluster'] == 1]; n2 = len(c2)
    f['X21'] = n2
    f['X22'] = c2['ShortAxis'].mean() if n2 > 0 else 0
    f['X23'] = c2['Organoids_Surface'].mean() if n2 > 0 else 0
    f['X24'] = c2['Organoids_Volume_Fill'].mean() if n2 > 0 else 0
    c3 = w[w['Cluster'] == 2]; n3 = len(c3)
    cv3 = c3['Cavity_Volume'].mean() if n3 > 0 else 0
    sa3 = c3['Organoids_Surface'].mean() if n3 > 0 else 1
    f['X31'] = cv3 / sa3 if sa3 > 0 else 0
    f['X32'] = n3
    f['X33'] = c3['Scatt_Mean'].mean() if n3 > 0 else 0
    f['X34'] = c3['ShortAxis'].mean() if n3 > 0 else 0
    f['X35'] = c3['Organoids_Surface'].mean() if n3 > 0 else 0
    c4 = w[w['Cluster'] == 3]; n4 = len(c4)
    f['X41'] = c4['LongAxis'].mean() if n4 > 0 else 0
    f['X42'] = c4['Scatt_Mean'].mean() if n4 > 0 else 0
    f['X43'] = c4['ShortAxis'].mean() if n4 > 0 else 0
    return f

rows = []
for _, r in df_all.groupby(['_well_id', '_day']).size().reset_index().iterrows():
    wid, dv = r['_well_id'], r['_day']
    feat = extract_features(df_all, wid, dv)
    feat['_well_id'] = wid
    feat['_day'] = dv
    rows.append(feat)

feat_df = pd.DataFrame(rows)
print(f'\n提取了 {len(feat_df)} 条特征记录')

# ============================================================
# 5. GC专用PCA
# ============================================================
fnames = ['X11', 'X12', 'X13', 'X21', 'X22', 'X23', 'X24',
          'X31', 'X32', 'X33', 'X34', 'X35', 'X41', 'X42', 'X43']

Xf = feat_df[fnames].fillna(0).values
scaler_f = StandardScaler().fit(Xf)
pca = PCA(n_components=4).fit(scaler_f.transform(Xf))
P = pca.transform(scaler_f.transform(Xf))

print(f'PCA解释方差: {pca.explained_variance_ratio_}')
print(f'累计: {pca.explained_variance_ratio_.cumsum()}')

for i in range(4):
    feat_df[f'P{i+1}'] = P[:, i]

# 论文F
w_paper = np.array([0.468, 0.247, 0.165, 0.120])
feat_df['F_paper'] = P @ w_paper

# ============================================================
# 6. 构造well-level Δ数据
# ============================================================
d1 = feat_df[feat_df['_day'] == '0701'][['_well_id', 'F_paper', 'P1', 'P2', 'P3', 'P4']].copy()
d1.columns = ['_well_id', 'F_paper_D1', 'P1_D1', 'P2_D1', 'P3_D1', 'P4_D1']

d2 = feat_df[feat_df['_day'] == '0703'][['_well_id', 'F_paper', 'P1', 'P2', 'P3', 'P4']].copy()
d2.columns = ['_well_id', 'F_paper_D2', 'P1_D2', 'P2_D2', 'P3_D2', 'P4_D2']

m = d1.merge(d2, on='_well_id')
m['Delta_F_paper'] = m['F_paper_D2'] - m['F_paper_D1']
for i in range(1, 5):
    m[f'Delta_P{i}'] = m[f'P{i}_D2'] - m[f'P{i}_D1']
m['ATP'] = m['_well_id'].map(atp_dict)
m['log10_ATP'] = np.log10(m['ATP'])

valid = m['ATP'].notna()
d = m[valid].copy()
print(f'\n有效wells: {len(d)}')

# ============================================================
# 7. 方向A: 重新拟合权重
# ============================================================
print(f'\n{"=" * 80}')
print(f'  方向A: 重新拟合 w1-w4 权重')
print(f'{"=" * 80}')

X_delta = d[['Delta_P1', 'Delta_P2', 'Delta_P3', 'Delta_P4']].values
y_log = d['log10_ATP'].values
y_raw = d['ATP'].values

# 全量拟合
lr_full = LinearRegression().fit(X_delta, y_log)
w_new = lr_full.coef_

print(f'\n  论文权重:   w1={w_paper[0]:.4f}  w2={w_paper[1]:.4f}  w3={w_paper[2]:.4f}  w4={w_paper[3]:.4f}')
print(f'  新拟合权重: w1={w_new[0]:.4f}  w2={w_new[1]:.4f}  w3={w_new[2]:.4f}  w4={w_new[3]:.4f}')

d['F_new'] = X_delta @ w_new

# 对比
r_old, p_old = pearsonr(d['Delta_F_paper'], d['log10_ATP'])
r_new, p_new = pearsonr(d['F_new'], d['log10_ATP'])
sp_old, _ = spearmanr(d['Delta_F_paper'], d['log10_ATP'])
sp_new, _ = spearmanr(d['F_new'], d['log10_ATP'])

print(f'\n  {"方法":35s} {"r(log10ATP)":>14s} {"p-value":>12s} {"Spearman ρ":>12s}')
print(f'  {"-"*75}')
print(f'  {"论文ΔF (固定权重)":35s} {r_old:>14.4f} {p_old:>12.2e} {sp_old:>12.4f}')
print(f'  {"新ΔF (全量拟合)":35s} {r_new:>14.4f} {p_new:>12.2e} {sp_new:>12.4f}')

# ============================================================
# 8. LOO交叉验证
# ============================================================
print(f'\n{"=" * 80}')
print(f'  方向A续: LOO交叉验证 (n={len(d)})')
print(f'{"=" * 80}')

loo = LeaveOneOut()
y_true_loo, y_pred_loo = [], []
for train_idx, test_idx in loo.split(X_delta):
    lr_cv = LinearRegression().fit(X_delta[train_idx], y_log[train_idx])
    y_pred_loo.append(lr_cv.predict(X_delta[test_idx])[0])
    y_true_loo.append(y_log[test_idx[0]])

y_pred_loo = np.array(y_pred_loo)
y_true_loo = np.array(y_true_loo)
r_cv, p_cv = pearsonr(y_pred_loo, y_true_loo)
sp_cv, spp_cv = spearmanr(y_pred_loo, y_true_loo)

print(f'  LOO CV 预测 log10(ATP):  r={r_cv:.4f}  p={p_cv:.2e}  Spearman ρ={sp_cv:.4f}')
print(f'  全量拟合 r={r_new:.4f}  →  LOO r={r_cv:.4f}  →  过拟合gap = {r_new - r_cv:.4f}')

# ============================================================
# 9. 方向B: 各分量单独/组合预测
# ============================================================
print(f'\n{"=" * 80}')
print(f'  方向B: 各分量组合预测 log10(ATP)')
print(f'{"=" * 80}')

# 9a: 单分量
print(f'\n  --- 单分量 ---')
print(f'  {"分量":10s} {"r(Day2)":>10s} {"p":>12s} {"r(Δ)":>10s} {"p":>12s}')
print(f'  {"-"*60}')
for i in range(1, 5):
    r_d2, p_d2 = pearsonr(d[f'P{i}_D2'], d['log10_ATP'])
    r_d, p_d = pearsonr(d[f'Delta_P{i}'], d['log10_ATP'])
    print(f'  {"P"+str(i):10s} {r_d2:>10.4f} {p_d2:>12.2e} {r_d:>10.4f} {p_d:>12.2e}')

# 9b: 等权组合
print(f'\n  --- 等权组合 (ΔP求和) ---')
print(f'  {"组合":20s} {"r":>10s} {"p-value":>12s} {"Spearman ρ":>12s}')
print(f'  {"-"*60}')
best_eq = ('', 0, 0, 0)
for k in range(1, 5):
    for combo in combinations([1, 2, 3, 4], k):
        cols = [f'Delta_P{i}' for i in combo]
        yc = d[cols].sum(axis=1).values
        r_c, p_c = pearsonr(yc, y_log)
        sp_c, _ = spearmanr(yc, y_log)
        name = '+'.join([f'ΔP{i}' for i in combo])
        if abs(r_c) > abs(best_eq[1]):
            best_eq = (name, r_c, p_c, sp_c)
        if abs(r_c) > 0.50:
            print(f'  {name:20s} {r_c:>10.4f} {p_c:>12.2e} {sp_c:>12.4f}')

print(f'\n  ★ 最佳等权组合: {best_eq[0]}  r={best_eq[1]:.4f}  p={best_eq[2]:.2e}')

# 9c: 回归拟合组合
print(f'\n  --- 回归拟合组合 (ΔP → log10ATP) ---')
print(f'  {"组合":20s} {"权重":>35s} {"r":>10s} {"p":>12s} {"ρ":>10s}')
print(f'  {"-"*85}')
best_reg = ('', 0, 0, 0)
for k in range(1, 5):
    for combo in combinations([1, 2, 3, 4], k):
        cols = [f'Delta_P{i}' for i in combo]
        Xc = d[cols].values
        lr = LinearRegression().fit(Xc, y_log)
        yc = lr.predict(Xc)
        r_c, p_c = pearsonr(yc, y_log)
        sp_c, _ = spearmanr(yc, y_log)
        name = '+'.join([f'ΔP{i}' for i in combo])
        w_str = ', '.join([f'{lr.coef_[j]:.3f}' for j in range(k)])
        if abs(r_c) > abs(best_reg[1]):
            best_reg = (name, r_c, p_c, sp_c)
        if abs(r_c) > 0.55:
            print(f'  {name:20s} [{w_str:35s}] {r_c:>10.4f} {p_c:>12.2e} {sp_c:>10.4f}')

print(f'\n  ★ 最佳回归组合: {best_reg[0]}  r={best_reg[1]:.4f}  p={best_reg[2]:.2e}')

# ============================================================
# 10. 最终汇总
# ============================================================
print(f'\n{"=" * 80}')
print(f'  ██ 最终汇总 ██')
print(f'{"=" * 80}')
print(f'')
print(f'  {"方法":45s} {"r(log10ATP)":>14s} {"p-value":>12s}')
print(f'  {"-"*75}')
print(f'  {"论文ΔF (固定权重 0.468/0.247/0.165/0.120)":45s} {r_old:>14.4f} {p_old:>12.2e}')
print(f'  {"4分量全量拟合 (含过拟合)":45s} {r_new:>14.4f} {p_new:>12.2e}')
print(f'  {"4分量LOO CV (去过拟合)":45s} {r_cv:>14.4f} {p_cv:>12.2e}')
print(f'  {"最佳等权组合: " + best_eq[0]:45s} {best_eq[1]:>14.4f} {best_eq[2]:>12.2e}')
print(f'  {"最佳回归组合: " + best_reg[0]:45s} {best_reg[1]:>14.4f} {best_reg[2]:>12.2e}')
print(f'')
print(f'  论文声称: r = 0.887')
print(f'  GC实测最佳: r = {max(abs(r_cv), abs(best_eq[1]), abs(best_reg[1])):.4f}')

best_r = max(abs(r_cv), abs(best_eq[1]), abs(best_reg[1]))
gap = abs(best_r - 0.887)
print(f'  差距: {gap:.4f}')
print(f'')
if gap < 0.10:
    print(f'  ✅ 论文声称在GC上得到验证！')
elif gap < 0.20:
    print(f'  ⚠️ 接近但未完全达到论文水平')
elif gap < 0.30:
    print(f'  ❌ 偏差较大，论文框架对GC的预测上限约r={best_r:.2f}')
else:
    print(f'  ❌ 论文框架在GC上预测能力有限')

# 保存
output = r'D:\Desktop\music\measure\GC_step1_results.xlsx'
d.to_excel(output, index=False)
print(f'\n结果已保存: {output}')
print('DONE')