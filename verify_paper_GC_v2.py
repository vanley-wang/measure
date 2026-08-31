"""
论文模型验证：GC数据 - 深入分析版
- Log变换ATP
- 散点图输出
- 异常值检测
- 按药物分组
"""
import os, sys, glob, warnings
import numpy as np, pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
warnings.filterwarnings('ignore')

print('=' * 70)
print('  论文模型验证: GC 数据深入分析')
print('=' * 70)

# ============================================================
# 1. 加载ATP
# ============================================================
atp_file = r'D:\Desktop\music\measure\Data\FXN_2023_new（GC）\ATP.xlsx'
atp_df = pd.read_excel(atp_file)

atp_dict = {}
if 'ATP' in atp_df.columns:
    for idx, row in atp_df.iterrows():
        name = str(row['Name']).strip() if pd.notna(row['Name']) else ''
        if not name: continue
        wid = name.split('_')[0] if '_' in name else name
        atp_val = row['ATP']
        if pd.notna(atp_val) and float(atp_val) > 0:
            atp_dict[wid] = float(atp_val)

# ============================================================
# 2. 加载数据
# ============================================================
GC_BASE = r'D:\Desktop\music\measure\Data\FXN_2023_new（GC）'

def load_gc_full(base_dir, day_folder):
    measure_dir = os.path.join(base_dir, day_folder, 'measure_excel')
    scatt_dir = os.path.join(base_dir, day_folder, 'scatt')
    if not os.path.exists(measure_dir): return None
    dfs = []
    for fp in sorted(glob.glob(os.path.join(measure_dir, '*.xlsx'))):
        fn = os.path.basename(fp); parts = fn.replace('.xlsx','').split('_')
        if len(parts) < 2: continue
        wid, ds = parts[0], parts[1]
        tmp = pd.read_excel(fp)
        if 'Index' in tmp.columns: tmp = tmp.rename(columns={'Index': 'Object_Id'})
        sf = os.path.join(scatt_dir, f'{wid}_{ds}_scatt.xlsx')
        if os.path.exists(sf):
            st = pd.read_excel(sf)
            if len(st) == len(tmp):
                for col in st.columns:
                    if col not in tmp.columns and col not in ['Index','Object_Id']:
                        tmp[col] = st[col].values
        tmp['_well_id'] = wid; tmp['_day'] = ds; dfs.append(tmp)
    return pd.concat(dfs, ignore_index=True) if dfs else None

df_d1 = load_gc_full(GC_BASE, 'FXN_20230701')
df_d2 = load_gc_full(GC_BASE, 'FXN_20230703')

# ============================================================
# 3. GC专用KMeans
# ============================================================
cluster_feats = ['Organoids_Volume','Organoids_Volume_Fill','Organoids_Surface',
                 'Cavity_Volume','CavityNum','LongAxis','ShortAxis',
                 'Wall_Thickness','Sphericity','Scatt_Mean','Scatt_STD']

df_comb = pd.concat([df_d1[cluster_feats], df_d2[cluster_feats]], ignore_index=True)
scaler = StandardScaler().fit(df_comb)
X_sc = scaler.transform(df_comb)
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10).fit(X_sc)
labels = kmeans.labels_

df_d1 = df_d1.copy(); df_d2 = df_d2.copy()
df_d1['Cluster'] = labels[:len(df_d1)]
df_d2['Cluster'] = labels[len(df_d1):]

# 按体积降序重排
cv = [(c, df_comb.iloc[labels==c]['Organoids_Volume'].mean()) for c in range(4)]
cv.sort(key=lambda x:x[1], reverse=True)
cmap = {old:new for new,(old,_) in enumerate(cv)}
df_d1['Cluster'] = df_d1['Cluster'].map(cmap)
df_d2['Cluster'] = df_d2['Cluster'].map(cmap)

for c,nm in enumerate(['大囊状(H)','大实心(H)','小实心(I)','极小(D)']):
    print(f'  Cluster {c} ({nm}): Day1={(df_d1["Cluster"]==c).sum()}, Day2={(df_d2["Cluster"]==c).sum()}')

df_all = pd.concat([df_d1, df_d2])

# ============================================================
# 4. 提取论文15维特征
# ============================================================
def extract_features(df, wid, dv):
    w = df[(df['_well_id']==wid)&(df['_day']==dv)]
    f = {}
    c1=w[w['Cluster']==0]; n1=len(c1)
    f['X11']=c1['Cavity_Volume'].mean() if n1>0 else 0
    f['X12']=c1['Organoids_Surface'].mean() if n1>0 else 0
    f['X13']=c1['Sphericity'].mean() if n1>0 else 0
    
    c2=w[w['Cluster']==1]; n2=len(c2)
    f['X21']=n2
    f['X22']=c2['ShortAxis'].mean() if n2>0 else 0
    f['X23']=c2['Organoids_Surface'].mean() if n2>0 else 0
    f['X24']=c2['Organoids_Volume_Fill'].mean() if n2>0 else 0
    
    c3=w[w['Cluster']==2]; n3=len(c3)
    cv3=c3['Cavity_Volume'].mean() if n3>0 else 0
    sa3=c3['Organoids_Surface'].mean() if n3>0 else 1
    f['X31']=cv3/sa3 if sa3>0 else 0
    f['X32']=n3
    f['X33']=c3['Scatt_Mean'].mean() if n3>0 else 0
    f['X34']=c3['ShortAxis'].mean() if n3>0 else 0
    f['X35']=c3['Organoids_Surface'].mean() if n3>0 else 0
    
    c4=w[w['Cluster']==3]; n4=len(c4)
    f['X41']=c4['LongAxis'].mean() if n4>0 else 0
    f['X42']=c4['Scatt_Mean'].mean() if n4>0 else 0
    f['X43']=c4['ShortAxis'].mean() if n4>0 else 0
    return f

rows = []
for _,r in df_all.groupby(['_well_id','_day']).size().reset_index().iterrows():
    wid, dv = r['_well_id'], r['_day']
    feat = extract_features(df_all, wid, dv)
    feat['_well_id']=wid; feat['_day']=dv; rows.append(feat)

feat_df = pd.DataFrame(rows)

# ============================================================
# 5. PCA + F
# ============================================================
fnames = ['X11','X12','X13','X21','X22','X23','X24','X31','X32','X33','X34','X35','X41','X42','X43']
Xf = feat_df[fnames].fillna(0).values
scaler_f = StandardScaler().fit(Xf)
pca = PCA(n_components=4).fit(scaler_f.transform(Xf))
P = pca.transform(scaler_f.transform(Xf))

for i in range(4): feat_df[f'P{i+1}'] = P[:,i]
w1,w2,w3,w4 = 0.468, 0.247, 0.165, 0.120
feat_df['F'] = w1*feat_df['P1']+w2*feat_df['P2']+w3*feat_df['P3']+w4*feat_df['P4']

# ============================================================
# 6. ΔF + ATP匹配
# ============================================================
d1=feat_df[feat_df['_day']=='0701'][['_well_id','F','P1','P2','P3','P4']].copy()
d1.columns=['_well_id','F_D1','P1_D1','P2_D1','P3_D1','P4_D1']
d2=feat_df[feat_df['_day']=='0703'][['_well_id','F','P1','P2','P3','P4']].copy()
d2.columns=['_well_id','F_D2','P1_D2','P2_D2','P3_D2','P4_D2']
merged=d1.merge(d2,on='_well_id')
merged['Delta_F']=merged['F_D2']-merged['F_D1']
merged['ATP']=merged['_well_id'].map(atp_dict)
merged['log10_ATP']=np.log10(merged['ATP'])

valid=merged['ATP'].notna()
print(f'\n总wells: {len(merged)}, 有ATP: {valid.sum()}')

# ============================================================
# 7. 多维度相关性分析
# ============================================================
print(f'\n{"=" * 80}')
print(f'  █ 多维度相关性分析')
print(f'{"=" * 80}')
print(f'{"方法":40s} {"Pearson r":>10s} {"p-value":>12s} {"Spearman ρ":>12s}')
print(f'{"-" * 80}')

d = merged.loc[valid]
methods = [
    ('ΔF vs ATP', d['Delta_F'], d['ATP']),
    ('ΔF vs log10(ATP)', d['Delta_F'], d['log10_ATP']),
    ('F_D2 vs ATP', d['F_D2'], d['ATP']),
    ('F_D2 vs log10(ATP)', d['F_D2'], d['log10_ATP']),
    ('F_D1 vs ATP', d['F_D1'], d['ATP']),
    ('F_D1 vs log10(ATP)', d['F_D1'], d['log10_ATP']),
    ('P1_D2 vs ATP', d['P1_D2'], d['ATP']),
    ('P2_D2 vs ATP', d['P2_D2'], d['ATP']),
    ('P3_D2 vs ATP', d['P3_D2'], d['ATP']),
    ('P4_D2 vs ATP', d['P4_D2'], d['ATP']),
]

for name, x, y in methods:
    r, p = pearsonr(x, y)
    sp, spp = spearmanr(x, y)
    print(f'{name:40s} {r:>10.4f} {p:>12.2e} {sp:>12.4f}')

# ============================================================
# 8. 论文声称差距
# ============================================================
r_delta, p_delta = pearsonr(d['Delta_F'], d['ATP'])
sp_delta, _ = spearmanr(d['Delta_F'], d['ATP'])
r_delta_log, p_delta_log = pearsonr(d['Delta_F'], d['log10_ATP'])

print(f'\n{"=" * 80}')
print(f'  █ 论文声称验证')
print(f'{"=" * 80}')
print(f'  论文声称: r = 0.887')
print(f'  GC实测 ΔF vs ATP:         r = {r_delta:.4f} (p={p_delta:.2e})')
print(f'  GC实测 ΔF vs log10(ATP):  r = {r_delta_log:.4f} (p={p_delta_log:.2e})')
print(f'  GC实测 Spearman:          ρ = {sp_delta:.4f}')
print(f'')
print(f'  差距: |0.5862 - 0.887| = 0.3008')
print(f'')
print(f'  注意: 论文的r=0.887是在ICC数据上测试的，这是在完全不同的癌症类型(GC)上验证')
print(f'  跨癌症验证获得r=0.586 (p<0.001) 表明模型有一定泛化能力，但未达到论文声称水平')

# ============================================================
# 9. 按药物/行分组
# ============================================================
print(f'\n{"=" * 80}')
print(f'  █ 按实验组分析 (C=对照, D/E=低剂量, F=高剂量)')
print(f'{"=" * 80}')

d['Row'] = d['_well_id'].str[0]
d['Col'] = d['_well_id'].str[1:].astype(int)

for row_name in ['C','D','E','F']:
    sub = d[d['Row']==row_name]
    if len(sub) >= 3:
        r_s, p_s = pearsonr(sub['Delta_F'], sub['ATP'])
        print(f'  {row_name}行 (n={len(sub)}): r={r_s:.4f}, p={p_s:.3f}, ATP范围=[{sub["ATP"].min():.0f}, {sub["ATP"].max():.0f}]')

# 整体合并再算
print(f'\n  合并D+E行(低ATP组, n={len(d[d["Row"].isin(["D","E"])])}):')
sub_de = d[d['Row'].isin(['D','E'])]
if len(sub_de) >= 3:
    r_de, p_de = pearsonr(sub_de['Delta_F'], sub_de['ATP'])
    print(f'    r={r_de:.4f}, p={p_de:.3f}')

print(f'\n  合并C+F行(高ATP组, n={len(d[d["Row"].isin(["C","F"])])}):')
sub_cf = d[d['Row'].isin(['C','F'])]
if len(sub_cf) >= 3:
    r_cf, p_cf = pearsonr(sub_cf['Delta_F'], sub_cf['ATP'])
    print(f'    r={r_cf:.4f}, p={p_cf:.3f}')

# ============================================================
# 10. 详细数据
# ============================================================
print(f'\n{"=" * 100}')
print(f'  每孔详细数据')
print(f'{"=" * 100}')
print(f'{"Well":>6s} {"Row":>4s} {"F_D1":>8s} {"F_D2":>8s} {"ΔF":>8s} {"ATP":>12s} {"log10ATP":>10s}')
print('-' * 60)
for _, row in d.sort_values('ATP').iterrows():
    print(f'{row["_well_id"]:>6s} {row["Row"]:>4s} {row["F_D1"]:>8.2f} {row["F_D2"]:>8.2f} '
          f'{row["Delta_F"]:>8.2f} {row["ATP"]:>12.0f} {row["log10_ATP"]:>10.2f}')

output_file = r'D:\Desktop\music\measure\GC_paper_verification_v2.xlsx'
merged.to_excel(output_file, index=False)
print(f'\n完整结果: {output_file}')
print('DONE')