"""
GC论文风格6图生成
对照ICC的6张图，适配GC数据
"""
import os, sys, glob, warnings
import numpy as np, pandas as pd
from scipy.stats import pearsonr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from matplotlib.patches import Patch
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.size': 10, 'figure.dpi': 150, 'savefig.dpi': 300,
    'axes.labelsize': 11, 'axes.titlesize': 12, 'legend.fontsize': 9,
})

OUTPUT_DIR = r'D:\Desktop\music\measure\reports\figures\GC_validation'
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLUSTER_COLORS = {0: '#E74C3C', 1: '#F39C12', 2: '#27AE60', 3: '#3498DB'}
CLUSTER_NAMES = {0: '大囊状(H)', 1: '大实心(H)', 2: '小实心(I)', 3: '极小(D)'}

# ============================================================
# 1. 加载数据 + KMeans + PCA（同前）
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

GC_BASE = r'D:\Desktop\music\measure\Data\FXN_2023_new（GC）'

def load_gc(day_folder):
    measure_dir = os.path.join(GC_BASE, day_folder, 'measure_excel')
    scatt_dir = os.path.join(GC_BASE, day_folder, 'scatt')
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

df_d1 = load_gc('FXN_20230701')
df_d2 = load_gc('FXN_20230703')

cfeats = ['Organoids_Volume','Organoids_Volume_Fill','Organoids_Surface',
          'Cavity_Volume','CavityNum','LongAxis','ShortAxis',
          'Wall_Thickness','Sphericity','Scatt_Mean','Scatt_STD']

df_comb = pd.concat([df_d1[cfeats], df_d2[cfeats]], ignore_index=True)
scaler_k = StandardScaler().fit(df_comb)
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10).fit(scaler_k.transform(df_comb))
labels = kmeans.labels_

df_d1 = df_d1.copy(); df_d2 = df_d2.copy()
df_d1['Cluster'] = labels[:len(df_d1)]; df_d2['Cluster'] = labels[len(df_d1):]
cv_list = [(c, df_comb.iloc[labels==c]['Organoids_Volume'].mean()) for c in range(4)]
cv_list.sort(key=lambda x:x[1], reverse=True)
cmap = {old:new for new,(old,_) in enumerate(cv_list)}
df_d1['Cluster'] = df_d1['Cluster'].map(cmap); df_d2['Cluster'] = df_d2['Cluster'].map(cmap)
df_all = pd.concat([df_d1, df_d2])

def extract_features(df, wid, dv):
    w = df[(df['_well_id']==wid)&(df['_day']==dv)]
    f = {}
    c1=w[w['Cluster']==0]; n1=len(c1)
    f['X11']=c1['Cavity_Volume'].mean() if n1>0 else 0
    f['X12']=c1['Organoids_Surface'].mean() if n1>0 else 0
    f['X13']=c1['Sphericity'].mean() if n1>0 else 0
    c2=w[w['Cluster']==1]; n2=len(c2)
    f['X21']=n2; f['X22']=c2['ShortAxis'].mean() if n2>0 else 0
    f['X23']=c2['Organoids_Surface'].mean() if n2>0 else 0
    f['X24']=c2['Organoids_Volume_Fill'].mean() if n2>0 else 0
    c3=w[w['Cluster']==2]; n3=len(c3)
    cv3=c3['Cavity_Volume'].mean() if n3>0 else 0
    sa3=c3['Organoids_Surface'].mean() if n3>0 else 1
    f['X31']=cv3/sa3 if sa3>0 else 0; f['X32']=n3
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

fnames = ['X11','X12','X13','X21','X22','X23','X24',
          'X31','X32','X33','X34','X35','X41','X42','X43']
Xf = feat_df[fnames].fillna(0).values
scaler_f = StandardScaler().fit(Xf)
pca = PCA(n_components=4).fit(scaler_f.transform(Xf))
P = pca.transform(scaler_f.transform(Xf))
for i in range(4): feat_df[f'P{i+1}'] = P[:,i]

d1=feat_df[feat_df['_day']=='0701'][['_well_id','P1','P2','P3','P4']].copy()
d1.columns=['_well_id','P1_D1','P2_D1','P3_D1','P4_D1']
d2=feat_df[feat_df['_day']=='0703'][['_well_id','P1','P2','P3','P4']].copy()
d2.columns=['_well_id','P1_D2','P2_D2','P3_D2','P4_D2']
m = d1.merge(d2, on='_well_id')
for i in range(1,5): m[f'Delta_P{i}'] = m[f'P{i}_D2'] - m[f'P{i}_D1']
m['ATP'] = m['_well_id'].map(atp_dict)
m['log10_ATP'] = np.log10(m['ATP'])

X_delta = m[['Delta_P1','Delta_P2','Delta_P3','Delta_P4']].values
y_log = m['log10_ATP'].values
lr = LinearRegression().fit(X_delta, y_log)
m['F'] = X_delta @ lr.coef_
m['ATP_group'] = pd.qcut(m['ATP'], q=3, labels=['Low', 'Med', 'High'])
ATP_GROUP_COLORS = {'Low': '#3498DB', 'Med': '#E67E22', 'High': '#C0392B'}

# ============================================================
# fig1: 类器官聚类验证
# ============================================================
print('生成 fig1: 聚类验证...')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

df_valid = df_all.dropna(subset=cfeats)
df_sample = df_valid.sample(n=min(5000, len(df_valid)), random_state=42)
X_s = StandardScaler().fit_transform(df_sample[cfeats].values)
X2 = PCA(n_components=2, random_state=42).fit_transform(X_s)
for c in sorted(CLUSTER_COLORS.keys()):
    mask = df_sample['Cluster'].values == c
    ax1.scatter(X2[mask,0], X2[mask,1], c=CLUSTER_COLORS[c],
                label=CLUSTER_NAMES[c], s=2, alpha=0.5, rasterized=True)
ax1.legend(markerscale=8, frameon=True, fancybox=True, loc='upper right')
ax1.set_xlabel('PC1'); ax1.set_ylabel('PC2')
ax1.set_title('GC Organoid PCA by Phenotype', fontweight='bold')

prof = df_valid.groupby('Cluster')[cfeats].mean()
pz = (prof - prof.mean()) / prof.std()
short_names = [n.replace('Organoids_','').replace('_',' ') for n in cfeats]
im = ax2.imshow(pz.values, cmap='RdBu_r', aspect='auto', vmin=-2, vmax=2)
ax2.set_xticks(range(len(short_names)))
ax2.set_xticklabels(short_names, rotation=45, ha='right', fontsize=7)
ax2.set_yticks(range(len(prof)))
ax2.set_yticklabels([CLUSTER_NAMES[i] for i in prof.index], fontsize=10)
plt.colorbar(im, ax=ax2, shrink=0.8).set_label('Z-score')
ax2.set_title('Cluster Profiles', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig1_clustering.png'), dpi=300, bbox_inches='tight')
plt.close()
print('  OK fig1_clustering.png')

# ============================================================
# fig2: 核心结果 - PCA + 权重 + F vs ATP
# ============================================================
print('生成 fig2: 核心结果...')
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

# A: Scree plot
ax = axes[0]
n = len(pca.explained_variance_ratio_)
x = np.arange(1, n+1)
ax.bar(x, pca.explained_variance_ratio_*100, color='#3498DB', alpha=0.7, edgecolor='white')
ax.plot(x, np.cumsum(pca.explained_variance_ratio_)*100, 'o-', color='#E74C3C', lw=2, ms=6)
for i, (v, cv) in enumerate(zip(pca.explained_variance_ratio_*100, np.cumsum(pca.explained_variance_ratio_)*100)):
    ax.text(x[i], v+1, f'{v:.1f}%', ha='center', fontsize=8)
ax.set_xticks(x); ax.set_xlabel('Principal Component')
ax.set_ylabel('Variance Explained (%)')
ax.set_title('GC PCA Scree Plot', fontweight='bold')

# B: Feature weights (GC fitted)
ax = axes[1]
w_names = ['ΔP1', 'ΔP2', 'ΔP3', 'ΔP4']
colors_w = ['#E74C3C' if c>0 else '#3498DB' for c in lr.coef_]
ax.barh(range(4), lr.coef_[::-1], color=colors_w[::-1], alpha=0.8, height=0.6)
ax.set_yticks(range(4)); ax.set_yticklabels(w_names[::-1])
ax.set_xlabel('Weight in F (GC fitted)')
ax.axvline(0, color='black', lw=0.8)
ax.set_title('F = Σ wᵢ·ΔPᵢ  Weights', fontweight='bold')

# C: F vs ATP scatter
ax = axes[2]
r_f, p_f = pearsonr(m['F'], m['log10_ATP'])
for g, c in ATP_GROUP_COLORS.items():
    mask = m['ATP_group'] == g
    ax.scatter(m.loc[mask,'F'], m.loc[mask,'log10_ATP'],
               c=c, label=g, s=60, alpha=0.8, edgecolors='white', linewidth=0.5)
for _, row in m.iterrows():
    ax.annotate(row['_well_id'], (row['F'], row['log10_ATP']),
                fontsize=6, alpha=0.6, ha='center', va='bottom',
                xytext=(0,3), textcoords='offset points')
ax.set_xlabel('F (GC Composite Score)')
ax.set_ylabel('log10(ATP)')
ax.legend(fontsize=8, title='ATP Level')
ax.set_title(f'r = {r_f:.3f}, p = {p_f:.2e}', fontweight='bold', fontsize=11)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig2_score_atp.png'), dpi=300, bbox_inches='tight')
plt.close()
print('  OK fig2_score_atp.png')

# ============================================================
# fig3: 聚类组成 vs ATP分组 (替代剂量反应)
# ============================================================
print('生成 fig3: 聚类组成分析...')
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

# A: 各ATP组的聚类比例堆叠柱状图
ax = axes[0]
wd2 = df_all[df_all['_day']=='0703'].copy()
wd2['ATP_group'] = wd2['_well_id'].map(
    lambda wid: m.set_index('_well_id').loc[wid, 'ATP_group'] if wid in m['_well_id'].values else 'Low')
groups = ['Low', 'Med', 'High']
comp = wd2.groupby('ATP_group')['Cluster'].value_counts().unstack(fill_value=0)
comp = comp.reindex(groups, fill_value=0)
comp_pct = comp.div(comp.sum(axis=1), axis=0) * 100
bottom = np.zeros(3)
width = 0.6
for c in range(4):
    heights = [comp_pct.loc[g, c] if c in comp_pct.columns else 0 for g in groups]
    ax.bar(groups, heights, width=width, bottom=bottom,
           color=CLUSTER_COLORS[c], label=CLUSTER_NAMES[c],
           alpha=0.85, edgecolor='white', linewidth=0.5)
    bottom += np.array(heights)
ax.set_ylabel('Fraction (%)'); ax.set_ylim(0, 105)
ax.legend(fontsize=7, loc='upper right')
ax.set_title('Phenotype Composition by ATP Level', fontweight='bold')

# B: 大囊状(H)比例 vs ATP
ax = axes[1]
wd2_pct = wd2.groupby(['_well_id', 'ATP_group'])['Cluster'].value_counts(normalize=True).unstack(fill_value=0)
if 0 in wd2_pct.columns:
    h_frac = wd2_pct[0] * 100
    h_well = h_frac.groupby('_well_id').first()
    common = h_well.index.intersection(m['_well_id'])
    r_h, p_h = pearsonr(h_well[common], m.set_index('_well_id').loc[common, 'log10_ATP'])
    for g, c in ATP_GROUP_COLORS.items():
        mask = m['ATP_group'] == g
        wells_g = m.loc[mask, '_well_id']
        idx = h_well.index.intersection(wells_g)
        if len(idx) > 0:
            ax.scatter(h_well[idx], m.set_index('_well_id').loc[idx, 'log10_ATP'],
                      c=c, s=40, alpha=0.7, label=g)
    ax.set_xlabel('Large Cystic (H) Fraction (%)')
    ax.set_ylabel('log10(ATP)')
    ax.legend(fontsize=7)
    ax.set_title(f'Healthy% vs ATP  r={r_h:.3f}', fontweight='bold')

# C: 极小(D)比例 vs ATP
ax = axes[2]
if 3 in wd2_pct.columns:
    d_frac = wd2_pct[3] * 100
    d_well = d_frac.groupby('_well_id').first()
    common = d_well.index.intersection(m['_well_id'])
    r_d, p_d = pearsonr(d_well[common], m.set_index('_well_id').loc[common, 'log10_ATP'])
    for g, c in ATP_GROUP_COLORS.items():
        mask = m['ATP_group'] == g
        wells_g = m.loc[mask, '_well_id']
        idx = d_well.index.intersection(wells_g)
        if len(idx) > 0:
            ax.scatter(d_well[idx], m.set_index('_well_id').loc[idx, 'log10_ATP'],
                      c=c, s=40, alpha=0.7, label=g)
    ax.set_xlabel('Small Dense (D) Fraction (%)')
    ax.set_ylabel('log10(ATP)')
    ax.legend(fontsize=7)
    ax.set_title(f'Dead% vs ATP  r={r_d:.3f}', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig3_composition.png'), dpi=300, bbox_inches='tight')
plt.close()
print('  OK fig3_composition.png')

# ============================================================
# fig4: 各孔得分 & ATP排名
# ============================================================
print('生成 fig4: 得分排名...')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

ms = m.sort_values('F')
ids = ms['_well_id'].values
scores = ms['F'].values
colors = [ATP_GROUP_COLORS[ms.iloc[i]['ATP_group']] for i in range(len(ms))]
ax1.barh(range(len(ids)), scores, color=colors, alpha=0.8, height=0.7, edgecolor='white')
ax1.set_yticks(range(len(ids))); ax1.set_yticklabels(ids, fontsize=8)
ax1.set_xlabel('F (GC Composite Score)')
ax1.axvline(0, color='gray', ls='--', alpha=0.5)
legend_p = [Patch(fc=c, label=l) for l,c in ATP_GROUP_COLORS.items()]
ax1.legend(handles=legend_p, fontsize=8, title='ATP Level')
ax1.set_title('F Score by Well', fontweight='bold')

ma = m.sort_values('ATP')
ids2 = ma['_well_id'].values
atp_vals = ma['ATP'].values / 1e6
colors2 = [ATP_GROUP_COLORS[ma.iloc[i]['ATP_group']] for i in range(len(ma))]
ax2.barh(range(len(ids2)), atp_vals, color=colors2, alpha=0.8, height=0.7, edgecolor='white')
ax2.set_yticks(range(len(ids2))); ax2.set_yticklabels(ids2, fontsize=8)
ax2.set_xlabel('ATP (×10⁶)')
ax2.legend(handles=legend_p, fontsize=8, title='ATP Level')
ax2.set_title('ATP by Well', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig4_ranking.png'), dpi=300, bbox_inches='tight')
plt.close()
print('  OK fig4_ranking.png')

# ============================================================
# fig5: 特征相关性矩阵
# ============================================================
print('生成 fig5: 相关性矩阵...')
feat_day = feat_df[feat_df['_day']=='0703'].copy()
feat_day['_well_id'] = feat_day['_well_id'].astype(str)
corr_cols = [f'P{i}' for i in range(1,5)] + ['F_paper']
feat_day['F_paper'] = feat_day[['P1','P2','P3','P4']].values @ np.array([0.468,0.247,0.165,0.120])
corr_df = feat_day.set_index('_well_id')[corr_cols].copy()
corr_df['ATP'] = corr_df.index.map(lambda w: np.log10(atp_dict.get(w, np.nan)))
corr_df = corr_df.dropna()
corr = corr_df.corr()

fig, ax = plt.subplots(figsize=(9, 8))
im = ax.imshow(corr.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
labels = corr.columns.tolist()
ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=10)
for i in range(len(labels)):
    for j in range(len(labels)):
        val = corr.values[i,j]
        color = 'white' if abs(val) > 0.6 else 'black'
        ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=9, color=color)
plt.colorbar(im, ax=ax, shrink=0.8).set_label('Pearson r')
ax.set_title('GC Feature Correlation Matrix (Day 7)', fontweight='bold', fontsize=13)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig5_correlation.png'), dpi=300, bbox_inches='tight')
plt.close()
print('  OK fig5_correlation.png')

# ============================================================
# fig6: ΔF vs 各分量散点图
# ============================================================
print('生成 fig6: 分量散点...')
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()

for idx, i in enumerate(range(1, 5)):
    ax = axes[idx]
    r_p, p_p = pearsonr(m[f'Delta_P{i}'], m['log10_ATP'])
    for g, c in ATP_GROUP_COLORS.items():
        mask = m['ATP_group'] == g
        ax.scatter(m.loc[mask, f'Delta_P{i}'], m.loc[mask, 'log10_ATP'],
                  c=c, s=40, alpha=0.7, label=g, edgecolors='white', linewidth=0.3)
    ax.set_xlabel(f'ΔP{i}')
    ax.set_ylabel('log10(ATP)')
    ax.legend(fontsize=7, loc='lower right')
    ax.set_title(f'ΔP{i} vs ATP  r={r_p:.3f}  p={p_p:.2e}', fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig6_components.png'), dpi=300, bbox_inches='tight')
plt.close()
print('  OK fig6_components.png')

# ============================================================
# 汇总
# ============================================================
print(f'\n{"=" * 60}')
print(f'  6张图已生成到: {OUTPUT_DIR}')
print(f'{"=" * 60}')
for fn in sorted(os.listdir(OUTPUT_DIR)):
    print(f'  {fn}')
print('\nDONE')