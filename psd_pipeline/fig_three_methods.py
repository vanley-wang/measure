# ============================================================
# 三方法对比：Score vs ATP 散点 + 特征Venn + PC1 Loading
# 约 15 分钟（3 × 50000 次搜索）
# ============================================================
import os, sys, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

warnings.filterwarnings('ignore')
BASE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(BASE, 'figures_three_methods')
os.makedirs(FIG_DIR, exist_ok=True)

ATP_DB = {
    'B10':601300,'B11':11180000,'B2':5391000,'B3':6538000,'B4':7103000,
    'B5':511900,'B6':404500,'B7':403900,'B8':312700,'B9':140300,
    'C10':211800,'C11':13930000,'C2':6336000,'C3':8336000,'C4':6800000,
    'C5':330900,'C6':238900,'C7':682100,'C8':211300,'C9':465900,
    'D11':11240000,'E11':14700000,'F10':21910000,'F11':11180000,
    'F2':18240000,'F3':14110000,'F4':13740000,'F5':17250000,
    'F6':20320000,'F7':20000000,'F8':17170000,'F9':15830000,
}
CF = ['Organoids_Volume','Organoids_Volume_Fill','Organoids_Surface',
      'Cavity_Volume','CavityNum','LongAxis','ShortAxis',
      'Wall_Thickness','Sphericity','Scatt_Mean','Scatt_STD']
N_MERGED = 4; N_SEARCH = 50000; N_PC = 4; RANDOM_SEED = 42

FEATURE_LONG = {
    'Organoids_Volume':'Volume_Avg','Organoids_Volume_Fill':'Volume_Fill_Avg',
    'Organoids_Surface':'Surface_Avg','Cavity_Volume':'Cavity_Volume_All',
    'CavityNum':'CavityNum_Avg','LongAxis':'Long_Axis_Avg',
    'ShortAxis':'Short_Axis_Avg','Wall_Thickness':'Cyst_Thick_Avg',
    'Sphericity':'Sphericity_Avg','Scatt_Mean':'Scatt_Mean_Avg',
    'Scatt_STD':'Scatt_STD_Avg',
}

# ---------------------------------------------------------------
def k6_to_k4(centers):
    dist = cdist(centers, centers); np.fill_diagonal(dist, np.inf)
    pairs = []
    for _ in range(2):
        i, j = np.unravel_index(np.argmin(dist), dist.shape)
        pairs.append((int(i), int(j)))
        dist[i,:]=np.inf; dist[:,i]=np.inf; dist[j,:]=np.inf; dist[:,j]=np.inf
    merge_map = {}; remaining = set(range(6))
    for a,b in pairs: merge_map[a]=a; merge_map[b]=a; remaining-={a,b}
    for r in sorted(remaining): merge_map[r]=r
    u = sorted(set(merge_map.values()))
    numeric_map = {k: u.index(merge_map[k]) for k in range(6)}
    return numeric_map, pairs

def aggregate_well(df, labels):
    rec = {}
    for cl in range(N_MERGED):
        mask = labels == cl; n = int(mask.sum())
        rec[f'Number_{cl+1}'] = n
        sub = df.loc[mask] if n>0 else df.iloc[:0]
        for src, alias in FEATURE_LONG.items():
            if src not in df.columns: continue
            if src == 'Cavity_Volume':
                rec[f'{alias}_{cl+1}'] = float(sub[src].sum())
            else:
                rec[f'{alias}_{cl+1}'] = float(sub[src].mean()) if n>0 else 0.0
    return rec

def search_features(X, y, feat_cols):
    np.random.seed(RANDOM_SEED)
    best = {'r': 0.0}
    for it in range(N_SEARCH):
        ns = np.random.randint(max(4,len(feat_cols)//3), len(feat_cols)+1)
        si = sorted(np.random.choice(len(feat_cols), ns, replace=False))
        Xs = X[:, si]; ss = StandardScaler(); Xss = ss.fit_transform(Xs)
        pc = PCA(n_components=N_PC)
        try: Xp = pc.fit_transform(Xss)
        except: continue
        wr = pc.explained_variance_ratio_[:N_PC]; wr = wr/wr.sum()
        r, p = pearsonr(Xp @ wr, y)
        if abs(r) > best['r']:
            best = {'r':abs(r),'sel_idx':si,'pca':pc,'scaler':ss,'weights':wr,'p':p}
    return best

def run_one_pipeline(name, d3_dir, d5_dir):
    print(f"\n{'='*60}\n  {name}\n{'='*60}")
    d3_all = pd.concat([pd.read_excel(os.path.join(d3_dir,f))[CF].fillna(0)
                         for f in sorted(os.listdir(d3_dir)) if f.endswith('.xlsx')], ignore_index=True)
    X_all = d3_all.values
    scaler = StandardScaler(); X_std = scaler.fit_transform(X_all)
    km = KMeans(n_clusters=6, random_state=RANDOM_SEED, n_init=10)
    k6 = km.fit_predict(X_std)
    nmap, pairs = k6_to_k4(km.cluster_centers_)
    sizes = dict(zip(*np.unique(k6, return_counts=True)))
    print(f"  K6 sizes: {sizes}, merge: {pairs}")

    d3f = {f.replace('.xlsx','').split('_')[0]:f for f in os.listdir(d3_dir) if f.endswith('.xlsx')}
    d5f = {f.replace('.xlsx','').split('_')[0]:f for f in os.listdir(d5_dir) if f.endswith('.xlsx')}
    common = sorted(set(d3f)&set(d5f))

    rows, wids, fnames = [], [], None
    for wid in common:
        d3 = pd.read_excel(os.path.join(d3_dir,d3f[wid]))
        d5 = pd.read_excel(os.path.join(d5_dir,d5f[wid]))
        X3 = d3[CF].fillna(0).values; X5 = d5[CF].fillna(0).values
        c3 = np.array([nmap[l] for l in km.predict(scaler.transform(X3))])
        c5 = np.array([nmap[l] for l in km.predict(scaler.transform(X5))])
        f3 = aggregate_well(d3,c3); f5 = aggregate_well(d5,c5)
        if fnames is None: fnames = sorted(f3.keys())
        rows.append([f5.get(k,0)-f3.get(k,0) for k in fnames])
        wids.append(wid)

    X = np.array(rows); y = np.array([ATP_DB[w] for w in wids])
    valid = [i for i,c in enumerate(fnames) if np.std(X[:,i])>1e-12]
    fc = [fnames[i] for i in valid]; X = X[:, valid]
    print(f"  Delta: {X.shape}, wells: {len(y)}")

    best = search_features(X, y, fc)
    bf = [fc[i] for i in best['sel_idx']]
    Xb = X[:, best['sel_idx']]; Xb_std = best['scaler'].transform(Xb)
    pcs = best['pca'].transform(Xb_std); score = pcs @ best['weights']
    r, pv = pearsonr(score, y); rho, _ = spearmanr(score, y)
    if r < 0:  # PCA方向修正
        score = -score; r = -r; rho = -rho; best['weights'] = -best['weights']
    cum = best['pca'].explained_variance_ratio_[:N_PC].sum()
    print(f"  |r|={abs(r):.4f}, rho={rho:.4f}, feats={len(bf)}, PCcum={cum:.1%}")
    return {'score':score,'atp':y,'r':r,'p':pv,'rho':rho,'features':bf,
            'pca':best['pca'],'weights':best['weights'],'n_feat':len(bf),
            'cum_var':cum,'wells':wids}

# ================================================================
# 1. 运行三个Pipeline (约 10-15 分钟)
# ================================================================
ROOT = os.path.dirname(BASE)

results = {}
results['Original'] = run_one_pipeline('Original ICC (Threshold)',
    os.path.join(ROOT,'Data','FXN_2023_new（ICC）','FXN_20230701','measure_excel'),
    os.path.join(ROOT,'Data','FXN_2023_new（ICC）','FXN_20230703','measure_excel'))

results['nnUNet'] = run_one_pipeline('nnUNet',
    os.path.join(ROOT,'nnUNet_FXN','FXN_0701','measure_excel'),
    os.path.join(ROOT,'nnUNet_FXN','FXN_0703','measure_excel'))

results['MedNext'] = run_one_pipeline('MedNext',
    os.path.join(ROOT,'MedNext_FXN_2023','FXN_0701','measure_excel'),
    os.path.join(ROOT,'MedNext_FXN_2023','FXN_0703','measure_excel'))

# ================================================================
# 2. 保存数据
# ================================================================
print("\n" + "="*60)
print("  SAVING DATA")
print("="*60)
df_all = pd.DataFrame()
for name, r in results.items():
    df = pd.DataFrame({'Method':name,'Score':r['score'],
                       'log10_ATP':np.log10(r['atp']),'ATP':r['atp'],'Well':r['wells']})
    df_all = pd.concat([df_all, df], ignore_index=True)
df_all.to_excel(os.path.join(FIG_DIR,'three_methods_scores.xlsx'), index=False)

feat_df = pd.DataFrame({k: pd.Series(v['features']) for k,v in results.items()})
feat_df.to_excel(os.path.join(FIG_DIR,'three_methods_features.xlsx'), index=False)
pickle.dump(results, open(os.path.join(FIG_DIR,'three_methods_results.pkl'),'wb'))

# ================================================================
# 图1: Score vs log10(ATP) 并排散点图
# ================================================================
print("\n  Figure 1: Score vs ATP scatter (3 panels)")
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
colors = {'Original':'#2E86AB','nnUNet':'#A23B72','MedNext':'#F18F01'}
labels = {'Original':'Original ICC (Threshold)','nnUNet':'nnUNet','MedNext':'MedNext'}

for ax, name in zip(axes, labels):
    r = results[name]; x, y = r['score'], np.log10(r['atp'])
    c = colors[name]
    ax.scatter(x, y, c=c, s=70, alpha=0.8, edgecolors='white', linewidths=0.8, zorder=3)
    z = np.polyfit(x, y, 1); pfit = np.poly1d(z)
    xl = np.linspace(x.min()-0.5, x.max()+0.5, 100)
    ax.plot(xl, pfit(xl), '--', color='#333', linewidth=1.8, alpha=0.7, zorder=2)
    ax.set_title(labels[name], fontsize=13, fontweight='bold', color=c, pad=10)
    ax.set_xlabel('PSD-Delta Score (F)', fontsize=11)
    if name == 'Original':
        ax.set_ylabel('log10(ATP) [pmol]', fontsize=11)
    txt = (f"r = {abs(r['r']):.4f}\np = {r['p']:.2e}\n"
           f"rho = {r['rho']:.4f}\n{r['n_feat']} features\nPC cum = {r['cum_var']:.1%}")
    ax.text(0.05, 0.95, txt, transform=ax.transAxes, fontsize=9, va='top',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.85, ec='#ccc'))
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(x.min()-0.8, x.max()+0.8); ax.set_ylim(3.8, 8.2)

fig.suptitle('PSD-Delta Score vs ATP Across Three Segmentation Methods',
             fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR,'fig1_score_vs_atp.png'), dpi=200, bbox_inches='tight')
plt.close()
print("  -> fig1_score_vs_atp.png")

# ================================================================
# 图2: 特征 Venn 图
# ================================================================
print("\n  Figure 2: Feature Venn diagram")
from matplotlib_venn import venn3

set_o = set(results['Original']['features'])
set_n = set(results['nnUNet']['features'])
set_m = set(results['MedNext']['features'])

fig, (ax_v, ax_t) = plt.subplots(1, 2, figsize=(20, 7),
                                  gridspec_kw={'width_ratios': [1.2, 1]})
v = venn3(subsets=(len(set_o-set_n-set_m), len(set_n-set_o-set_m),
                     len((set_o&set_n)-set_m), len(set_m-set_o-set_n),
                     len((set_o&set_m)-set_n), len((set_n&set_m)-set_o),
                     len(set_o&set_n&set_m)),
          set_labels=('Original ICC\n(Threshold)', 'nnUNet', 'MedNext'),
          set_colors=('#2E86AB', '#A23B72', '#F18F01'), alpha=0.6, ax=ax_v)
for text in v.set_labels:
    if text:
        text.set_fontsize(10); text.set_fontweight('bold')
for text in v.subset_labels:
    if text:
        text.set_fontsize(11); text.set_fontweight('bold')
ax_v.set_title('Feature Overlap Across Segmentation Methods',
               fontsize=14, fontweight='bold', pad=15)

ax_t.axis('off'); ax_t.set_xlim(0, 1); ax_t.set_ylim(0, 1)
yp = 0.95
sections = [
    (f'Intersection (all 3): {len(set_o&set_n&set_m)}', set_o&set_n&set_m, '#333'),
    (f'Original & nnUNet only: {len((set_o&set_n)-set_m)}', (set_o&set_n)-set_m, '#7B2D8E'),
    (f'Original & MedNext only: {len((set_o&set_m)-set_n)}', (set_o&set_m)-set_n, '#C75B39'),
    (f'nnUNet & MedNext only: {len((set_n&set_m)-set_o)}', (set_n&set_m)-set_o, '#D4A017'),
    (f'Original only: {len(set_o-set_n-set_m)}', set_o-set_n-set_m, colors['Original']),
    (f'nnUNet only: {len(set_n-set_o-set_m)}', set_n-set_o-set_m, colors['nnUNet']),
    (f'MedNext only: {len(set_m-set_o-set_n)}', set_m-set_o-set_n, colors['MedNext']),
]
for title, feats, color in sections:
    if len(feats) == 0:
        continue
    ax_t.text(0.02, yp, title, fontsize=10, fontweight='bold', color=color,
              transform=ax_t.transAxes)
    yp -= 0.04
    for f in sorted(feats)[:15]:
        ax_t.text(0.06, yp, f'  \u2022 {f}', fontsize=8, color='#444',
                  transform=ax_t.transAxes)
        yp -= 0.025
    if len(feats) > 15:
        ax_t.text(0.06, yp, f'  ... and {len(feats)-15} more', fontsize=8,
                  color='#888', transform=ax_t.transAxes, fontstyle='italic')
        yp -= 0.03
    yp -= 0.02

fig.suptitle('Selected Feature Comparison: Venn Diagram',
             fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig2_feature_venn.png'), dpi=200, bbox_inches='tight')
plt.close()
print("  -> fig2_feature_venn.png")

# ================================================================
# 图3: PC1 Loading 条形图
# ================================================================
print("\n  Figure 3: PC1 Loading bar chart")
fig, axes = plt.subplots(1, 3, figsize=(20, 7))

for ax, name in zip(axes, labels):
    r = results[name]; pc = r['pca']; feats = r['features']
    loading = pc.components_[0]; evr = pc.explained_variance_ratio_[0]
    si = np.argsort(loading); n_show = min(len(feats), 20)
    sl = loading[si][-n_show:]; sf = [feats[i] for i in si[-n_show:]]
    bar_c = [colors[name] if v > 0 else '#D64933' for v in sl]
    ax.barh(range(n_show), sl, color=bar_c, edgecolor='white', linewidth=0.5)
    ax.set_yticks(range(n_show)); ax.set_yticklabels(sf, fontsize=8)
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.set_title(f'{labels[name]}\nPC1 ({evr:.1%})', fontsize=12, fontweight='bold',
                 color=colors[name], pad=10)
    ax.set_xlabel('PC1 Loading', fontsize=10)
    ax.grid(True, alpha=0.3, axis='x', linestyle='--')

fig.suptitle('PC1 Loading Comparison Across Three Segmentation Methods',
             fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig3_pc1_loading.png'), dpi=200, bbox_inches='tight')
plt.close()
print("  -> fig3_pc1_loading.png")

# ================================================================
# 最终对比
# ================================================================
print("\n" + "=" * 70)
print("  FINAL COMPARISON")
print("=" * 70)
print(f"  {'Method':<28s} {'|r|':>8s} {'rho':>8s} {'p':>12s} {'nFeat':>6s} {'PC cum':>8s}")
print(f"  {'-'*70}")
for name in ['Original', 'nnUNet', 'MedNext']:
    r = results[name]
    print(f"  {labels[name]:<28s} {abs(r['r']):>8.4f} {r['rho']:>8.4f} "
          f"{r['p']:>12.2e} {r['n_feat']:>6d} {r['cum_var']:>7.1%}")
print(f"\n  All figures + data saved to: {FIG_DIR}")
print("=" * 70)