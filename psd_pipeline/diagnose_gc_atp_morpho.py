# ============================================================
# 核心诊断: GC的ATP差异大但聚类同质 → 形态 ≠ ATP ?
# ============================================================
import os, sys
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

GC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'Data','FXN_2023_new（GC）')
D3_DIR = os.path.join(GC_DIR, 'FXN_20230701','measure_excel')
D5_DIR = os.path.join(GC_DIR, 'FXN_20230703','measure_excel')

CF = ['Organoids_Volume','Organoids_Volume_Fill','Organoids_Surface',
      'Cavity_Volume','CavityNum','LongAxis','ShortAxis',
      'Wall_Thickness','Sphericity','Scatt_Mean','Scatt_STD']

# Load ATP
atp_df = pd.read_excel(os.path.join(GC_DIR,'ATP.xlsx'))
gc_atp = {}
for _, row in atp_df.iterrows():
    wid = str(row['Name']).split('_')[0]
    val = row['ATP']
    if pd.notna(val): gc_atp[wid] = float(val)

# ================================================================
# 1. 每个孔 Day3 的形态均值 vs ATP — 看看原始形态和ATP有没有关系
# ================================================================
print('='*60)
print('  Day3 全孔形态均值 vs ATP (不聚类，直接看)')
print('='*60)
print(f'  {"Feature":<20s} {"r":>8s} {"p":>10s}')
print('-' * 42)

well_stats = {}
for f in sorted(os.listdir(D3_DIR)):
    if not f.endswith('.xlsx'): continue
    wid = f.replace('.xlsx','').split('_')[0]
    df = pd.read_excel(os.path.join(D3_DIR,f))
    s = {}
    for col in CF:
        s[f'{col}_mean'] = df[col].mean()
        s[f'{col}_std'] = df[col].std()
        s[f'{col}_cv'] = df[col].std()/df[col].mean() if df[col].mean()!=0 else 0
    s['N_organoids'] = len(df)
    well_stats[wid] = s

common = [w for w in well_stats if w in gc_atp]
for feat in CF:
    x = [well_stats[w][f'{feat}_mean'] for w in common]
    y = [gc_atp[w] for w in common]
    r, p = pearsonr(x, y)
    sig = '**' if p<0.01 else ('*' if p<0.05 else '')
    print(f'  {feat:<20s} {r:>+8.3f} {p:>10.4f} {sig}')

# ================================================================
# 2. Day3 vs Day5 每个孔的形态差异 vs ATP
# ================================================================
print(f'\n{"="*60}')
print(f'  Day5−Day3 全孔 Delta vs ATP')
print(f'{"="*60}')
print(f'  {"Feature":<20s} {"r":>8s} {"p":>10s}')
print('-' * 42)

for feat in CF:
    deltas = []
    for w in common:
        d5 = pd.read_excel(os.path.join(D5_DIR, w+'_0703.xlsx'))
        d3 = pd.read_excel(os.path.join(D3_DIR, w+'_0701.xlsx'))
        deltas.append(d5[feat].mean() - d3[feat].mean())
    r, p = pearsonr(deltas, [gc_atp[w] for w in common])
    sig = '**' if p<0.01 else ('*' if p<0.05 else '')
    print(f'  Δ{feat:<19s} {r:>+8.3f} {p:>10.4f} {sig}')

# ================================================================
# 3. 高ATP组 vs 低ATP组 形态对比
# ================================================================
median_atp = np.median([gc_atp[w] for w in common])
hi = [w for w in common if gc_atp[w] > median_atp]
lo = [w for w in common if gc_atp[w] <= median_atp]
print(f'\n{"="*60}')
print(f'  高ATP组 (n={len(hi)}, >{median_atp/1e6:.1f}M) vs 低ATP组 (n={len(lo)})')
print(f'{"="*60}')
print(f'  {"Feature":<22s} {"High ATP":>10s} {"Low ATP":>10s} {"Diff":>10s}')
print('-' * 56)

for feat in CF:
    hi_vals = np.array([well_stats[w][f'{feat}_mean'] for w in hi])
    lo_vals = np.array([well_stats[w][f'{feat}_mean'] for w in lo])
    print(f'  {feat:<22s} {np.mean(hi_vals):>10.3e} {np.mean(lo_vals):>10.3e} '
          f'{(np.mean(hi_vals)-np.mean(lo_vals)):>+10.3e}')

# ================================================================
# 4. 孔级 CV (异质性) vs ATP — 也许变化不在均值，在异质性
# ================================================================
print(f'\n{"="*60}')
print(f'  Day3 异质性 (CV) vs ATP')
print(f'{"="*60}')
print(f'  {"Feature":<20s} {"r":>8s} {"p":>10s}')
print('-' * 42)
for feat in CF:
    x = [well_stats[w][f'{feat}_cv'] for w in common]
    y = [gc_atp[w] for w in common]
    r, p = pearsonr(x, y)
    sig = '**' if p<0.01 else ('*' if p<0.05 else '')
    print(f'  CV_{feat:<16s} {r:>+8.3f} {p:>10.4f} {sig}')

gc_atp_log = {w: np.log10(v) for w,v in gc_atp.items()}
print(f'\n  (log10 ATP)')
for feat in CF:
    x = [well_stats[w][f'{feat}_cv'] for w in common]
    y = [gc_atp_log[w] for w in common]
    r, p = pearsonr(x, y)
    sig = '**' if p<0.01 else ('*' if p<0.05 else '')
    print(f'  CV_{feat:<16s} {r:>+8.3f} {p:>10.4f} {sig}')