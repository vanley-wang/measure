# ============================================================
# 深度诊断: nnUNet vs Original — 同一个类器官被两种分割"看到"了什么
# ============================================================
import os, sys
import numpy as np
import pandas as pd

ORIG_D3 = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'Data', 'FXN_2023_new（ICC）', 'FXN_20230701', 'measure_excel')
NNUNET_D3 = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          'Data', 'nnUNet_FXN_2023', 'FXN_0701', 'measure_excel')
CF = ['Organoids_Volume','Organoids_Volume_Fill','Organoids_Surface',
      'Cavity_Volume','CavityNum','LongAxis','ShortAxis',
      'Wall_Thickness','Sphericity','Scatt_Mean','Scatt_STD']

# 1. 逐个well对比类器官数量
print('='*60)
print('  1. Organoid Count per Well')
print('='*60)
d3o = {f.replace('.xlsx','').split('_')[0]: f for f in os.listdir(ORIG_D3) if f.endswith('.xlsx')}
d3n = {f.replace('.xlsx','').split('_')[0]: f for f in os.listdir(NNUNET_D3) if f.endswith('.xlsx')}
common = sorted(set(d3o)&set(d3n))

cnts_o, cnts_n = [], []
for w in common:
    no = len(pd.read_excel(os.path.join(ORIG_D3, d3o[w])))
    nn = len(pd.read_excel(os.path.join(NNUNET_D3, d3n[w])))
    cnts_o.append(no); cnts_n.append(nn)
    if abs(nn-no) > 100:
        print(f'  {w}: Orig={no}, nnU={nn}, Δ={nn-no:+d} ***')
print(f'\n  Total: Orig={sum(cnts_o)}, nnU={sum(cnts_n)}, Δ={sum(cnts_n)-sum(cnts_o):+d}')

# 2. 全局特征差异
print(f'\n{"="*60}')
print(f'  2. Global Morphology Stats (all organoids pooled)')
print(f'{"="*60}')
print(f'  {"Feature":<22s} {"Orig":>12s} {"nnUNet":>12s} {"Ratio":>8s} {"CV Orig":>8s} {"CV nnU":>8s}')
print('-'*72)

for feat in CF:
    all_o, all_n = [], []
    for w in common:
        do = pd.read_excel(os.path.join(ORIG_D3, d3o[w]))
        dn = pd.read_excel(os.path.join(NNUNET_D3, d3n[w]))
        all_o.extend(do[feat].dropna().tolist())
        all_n.extend(dn[feat].dropna().tolist())
    ao, an = np.array(all_o), np.array(all_n)
    cv_o = ao.std()/ao.mean() if ao.mean()!=0 else 0
    cv_n = an.std()/an.mean() if an.mean()!=0 else 0
    print(f'  {feat:<22s} {ao.mean():>12.3e} {an.mean():>12.3e} '
          f'{an.mean()/ao.mean():>7.1f}x  {cv_o:>7.3f}  {cv_n:>7.3f}')

# 3. Cavity 专项分析
print(f'\n{"="*60}')
print(f'  3. Cavity Analysis — Why CavityNum ×{43:.0f}?')
print(f'{"="*60}')

# 统计有多少类器官有 cavity
orig_has_cav, nnu_has_cav = 0, 0
orig_total, nnu_total = 0, 0
for w in common:
    do = pd.read_excel(os.path.join(ORIG_D3, d3o[w]))
    dn = pd.read_excel(os.path.join(NNUNET_D3, d3n[w]))
    orig_has_cav += (do['CavityNum'] > 0).sum()
    orig_total += len(do)
    nnu_has_cav += (dn['CavityNum'] > 0).sum()
    nnu_total += len(dn)

print(f'  Original: {orig_has_cav}/{orig_total} = {orig_has_cav/orig_total*100:.1f}% have cavities')
print(f'  nnUNet:   {nnu_has_cav}/{nnu_total} = {nnu_has_cav/nnu_total*100:.1f}% have cavities')

# CavityNum分布
orig_cavnums, nnu_cavnums = [], []
for w in common:
    do = pd.read_excel(os.path.join(ORIG_D3, d3o[w]))
    dn = pd.read_excel(os.path.join(NNUNET_D3, d3n[w]))
    orig_cavnums.extend(do['CavityNum'].dropna().tolist())
    nnu_cavnums.extend(dn['CavityNum'].dropna().tolist())

ao_cn, an_cn = np.array(orig_cavnums), np.array(nnu_cavnums)
print(f'\n  CavityNum distribution:')
print(f'  {"":>10s} {"Orig":>10s} {"nnUNet":>10s}')
print(f'  {"Mean":>10s} {ao_cn.mean():>10.4f} {an_cn.mean():>10.4f}')
print(f'  {"Median":>10s} {np.median(ao_cn):>10.4f} {np.median(an_cn):>10.4f}')
print(f'  {"Max":>10s} {ao_cn.max():>10.4f} {an_cn.max():>10.4f}')

# CavityNum > 0 的类器官的平均CavityNum
orig_nonzero = ao_cn[ao_cn > 0]
nnu_nonzero = an_cn[an_cn > 0]
print(f'\n  Among organoids WITH cavities:')
print(f'    Orig: {len(orig_nonzero)} organoids, mean CavNum={orig_nonzero.mean():.2f}')
print(f'    nnU:  {len(nnu_nonzero)} organoids, mean CavNum={nnu_nonzero.mean():.2f}')

# Cavity Volume 在有cavity的organoids中
orig_cavvol, nnu_cavvol = [], []
for w in common:
    do = pd.read_excel(os.path.join(ORIG_D3, d3o[w]))
    dn = pd.read_excel(os.path.join(NNUNET_D3, d3n[w]))
    orig_cavvol.extend(do.loc[do['CavityNum']>0, 'Cavity_Volume'].dropna().tolist())
    nnu_cavvol.extend(dn.loc[dn['CavityNum']>0, 'Cavity_Volume'].dropna().tolist())
ao_cv = np.array(orig_cavvol); an_cv = np.array(nnu_cavvol)
print(f'\n  Cavity Volume among organoids with cavities:')
print(f'    Orig: mean={ao_cv.mean():.1f}, median={np.median(ao_cv):.1f}, max={ao_cv.max():.1f}')
print(f'    nnU:  mean={an_cv.mean():.1f}, median={np.median(an_cv):.1f}, max={an_cv.max():.1f}')

# 4. Scattering 专项分析
print(f'\n{"="*60}')
print(f'  4. Scattering Analysis — Why Scatt ×0.1?')
print(f'{"="*60}')

orig_scatt, nnu_scatt = [], []
for w in common:
    do = pd.read_excel(os.path.join(ORIG_D3, d3o[w]))
    dn = pd.read_excel(os.path.join(NNUNET_D3, d3n[w]))
    orig_scatt.extend(do['Scatt_Mean'].dropna().tolist())
    nnu_scatt.extend(dn['Scatt_Mean'].dropna().tolist())
ao_s, an_s = np.array(orig_scatt), np.array(nnu_scatt)
print(f'  Scatt_Mean: Orig mean={ao_s.mean():.1f}, nnU mean={an_s.mean():.1f}, ratio={an_s.mean()/ao_s.mean():.3f}')
print(f'  Scatt_Mean: Orig std={ao_s.std():.1f}, nnU std={an_s.std():.1f}, ratio={an_s.std()/ao_s.std():.3f}')

orig_scatt_s, nnu_scatt_s = [], []
for w in common:
    do = pd.read_excel(os.path.join(ORIG_D3, d3o[w]))
    dn = pd.read_excel(os.path.join(NNUNET_D3, d3n[w]))
    orig_scatt_s.extend(do['Scatt_STD'].dropna().tolist())
    nnu_scatt_s.extend(dn['Scatt_STD'].dropna().tolist())
ao_ss, an_ss = np.array(orig_scatt_s), np.array(nnu_scatt_s)
print(f'  Scatt_STD:  Orig mean={ao_ss.mean():.1f}, nnU mean={an_ss.mean():.1f}, ratio={an_ss.mean()/ao_ss.mean():.3f}')

# 5. Surface 分析 - 球形度与面积的关系
print(f'\n{"="*60}')
print(f'  5. Surface/Sphericity — Boundary smoothness')
print(f'{"="*60}')

# 对相同体积范围的类器官比较Surface
orig_surf, orig_vol, orig_spher = [], [], []
nnu_surf, nnu_vol, nnu_spher = [], [], []
for w in common:
    do = pd.read_excel(os.path.join(ORIG_D3, d3o[w]))
    dn = pd.read_excel(os.path.join(NNUNET_D3, d3n[w]))
    orig_surf.extend(do['Organoids_Surface'].dropna().tolist())
    orig_vol.extend(do['Organoids_Volume'].dropna().tolist())
    orig_spher.extend(do['Sphericity'].dropna().tolist())
    nnu_surf.extend(dn['Organoids_Surface'].dropna().tolist())
    nnu_vol.extend(dn['Organoids_Volume'].dropna().tolist())
    nnu_spher.extend(dn['Sphericity'].dropna().tolist())

print(f'  Surface/Volume ratio:')
print(f'    Orig: Surf/Vol mean={np.mean(np.array(orig_surf)/np.array(orig_vol)):.4f}')
print(f'    nnU:  Surf/Vol mean={np.mean(np.array(nnu_surf)/np.array(nnu_vol)):.4f}')
print(f'  Sphericity:')
print(f'    Orig: mean={np.mean(orig_spher):.4f}, CV={np.std(orig_spher)/np.mean(orig_spher):.3f}')
print(f'    nnU:  mean={np.mean(nnu_spher):.4f}, CV={np.std(nnu_spher)/np.mean(nnu_spher):.3f}')

# 6. CV对比 — 哪个分割保留了更多形态差异
print(f'\n{"="*60}')
print(f'  6. Coefficient of Variation (CV) — Morphological Diversity')
print(f'{"="*60}')
print(f'  Higher CV = more morphological diversity preserved')
print(f'  {"Feature":<22s} {"CV Orig":>8s} {"CV nnU":>8s} {"nnU/Orig":>10s}')
print('-'*52)
for feat in CF:
    ao_vals, an_vals = [], []
    for w in common:
        do = pd.read_excel(os.path.join(ORIG_D3, d3o[w]))
        dn = pd.read_excel(os.path.join(NNUNET_D3, d3n[w]))
        ao_vals.extend(do[feat].dropna().tolist())
        an_vals.extend(dn[feat].dropna().tolist())
    ao, an = np.array(ao_vals), np.array(an_vals)
    cv_o = ao.std()/ao.mean() if ao.mean()!=0 else 0
    cv_n = an.std()/an.mean() if an.mean()!=0 else 0
    arrow = '↓' if cv_n/cv_o < 0.9 else ('↑' if cv_n/cv_o > 1.1 else '≈')
    print(f'  {feat:<22s} {cv_o:>8.3f} {cv_n:>8.3f} {cv_n/cv_o:>9.3f} {arrow}')

print()