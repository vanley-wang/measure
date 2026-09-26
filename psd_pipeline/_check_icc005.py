"""Compare ICC005 new measurements with Original FXN measurements for matched wells"""
import os, pickle
import numpy as np
import pandas as pd

MORPH_FEATURES = [
    'Organoids_Volume', 'Organoids_Volume_Fill', 'Organoids_Surface',
    'Cavity_Volume', 'CavityNum', 'LongAxis', 'ShortAxis',
    'Wall_Thickness', 'Sphericity'
]

ICC_DIR = r'D:\Desktop\music\measure\Data'
# Find the ICC directory with Chinese characters
subs = [f for f in os.listdir(ICC_DIR) if 'ICC' in f and 'new' in f]
ICC_DATA = os.path.join(ICC_DIR, subs[0])

WELLS = ['B4','B6','B9','B10','B11','F4','F5','F7','F8','F10','G4','G6','G9','G11']
DATE_TAGS = ['0701','0703']

print('=' * 70)
print('  Compare ICC005 (new seg) vs Original FXN measurements')
print('=' * 70)

new_dir = r'D:\Desktop\music\measure\psd_pipeline\icc005_results'

for date_tag in DATE_TAGS:
    print(f'\n--- {date_tag} ---')
    print(f'{"Well":<6s} {"N_new":>6s} {"N_old":>6s}  {"Vol_new":>10s} {"Vol_old":>10s}  {"Surf_new":>10s} {"Surf_old":>10s}  {"LongAx_new":>10s} {"LongAx_old":>10s}')
    print('-' * 70)

    for well in WELLS:
        new_fp = os.path.join(new_dir, well, f'{well}_{date_tag}.xlsx')
        old_fp = os.path.join(ICC_DATA, f'FXN_2023{date_tag}', 'measure_excel', f'{well}_{date_tag}.xlsx')

        new_stats = {'N': 0, 'Vol': 0, 'Surf': 0, 'LongAx': 0}
        old_stats = {'N': 0, 'Vol': 0, 'Surf': 0, 'LongAx': 0}

        if os.path.exists(new_fp):
            df = pd.read_excel(new_fp)
            new_stats['N'] = len(df)
            new_stats['Vol'] = df['Organoids_Volume'].mean()
            new_stats['Surf'] = df['Organoids_Surface'].mean()
            new_stats['LongAx'] = df['LongAxis'].mean()

        if os.path.exists(old_fp):
            df = pd.read_excel(old_fp)
            old_stats['N'] = len(df)
            old_stats['Vol'] = df['Organoids_Volume'].mean()
            old_stats['Surf'] = df['Organoids_Surface'].mean()
            old_stats['LongAx'] = df['LongAxis'].mean()

        print(f'{well:<6s} {new_stats["N"]:>6d} {old_stats["N"]:>6d}  '
              f'{new_stats["Vol"]:>10.1f} {old_stats["Vol"]:>10.1f}  '
              f'{new_stats["Surf"]:>10.1f} {old_stats["Surf"]:>10.1f}  '
              f'{new_stats["LongAx"]:>10.1f} {old_stats["LongAx"]:>10.1f}')

# Summary ratio
all_new_vol, all_old_vol = [], []
all_new_surf, all_old_surf = [], []
for date_tag in DATE_TAGS:
    for well in WELLS:
        new_fp = os.path.join(new_dir, well, f'{well}_{date_tag}.xlsx')
        old_fp = os.path.join(ICC_DATA, f'FXN_2023{date_tag}', 'measure_excel', f'{well}_{date_tag}.xlsx')
        if os.path.exists(new_fp) and os.path.exists(old_fp):
            dn = pd.read_excel(new_fp)
            do = pd.read_excel(old_fp)
            all_new_vol.extend(dn['Organoids_Volume'].tolist())
            all_old_vol.extend(do['Organoids_Volume'].tolist())
            all_new_surf.extend(dn['Organoids_Surface'].tolist())
            all_old_surf.extend(do['Organoids_Surface'].tolist())

anv = np.array(all_new_vol); aov = np.array(all_old_vol)
ans = np.array(all_new_surf); aos = np.array(all_old_surf)
print(f'\n{"="*70}')
print(f'  Summary (all organoids)')
print(f'  Volume:  new_mean={anv.mean():.1f}  old_mean={aov.mean():.1f}  ratio={anv.mean()/aov.mean():.3f}')
print(f'  Surface: new_mean={ans.mean():.1f}  old_mean={aos.mean():.1f}  ratio={ans.mean()/aos.mean():.3f}')
print(f'  Organoid count: new={len(anv)}  old={len(aov)}')
print(f'  Correlation new vs old volume: {np.corrcoef(anv, aov)[0,1]:.4f}')