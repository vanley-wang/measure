# 快速诊断：GC vs ICC 表型分布差异
import os, sys, pickle
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

GC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'Data', 'FXN_2023_new（GC）')
ICC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'Data', 'FXN_2023_new（ICC）')
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model')

CF = ['Organoids_Volume','Organoids_Volume_Fill','Organoids_Surface',
      'Cavity_Volume','CavityNum','LongAxis','ShortAxis',
      'Wall_Thickness','Sphericity','Scatt_Mean','Scatt_STD']
NM = {3:0,5:0,1:1,0:2,4:2,2:3}
NAMES = {0:'巨大囊泡',1:'中等过渡',2:'小体积基准',3:'高散射实心'}

with open(os.path.join(MODEL_DIR,'kmeans_k6.pkl'),'rb') as f: km = pickle.load(f)
with open(os.path.join(MODEL_DIR,'scaler_k6.pkl'),'rb') as f: sc = pickle.load(f)

def count_phenotypes(day_dir):
    wells = {}
    for f in os.listdir(day_dir):
        if not f.endswith('.xlsx'): continue
        wid = f.replace('.xlsx','').split('_')[0]
        df = pd.read_excel(os.path.join(day_dir,f))
        X = df[CF].fillna(0).values
        k6 = km.predict(sc.transform(X))
        k4 = np.array([NM[l] for l in k6])
        counts = {i: int(np.sum(k4==i)) for i in range(4)}
        counts['total'] = len(k4)
        wells[wid] = counts
    return wells

for name, d3_dir, d5_dir in [
    ('ICC', os.path.join(ICC_DIR,'FXN_20230701','measure_excel'),
            os.path.join(ICC_DIR,'FXN_20230703','measure_excel')),
    ('GC',  os.path.join(GC_DIR,'FXN_20230701','measure_excel'),
            os.path.join(GC_DIR,'FXN_20230703','measure_excel')),
]:
    d3 = count_phenotypes(d3_dir)
    print(f'\n{"="*50}')
    print(f'  {name} — Day3 → Day5 Phenotype Shift')
    print(f'{"="*50}')
    for i in range(4):
        d3_total = sum(w[i] for w in d3.values())
        d3_pct = d3_total / sum(w['total'] for w in d3.values()) * 100
        print(f'  {NAMES[i]:8s}  Day3: {d3_total:>7,} ({d3_pct:>5.1f}%)')
    total = sum(w['total'] for w in d3.values())
    print(f'  {"Total":8s}  Day3: {total:>7,}')

    # Day3→Day5 比例变化
    common = sorted(set(d3)&set([f.replace('.xlsx','').split('_')[0] for f in os.listdir(d5_dir) if f.endswith('.xlsx')]))
    d5 = count_phenotypes(d5_dir)
    for day_label, day_data in [('Day3', d3), ('Day5', d5)]:
        d_total = sum(day_data[w]['total'] for w in common if w in day_data)
        print(f'\n  [{day_label}] ', end='')
        for i in range(4):
            cnt = sum(day_data[w][i] for w in common if w in day_data)
            pct = cnt/d_total*100 if d_total>0 else 0
            print(f'{NAMES[i]}: {pct:.1f}%  ', end='')
    print()

    # 每种表型的 delta 体积 (Day5-Day3)
    FEATURE_LONG = {
        'Organoids_Volume':'Volume_Avg','Organoids_Volume_Fill':'Volume_Fill_Avg',
        'Organoids_Surface':'Surface_Avg','Cavity_Volume':'Cavity_Volume_All',
        'CavityNum':'CavityNum_Avg','LongAxis':'Long_Axis_Avg',
        'ShortAxis':'Short_Axis_Avg','Wall_Thickness':'Cyst_Thick_Avg',
        'Sphericity':'Sphericity_Avg','Scatt_Mean':'Scatt_Mean_Avg',
        'Scatt_STD':'Scatt_STD_Avg',
    }
    # 每个表型总体积变化
    for cl in range(4):
        vols_d3, vols_d5 = [], []
        for w in common:
            wd3 = pd.read_excel(os.path.join(d3_dir, w + ('_0701.xlsx' if name=='GC' else '_0701.xlsx')))
            wd5 = pd.read_excel(os.path.join(d5_dir, w + ('_0703.xlsx' if name=='GC' else '_0703.xlsx')))
            X3 = wd3[CF].fillna(0).values; X5 = wd5[CF].fillna(0).values
            k4_3 = np.array([NM[l] for l in km.predict(sc.transform(X3))])
            k4_5 = np.array([NM[l] for l in km.predict(sc.transform(X5))])
            vols_d3.append(wd3.loc[k4_3==cl,'Organoids_Volume'].mean() if (k4_3==cl).sum()>0 else np.nan)
            vols_d5.append(wd5.loc[k4_5==cl,'Organoids_Volume'].mean() if (k4_5==cl).sum()>0 else np.nan)
        v3 = np.nanmean(vols_d3); v5 = np.nanmean(vols_d5)
        print(f'  {NAMES[cl]:8s}  Vol D3→D5: {v3/1e6:>6.1f}M → {v5/1e6:>6.1f}M  (Δ={v5-v3:+.0f})')

print()