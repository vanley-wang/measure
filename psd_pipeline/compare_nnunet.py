# ============================================================
# nnUNet vs Original ICC — 预测对比 + 形态差异诊断
# ============================================================
import os, sys, pickle
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

ORIG_D3 = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'Data', 'FXN_2023_new（ICC）', 'FXN_20230701', 'measure_excel')
ORIG_D5 = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'Data', 'FXN_2023_new（ICC）', 'FXN_20230703', 'measure_excel')
NNUNET_D3 = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          'Data', 'nnUNet_FXN_2023', 'FXN_0701', 'measure_excel')
NNUNET_D5 = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          'Data', 'nnUNet_FXN_2023', 'FXN_0703', 'measure_excel')
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model')

CF = ['Organoids_Volume','Organoids_Volume_Fill','Organoids_Surface',
      'Cavity_Volume','CavityNum','LongAxis','ShortAxis',
      'Wall_Thickness','Sphericity','Scatt_Mean','Scatt_STD']
CF_SHORT = ['Vol','FillVol','Surf','CavVol','CavNum',
            'LongAx','ShortAx','WallTh','Spher','ScattM','ScattS']
NUMERIC_MAP = {3:0,5:0,1:1,0:2,4:2,2:3}
N_MERGED = 4
FEATURE_LONG = {
    'Organoids_Volume':'Volume_Avg','Organoids_Volume_Fill':'Volume_Fill_Avg',
    'Organoids_Surface':'Surface_Avg','Cavity_Volume':'Cavity_Volume_All',
    'CavityNum':'CavityNum_Avg','LongAxis':'Long_Axis_Avg',
    'ShortAxis':'Short_Axis_Avg','Wall_Thickness':'Cyst_Thick_Avg',
    'Sphericity':'Sphericity_Avg','Scatt_Mean':'Scatt_Mean_Avg',
    'Scatt_STD':'Scatt_STD_Avg',
}

ATP_DB = {
    'B10':601300,'B11':11180000,'B2':5391000,'B3':6538000,
    'B4':7103000,'B5':511900,'B6':404500,'B7':403900,
    'B8':312700,'B9':140300,'C10':211800,'C11':13930000,
    'C2':6336000,'C3':8336000,'C4':6800000,'C5':330900,
    'C6':238900,'C7':682100,'C8':211300,'C9':465900,
    'D11':11240000,'E11':14700000,'F10':21910000,'F11':11180000,
    'F2':18240000,'F3':14110000,'F4':13740000,'F5':17250000,
    'F6':20320000,'F7':20000000,'F8':17170000,'F9':15830000,
}

# 1. 加载模型
with open(os.path.join(MODEL_DIR,'kmeans_k6.pkl'),'rb') as f: kmeans = pickle.load(f)
with open(os.path.join(MODEL_DIR,'scaler_k6.pkl'),'rb') as f: clus_scaler = pickle.load(f)
with open(os.path.join(MODEL_DIR,'paper_delta_deploy.pkl'),'rb') as f: deploy = pickle.load(f)

# 2. 预测函数
def predict_one_dataset(d3_dir, d5_dir):
    d3_files = {f.replace('.xlsx','').split('_')[0]: f for f in os.listdir(d3_dir) if f.endswith('.xlsx')}
    d5_files = {f.replace('.xlsx','').split('_')[0]: f for f in os.listdir(d5_dir) if f.endswith('.xlsx')}
    common = sorted(set(d3_files)&set(d5_files))
    def agg(df, labels):
        rec = {}
        for cl in range(N_MERGED):
            mask = labels == cl
            n = int(mask.sum())
            rec[f'Number_{cl+1}'] = n
            sub = df.loc[mask] if n>0 else df.iloc[:0]
            for src, alias in FEATURE_LONG.items():
                if src not in df.columns: continue
                rec[f'{alias}_{cl+1}'] = float(sub[src].sum()) if src=='Cavity_Volume' else (float(sub[src].mean()) if n>0 else 0.0)
        return rec
    scores = {}
    for wid in common:
        d3 = pd.read_excel(os.path.join(d3_dir, d3_files[wid]))
        d5 = pd.read_excel(os.path.join(d5_dir, d5_files[wid]))
        X3 = d3[CF].fillna(0).values; X5 = d5[CF].fillna(0).values
        c3 = np.array([NUMERIC_MAP[l] for l in kmeans.predict(clus_scaler.transform(X3))])
        c5 = np.array([NUMERIC_MAP[l] for l in kmeans.predict(clus_scaler.transform(X5))])
        feats_d3 = agg(d3, c3); feats_d5 = agg(d5, c5)
        delta = np.array([feats_d5.get(f,0)-feats_d3.get(f,0) for f in deploy['features']])
        x_std = deploy['scaler'].transform(delta.reshape(1,-1))[0]
        scores[wid] = float(np.dot(deploy['beta'], x_std))
    return scores

print('Predicting...')
scores_orig = predict_one_dataset(ORIG_D3, ORIG_D5)
scores_nnu = predict_one_dataset(NNUNET_D3, NNUNET_D5)

# 3. 相关性对比
common_w = sorted(set(scores_orig)&set(scores_nnu)&set(ATP_DB))
atp = [ATP_DB[w] for w in common_w]
r_orig, _ = pearsonr([scores_orig[w] for w in common_w], atp)
r_nnu, _ = pearsonr([scores_nnu[w] for w in common_w], atp)
print(f'\n{"="*50}')
print(f'  Prediction Comparison (same ICC wells)')
print(f'{"="*50}')
print(f'  Original r = {r_orig:.4f}')
print(f'  nnUNet   r = {r_nnu:.4f}')
print(f'  Δr       = {r_orig - abs(r_nnu):+.4f}')

# 4. 对比每个孔的 Score
print(f'\n{"="*50}')
print(f'  Per-well Score Comparison')
print(f'{"="*50}')
print(f'  {"Well":<6s} {"Orig":>10s} {"nnUNet":>10s} {"Diff":>10s}')
print('-'*40)
for w in common_w:
    d = scores_orig[w] - scores_nnu[w]
    print(f'  {w:<6s} {scores_orig[w]:>10.4f} {scores_nnu[w]:>10.4f} {d:>+10.4f}')

# 5. 形态特征对比：同一个孔 Day3 均值差异
print(f'\n{"="*60}')
print(f'  Morphology Bias: nnUNet − Original (per-organoid mean)')
print(f'{"="*60}')
biases = {f: [] for f in CF}
for w in common_w:
    d3o = pd.read_excel(os.path.join(ORIG_D3, f'{w}_0701.xlsx'))
    d3n = pd.read_excel(os.path.join(NNUNET_D3, f'{w}_0701.xlsx'))
    for f in CF:
        biases[f].append(d3n[f].mean() - d3o[f].mean())

# 百分比偏差
print(f'  {"Feature":<22s} {"Bias":>12s} {"Orig Mean":>12s} {"Rel%":>8s}')
print('-'*58)
for f in CF:
    all_orig = []
    for w in common_w:
        d3o = pd.read_excel(os.path.join(ORIG_D3, f'{w}_0701.xlsx'))
        all_orig.append(d3o[f].mean())
    orig_mean = np.mean(all_orig)
    bias_mean = np.mean(biases[f])
    rel = bias_mean / orig_mean * 100 if orig_mean != 0 else 0
    print(f'  {f:<22s} {bias_mean:>+12.3e} {orig_mean:>12.3e} {rel:>+7.1f}%')

# 6. Delta 特征对比
print(f'\n{"="*60}')
print(f'  Delta Bias: nnUNet Delta − Original Delta')
print(f'{"="*60}')
for f in CF:
    delta_orig = []; delta_nnu = []
    for w in common_w:
        d3o=pd.read_excel(os.path.join(ORIG_D3,f'{w}_0701.xlsx'));d5o=pd.read_excel(os.path.join(ORIG_D5,f'{w}_0703.xlsx'))
        d3n=pd.read_excel(os.path.join(NNUNET_D3,f'{w}_0701.xlsx'));d5n=pd.read_excel(os.path.join(NNUNET_D5,f'{w}_0703.xlsx'))
        delta_orig.append(d5o[f].mean()-d3o[f].mean()); delta_nnu.append(d5n[f].mean()-d3n[f].mean())
    r_o, _ = pearsonr(delta_orig, atp); r_n, _ = pearsonr(delta_nnu, atp)
    print(f'  Δ{CF_SHORT[CF.index(f)]:<9s}  Orig r={r_o:+.3f}  nnU r={r_n:+.3f}  Δr={abs(r_o)-abs(r_n):+.3f}')

# 7. 类器官数量对比
print(f'\n{"="*60}')
print(f'  Organoid Count Bias: nnUNet − Original')
print(f'{"="*60}')
for w in common_w[:5]:
    no=len(pd.read_excel(os.path.join(ORIG_D3,f'{w}_0701.xlsx')))
    nn=len(pd.read_excel(os.path.join(NNUNET_D3,f'{w}_0701.xlsx')))
    print(f'  {w}: Orig={no}, nnU={nn}, Δ={nn-no:+d}')
# 全量统计
cnts_orig=[]; cnts_nnu=[]
for w in common_w:
    cnts_orig.append(len(pd.read_excel(os.path.join(ORIG_D3,f'{w}_0701.xlsx'))))
    cnts_nnu.append(len(pd.read_excel(os.path.join(NNUNET_D3,f'{w}_0701.xlsx'))))
print(f'  Total: Orig={sum(cnts_orig)}, nnU={sum(cnts_nnu)}, Δ={sum(cnts_nnu)-sum(cnts_orig):+d}')