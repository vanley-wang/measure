# ============================================================
# Step 2: 逐孔、逐表型聚合统计 (复现 cluster_analysis_1.py)
# 论文方法: 每个孔 X 每个合并表型(0-3)，计算均数/总数
#          Cavity_Volume 用总和(Cavity_Volume_All)，其余用均值
#          输出为扁平表，每孔每天一行，存入 FXN_2023_Analysis.xlsx
# ============================================================
import os
import sys
import pickle
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, 'Data', 'FXN_2023_new（ICC）')
DAY3_MEASURE_DIR = os.path.join(DATA_DIR, 'FXN_20230701', 'measure_excel')
DAY5_MEASURE_DIR = os.path.join(DATA_DIR, 'FXN_20230703', 'measure_excel')
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

FEATURES = [
    'Organoids_Volume', 'Organoids_Volume_Fill', 'Organoids_Surface',
    'Cavity_Volume', 'CavityNum', 'LongAxis', 'ShortAxis',
    'Wall_Thickness', 'Sphericity', 'Scatt_Mean', 'Scatt_STD'
]

NUMERIC_MAP = {3: 0, 5: 0, 1: 1, 0: 2, 4: 2, 2: 3}
N_MERGED = 4

# 聚合使用的参数 (与论文一致)
# Cavity_Volume → 总和; 其余 → 均值
AGGREGATES = {
    'Cavity_Volume_All':  ('sum',  'Cavity_Volume'),
    'Volume_Fill_Avg':    ('mean', 'Organoids_Volume_Fill'),
    'Volume_Avg':         ('mean', 'Organoids_Volume'),
    'Surface_Avg':        ('mean', 'Organoids_Surface'),
    'Long_Axis_Avg':      ('mean', 'LongAxis'),
    'Short_Axis_Avg':     ('mean', 'ShortAxis'),
    'Cyst_Thick_Avg':     ('mean', 'Wall_Thickness'),
    'Sphericity_Avg':     ('mean', 'Sphericity'),
    'Scatt_Mean_Avg':     ('mean', 'Scatt_Mean'),
    'Scatt_STD_Avg':      ('mean', 'Scatt_STD'),
    'CavityNum_Avg':      ('mean', 'CavityNum'),
}


def compute_well_stats(df_well):
    cluster = df_well['Cluster'].values
    result = {}
    for c in range(N_MERGED):
        mask = cluster == c
        n = mask.sum()
        result[f'Number_{c+1}'] = int(n)
        if n > 0:
            sub = df_well.loc[mask]
            for out_col, (agg_func, src_col) in AGGREGATES.items():
                vals = sub[src_col].values
                if agg_func == 'sum':
                    result[f'{out_col}_{c+1}'] = float(np.sum(vals))
                else:
                    result[f'{out_col}_{c+1}'] = float(np.mean(vals))
        else:
            for out_col in AGGREGATES:
                result[f'{out_col}_{c+1}'] = 0.0
    return result


def load_measure_files(measure_dir):
    dfs = {}
    for fname in os.listdir(measure_dir):
        if not fname.endswith('.xlsx'):
            continue
        fpath = os.path.join(measure_dir, fname)
        df = pd.read_excel(fpath)
        dfs[fname] = df
    return dfs


def main():
    with open(os.path.join(MODEL_DIR, 'scaler_k6.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    with open(os.path.join(MODEL_DIR, 'kmeans_k6.pkl'), 'rb') as f:
        kmeans = pickle.load(f)

    all_rows = []
    well_count = 0

    for measure_dir, day_label in [(DAY3_MEASURE_DIR, '0701'), (DAY5_MEASURE_DIR, '0703')]:
        dfs = load_measure_files(measure_dir)
        for fname, df in sorted(dfs.items()):
            well_name = fname.replace('.xlsx', '')
            if all(col in df.columns for col in FEATURES):
                X = df[FEATURES].fillna(0).values
                X_std = scaler.transform(X)
                labels_k6 = kmeans.predict(X_std)
                df['Cluster'] = [NUMERIC_MAP.get(l, -1) for l in labels_k6]
            else:
                missing = [c for c in FEATURES if c not in df.columns]
                print(f'  [SKIP] {fname}: missing {missing}')
                continue

            stats = compute_well_stats(df)
            stats['Name'] = f'{well_name}_{day_label}'
            all_rows.append(stats)
            well_count += 1

    df_analysis = pd.DataFrame(all_rows)
    # 按论文格式排序列: Name + Number_{1..4} + 各参数_{1..4}
    param_cols = []
    for c in range(1, N_MERGED + 1):
        param_cols.append(f'Number_{c}')
    for out_col in AGGREGATES:
        for c in range(1, N_MERGED + 1):
            param_cols.append(f'{out_col}_{c}')
    available = ['Name'] + [p for p in param_cols if p in df_analysis.columns]
    df_analysis = df_analysis[available]

    # 按 Well_ID 排序
    def sort_key(name):
        parts = str(name).split('_')
        well = parts[0]
        day = parts[-1] if len(parts) > 1 else '0000'
        return (well, day)
    df_analysis = df_analysis.sort_values('Name', key=lambda s: s.map(sort_key)).reset_index(drop=True)

    out_path = os.path.join(OUTPUT_DIR, 'FXN_2023_Analysis.xlsx')
    df_analysis.to_excel(out_path, index=False)
    print(f'Analysis table saved: {out_path}')
    print(f'  Wells: {well_count}, Features: {len(df_analysis.columns)-1}')
    print('Step 2 complete.')


if __name__ == '__main__':
    main()