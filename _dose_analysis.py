import sys, os, glob, warnings
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')

import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, linregress
from cluster_utils import RAW_FEATURES, load_model_package

# ===== Config =====
DATA_FOLDERS = [
    os.path.join('Data', 'nnUNet_FXN_2023', 'FXN_0701', 'measure_excel'),
    os.path.join('Data', 'nnUNet_FXN_2023', 'FXN_0703', 'measure_excel'),
]

MODEL_PATHS = {
    'KMeans-10d': 'model/Kmeans-scatt.pickle',
    'GMM-10d': 'model/GMM-scatt.pickle',
    'KMeans-5d': 'model/Kmeans-5d.pickle',
    'GMM-5d': 'model/GMM-5d.pickle',
}

WELL_CONC_MAP = {
    'E11': 0, 'F2': 0, 'F3': 0, 'F4': 0, 'F5': 0, 'F6': 0, 'F7': 0, 'F8': 0, 'F9': 0, 'F10': 0, 'F11': 0,
    'B11': 0, 'C11': 0, 'D11': 0,
    'B2': 20, 'B3': 20, 'B4': 20, 'C2': 20, 'C3': 20, 'C4': 20,
    'B5': 40, 'B6': 40, 'B7': 40, 'C5': 40, 'C6': 40, 'C7': 40,
    'B8': 80, 'B9': 80, 'B10': 80, 'C8': 80, 'C9': 80, 'C10': 80,
}

WELL_VOLUME_CHANGE = {0: 213.3, 20: 47.9, 40: 9.1, 80: 13.7}

REPORTS_DIR = 'reports'
FIGURES_DIR = os.path.join(REPORTS_DIR, 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

# ===== Functions =====
def infer_concentration(well_name):
    prefix = well_name.split('_')[0].upper()
    return WELL_CONC_MAP.get(prefix, -1)

def load_all_data():
    df_list = []
    for folder in DATA_FOLDERS:
        if not os.path.exists(folder):
            continue
        for fp in glob.glob(os.path.join(folder, '*.xlsx')):
            df = pd.read_excel(fp)
            df['_well'] = os.path.basename(fp).replace('.xlsx', '')
            df_list.append(df)
    df = pd.concat(df_list, ignore_index=True)
    before = len(df)
    df = df.dropna(subset=RAW_FEATURES)
    after = len(df)
    print(f"Loaded {after} rows (dropped {before - after} NaN)")
    return df

def get_model_labels(df, model_path):
    if not os.path.exists(model_path):
        return None
    try:
        pkg = load_model_package(model_path)
        features = pkg.get('feature_names', RAW_FEATURES)
        preprocessor = pkg.get('preprocessor')
        scaler = pkg.get('scaler')
        model = pkg.get('kmeans') or pkg.get('gmm') or pkg['model']
        raw_to_final = pkg.get('raw_to_final', {})

        if preprocessor is not None:
            if not hasattr(preprocessor, 'mode'):
                preprocessor.mode = 'full'
            if not hasattr(preprocessor, '_log_features'):
                if preprocessor.mode == 'reduced':
                    preprocessor._log_features = ['Organoids_Volume_Fill']
                    preprocessor._processed_features = ['Organoids_Volume_Fill', 'Sphericity', 'Scatt_Mean', 'Scatt_STD', 'Cavity_Ratio']
                else:
                    preprocessor._log_features = ['Organoids_Volume_Fill', 'Organoids_Surface', 'LongAxis', 'ShortAxis', 'Cavity_Volume']
                    preprocessor._processed_features = ['Organoids_Volume_Fill', 'Organoids_Surface', 'LongAxis', 'ShortAxis', 'Cavity_Volume', 'Sphericity', 'Scatt_Mean', 'Scatt_STD', 'CavityNum', 'Cavity_Ratio']
            X = preprocessor.transform(df[features])
        elif scaler is not None:
            X = scaler.transform(df[features])
        else:
            return None

        raw_labels = model.predict(X)
        if raw_to_final:
            return np.array([raw_to_final.get(l, l) for l in raw_labels])
        return raw_labels
    except Exception as e:
        print(f"  [WARN] {model_path}: {e}")
        return None

def compute_stats(df, labels):
    df = df.copy()
    df['Cluster'] = labels
    df['_conc'] = df['_well'].apply(infer_concentration)

    well_stats = df.groupby('_well').agg(
        Conc=('_conc', 'first'),
        Total=('Cluster', 'size'),
        Red=('Cluster', lambda s: (s == 0).sum()),
        Yellow=('Cluster', lambda s: (s == 1).sum()),
        Blue=('Cluster', lambda s: (s == 3).sum()),
        Volume_Mean=('Organoids_Volume_Fill', 'mean'),
    )
    well_stats['Healthy'] = well_stats['Red'] + well_stats['Yellow']
    well_stats['Healthy_Fraction'] = well_stats['Healthy'] / well_stats['Total']
    well_stats['Blue_Fraction'] = well_stats['Blue'] / well_stats['Total']

    conc_stats = well_stats.groupby('Conc').agg(
        N_Wells=('Blue', 'size'),
        Total_Organoids=('Total', 'sum'),
        Healthy_Mean=('Healthy_Fraction', 'mean'),
        Healthy_Std=('Healthy_Fraction', 'std'),
        Blue_Mean=('Blue_Fraction', 'mean'),
        Blue_Std=('Blue_Fraction', 'std'),
        Volume_Mean=('Volume_Mean', 'mean'),
    ).reset_index()

    return well_stats, conc_stats

# ===== Main =====
print("=" * 60)
print("Dose Response Analysis (Plan C: Healthy Fraction)")
print("=" * 60)

df = load_all_data()

all_conc = {}
all_well = {}

for name, path in MODEL_PATHS.items():
    print(f"\n{name}...")
    labels = get_model_labels(df, path)
    if labels is None:
        print("  [SKIP]")
        continue
    well_stats, conc_stats = compute_stats(df, labels)
    all_conc[name] = conc_stats
    all_well[name] = well_stats

    print(f"  {'Conc':>5s}  {'Healthy%':>10s}  {'Blue%':>10s}  {'Wells':>6s}  {'Organoids':>10s}")
    for _, row in conc_stats.iterrows():
        print(f"  {int(row['Conc']):5d} uM  {row['Healthy_Mean']*100:9.1f}%  {row['Blue_Mean']*100:9.1f}%  {int(row['N_Wells']):6d}  {int(row['Total_Organoids']):10d}")

# ===== Plots =====
print("\n--- Generating plots ---")

# Healthy dose response (4 subplots)
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
colors = {'KMeans-10d': '#1f77b4', 'GMM-10d': '#ff7f0e', 'KMeans-5d': '#2ca02c', 'GMM-5d': '#d62728'}
for idx, (name, cs) in enumerate(all_conc.items()):
    ax = axes[idx // 2, idx % 2]
    cs = cs[cs['Conc'] >= 0].sort_values('Conc')
    x = cs['Conc'].values
    y = cs['Healthy_Mean'].values
    yerr = cs['Healthy_Std'].fillna(0).values
    ax.errorbar(x, y * 100, yerr=yerr * 100, fmt='o-', color=colors[name], capsize=5, lw=2, markersize=8)
    ax.set_xlabel('Icaritin (uM)')
    ax.set_ylabel('Healthy (Red+Yellow) %')
    ax.set_title(name, fontweight='bold')
    ax.set_xticks([0, 20, 40, 80])
    ax.set_xlim(-5, 95)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for conc, vol in WELL_VOLUME_CHANGE.items():
        if conc in x:
            idx_x = list(x).index(conc)
            ax.annotate(f'+{vol}%', xy=(conc, y[idx_x] * 100), xytext=(0, 12),
                       textcoords='offset points', ha='center', fontsize=8, color='gray')
    if len(x) >= 3:
        rho, pval = spearmanr(x, y)
        lr = linregress(x, y)
        ax.axline((x[0], lr.intercept + lr.slope * x[0]), slope=lr.slope, color=colors[name], linestyle='--', alpha=0.5)
        sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''
        ax.text(0.95, 0.05, f"rho={rho:.3f}{sig}\np={pval:.3f}", transform=ax.transAxes, ha='right', va='bottom',
               fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'healthy_dose_response.png'), dpi=300, bbox_inches='tight')
plt.close()
print("  Saved healthy_dose_response.png")

# Healthy combined plot
fig, ax1 = plt.subplots(figsize=(10, 6))
for name, cs in all_conc.items():
    cs = cs[cs['Conc'] >= 0].sort_values('Conc')
    ax1.plot(cs['Conc'].values, cs['Healthy_Mean'].values * 100, 'o-', color=colors[name], lw=2, markersize=8, label=name)
ax1.set_xlabel('Icaritin (uM)', fontsize=12)
ax1.set_ylabel('Healthy (Red+Yellow) %', fontsize=12)
ax1.set_xticks([0, 20, 40, 80])
ax1.set_xlim(-5, 95)
ax1.spines['top'].set_visible(False)
ax2 = ax1.twinx()
vol_x = list(WELL_VOLUME_CHANGE.keys())
vol_y = list(WELL_VOLUME_CHANGE.values())
ax2.bar(vol_x, vol_y, width=8, alpha=0.2, color='gray', label='Well volume change')
ax2.set_ylabel('Well Volume Change %', fontsize=12, color='gray')
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower left', fontsize=9)
ax1.set_title('Healthy (Red+Yellow) Organoid Fraction vs Icaritin\n(4 Models + Well Volume Change)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'healthy_dose_response_combined.png'), dpi=300, bbox_inches='tight')
plt.close()
print("  Saved healthy_dose_response_combined.png")

# ===== Summary =====
print("\n" + "=" * 70)
print("=== Healthy (Red+Yellow) Dose Response ===")
print("=" * 70)
rows = []
for name, cs in all_conc.items():
    cs = cs[cs['Conc'] >= 0].sort_values('Conc')
    x = cs['Conc'].values
    y = cs['Healthy_Mean'].values
    if len(x) >= 3:
        rho, pval = spearmanr(x, y)
        lr = linregress(x, y)
        direction = 'DOWN (expected)' if rho < 0 else 'UP (unexpected)'
        rows.append({
            'Model': name,
            'Control': f"{y[0]*100:.1f}%",
            '20uM': f"{y[1]*100:.1f}%" if len(y) > 1 else 'N/A',
            '40uM': f"{y[2]*100:.1f}%" if len(y) > 2 else 'N/A',
            '80uM': f"{y[3]*100:.1f}%" if len(y) > 3 else 'N/A',
            'Spearman_Rho': f"{rho:.3f}",
            'P_Value': f"{pval:.3f}",
            'R2': f"{lr.rvalue**2:.3f}",
            'Direction': direction,
        })
df_summary = pd.DataFrame(rows)
print(df_summary.to_string(index=False))

print("\n" + "=" * 70)
print("=== Blue (Damaged) Dose Response (auxiliary) ===")
print("=" * 70)
blue_rows = []
for name, cs in all_conc.items():
    cs = cs[cs['Conc'] >= 0].sort_values('Conc')
    x = cs['Conc'].values
    y = cs['Blue_Mean'].values
    if len(x) >= 3:
        rho, pval = spearmanr(x, y)
        blue_rows.append({
            'Model': name,
            'Control': f"{y[0]*100:.1f}%",
            '20uM': f"{y[1]*100:.1f}%" if len(y) > 1 else 'N/A',
            '40uM': f"{y[2]*100:.1f}%" if len(y) > 2 else 'N/A',
            '80uM': f"{y[3]*100:.1f}%" if len(y) > 3 else 'N/A',
            'Spearman_Rho': f"{rho:.3f}",
            'P_Value': f"{pval:.3f}",
        })
df_blue = pd.DataFrame(blue_rows)
print(df_blue.to_string(index=False))

# ===== Save Excel =====
print("\n--- Saving Excel ---")
excel_path = os.path.join(REPORTS_DIR, 'dose_response_analysis.xlsx')
with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
    df_summary.to_excel(writer, sheet_name='Healthy_Summary', index=False)
    df_blue.to_excel(writer, sheet_name='Blue_Summary', index=False)
    for name, cs in all_conc.items():
        cs.to_excel(writer, sheet_name=f'{name.replace("-", "_")}_Conc', index=False)
    for name, ws in all_well.items():
        ws.to_excel(writer, sheet_name=f'{name.replace("-", "_")}_Well', index=False)
print(f"  Saved {excel_path}")

print("\n[Done] Plan C analysis complete.")