"""
Paper Figures Generator

Generates publication-quality figures for the organoid phenotyping study.

Output: reports/figures/
  fig1_clustering_overview.png    - PCA 2D projection + per-cluster feature heatmap
  fig2_score_atp_bridge.png       - 3-panel: Score~ATP / Healthy%~Score / Healthy%~ATP
  fig3_dose_response.png          - Healthy% & Blue% vs concentration
  fig4_cluster_composition.png    - Stacked bar: cluster proportions per dose
  fig5_feature_radar.png          - Radar chart: feature profiles per cluster
"""
import sys, os, glob, warnings
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Patch
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

from cluster_utils import RAW_FEATURES, load_model_package

# ============================================================
# Config
# ============================================================
DATA_FOLDERS = [
    os.path.join('Data', 'nnUNet_FXN_2023', 'FXN_0701', 'measure_excel'),
    os.path.join('Data', 'nnUNet_FXN_2023', 'FXN_0703', 'measure_excel'),
]
ATP_FILE = os.path.join('Data', 'nnUNet_FXN_2023', 'nnUNet_Analysis.xlsx')
MODEL_PATH = 'model/Kmeans-scatt.pickle'

PCA_FEATURES = [
    'Cavity_Volume_All_1', 'Cyst_Thick_Avg_3', 'Long_Axis_Avg_4', 'Number_2', 'Number_3',
    'Roughness_All',
    'Scatt_Mean_Avg_3', 'Scatt_Mean_Avg_4',
    'Short_Axis_Avg_2', 'Short_Axis_Avg_3', 'Short_Axis_Avg_4',
    'Surface_Avg_1', 'Surface_Avg_2', 'Surface_Avg_3', 'Volume_Fill_Avg_2'
]

CONC_MAP = {
    'E11': 0, 'F2': 0, 'F3': 0, 'F4': 0, 'F5': 0, 'F6': 0, 'F7': 0, 'F8': 0, 'F9': 0, 'F10': 0, 'F11': 0,
    'B11': 0, 'C11': 0, 'D11': 0,
    'B2': 20, 'B3': 20, 'B4': 20, 'C2': 20, 'C3': 20, 'C4': 20,
    'B5': 40, 'B6': 40, 'B7': 40, 'C5': 40, 'C6': 40, 'C7': 40,
    'B8': 80, 'B9': 80, 'B10': 80, 'C8': 80, 'C9': 80, 'C10': 80,
}

CLUSTER_COLORS = {0: '#E74C3C', 1: '#F1C40F', 2: '#2ECC71', 3: '#3498DB'}
CLUSTER_NAMES = {0: 'Red (Healthy)', 1: 'Yellow (Healthy)', 2: 'Green (Intermediate)', 3: 'Blue (Damaged)'}
CLUSTER_SHORT = {0: 'Red', 1: 'Yellow', 2: 'Green', 3: 'Blue'}
HEALTHY_CLUSTERS = {0, 1}

FIGURES_DIR = os.path.join('reports', 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

# Global style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 10,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

# ============================================================
# Data Loading
# ============================================================
def load_all_data():
    print("Loading data...")
    df_list = []
    for folder in DATA_FOLDERS:
        if not os.path.exists(folder):
            continue
        for fp in glob.glob(os.path.join(folder, '*.xlsx')):
            df = pd.read_excel(fp)
            df['_well'] = os.path.basename(fp).replace('.xlsx', '')
            df_list.append(df)
    df = pd.concat(df_list, ignore_index=True)
    df = df.dropna(subset=RAW_FEATURES)
    print(f"  {len(df)} organoids from {len(df_list)} files")

    pkg = load_model_package(MODEL_PATH)
    preprocessor = pkg.get('preprocessor')
    scaler_km = pkg.get('scaler')
    model = pkg.get('kmeans') or pkg.get('gmm') or pkg.get('model')
    raw_to_final = pkg.get('raw_to_final', {})

    if preprocessor is not None:
        if not hasattr(preprocessor, 'mode'):
            preprocessor.mode = 'full'
        if not hasattr(preprocessor, '_log_features'):
            preprocessor._log_features = ['Organoids_Volume_Fill', 'Organoids_Surface',
                                           'LongAxis', 'ShortAxis', 'Cavity_Volume']
            preprocessor._processed_features = ['Organoids_Volume_Fill', 'Organoids_Surface',
                                                 'LongAxis', 'ShortAxis', 'Cavity_Volume',
                                                 'Sphericity', 'Scatt_Mean', 'Scatt_STD',
                                                 'CavityNum', 'Cavity_Ratio']
        X = preprocessor.transform(df[RAW_FEATURES])
    elif scaler_km is not None:
        X = scaler_km.transform(df[RAW_FEATURES])
    else:
        raise RuntimeError("No preprocessor or scaler in model package")

    raw_labels = model.predict(X)
    labels = np.array([raw_to_final.get(l, l) for l in raw_labels])
    df['Cluster'] = labels

    df['_conc'] = df['_well'].apply(lambda w: CONC_MAP.get(str(w).split('_')[0], -1))
    df['_day'] = df['_well'].apply(lambda w: str(w).split('_')[1] if len(str(w).split('_')) > 1 else '??')

    return df

def load_atp_pca():
    print("Loading ATP + PCA Score...")
    df_a = pd.read_excel(ATP_FILE)
    X = df_a[PCA_FEATURES].fillna(0).values
    X_s = StandardScaler().fit_transform(X)
    pca = PCA(n_components=4, random_state=42)
    X_p = pca.fit_transform(X_s)
    weights = pca.explained_variance_ratio_ / np.sum(pca.explained_variance_ratio_)
    df_a['PCA_Score'] = np.dot(X_p, weights)
    df_a['_match_key'] = df_a['Name'].astype(str).str.strip()
    return df_a, pca, weights

# ============================================================
# Figure 1: Clustering Overview
# ============================================================
def fig1_clustering(df):
    print("\n[Fig 1] Clustering Overview...")
    fig = plt.figure(figsize=(16, 7))

    # Panel A: t-SNE 2D projection
    ax1 = fig.add_subplot(1, 2, 1)
    X = df[RAW_FEATURES].values
    X_s = StandardScaler().fit_transform(X)

    tsne = TSNE(n_components=2, random_state=42, perplexity=50, max_iter=1000)
    X_2d = tsne.fit_transform(X_s)

    for c in sorted(CLUSTER_COLORS.keys()):
        mask = df['Cluster'] == c
        ax1.scatter(X_2d[mask, 0], X_2d[mask, 1], c=CLUSTER_COLORS[c],
                    label=CLUSTER_SHORT[c], s=2, alpha=0.6, rasterized=True)
    ax1.set_xlabel('t-SNE 1')
    ax1.set_ylabel('t-SNE 2')
    ax1.set_title('A   t-SNE Projection of Organoids', fontweight='bold', loc='left')
    ax1.legend(markerscale=5, frameon=True, fancybox=True, loc='upper right')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Panel B: Feature heatmap per cluster
    ax2 = fig.add_subplot(1, 2, 2)
    feature_labels = ['Volume', 'Surface', 'LongAxis', 'ShortAxis', 'CavityVol',
                      'Thickness', 'Sphericity', 'CavityNum', 'ScattMean', 'ScattSTD']

    profile = df.groupby('Cluster')[RAW_FEATURES].mean()
    profile_z = (profile - profile.mean()) / profile.std()

    im = ax2.imshow(profile_z.values, cmap='RdBu_r', aspect='auto', vmin=-2, vmax=2)
    ax2.set_xticks(range(len(feature_labels)))
    ax2.set_xticklabels(feature_labels, rotation=45, ha='right', fontsize=8)
    ax2.set_yticks(range(4))
    ax2.set_yticklabels([CLUSTER_SHORT[i] for i in range(4)])
    ax2.set_title('B   Cluster Feature Profiles (Z-score)', fontweight='bold', loc='left')

    for i in range(4):
        for j in range(len(feature_labels)):
            val = profile_z.values[i, j]
            color = 'white' if abs(val) > 1.2 else 'black'
            ax2.text(j, i, f'{val:.1f}', ha='center', va='center', fontsize=6, color=color)

    cbar = plt.colorbar(im, ax=ax2, shrink=0.8, pad=0.02)
    cbar.set_label('Z-score')

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'fig1_clustering_overview.png'))
    plt.close()
    print("  Saved: fig1_clustering_overview.png")

# ============================================================
# Figure 2: Score-ATP Bridge
# ============================================================
def fig2_bridge(df, df_atp):
    print("\n[Fig 2] Score-ATP Bridge...")

    well_stats = df.groupby('_well').agg(
        Total=('Cluster', 'size'),
        Red=('Cluster', lambda s: (s == 0).sum()),
        Yellow=('Cluster', lambda s: (s == 1).sum()),
    ).reset_index()
    well_stats['Healthy_Fraction'] = (well_stats['Red'] + well_stats['Yellow']) / well_stats['Total']
    well_stats['_match_key'] = well_stats['_well'].apply(
        lambda x: '_'.join(str(x).split('_')[:2])
    )

    merged = df_atp[['_match_key', 'PCA_Score', 'ATP']].merge(
        well_stats[['_match_key', 'Healthy_Fraction']], on='_match_key', how='inner'
    )

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # Panel A: Score ~ ATP
    ax = axes[0]
    x = merged['PCA_Score'].values
    y = merged['ATP'].values / 1e6
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]

    slope, intercept, r, p, _ = stats.linregress(x, y)
    r2 = r ** 2
    spearman_rho, spearman_p = spearmanr(x, y)

    ax.scatter(x, y, c='#2C3E50', s=60, edgecolors='white', linewidth=0.5, zorder=3)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, intercept + slope * x_line, '--', color='#E74C3C', lw=2, zorder=2)
    ax.fill_between(x_line, intercept + slope * x_line - stats.sem(y - (intercept + slope * x)),
                     intercept + slope * x_line + stats.sem(y - (intercept + slope * x)),
                     alpha=0.15, color='#E74C3C')

    ax.set_xlabel('PCA Composite Score')
    ax.set_ylabel('ATP (×10\u2076)')
    ax.set_title('A   Score vs ATP', fontweight='bold', loc='left')
    ax.text(0.05, 0.95, f'R\u00b2 = {r2:.3f}\nSpearman \u03c1 = {spearman_rho:.3f}, p = {spearman_p:.3f}',
            transform=ax.transAxes, va='top', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Panel B: Healthy% ~ Score
    ax = axes[1]
    x = merged['PCA_Score'].values
    y = merged['Healthy_Fraction'].values * 100
    slope, intercept, r, p, _ = stats.linregress(x, y)
    r2 = r ** 2

    ax.scatter(x, y, c='#27AE60', s=60, edgecolors='white', linewidth=0.5, zorder=3)
    ax.plot(x_line, intercept + slope * x_line, '--', color='#E74C3C', lw=2, zorder=2)
    ax.set_xlabel('PCA Composite Score')
    ax.set_ylabel('Healthy Fraction (%)')
    ax.set_title('B   Healthy% vs Score (Bridge)', fontweight='bold', loc='left')
    ax.text(0.05, 0.95, f'R\u00b2 = {r2:.3f}\np = {p:.2e}',
            transform=ax.transAxes, va='top', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Panel C: Healthy% ~ ATP
    ax = axes[2]
    x = merged['Healthy_Fraction'].values * 100
    y = merged['ATP'].values / 1e6
    slope, intercept, r, p, _ = stats.linregress(x, y)
    r2 = r ** 2
    spearman_rho, spearman_p = spearmanr(x, y)

    ax.scatter(x, y, c='#8E44AD', s=60, edgecolors='white', linewidth=0.5, zorder=3)
    x_line2 = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line2, intercept + slope * x_line2, '--', color='#E74C3C', lw=2, zorder=2)
    ax.set_xlabel('Healthy Fraction (%)')
    ax.set_ylabel('ATP (×10\u2076)')
    ax.set_title('C   Healthy% vs ATP (Direct)', fontweight='bold', loc='left')
    ax.text(0.05, 0.95, f'R\u00b2 = {r2:.3f}\nSpearman \u03c1 = {spearman_rho:.3f}, p = {spearman_p:.3f}',
            transform=ax.transAxes, va='top', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'fig2_score_atp_bridge.png'))
    plt.close()
    print("  Saved: fig2_score_atp_bridge.png")

# ============================================================
# Figure 3: Dose Response
# ============================================================
def fig3_dose_response(df):
    print("\n[Fig 3] Dose Response...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    concentrations = [0, 20, 40, 80]

    # Panel A: Healthy% per concentration
    ax = axes[0]
    healthy_data = []
    for conc in concentrations:
        mask = df['_conc'] == conc
        well_healthy = df[mask].groupby('_well').apply(
            lambda g: (g['Cluster'].isin(HEALTHY_CLUSTERS).sum()) / len(g)
        )
        healthy_data.append(well_healthy.values)

    bp = ax.boxplot(healthy_data, positions=concentrations, widths=8,
                    patch_artist=True, showfliers=False,
                    medianprops=dict(color='black', lw=2))
    colors = ['#E8F8F5', '#D1F2EB', '#A3E4D7', '#76D7C4']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    for i, (conc, vals) in enumerate(zip(concentrations, healthy_data)):
        jitter = np.random.normal(0, 1.5, len(vals))
        ax.scatter([conc] * len(vals) + jitter, vals, s=30, alpha=0.7,
                   c='#1ABC9C', edgecolors='white', linewidth=0.5, zorder=3)

    # Trend line
    means = [np.mean(v) for v in healthy_data]
    ax.plot(concentrations, means, '-o', color='#E74C3C', lw=2.5, markersize=8, zorder=4)

    ax.set_xlabel('Concentration (μM)')
    ax.set_ylabel('Healthy Fraction')
    ax.set_title('A   Healthy Fraction vs Dose', fontweight='bold', loc='left')
    ax.set_xticks(concentrations)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Panel B: Blue% per concentration
    ax = axes[1]
    blue_data = []
    for conc in concentrations:
        mask = df['_conc'] == conc
        well_blue = df[mask].groupby('_well').apply(
            lambda g: (g['Cluster'] == 3).sum() / len(g)
        )
        blue_data.append(well_blue.values)

    bp = ax.boxplot(blue_data, positions=concentrations, widths=8,
                    patch_artist=True, showfliers=False,
                    medianprops=dict(color='black', lw=2))
    colors_b = ['#EBF5FB', '#D6EAF8', '#AED6F1', '#85C1E9']
    for patch, color in zip(bp['boxes'], colors_b):
        patch.set_facecolor(color)

    for i, (conc, vals) in enumerate(zip(concentrations, blue_data)):
        jitter = np.random.normal(0, 1.5, len(vals))
        ax.scatter([conc] * len(vals) + jitter, vals, s=30, alpha=0.7,
                   c='#3498DB', edgecolors='white', linewidth=0.5, zorder=3)

    means_b = [np.mean(v) for v in blue_data]
    ax.plot(concentrations, means_b, '-o', color='#E74C3C', lw=2.5, markersize=8, zorder=4)

    ax.set_xlabel('Concentration (μM)')
    ax.set_ylabel('Blue (Damaged) Fraction')
    ax.set_title('B   Damaged Fraction vs Dose', fontweight='bold', loc='left')
    ax.set_xticks(concentrations)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Panel C: Stacked bar of cluster composition
    ax = axes[2]
    comp = df.groupby(['_conc', 'Cluster']).size().unstack(fill_value=0)
    comp_pct = comp.div(comp.sum(axis=1), axis=0)

    bar_width = 12
    bottom = np.zeros(len(concentrations))
    for c in [0, 1, 2, 3]:
        vals = [comp_pct.loc[conc, c] if conc in comp_pct.index and c in comp_pct.columns else 0
                for conc in concentrations]
        ax.bar(concentrations, vals, bar_width, bottom=bottom,
               color=CLUSTER_COLORS[c], label=CLUSTER_SHORT[c], edgecolor='white', linewidth=0.5)
        bottom += np.array(vals)

    ax.set_xlabel('Concentration (μM)')
    ax.set_ylabel('Proportion')
    ax.set_title('C   Cluster Composition by Dose', fontweight='bold', loc='left')
    ax.set_xticks(concentrations)
    ax.legend(loc='upper right', frameon=True, fancybox=True)
    ax.set_ylim(0, 1.05)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'fig3_dose_response.png'))
    plt.close()
    print("  Saved: fig3_dose_response.png")

# ============================================================
# Figure 4: Cluster Composition by Day
# ============================================================
def fig4_day_comparison(df):
    print("\n[Fig 4] Day 3 vs Day 5 Comparison...")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    for idx, day in enumerate(['0701', '0703']):
        ax = axes[idx]
        day_label = 'Day 3' if day == '0701' else 'Day 5'
        day_df = df[df['_day'] == day]

        concentrations = [0, 20, 40, 80]
        comp = day_df.groupby(['_conc', 'Cluster']).size().unstack(fill_value=0)
        comp_pct = comp.div(comp.sum(axis=1), axis=0)

        bar_width = 12
        bottom = np.zeros(len(concentrations))
        for c in [0, 1, 2, 3]:
            vals = [comp_pct.loc[conc, c] if conc in comp_pct.index and c in comp_pct.columns else 0
                    for conc in concentrations]
            ax.bar(concentrations, vals, bar_width, bottom=bottom,
                   color=CLUSTER_COLORS[c], label=CLUSTER_SHORT[c] if idx == 0 else None,
                   edgecolor='white', linewidth=0.5)
            bottom += np.array(vals)

        ax.set_xlabel('Concentration (μM)')
        ax.set_ylabel('Proportion')
        ax.set_title(f'{day_label}', fontweight='bold', loc='left')
        ax.set_xticks(concentrations)
        ax.set_ylim(0, 1.05)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    handles = [Patch(color=CLUSTER_COLORS[c], label=CLUSTER_SHORT[c]) for c in [0, 1, 2, 3]]
    fig.legend(handles=handles, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.02),
               frameon=False, fontsize=10)

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'fig4_day_comparison.png'))
    plt.close()
    print("  Saved: fig4_day_comparison.png")

# ============================================================
# Figure 5: Per-Cluster Feature Radar
# ============================================================
def fig5_radar(df):
    print("\n[Fig 5] Feature Radar...")

    feature_labels = ['Volume', 'Surface', 'LongAxis', 'ShortAxis', 'CavityVol',
                      'WallThick', 'Sphericity', 'CavityNum', 'ScattMean', 'ScattSTD']

    profile = df.groupby('Cluster')[RAW_FEATURES].mean()
    profile_norm = (profile - profile.min()) / (profile.max() - profile.min())

    angles = np.linspace(0, 2 * np.pi, len(feature_labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for c in [0, 1, 2, 3]:
        values = profile_norm.loc[c].values.tolist()
        values += values[:1]
        ax.fill(angles, values, color=CLUSTER_COLORS[c], alpha=0.15)
        ax.plot(angles, values, color=CLUSTER_COLORS[c], lw=2, label=CLUSTER_SHORT[c])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(feature_labels, fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['0.25', '0.50', '0.75', '1.00'], fontsize=7)
    ax.set_title('Per-Cluster Morphological Feature Profiles', fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), frameon=True, fancybox=True)

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'fig5_feature_radar.png'))
    plt.close()
    print("  Saved: fig5_feature_radar.png")

# ============================================================
# Figure 6: PCA Loadings
# ============================================================
def fig6_pca_loadings(pca, weights):
    print("\n[Fig 6] PCA Loadings...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel A: Scree plot
    ax = axes[0]
    ev = pca.explained_variance_ratio_
    cum = np.cumsum(ev)
    x = range(1, len(ev) + 1)

    ax.bar(x, ev * 100, color='#3498DB', alpha=0.7, edgecolor='white', label='Individual')
    ax.plot(x, cum * 100, 'o-', color='#E74C3C', lw=2, markersize=8, label='Cumulative')
    ax.axhline(y=80, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Explained Variance (%)')
    ax.set_title('A   PCA Scree Plot', fontweight='bold', loc='left')
    ax.set_xticks(x)
    ax.legend(frameon=True, fancybox=True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Panel B: PC1 loadings
    ax = axes[1]
    final_coef = np.dot(pca.components_.T, weights)
    idx = np.argsort(np.abs(final_coef))[::-1]
    features_sorted = [PCA_FEATURES[i] for i in idx]
    coef_sorted = final_coef[idx]

    short_names = [f.replace('_Avg_', '_').replace('_All', '') for f in features_sorted]

    colors = ['#E74C3C' if c > 0 else '#3498DB' for c in coef_sorted]
    ax.barh(range(len(coef_sorted)), coef_sorted, color=colors, edgecolor='white', height=0.7)
    ax.set_yticks(range(len(coef_sorted)))
    ax.set_yticklabels(short_names, fontsize=8)
    ax.set_xlabel('Coefficient')
    ax.set_title('B   Composite Score Feature Weights', fontweight='bold', loc='left')
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.invert_yaxis()

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'fig6_pca_loadings.png'))
    plt.close()
    print("  Saved: fig6_pca_loadings.png")

# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    df = load_all_data()
    df_atp, pca_model, pca_weights = load_atp_pca()

    fig1_clustering(df)
    fig2_bridge(df, df_atp)
    fig3_dose_response(df)
    fig4_day_comparison(df)
    fig5_radar(df)
    fig6_pca_loadings(pca_model, pca_weights)

    print(f"\n{'='*60}")
    print("All figures saved to reports/figures/")
    print(f"{'='*60}")