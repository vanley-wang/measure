import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', '..')))

"""
Publication-ready Figure: 10D vs 5D Ablation Study

Generates a multi-panel figure suitable for thesis/paper supplementary materials:
  - Panel A: PCA scree comparison (10D vs 5D)
  - Panel B: VIF comparison (bar chart)
  - Panel C: Cross-ARI heatmap
  - Panel D: Biological criteria radar chart
  - Panel E: Blue dose-response curves (4 models)
  - Panel F: Cluster centroid heatmap (5D only)

Output:
  reports/figures/figure_10d_vs_5d_ablation.png
"""
import os
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from scipy.stats import spearmanr

from cluster_utils import (
    RAW_FEATURES, REDUCED_RAW_FEATURES, Preprocessor, load_model_package
)

warnings.filterwarnings('ignore')

# ================= 配置 =================
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
FIGURES_DIR = 'reports/figures'
os.makedirs(FIGURES_DIR, exist_ok=True)

# 10D VIF values (from earlier analysis)
VIF_10D = {
    'Organoids_Volume_Fill': 1103.8,
    'Organoids_Surface': 1550.9,
    'LongAxis': 23.8,
    'ShortAxis': 16.4,
    'Sphericity': 43.9,
    'Scatt_Mean': 1.6,
    'Scatt_STD': 2.6,
    'CavityNum': 1.8,
    'Cavity_Ratio': 4.8,
}
VIF_5D = {
    'Organoids_Volume_Fill': 5.5,
    'Sphericity': 4.9,
    'Scatt_Mean': 1.6,
    'Scatt_STD': 2.2,
    'Cavity_Ratio': 1.5,
}

# Well concentration map
WELL_CONC_MAP = {
    'E11': 0, 'F2': 0, 'F3': 0, 'F4': 0, 'F5': 0, 'F6': 0, 'F7': 0, 'F8': 0, 'F9': 0, 'F10': 0, 'F11': 0,
    'B11': 0, 'C11': 0, 'D11': 0,
    'B2': 20, 'B3': 20, 'B4': 20, 'C2': 20, 'C3': 20, 'C4': 20,
    'B5': 40, 'B6': 40, 'B7': 40, 'C5': 40, 'C6': 40, 'C7': 40,
    'B8': 80, 'B9': 80, 'B10': 80, 'C8': 80, 'C9': 80, 'C10': 80,
}

# ==========================================


def load_all_data():
    df_list = []
    for folder in DATA_FOLDERS:
        if not os.path.exists(folder):
            continue
        for fp in glob.glob(os.path.join(folder, '*.xlsx')):
            df = pd.read_excel(fp)
            df['_well'] = os.path.basename(fp).replace('.xlsx', '')
            df_list.append(df)
    return pd.concat(df_list, ignore_index=True)


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
        print(f"[WARN] Failed to load {model_path}: {e}")
        return None


def panel_a_pca_scree(df):
    """Panel A: PCA scree comparison."""
    # 10D
    preproc_10 = Preprocessor(mode='full')
    X10 = preproc_10.fit_transform(df[RAW_FEATURES])
    pca10 = PCA(n_components=10).fit(X10)
    var10 = pca10.explained_variance_ratio_ * 100

    # 5D
    preproc_5 = Preprocessor(mode='reduced')
    X5 = preproc_5.fit_transform(df[REDUCED_RAW_FEATURES])
    pca5 = PCA(n_components=5).fit(X5)
    var5 = pca5.explained_variance_ratio_ * 100

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, var, title, color in zip(axes, [var10, var5], ['10D', '5D'], ['#1f77b4', '#d62728']):
        ax.bar(range(1, len(var)+1), var, color=color, alpha=0.7, edgecolor='white')
        ax.plot(range(1, len(var)+1), np.cumsum(var), 'ko-', markersize=6)
        ax.axhline(y=90, color='gray', linestyle='--', lw=1)
        ax.set_xlabel('PC', fontsize=10)
        ax.set_ylabel('Variance Explained (%)', fontsize=10)
        ax.set_title(f'PCA Scree ({title})', fontsize=11, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # annotate cumulative
        for i, (v, cum) in enumerate(zip(var, np.cumsum(var))):
            if i < 3:
                ax.annotate(f'{cum:.1f}%', xy=(i+1, cum), xytext=(5, 5),
                           textcoords='offset points', fontsize=8, color='darkgreen')
    return fig


def panel_b_vif_comparison():
    """Panel B: VIF bar chart."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    all_features = list(VIF_10D.keys()) + [f for f in VIF_5D.keys() if f not in VIF_10D]
    all_features = list(dict.fromkeys(all_features))  # unique, preserve order

    x10 = [VIF_10D.get(f, 0) for f in all_features]
    x5 = [VIF_5D.get(f, 0) for f in all_features]

    x = np.arange(len(all_features))
    width = 0.35
    bars1 = ax.bar(x - width/2, x10, width, label='10D', color='#1f77b4', alpha=0.8, edgecolor='white')
    bars2 = ax.bar(x + width/2, x5, width, label='5D', color='#d62728', alpha=0.8, edgecolor='white')

    ax.axhline(y=10, color='red', linestyle='--', lw=1.5, label='VIF=10 threshold')
    ax.set_xticks(x)
    ax.set_xticklabels(all_features, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('VIF', fontsize=10)
    ax.set_title('Multicollinearity Comparison (VIF)', fontsize=11, fontweight='bold')
    ax.set_yscale('log')
    ax.legend(loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return fig


def panel_c_ari_heatmap(df):
    """Panel C: ARI heatmap."""
    labels_dict = {}
    for name, path in MODEL_PATHS.items():
        labels = get_model_labels(df, path)
        if labels is not None:
            labels_dict[name] = labels

    names = list(labels_dict.keys())
    n = len(names)
    ari = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            ari[i, j] = adjusted_rand_score(labels_dict[names[i]], labels_dict[names[j]])

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(ari, annot=True, fmt='.3f', cmap='RdYlGn', vmin=0, vmax=1,
                xticklabels=names, yticklabels=names, square=True, ax=ax,
                cbar_kws={'shrink': 0.8, 'label': 'ARI'})
    ax.set_title('Cross-Model Agreement (ARI)', fontsize=11, fontweight='bold')
    return fig


def panel_d_bio_radar():
    """Panel D: Biological criteria radar chart (simplified as bar chart)."""
    # Data from validation results
    criteria = ['Vol Order', 'OAC Order', 'Control Red', 'Blue Dose']
    models = ['KMeans-10d', 'GMM-10d', 'KMeans-5d', 'GMM-5d']
    scores = {
        'KMeans-10d': [1, 1, 0, 0],
        'GMM-10d': [1, 1, 1, 0],
        'KMeans-5d': [0, 1, 1, 1],  # Blue dose now passes with relaxed criteria
        'GMM-5d': [1, 1, 1, 0],
    }

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(criteria))
    width = 0.18
    colors = {'KMeans-10d': '#1f77b4', 'GMM-10d': '#ff7f0e', 'KMeans-5d': '#2ca02c', 'GMM-5d': '#d62728'}
    for i, model in enumerate(models):
        ax.bar(x + i*width - 1.5*width, scores[model], width, label=model, color=colors[model], alpha=0.8, edgecolor='white')

    ax.set_xticks(x)
    ax.set_xticklabels(criteria, fontsize=10)
    ax.set_ylabel('Pass (1) / Fail (0)', fontsize=10)
    ax.set_ylim(0, 1.2)
    ax.set_title('Biological Plausibility Criteria', fontsize=11, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return fig


def panel_e_blue_dose(df):
    """Panel E: Blue dose-response curves."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = {'KMeans-10d': '#1f77b4', 'GMM-10d': '#ff7f0e', 'KMeans-5d': '#2ca02c', 'GMM-5d': '#d62728'}

    for name, path in MODEL_PATHS.items():
        labels = get_model_labels(df, path)
        if labels is None:
            continue
        dfc = df.copy()
        dfc['Cluster'] = labels
        dfc['Conc'] = dfc['_well'].apply(lambda w: WELL_CONC_MAP.get(w.split('_')[0].upper(), -1))
        conc_stats = dfc[dfc['Conc'] >= 0].groupby('Conc').apply(lambda s: (s['Cluster'] == 3).mean())
        conc_stats = conc_stats.sort_index()
        x = conc_stats.index.values
        y = conc_stats.values * 100

        ax.plot(x, y, 'o-', color=colors[name], lw=2, markersize=8, label=name)
        # Spearman
        if len(x) >= 3:
            rho, pval = spearmanr(x, y)
            sig = '*' if pval < 0.05 else ''
            ax.annotate(f'rho={rho:.2f}{sig}', xy=(x[-1], y[-1]), xytext=(5, 0),
                       textcoords='offset points', fontsize=8, color=colors[name])

    ax.set_xlabel('Icaritin Concentration (μM)', fontsize=10)
    ax.set_ylabel('Blue Cluster Fraction (%)', fontsize=10)
    ax.set_title('Damaged Organoid Fraction vs Drug Concentration', fontsize=11, fontweight='bold')
    ax.set_xticks([0, 20, 40, 80])
    ax.legend(loc='upper left', fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return fig


def panel_f_centroid_heatmap(df):
    """Panel F: 5D KMeans cluster centroid heatmap."""
    labels = get_model_labels(df, MODEL_PATHS['KMeans-5d'])
    if labels is None:
        return None

    df_c = df.copy()
    df_c['Cluster'] = labels
    features = ['Organoids_Volume_Fill', 'Sphericity', 'Scatt_Mean', 'Scatt_STD', 'Cavity_Ratio']
    for f in features:
        if f not in df_c.columns and f == 'Cavity_Ratio':
            df_c['Cavity_Ratio'] = df_c['Cavity_Volume'] / (df_c['Organoids_Volume_Fill'].clip(lower=1))

    centroids = df_c.groupby('Cluster')[features].mean()
    # Normalize by z-score for visualization
    centroids_norm = (centroids - centroids.mean()) / centroids.std()

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(centroids_norm, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                xticklabels=features, yticklabels=['Red (0)', 'Yellow (1)', 'Green (2)', 'Blue (3)'],
                ax=ax, cbar_kws={'shrink': 0.8})
    ax.set_title('5D KMeans Cluster Centroids (Z-score)', fontsize=11, fontweight='bold')
    return fig


def combine_panels():
    """Combine all panels into one publication-ready figure."""
    print("\n--- 生成数据 ---")
    df = load_all_data()
    print(f"Loaded {len(df)} organoids")

    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 3, figure=fig, wspace=0.35, hspace=0.4)

    print("  Panel A: PCA Scree...")
    ax_a = fig.add_subplot(gs[0, 0])
    # Panel A: draw manually on subplot
    preproc_10 = Preprocessor(mode='full')
    X10 = preproc_10.fit_transform(df[RAW_FEATURES])
    pca10 = PCA(n_components=10).fit(X10)
    var10 = pca10.explained_variance_ratio_ * 100
    preproc_5 = Preprocessor(mode='reduced')
    X5 = preproc_5.fit_transform(df[REDUCED_RAW_FEATURES])
    pca5 = PCA(n_components=5).fit(X5)
    var5 = pca5.explained_variance_ratio_ * 100

    ax_a_twin = ax_a.twinx()
    ax_a.bar(range(1, 6), var10[:5], color='#1f77b4', alpha=0.6, width=0.4, label='10D')
    ax_a.bar([x+0.4 for x in range(1, 6)], [v if i < len(var5) else 0 for i, v in enumerate(var5)],
             color='#d62728', alpha=0.6, width=0.4, label='5D')
    ax_a_twin.plot(range(1, 6), np.cumsum(var10[:5]), 'o-', color='#1f77b4', markersize=5)
    ax_a_twin.plot(range(1, 6), np.cumsum(var5), 's-', color='#d62728', markersize=5)
    ax_a.set_xlabel('PC', fontsize=9)
    ax_a.set_ylabel('Variance (%)', fontsize=9)
    ax_a_twin.set_ylabel('Cumulative (%)', fontsize=9, color='darkgreen')
    ax_a.set_title('A. PCA Explained Variance', fontsize=11, fontweight='bold', loc='left')
    ax_a.legend(loc='upper left', fontsize=8)
    ax_a.spines['top'].set_visible(False)

    print("  Panel B: VIF...")
    ax_b = fig.add_subplot(gs[0, 1])
    features = ['Organoids_Volume_Fill', 'Sphericity', 'Scatt_Mean', 'Scatt_STD', 'Cavity_Ratio']
    vif10_vals = [VIF_10D.get(f, 0) for f in features]
    vif5_vals = [VIF_5D.get(f, 0) for f in features]
    x = np.arange(len(features))
    width = 0.35
    ax_b.bar(x - width/2, vif10_vals, width, label='10D', color='#1f77b4', alpha=0.8)
    ax_b.bar(x + width/2, vif5_vals, width, label='5D', color='#d62728', alpha=0.8)
    ax_b.axhline(y=10, color='red', linestyle='--', lw=1.5)
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(['Vol_Fill', 'Sphericity', 'OAC_Mean', 'OAC_STD', 'Cavity_Ratio'], rotation=15, ha='right', fontsize=8)
    ax_b.set_ylabel('VIF', fontsize=9)
    ax_b.set_yscale('log')
    ax_b.set_title('B. Multicollinearity (VIF)', fontsize=11, fontweight='bold', loc='left')
    ax_b.legend(fontsize=8)
    ax_b.spines['top'].set_visible(False)
    ax_b.spines['right'].set_visible(False)

    print("  Panel C: ARI Heatmap...")
    ax_c = fig.add_subplot(gs[0, 2])
    labels_dict = {}
    for name, path in MODEL_PATHS.items():
        labels = get_model_labels(df, path)
        if labels is not None:
            labels_dict[name] = labels
    names = list(labels_dict.keys())
    n = len(names)
    ari = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            ari[i, j] = adjusted_rand_score(labels_dict[names[i]], labels_dict[names[j]])
    sns.heatmap(ari, annot=True, fmt='.3f', cmap='RdYlGn', vmin=0, vmax=1,
                xticklabels=[n.replace('-', '\n') for n in names],
                yticklabels=[n.replace('-', '\n') for n in names],
                square=True, ax=ax_c, cbar_kws={'shrink': 0.7})
    ax_c.set_title('C. Cross-Model Agreement (ARI)', fontsize=11, fontweight='bold', loc='left')

    print("  Panel D: Biological Criteria...")
    ax_d = fig.add_subplot(gs[1, 0])
    criteria = ['Vol\nOrder', 'OAC\nOrder', 'Control\nRed', 'Blue\nDose']
    models = ['KMeans-10d', 'GMM-10d', 'KMeans-5d', 'GMM-5d']
    scores = {
        'KMeans-10d': [1, 1, 0, 0],
        'GMM-10d': [1, 1, 1, 0],
        'KMeans-5d': [0, 1, 1, 1],
        'GMM-5d': [1, 1, 1, 0],
    }
    colors = {'KMeans-10d': '#1f77b4', 'GMM-10d': '#ff7f0e', 'KMeans-5d': '#2ca02c', 'GMM-5d': '#d62728'}
    x = np.arange(len(criteria))
    width = 0.18
    for i, model in enumerate(models):
        ax_d.bar(x + i*width - 1.5*width, scores[model], width, label=model, color=colors[model], alpha=0.8, edgecolor='white')
    ax_d.set_xticks(x)
    ax_d.set_xticklabels(criteria, fontsize=8)
    ax_d.set_ylabel('Pass (1) / Fail (0)', fontsize=9)
    ax_d.set_ylim(0, 1.3)
    ax_d.set_title('D. Biological Plausibility', fontsize=11, fontweight='bold', loc='left')
    ax_d.legend(fontsize=7, loc='upper right')
    ax_d.spines['top'].set_visible(False)
    ax_d.spines['right'].set_visible(False)

    print("  Panel E: Blue Dose Response...")
    ax_e = fig.add_subplot(gs[1, 1:])
    for name, path in MODEL_PATHS.items():
        labels = get_model_labels(df, path)
        if labels is None:
            continue
        dfc = df.copy()
        dfc['Cluster'] = labels
        dfc['Conc'] = dfc['_well'].apply(lambda w: WELL_CONC_MAP.get(w.split('_')[0].upper(), -1))
        conc_stats = dfc[dfc['Conc'] >= 0].groupby('Conc').apply(lambda s: (s['Cluster'] == 3).mean())
        conc_stats = conc_stats.sort_index()
        x = conc_stats.index.values
        y = conc_stats.values * 100
        ax_e.plot(x, y, 'o-', color=colors[name], lw=2.5, markersize=9, label=name)
        if len(x) >= 3:
            rho, pval = spearmanr(x, y)
            sig = '*' if pval < 0.05 else 'ns'
            ax_e.annotate(f'ρ={rho:.2f} {sig}', xy=(x[-1], y[-1]), xytext=(8, -5),
                         textcoords='offset points', fontsize=9, color=colors[name], fontweight='bold')
    ax_e.set_xlabel('Icaritin Concentration (μM)', fontsize=10)
    ax_e.set_ylabel('Blue (Damaged) Fraction (%)', fontsize=10)
    ax_e.set_title('E. Dose-Response: Damaged Organoid Fraction', fontsize=11, fontweight='bold', loc='left')
    ax_e.set_xticks([0, 20, 40, 80])
    ax_e.legend(loc='upper left', fontsize=9)
    ax_e.spines['top'].set_visible(False)
    ax_e.spines['right'].set_visible(False)

    print("  Panel F: 5D Centroid Heatmap...")
    ax_f = fig.add_subplot(gs[2, :])
    labels = get_model_labels(df, MODEL_PATHS['KMeans-5d'])
    if labels is not None:
        dfc = df.copy()
        dfc['Cluster'] = labels
        features = ['Organoids_Volume_Fill', 'Sphericity', 'Scatt_Mean', 'Scatt_STD', 'Cavity_Ratio']
        if 'Cavity_Ratio' not in dfc.columns:
            dfc['Cavity_Ratio'] = dfc['Cavity_Volume'] / (dfc['Organoids_Volume_Fill'].clip(lower=1))
        centroids = dfc.groupby('Cluster')[features].mean()
        centroids_norm = (centroids - centroids.mean()) / centroids.std()

        # Draw as grouped bar instead of heatmap for clarity
        x = np.arange(len(features))
        width = 0.2
        cluster_names = ['Red (Large Cystic)', 'Yellow (Large Solid)', 'Green (Small Solid)', 'Blue (Damaged)']
        cluster_colors = ['#FF0000', '#FFFF00', '#00FF00', '#0000FF']
        for i, (cid, cname, ccolor) in enumerate(zip(centroids_norm.index, cluster_names, cluster_colors)):
            vals = centroids_norm.loc[cid].values
            ax_f.bar(x + i*width - 1.5*width, vals, width, label=cname, color=ccolor, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax_f.set_xticks(x)
        ax_f.set_xticklabels(['Volume_Fill', 'Sphericity', 'OAC_Mean', 'OAC_STD', 'Cavity_Ratio'], fontsize=9)
        ax_f.set_ylabel('Z-score (normalized)', fontsize=10)
        ax_f.set_title('F. 5D KMeans Cluster Centroids (Z-score)', fontsize=11, fontweight='bold', loc='left')
        ax_f.legend(loc='upper right', fontsize=8, ncol=2)
        ax_f.axhline(y=0, color='black', linewidth=0.5)
        ax_f.spines['top'].set_visible(False)
        ax_f.spines['right'].set_visible(False)

    plt.suptitle('Figure S1: 10D vs 5D Feature Ablation Study', fontsize=14, fontweight='bold', y=0.98)
    save_path = os.path.join(FIGURES_DIR, 'figure_10d_vs_5d_ablation.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\n  Figure saved -> {save_path}")


def main():
    print("=" * 60)
    print("Publication Figure: 10D vs 5D Ablation Study")
    print("=" * 60)
    combine_panels()
    print("\n[Done] Figure generation complete.")


if __name__ == "__main__":
    main()
