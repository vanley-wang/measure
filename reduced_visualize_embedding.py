"""
5维精简特征降维可视化脚本

输出：
  reports/figures/reduced_pca_scree.png
  reports/figures/reduced_pca_2d.png
  reports/figures/reduced_pca_3d.html
  reports/figures/reduced_tsne_comparison.png
  reports/figures/reduced_umap_comparison.png
"""
import os
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from cluster_utils import REDUCED_RAW_FEATURES, Preprocessor, load_model_package

warnings.filterwarnings('ignore')

# ================= 配置 =================
DATA_FOLDERS = [
    os.path.join('Data', 'nnUNet_FXN_2023', 'FXN_0701', 'measure_excel'),
    os.path.join('Data', 'nnUNet_FXN_2023', 'FXN_0703', 'measure_excel'),
]
REPORTS_DIR = 'reports'
FIGURES_DIR = os.path.join(REPORTS_DIR, 'figures')
MODEL_PATH = 'model/Kmeans-scatt.pickle'

os.makedirs(FIGURES_DIR, exist_ok=True)

HAVE_UMAP = False
try:
    import umap
    HAVE_UMAP = True
except ImportError:
    print("[WARN] umap-learn 未安装，跳过 UMAP 可视化。")

HAVE_PLOTLY = False
try:
    import plotly.express as px
    HAVE_PLOTLY = True
except ImportError:
    print("[WARN] plotly 未安装，跳过 3D 交互图。")

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


def get_kmeans_labels(df, features):
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        pkg = load_model_package(MODEL_PATH)
        preprocessor = pkg.get('preprocessor')
        scaler = pkg.get('scaler')
        model = pkg['model']
        raw_to_final = pkg.get('raw_to_final', {})

        if preprocessor is not None:
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
        print(f"[WARN] 加载 K-means 模型失败: {e}")
        return None


def plot_pca_scree(pca, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    var = pca.explained_variance_ratio_ * 100
    ax = axes[0]
    ax.bar(range(1, len(var) + 1), var, color='steelblue', edgecolor='white')
    ax.plot(range(1, len(var) + 1), np.cumsum(var), 'ro-', markersize=6)
    ax.axhline(y=90, color='gray', linestyle='--', lw=1)
    ax.set_xlabel('Principal Component', fontsize=11)
    ax.set_ylabel('Explained Variance (%)', fontsize=11)
    ax.set_title('PCA Scree Plot (5D)', fontsize=12, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax = axes[1]
    cumvar = np.cumsum(var)
    ax.plot(range(1, len(cumvar) + 1), cumvar, 'o-', color='darkgreen', markersize=7)
    ax.axhline(y=70, color='gray', linestyle='--', lw=1, label='70%')
    ax.axhline(y=90, color='gray', linestyle='--', lw=1, label='90%')
    ax.set_xlabel('Number of Components', fontsize=11)
    ax.set_ylabel('Cumulative Variance (%)', fontsize=11)
    ax.set_title('Cumulative Explained Variance (5D)', fontsize=12, fontweight='bold')
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  PCA 碎石图已保存 -> {save_path}")


def plot_pca_2d(df_pca, labels, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    palette = ['#FF0000', '#FFFF00', '#00FF00', '#0000FF', '#888888', '#FF00FF']

    ax = axes[0]
    ax.scatter(df_pca['PC1'], df_pca['PC2'], c='steelblue', s=8, alpha=0.4, edgecolors='none')
    ax.set_xlabel(f"PC1 ({df_pca['PC1_var'].iloc[0]:.1f}%)", fontsize=11)
    ax.set_ylabel(f"PC2 ({df_pca['PC2_var'].iloc[0]:.1f}%)", fontsize=11)
    ax.set_title('PCA 2D (No Labels, 5D)', fontsize=12, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax = axes[1]
    if labels is not None:
        for cid in sorted(np.unique(labels)):
            mask = labels == cid
            color = palette[cid % len(palette)]
            ax.scatter(df_pca.loc[mask, 'PC1'], df_pca.loc[mask, 'PC2'],
                       c=color, s=8, alpha=0.5, edgecolors='none', label=f'C{cid}')
        ax.legend(title='Cluster', markerscale=2)
    else:
        ax.scatter(df_pca['PC1'], df_pca['PC2'], c='steelblue', s=8, alpha=0.4, edgecolors='none')
    ax.set_xlabel(f"PC1 ({df_pca['PC1_var'].iloc[0]:.1f}%)", fontsize=11)
    ax.set_ylabel(f"PC2 ({df_pca['PC2_var'].iloc[0]:.1f}%)", fontsize=11)
    ax.set_title('PCA 2D (K-means 4 Clusters, 5D)', fontsize=12, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  PCA 2D 图已保存 -> {save_path}")


def plot_pca_3d_html(df_pca, labels, save_path):
    if not HAVE_PLOTLY:
        return
    df_plot = df_pca[['PC1', 'PC2', 'PC3']].copy()
    if labels is not None:
        df_plot['Cluster'] = labels.astype(str)
        fig = px.scatter_3d(df_plot, x='PC1', y='PC2', z='PC3', color='Cluster',
                            opacity=0.5, title='PCA 3D (5D Features)')
    else:
        fig = px.scatter_3d(df_plot, x='PC1', y='PC2', z='PC3',
                            opacity=0.5, title='PCA 3D (5D Features, No Labels)')
    fig.update_traces(marker_size=2)
    fig.write_html(save_path)
    print(f"  PCA 3D 交互图已保存 -> {save_path}")


def plot_tsne_comparison(X, labels, save_path):
    perplexities = [30, 50, 100]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    palette = ['#FF0000', '#FFFF00', '#00FF00', '#0000FF', '#888888', '#FF00FF']

    for idx, perp in enumerate(perplexities):
        print(f"  计算 t-SNE (perplexity={perp})...")
        tsne = TSNE(n_components=2, perplexity=perp, random_state=42, init='pca', learning_rate='auto')
        Y = tsne.fit_transform(X)

        ax = axes[idx]
        if labels is not None:
            for cid in sorted(np.unique(labels)):
                mask = labels == cid
                color = palette[cid % len(palette)]
                ax.scatter(Y[mask, 0], Y[mask, 1], c=color, s=5, alpha=0.5, edgecolors='none', label=f'C{cid}')
            ax.legend(title='Cluster', markerscale=2, fontsize=7)
        else:
            ax.scatter(Y[:, 0], Y[:, 1], c='steelblue', s=5, alpha=0.4, edgecolors='none')
        ax.set_title(f't-SNE (perplexity={perp}, 5D)', fontsize=11, fontweight='bold')
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  t-SNE 对比图已保存 -> {save_path}")


def plot_umap_comparison(X, labels, save_path):
    if not HAVE_UMAP:
        return
    configs = [
        {'n_neighbors': 15, 'min_dist': 0.1},
        {'n_neighbors': 30, 'min_dist': 0.1},
        {'n_neighbors': 50, 'min_dist': 0.5},
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    palette = ['#FF0000', '#FFFF00', '#00FF00', '#0000FF', '#888888', '#FF00FF']

    for idx, cfg in enumerate(configs):
        print(f"  计算 UMAP (n_neighbors={cfg['n_neighbors']}, min_dist={cfg['min_dist']})...")
        reducer = umap.UMAP(n_components=2, random_state=42, **cfg)
        Y = reducer.fit_transform(X)

        ax = axes[idx]
        if labels is not None:
            for cid in sorted(np.unique(labels)):
                mask = labels == cid
                color = palette[cid % len(palette)]
                ax.scatter(Y[mask, 0], Y[mask, 1], c=color, s=5, alpha=0.5, edgecolors='none', label=f'C{cid}')
            ax.legend(title='Cluster', markerscale=2, fontsize=7)
        else:
            ax.scatter(Y[:, 0], Y[:, 1], c='steelblue', s=5, alpha=0.4, edgecolors='none')
        ax.set_title(f"UMAP (n={cfg['n_neighbors']}, d={cfg['min_dist']}, 5D)", fontsize=11, fontweight='bold')
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  UMAP 对比图已保存 -> {save_path}")


def print_pca_loadings(pca, feature_names):
    loadings = pd.DataFrame(
        pca.components_[:3].T,
        columns=['PC1', 'PC2', 'PC3'],
        index=feature_names
    )
    print("\n--- PCA 载荷矩阵 (前 3 主成分, 5D) ---")
    print(loadings.round(3).to_string())
    print("\n各主成分主导特征:")
    for pc in ['PC1', 'PC2', 'PC3']:
        top = loadings[pc].abs().idxmax()
        print(f"  {pc}: {top} (loading={loadings.loc[top, pc]:.3f})")


def main():
    print("=" * 60)
    print("5维精简特征 - 降维可视化")
    print("=" * 60)

    print("\n--- 加载数据 ---")
    df = load_all_data()
    print(f"总类器官数: {len(df)}")

    print("\n--- 预处理 (5D: log1p + StandardScaler) ---")
    preprocessor = Preprocessor(mode='reduced')
    X = preprocessor.fit_transform(df[REDUCED_RAW_FEATURES])
    feature_names = preprocessor.get_feature_names()
    print(f"特征维度: {X.shape[1]} ({feature_names})")

    labels = get_kmeans_labels(df, REDUCED_RAW_FEATURES)
    if labels is not None:
        print(f"已加载 K-means 标签，类别分布: {dict(pd.Series(labels).value_counts().sort_index())}")

    print("\n--- PCA (5D) ---")
    pca = PCA(n_components=5, random_state=42)
    pca_coords = pca.fit_transform(X)
    var = pca.explained_variance_ratio_ * 100
    print(f"  前 3 维方差贡献: PC1={var[0]:.1f}%, PC2={var[1]:.1f}%, PC3={var[2]:.1f}%")
    print(f"  前 3 维累计: {var[:3].sum():.1f}%")
    print(f"  前 5 维累计: {var.sum():.1f}%")

    print_pca_loadings(pca, feature_names)

    df_pca = pd.DataFrame({
        'PC1': pca_coords[:, 0],
        'PC2': pca_coords[:, 1],
        'PC3': pca_coords[:, 2],
        'PC1_var': var[0],
        'PC2_var': var[1],
        'PC3_var': var[2],
    })

    plot_pca_scree(pca, os.path.join(FIGURES_DIR, 'reduced_pca_scree.png'))
    plot_pca_2d(df_pca, labels, os.path.join(FIGURES_DIR, 'reduced_pca_2d.png'))
    if HAVE_PLOTLY:
        plot_pca_3d_html(df_pca, labels, os.path.join(FIGURES_DIR, 'reduced_pca_3d.html'))

    print("\n--- t-SNE (5D) ---")
    plot_tsne_comparison(X, labels, os.path.join(FIGURES_DIR, 'reduced_tsne_comparison.png'))

    if HAVE_UMAP:
        print("\n--- UMAP (5D) ---")
        plot_umap_comparison(X, labels, os.path.join(FIGURES_DIR, 'reduced_umap_comparison.png'))
    else:
        print("\n--- UMAP (跳过，未安装 umap-learn) ---")

    print("\n[Done] 5维降维可视化完成。")


if __name__ == "__main__":
    main()
