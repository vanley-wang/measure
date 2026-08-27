import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from cluster_utils import PROCESSED_FEATURES
from .config import CLUSTER_COLORS, CLUSTER_NAMES, CONC_MAP, FIGURES_DIR

plt.rcParams.update({
    'font.size': 10, 'figure.dpi': 150, 'savefig.dpi': 300,
    'axes.labelsize': 11, 'axes.titlesize': 12, 'legend.fontsize': 9,
})

CONC_COLORS = {0: '#3498DB', 20: '#27AE60', 40: '#E67E22', 80: '#C0392B'}
CONC_LABELS = {0: 'Control', 20: '20 \u03bcM', 40: '40 \u03bcM', 80: '80 \u03bcM'}


def _shorten(name):
    return (name.replace('Delta_Healthy_', '\u0394 ')
            .replace('Organoids_', '')
            .replace('_Frac', '%')
            .replace('_', ' '))


def _panel_label(ax, label):
    ax.text(-0.08, 1.08, label, transform=ax.transAxes, fontsize=13,
            fontweight='bold', va='top', ha='left')


def generate_figures(df, ws, fm, merged, res, cdf, pca=None, sel_feats=None):
    print('\n' + '=' * 70)
    print('Step 6: Generating Publication Figures')
    print('=' * 70)

    os.makedirs(FIGURES_DIR, exist_ok=True)

    figs = [
        ('fig1_clustering.png', fig1),
        ('fig2_score_atp.png', fig2),
        ('fig3_dose_response.png', fig3),
        ('fig4_score_dist.png', fig4),
        ('fig5_radar.png', fig5),
        ('fig6_residual.png', fig6),
    ]

    for fn, func in figs:
        try:
            path = os.path.join(FIGURES_DIR, fn)
            func(df, ws, fm, merged, res, cdf, pca, sel_feats, path)
            print(f'OK {fn}')
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f'ERR {fn}: {e}')


# ===========================================================================
# fig1: Organoid clustering validation (PCA + heatmap)
# ===========================================================================
def fig1(df, ws, fm, merged, res, cdf, pca, sel_feats, fp):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # --- A: PCA of organoid features coloured by cluster ---
    cluster_feats = [f for f in PROCESSED_FEATURES if f in df.columns]
    df_valid = df.dropna(subset=cluster_feats)
    if len(df_valid) > 100:
        n_sample = min(5000, len(df_valid))
        df_sample = df_valid.sample(n=n_sample, random_state=42)
        X = StandardScaler().fit_transform(df_sample[cluster_feats].values)
        X2 = PCA(n_components=2, random_state=42).fit_transform(X)
        for c in sorted(CLUSTER_COLORS.keys()):
            mask = df_sample['Cluster'].values == c
            ax1.scatter(X2[mask, 0], X2[mask, 1], c=CLUSTER_COLORS[c],
                        label=CLUSTER_NAMES[c], s=3, alpha=0.5, rasterized=True)
        ax1.legend(markerscale=6, frameon=True, fancybox=True,
                   loc='upper right', title='Phenotype')
        ax1.set_xlabel('PC1')
        ax1.set_ylabel('PC2')
    _panel_label(ax1, 'A')
    ax1.set_title('Organoid PCA by Phenotype', fontweight='bold', loc='left')

    # --- B: Cluster centroid heatmap (Z-score) ---
    prof = df_valid.groupby('Cluster')[cluster_feats].mean()
    pz = (prof - prof.mean()) / prof.std()
    short_names = [_shorten(f) for f in cluster_feats]
    im = ax2.imshow(pz.values, cmap='RdBu_r', aspect='auto', vmin=-2, vmax=2)
    ax2.set_xticks(range(len(short_names)))
    ax2.set_xticklabels(short_names, rotation=45, ha='right', fontsize=8)
    ax2.set_yticks(range(len(prof)))
    ax2.set_yticklabels([CLUSTER_NAMES[i] for i in prof.index], fontsize=10)
    cbar = plt.colorbar(im, ax=ax2, shrink=0.8)
    cbar.set_label('Z-score', fontsize=9)
    _panel_label(ax2, 'B')
    ax2.set_title('Cluster Profiles', fontweight='bold', loc='left')

    plt.tight_layout()
    plt.savefig(fp, dpi=300, bbox_inches='tight')
    plt.close()


# ===========================================================================
# fig2: PCA construction + ATP validation (core result)
# ===========================================================================
def fig2(df, ws, fm, merged, res, cdf, pca, sel_feats, fp):
    if res is None or merged is None:
        return _placeholder(fp, 'Score-ATP Validation', 'No data')

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # --- A: Scree plot ---
    ax = axes[0]
    if pca is not None:
        n = len(pca.explained_variance_ratio_)
        x = np.arange(1, n + 1)
        bars = ax.bar(x, pca.explained_variance_ratio_ * 100, color='#3498DB',
                      alpha=0.7, edgecolor='white', label='Individual')
        ax.plot(x, np.cumsum(pca.explained_variance_ratio_) * 100, 'o-',
                color='#E74C3C', lw=2, ms=6, label='Cumulative')
        ax.axhline(y=1.0 / n * 100, color='gray', ls='--', lw=1,
                   label=f'Kaiser (={1.0/n*100:.1f}%)')
        for i, (v, cv) in enumerate(zip(pca.explained_variance_ratio_ * 100,
                                         np.cumsum(pca.explained_variance_ratio_) * 100)):
            ax.text(x[i], v + 1, f'{v:.1f}%', ha='center', fontsize=8)
        ax.set_xticks(x)
        ax.set_xlabel('Principal Component')
        ax.set_ylabel('Variance Explained (%)')
        ax.legend(fontsize=8, loc='center right')
    _panel_label(ax, 'A')
    ax.set_title('PCA Scree Plot', fontweight='bold', loc='left')

    # --- B: Feature weights ---
    ax = axes[1]
    if cdf is not None and len(cdf) > 0:
        tn = min(12, len(cdf))
        tc = cdf.head(tn).iloc[::-1]
        colors = ['#E74C3C' if c > 0 else '#3498DB' for c in tc['Coef']]
        ax.barh(range(tn), tc['Coef'], color=colors, alpha=0.8, height=0.65)
        ax.set_yticks(range(tn))
        ax.set_yticklabels([_shorten(f) for f in tc['Feature']], fontsize=8)
        ax.set_xlabel('Weight in Composite Score')
        ax.axvline(0, color='black', lw=0.8)
        ax.invert_yaxis()
    _panel_label(ax, 'B')
    ax.set_title('Feature Weights', fontweight='bold', loc='left')

    # --- C: Score vs ATP scatter ---
    ax = axes[2]
    m = merged.dropna(subset=['Score', 'ATP'])
    if len(m) >= 5:
        ax.scatter(m['Score'], m['ATP'] / 1e6, c='#2C3E50', alpha=0.7, s=70,
                   edgecolors='white', linewidth=0.5)
        for _, row in m.iterrows():
            ax.annotate(row['Well_ID'], (row['Score'], row['ATP'] / 1e6),
                        fontsize=6, alpha=0.6, ha='center', va='bottom',
                        xytext=(0, 4), textcoords='offset points')
        ax.set_xlabel('PCA Composite Score')
        ax.set_ylabel('ATP (\u00d710\u2076)')
        ax.set_title(f"r = {res['pearson_r']:.3f}, p = {res['pearson_p']:.2e}",
                     fontweight='bold', loc='center', fontsize=11)
    _panel_label(ax, 'C')
    ax.set_title('Score vs ATP', fontweight='bold', loc='left')

    plt.tight_layout()
    plt.savefig(fp, dpi=300, bbox_inches='tight')
    plt.close()


# ===========================================================================
# fig3: Dose-response (OAC concentration)
# ===========================================================================
def fig3(df, ws, fm, merged, res, cdf, pca, sel_feats, fp):
    concs = [0, 20, 40, 80]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    wd5 = ws[ws['_day'] == '0703'].copy()
    wd5['Conc'] = wd5['_well_id'].map(CONC_MAP)

    # --- A: Healthy% dose response ---
    ax = axes[0]
    data_h, positions_h = [], []
    for i, c in enumerate(concs):
        vals = wd5.loc[wd5['Conc'] == c, 'Healthy_Fraction'].values * 100
        if len(vals) > 0:
            data_h.append(vals)
            positions_h.append(c)
            ax.scatter(np.full(len(vals), c + np.random.uniform(-2, 2, len(vals))),
                       vals, c=CONC_COLORS[c], alpha=0.4, s=30, zorder=2)
    bp = ax.boxplot(data_h, positions=positions_h, widths=7,
                    patch_artist=True, showfliers=False, zorder=1)
    for patch, c in zip(bp['boxes'], concs):
        patch.set_facecolor(CONC_COLORS[c])
        patch.set_alpha(0.35)
    ax.set_xticks(concs)
    ax.set_xticklabels([CONC_LABELS[c] for c in concs])
    ax.set_ylabel('Healthy Fraction (%)')
    _panel_label(ax, 'A')
    ax.set_title('Healthy% Dose Response', fontweight='bold', loc='left')

    # --- B: Blue% dose response ---
    ax = axes[1]
    data_b, positions_b = [], []
    for i, c in enumerate(concs):
        vals = wd5.loc[wd5['Conc'] == c, 'Blue_Fraction'].values * 100
        if len(vals) > 0:
            data_b.append(vals)
            positions_b.append(c)
            ax.scatter(np.full(len(vals), c + np.random.uniform(-2, 2, len(vals))),
                       vals, c=CONC_COLORS[c], alpha=0.4, s=30, zorder=2)
    bp = ax.boxplot(data_b, positions=positions_b, widths=7,
                    patch_artist=True, showfliers=False, zorder=1)
    for patch, c in zip(bp['boxes'], concs):
        patch.set_facecolor(CONC_COLORS[c])
        patch.set_alpha(0.35)
    ax.set_xticks(concs)
    ax.set_xticklabels([CONC_LABELS[c] for c in concs])
    ax.set_ylabel('Blue (Dead) Fraction (%)')
    _panel_label(ax, 'B')
    ax.set_title('Blue% Dose Response', fontweight='bold', loc='left')

    # --- C: Cluster composition stacked bar ---
    ax = axes[2]
    comp = wd5.groupby('Conc').agg({'Red': 'sum', 'Yellow': 'sum',
                                     'Green': 'sum', 'Blue': 'sum'})
    total = comp.sum(axis=1)
    comp_pct = comp.div(total, axis=0) * 100
    cluster_order = ['Red', 'Yellow', 'Green', 'Blue']
    bottom = np.zeros(len(concs))
    bar_width = 12
    for cn in cluster_order:
        if cn in comp_pct.columns:
            heights = [comp_pct.loc[c, cn] if c in comp_pct.index else 0
                       for c in concs]
            ax.bar(concs, heights, width=bar_width, bottom=bottom,
                   color=CLUSTER_COLORS[cluster_order.index(cn)],
                   label=CLUSTER_NAMES[cluster_order.index(cn)],
                   alpha=0.85, edgecolor='white', linewidth=0.5)
            bottom += np.array(heights)
    ax.set_xticks(concs)
    ax.set_xticklabels([CONC_LABELS[c] for c in concs])
    ax.set_ylabel('Fraction (%)')
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8, loc='upper right', frameon=True)
    _panel_label(ax, 'C')
    ax.set_title('Phenotype Composition', fontweight='bold', loc='left')

    plt.tight_layout()
    plt.savefig(fp, dpi=300, bbox_inches='tight')
    plt.close()


# ===========================================================================
# fig4: Score & ATP by well (ranked, coloured by concentration)
# ===========================================================================
def fig4(df, ws, fm, merged, res, cdf, pca, sel_feats, fp):
    if merged is None or 'Score' not in merged.columns:
        return _placeholder(fp, 'Score & ATP by Well', 'No data')

    m = merged.copy()
    m['Conc'] = m['Well_ID'].map(CONC_MAP)
    m = m.sort_values('Score')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # --- A: Score by well ---
    ids = m['Well_ID'].values
    scores = m['Score'].values
    colors = [CONC_COLORS.get(m.iloc[i]['Conc'], '#95A5A6') for i in range(len(m))]
    ax1.barh(range(len(ids)), scores, color=colors, alpha=0.8, height=0.7,
             edgecolor='white', linewidth=0.3)
    ax1.set_yticks(range(len(ids)))
    ax1.set_yticklabels(ids, fontsize=8)
    ax1.set_xlabel('PCA Composite Score')
    ax1.axvline(0, color='gray', ls='--', alpha=0.5)
    legend_patches = [Patch(fc=CONC_COLORS[c], label=CONC_LABELS[c])
                      for c in [0, 20, 40, 80]]
    ax1.legend(handles=legend_patches, fontsize=8, loc='lower right',
               title='OAC Concentration')
    _panel_label(ax1, 'A')
    ax1.set_title('Score by Well', fontweight='bold', loc='left')

    # --- B: ATP by well ---
    m_atp = m.sort_values('ATP')
    ids2 = m_atp['Well_ID'].values
    atp_vals = m_atp['ATP'].values / 1e6
    colors2 = [CONC_COLORS.get(m_atp.iloc[i]['Conc'], '#95A5A6')
               for i in range(len(m_atp))]
    ax2.barh(range(len(ids2)), atp_vals, color=colors2, alpha=0.8, height=0.7,
             edgecolor='white', linewidth=0.3)
    ax2.set_yticks(range(len(ids2)))
    ax2.set_yticklabels(ids2, fontsize=8)
    ax2.set_xlabel('ATP (\u00d710\u2076)')
    ax2.legend(handles=legend_patches, fontsize=8, loc='lower right',
               title='OAC Concentration')
    _panel_label(ax2, 'B')
    ax2.set_title('ATP by Well', fontweight='bold', loc='left')

    plt.tight_layout()
    plt.savefig(fp, dpi=300, bbox_inches='tight')
    plt.close()


# ===========================================================================
# fig5: Feature correlation matrix (12D + Score + ATP)
# ===========================================================================
def fig5(df, ws, fm, merged, res, cdf, pca, sel_feats, fp):
    if sel_feats is None or merged is None:
        return _placeholder(fp, 'Feature Correlation', 'No data')

    common_wells = set(merged['Well_ID']) & set(fm['Well_ID'] if 'Well_ID' in fm.columns else fm.index)
    common_wells = sorted(common_wells)

    m = merged.set_index('Well_ID').loc[common_wells]
    fm_idx = fm.set_index('Well_ID') if 'Well_ID' in fm.columns else fm
    fm_idx = fm_idx.loc[common_wells]

    cols = [f for f in sel_feats if f in fm_idx.columns]
    if len(cols) < 2:
        return _placeholder(fp, 'Feature Correlation', 'Insufficient features')

    corr_df = fm_idx[cols].copy()
    if 'Score' in m.columns:
        corr_df['Score'] = m['Score']
    if 'ATP' in m.columns:
        corr_df['ATP'] = m['ATP']

    corr = corr_df.corr()

    short_labels = [_shorten(c) for c in corr.columns]

    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(corr.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(range(len(short_labels)))
    ax.set_xticklabels(short_labels, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(short_labels)))
    ax.set_yticklabels(short_labels, fontsize=8)

    for i in range(len(short_labels)):
        for j in range(len(short_labels)):
            val = corr.values[i, j]
            color = 'white' if abs(val) > 0.6 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=7, color=color)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Pearson r', fontsize=9)
    ax.set_title('Feature Correlation Matrix', fontweight='bold', fontsize=13, pad=15)

    plt.tight_layout()
    plt.savefig(fp, dpi=300, bbox_inches='tight')
    plt.close()


# ===========================================================================
# fig6: Score vs key individual features (validation)
# ===========================================================================
def fig6(df, ws, fm, merged, res, cdf, pca, sel_feats, fp):
    if merged is None or sel_feats is None:
        return _placeholder(fp, 'Score vs Features', 'No data')

    common_wells = set(merged['Well_ID']) & set(fm['Well_ID'] if 'Well_ID' in fm.columns else fm.index)
    common_wells = sorted(common_wells)

    m = merged.set_index('Well_ID').loc[common_wells]
    fm_idx = fm.set_index('Well_ID') if 'Well_ID' in fm.columns else fm
    fm_idx = fm_idx.loc[common_wells]

    key_features = [
        ('Red_Frac_D5', 'Red(H) Fraction Day5'),
        ('Delta_Healthy_Scatt_Mean', '\u0394 Healthy Scatt Mean'),
        ('Delta_Healthy_Frac', '\u0394 Healthy Fraction'),
        ('Delta_Healthy_Sphericity', '\u0394 Healthy Sphericity'),
    ]

    available = [(col, label) for col, label in key_features
                 if col in fm_idx.columns]
    if len(available) < 2:
        return _placeholder(fp, 'Score vs Features', 'Insufficient features')

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for idx, (col, label) in enumerate(available[:4]):
        ax = axes[idx]
        valid = pd.DataFrame({
            'Score': m['Score'].values,
            'Feature': fm_idx[col].values,
        }).dropna()

        if len(valid) < 5:
            ax.text(0.5, 0.5, 'Insufficient data', transform=ax.transAxes,
                    ha='center', va='center', color='gray')
            continue

        ax.scatter(valid['Feature'], valid['Score'], c='#2C3E50', alpha=0.7,
                   s=60, edgecolors='white', linewidth=0.5)
        r_val, p_val = pearsonr(valid['Feature'], valid['Score'])
        ax.set_xlabel(label, fontsize=10)
        ax.set_ylabel('PCA Composite Score', fontsize=10)
        ax.set_title(f'r = {r_val:.3f}, p = {p_val:.2e}',
                     fontweight='bold', loc='center', fontsize=10)
        _panel_label(ax, chr(65 + idx))

    for idx in range(len(available), 4):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig(fp, dpi=300, bbox_inches='tight')
    plt.close()


def _placeholder(fp, title, msg):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.text(0.5, 0.5, f'{title}\n\n{msg}', transform=ax.transAxes,
            ha='center', va='center', fontsize=16, color='gray',
            bbox=dict(boxstyle='round', fc='lightgray', alpha=0.3))
    ax.axis('off')
    ax.set_title(title, fontweight='bold')
    plt.savefig(fp, dpi=150, bbox_inches='tight')
    plt.close()