import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy.stats import pearsonr
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from cluster_utils import PROCESSED_FEATURES
from .config import CLUSTER_COLORS, CONC_MAP, FIGURES_DIR

plt.rcParams.update({'font.size': 10, 'figure.dpi': 150, 'savefig.dpi': 300})


def generate_figures(df, ws, fm, merged, res, cdf):
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
            func(df, ws, fm, merged, res, cdf, path)
            print(f'OK {fn}')
        except Exception as e:
            print(f'ERR {fn}: {e}')


def fig1(df, ws, fm, merged, res, cdf, fp):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    feats = [f for f in PROCESSED_FEATURES if f in df.columns]
    X = df[feats].dropna().values
    if len(X) > 100:
        Xs = StandardScaler().fit_transform(X)
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(50, len(Xs) - 1))
        X2 = tsne.fit_transform(Xs)
        lbl = df.dropna(subset=feats)['Cluster'].values
        for c in sorted(CLUSTER_COLORS.keys()):
            m = lbl == c
            ax1.scatter(X2[m, 0], X2[m, 1], c=CLUSTER_COLORS[c], label=f'{c}', s=2, alpha=0.5, rasterized=True)
        ax1.set_title('A t-SNE Projection', fontweight='bold', loc='left')
        ax1.legend(markerscale=5)

    prof = df.groupby('Cluster')[feats].mean()
    if len(prof) > 0:
        pz = (prof - prof.mean()) / prof.std()
        im = ax2.imshow(pz.values, cmap='RdBu_r', aspect='auto', vmin=-2, vmax=2)
        ax2.set_xticks(range(len(feats)))
        ax2.set_xticklabels([f.replace('_', '\n') for f in feats], rotation=45, ha='right', fontsize=8)
        ax2.set_title('B Cluster Profiles (Z-score)', fontweight='bold', loc='left')
        plt.colorbar(im, ax=ax2, shrink=0.8)

    plt.tight_layout()
    plt.savefig(fp, dpi=300, bbox_inches='tight')
    plt.close()


def fig2(df, ws, fm, merged, res, cdf, fp):
    if res is None or merged is None:
        return placeholder(fp, 'Score-ATP Correlation', 'No data')

    m = merged.merge(
        ws[ws['_day'] == '0703'][['_well_id', 'Healthy_Fraction']],
        left_on='Well_ID',
        right_on='_well_id',
        how='inner',
    )
    if len(m) < 5:
        return placeholder(fp, 'Score-ATP Correlation', 'Insufficient')

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    xs, at, h = m['Score'].values, m['ATP'].values, m['Healthy_Fraction'].values * 100

    axes[0].scatter(xs, at, c='#3498DB', alpha=0.6, s=60, edgecolors='white')
    axes[0].set_xlabel('PCA Score')
    axes[0].set_ylabel('ATP')
    axes[0].set_title(f"A Score vs ATP (r={res['pearson_r']:.3f}, p={res['pearson_p']:.2e})", fontweight='bold', loc='left')

    axes[1].scatter(h, xs, c='#2ECC71', alpha=0.6, s=60, edgecolors='white')
    z = np.polyfit(h, xs, 1)
    p = np.poly1d(z)
    xf = np.linspace(h.min(), h.max(), 100)
    axes[1].plot(xf, p(xf), 'r--', lw=2)
    r, _ = pearsonr(h, xs)
    axes[1].set_xlabel('Healthy%')
    axes[1].set_ylabel('Score')
    axes[1].set_title(f'B Healthy%->Score (r={r:.3f})', fontweight='bold', loc='left')

    axes[2].scatter(h, at, c='#E74C3C', alpha=0.6, s=60, edgecolors='white')
    z = np.polyfit(h, at, 1)
    p = np.poly1d(z)
    axes[2].plot(xf, p(xf), 'r--', lw=2)
    r, _ = pearsonr(h, at)
    axes[2].set_xlabel('Healthy%')
    axes[2].set_ylabel('ATP')
    axes[2].set_title(f'C Healthy%->ATP (r={r:.3f})', fontweight='bold', loc='left')

    plt.tight_layout()
    plt.savefig(fp, dpi=300, bbox_inches='tight')
    plt.close()


def fig3(df, ws, fm, merged, res, cdf, fp):
    concs = [0, 20, 40, 80]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    wd5 = ws[ws['_day'] == '0703'].copy()
    wd5['Conc'] = wd5['_well_id'].map(CONC_MAP)

    for idx, met in enumerate(['Healthy_Fraction', 'Blue_Fraction']):
        ax = axes[idx]
        data = []
        for c in concs:
            v = wd5.loc[wd5['Conc'] == c, met] * 100
            data.append(v)
        bp = ax.boxplot(
            data,
            positions=concs,
            widths=8,
            patch_artist=True,
            showfliers=True,
            flierprops=dict(marker='o', ms=4, alpha=0.5),
        )
        for p in bp['boxes']:
            p.set_facecolor('#A3E4D7' if idx == 0 else '#85C1E9')
        means = [np.mean(d) for d in data]
        ax.plot(concs, means, '-o', color='#E74C3C', lw=2.5, ms=8)
        ax.set_xlabel('Conc (uM)')
        ax.set_ylabel(met.replace('_', ' ') + ' (%)')
        ax.set_title(('A Healthy%' if idx == 0 else 'B Blue%') + ' Dose Response', fontweight='bold', loc='left')
        ax.set_xticks(concs)

    ax = axes[2]
    comp = wd5.groupby('Conc').agg({'Red': 'sum', 'Yellow': 'sum', 'Green': 'sum', 'Blue': 'sum'})
    total = comp.sum(axis=1)
    comp_pct = comp.div(total, axis=0) * 100
    bottom = np.zeros(4)
    for cn, cl in zip(['Red', 'Yellow', 'Green', 'Blue'], CLUSTER_COLORS.values()):
        if cn in comp_pct.columns:
            h = [comp_pct.loc[c, cn] if c in comp_pct.index else 0 for c in concs]
            ax.bar(concs, h, width=12, bottom=bottom, color=cl, label=cn, alpha=0.85)
            bottom += np.array(h)
    ax.set_xlabel('Conc (uM)')
    ax.set_ylabel('%')
    ax.set_title('C Composition by Dose', fontweight='bold', loc='left')
    ax.set_xticks(concs)
    ax.legend()
    ax.set_ylim(0, 105)

    plt.tight_layout()
    plt.savefig(fp, dpi=300, bbox_inches='tight')
    plt.close()


def fig4(df, ws, fm, merged, res, cdf, fp):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    if merged is not None and 'Score' in merged.columns:
        ids = merged['Well_ID'].values
        scores = merged['Score'].values
        cols = ['#3498DB' if CONC_MAP.get(w, -1) == 0 else '#E74C3C' for w in ids]
        ax1.barh(range(len(ids)), scores, color=cols, alpha=0.7, height=0.7)
        ax1.set_yticks(range(len(ids)))
        ax1.set_yticklabels(ids, fontsize=8)
        ax1.set_xlabel('Score')
        ax1.set_title('A Score by Well', fontweight='bold', loc='left')
        ax1.axvline(0, color='gray', ls='--', alpha=0.5)
        ax1.legend([Patch(fc='#3498DB'), Patch(fc='#E74C3C')], ['Control', 'Treated'])

    if cdf is not None and len(cdf) > 0:
        tn = min(12, len(cdf))
        tc = cdf.head(tn)
        colors = ['#E74C3C' if c > 0 else '#3498DB' for c in tc['Coef']]
        ax2.barh(range(tn), tc['Coef'], color=colors, alpha=0.75)
        ax2.set_yticks(range(tn))
        ax2.set_yticklabels([f.replace('_', ' ')[:18] for f in tc['Feature']], fontsize=8)
        ax2.set_xlabel('Weight')
        ax2.set_title('B Feature Weights', fontweight='bold', loc='left')
        ax2.axvline(0, color='black', lw=0.8)
        ax2.invert_yaxis()

    plt.tight_layout()
    plt.savefig(fp, dpi=300, bbox_inches='tight')
    plt.close()


def fig5(df, ws, fm, merged, res, cdf, fp):
    if cdf is None or len(cdf) < 3:
        return placeholder(fp, 'Radar', 'No data')

    tn = min(10, len(cdf))
    tv = cdf.head(tn)['AbsCoef'].values
    tvn = tv / tv.max()
    angles = np.linspace(0, 2 * np.pi, tn, endpoint=False).tolist() + [0]
    values = list(tvn) + [tvn[0]]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    ax.fill(angles, values, color='#3498DB', alpha=0.25)
    ax.plot(angles, values, color='#3498DB', lw=2)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(
        [f.replace('Delta_Healthy_', 'D ').replace('_', ' ')[:18] for f in cdf.head(tn)['Feature']],
        fontsize=9,
    )
    ax.set_title('Top 10 Feature Weights', fontweight='bold', pad=25)

    plt.tight_layout()
    plt.savefig(fp, dpi=300, bbox_inches='tight')
    plt.close()


def fig6(df, ws, fm, merged, res, cdf, fp):
    if merged is None or 'Score' not in merged.columns or 'ATP' not in merged.columns:
        return placeholder(fp, 'Score Distribution', 'No data')

    m = merged.copy()
    m['Conc'] = m['Well_ID'].map(CONC_MAP)
    concs = [0, 20, 40, 80]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    data = [m.loc[m['Conc'] == c, 'Score'].values for c in concs]
    bp = axes[0, 0].boxplot(
        data,
        positions=concs,
        widths=8,
        patch_artist=True,
        showfliers=True,
        flierprops=dict(marker='o', ms=4, alpha=0.5),
    )
    colors = ['#3498DB', '#2ECC71', '#F39C12', '#E74C3C']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    axes[0, 0].set_xlabel('Concentration (uM)')
    axes[0, 0].set_ylabel('PCA Score')
    axes[0, 0].set_title('A Score by Concentration', fontweight='bold', loc='left')
    axes[0, 0].set_xticks(concs)

    axes[0, 1].scatter(m['ATP'], m['Score'], c='#3498DB', alpha=0.6, s=50, edgecolors='white')
    axes[0, 1].set_xlabel('ATP')
    axes[0, 1].set_ylabel('PCA Score')
    axes[0, 1].set_title(f"B Score vs ATP (r={res['pearson_r']:.3f})", fontweight='bold', loc='left')

    scores = m['Score'].values
    axes[1, 0].hist(scores, bins=12, color='#3498DB', alpha=0.7, edgecolor='white')
    axes[1, 0].axvline(0, color='red', ls='--', lw=1.5)
    axes[1, 0].set_xlabel('PCA Score')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('C Score Distribution', fontweight='bold', loc='left')

    atp_vals = m['ATP'].values / 1e6
    axes[1, 1].hist(atp_vals, bins=12, color='#E74C3C', alpha=0.7, edgecolor='white')
    axes[1, 1].set_xlabel('ATP (x10^6)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('D ATP Distribution', fontweight='bold', loc='left')

    plt.tight_layout()
    plt.savefig(fp, dpi=300, bbox_inches='tight')
    plt.close()


def placeholder(fp, title, msg):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.text(
        0.5,
        0.5,
        f'{title}\n\n{msg}',
        transform=ax.transAxes,
        ha='center',
        va='center',
        fontsize=16,
        color='gray',
        bbox=dict(boxstyle='round', fc='lightgray', alpha=0.3),
    )
    ax.axis('off')
    ax.set_title(title, fontweight='bold')
    plt.savefig(fp, dpi=150, bbox_inches='tight')
    plt.close()