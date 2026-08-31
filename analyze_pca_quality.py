"""
PCA Quality Assessment: KMO, Bartlett, Collinearity & Feature Subset Comparison

Evaluates the ~45D extended feature matrix for PCA suitability, identifies
redundant features, and compares PCA performance across different feature subsets.

Output:
  reports/figures/v2_stratified_v.2.0/
    kmo_bartlett_report.txt           - KMO & Bartlett test results
    feature_correlation_heatmap.png   - Full correlation matrix heatmap
    feature_subset_benchmark.png      - PCA performance across subsets
    subset_pca_scree_grid.png         - Scree plots for each subset
"""

import os, sys, warnings
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import stats
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from step7_pca_atp_v2_stratified.config import (
    ATP_DATABASE, CLUSTER_COLORS, CLUSTER_NAMES, HEALTHY_CLUSTERS,
    MIN_HEALTHY_SAMPLES, MODEL_PATH, FIGURES_DIR,
)
from step7_pca_atp_v2_stratified.data_io import load_organoid_data, validate_data_structure
from step7_pca_atp_v2_stratified.modeling import (
    apply_clustering, stratified_median_aggregation,
)

os.makedirs(FIGURES_DIR, exist_ok=True)

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 9,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})


def load_extended_feature_matrix():
    print('=' * 70)
    print('Step 0: Loading Extended Feature Matrix')
    print('=' * 70)
    validate_data_structure()
    df, feats, wells = load_organoid_data()
    df, ws = apply_clustering(df, feats)
    fm, extended_sel, d3_feats, d5_feats = stratified_median_aggregation(
        df, ws, feats, wells
    )
    fm['ATP'] = fm['Well_ID'].map(ATP_DATABASE)
    return fm, extended_sel, d3_feats, d5_feats


def compute_kmo(X):
    """
    Kaiser-Meyer-Olkin Measure of Sampling Adequacy.

    KMO_j = Σ_{i≠j} r_{ij}^2 / (Σ_{i≠j} r_{ij}^2 + Σ_{i≠j} a_{ij}^2)

    where r_{ij} = correlation, a_{ij} = partial correlation (anti-image).

    Returns:
      kmo_per_feature: dict {feature_name: KMO_value}
      kmo_overall: float (overall KMO)
    """
    corr = np.corrcoef(X, rowvar=False)
    corr_reg = corr + np.eye(corr.shape[0]) * 1e-8
    n = corr_reg.shape[0]

    inv_corr = np.linalg.pinv(corr_reg)
    aic = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                aic[i, j] = -inv_corr[i, j] / np.sqrt(inv_corr[i, i] * inv_corr[j, j])

    kmo_per_feature = {}
    kmo_sum = 0.0
    kmo_count = 0
    for i in range(n):
        num = np.sum(corr[i, :] ** 2) - 1.0
        denom = num + (np.sum(aic[i, :] ** 2) - 1.0)
        kmo_i = num / denom if denom > 0 else 0.0
        kmo_per_feature[f'feat_{i}'] = kmo_i
        kmo_sum += num
        kmo_count += denom

    overall_kmo = kmo_sum / kmo_count if kmo_count > 0 else 0.0
    return kmo_per_feature, overall_kmo


def bartlett_test(X):
    """
    Bartlett's test of sphericity.

    H0: The correlation matrix is an identity matrix (no correlation).
    Test statistic: χ² = -(n - 1 - (2p + 5)/6) * ln|R|
    df = p(p - 1) / 2

    Returns:
      chi2: test statistic
      p_value: p-value
      df: degrees of freedom
    """
    n, p = X.shape
    corr = np.corrcoef(X, rowvar=False)
    corr_reg = corr + np.eye(p) * 1e-8
    det_corr = np.linalg.det(corr_reg)
    if det_corr <= 0:
        det_corr = 1e-10
    chi2 = -(n - 1 - (2 * p + 5) / 6) * np.log(det_corr)
    df = p * (p - 1) / 2
    p_value = 1 - stats.chi2.cdf(chi2, df)
    return chi2, p_value, df


def compute_vif(X):
    """
    Variance Inflation Factor for each feature.
    VIF_j = 1 / (1 - R_j^2) where R_j^2 is the R² of regressing feature j on all others.

    VIF > 10: severe multicollinearity
    VIF > 5:  moderate multicollinearity
    """
    n, p = X.shape
    vif_values = np.zeros(p)
    for i in range(p):
        y = X[:, i]
        X_others = np.delete(X, i, axis=1)
        X_others = np.column_stack([np.ones(n), X_others])
        try:
            coeffs, _, _, _ = np.linalg.lstsq(X_others, y, rcond=None)
            y_pred = X_others @ coeffs
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            vif_values[i] = 1 / (1 - r_squared) if r_squared < 1 else np.inf
        except np.linalg.LinAlgError:
            vif_values[i] = np.inf
    return vif_values


def compute_condition_number(X):
    """
    Condition number of the correlation matrix.
    κ = λ_max / λ_min

    κ < 100:  no serious multicollinearity
    κ 100-1000: moderate to strong
    κ > 1000:  severe
    """
    corr = np.corrcoef(X, rowvar=False)
    corr_reg = corr + np.eye(corr.shape[0]) * 1e-8
    eigenvalues = np.linalg.eigvalsh(corr_reg)
    eigenvalues = np.maximum(eigenvalues, 1e-10)
    return np.sqrt(eigenvalues.max() / eigenvalues.min())


def _filter_and_impute(fm, feature_names, min_completeness=0.5):
    completeness = fm[feature_names].notna().mean()
    valid_feats = completeness[completeness >= min_completeness].index.tolist()
    dropped = [f for f in feature_names if f not in valid_feats]
    if dropped:
        print(f'  Dropped {len(dropped)} features with <{min_completeness:.0%} completeness:')
        for d in dropped:
            print(f'    - {d} ({completeness[d]:.0%})')

    X_raw = fm[valid_feats].copy()
    for col in valid_feats:
        if X_raw[col].isna().any():
            X_raw[col] = X_raw[col].fillna(X_raw[col].median())
    valid_rows = X_raw.notna().all(axis=1)
    X = X_raw[valid_rows].values
    return X, valid_feats, valid_rows


def run_kmo_bartlett(fm, feature_names):
    print('\n' + '=' * 70)
    print('ANALYSIS 1: KMO & Bartlett Test')
    print('=' * 70)

    X, valid_feats, valid_rows = _filter_and_impute(fm, feature_names, min_completeness=0.5)
    X_s = StandardScaler().fit_transform(X)
    n, p = X_s.shape

    print(f'Samples: {n}, Features: {p} (from {len(feature_names)} total)')
    print(f'd/n ratio: {p/n:.2f}')

    kmo_feat, kmo_overall = compute_kmo(X_s)
    chi2, p_val, df = bartlett_test(X_s)

    print(f'\n--- KMO Test ---')
    print(f'Overall KMO: {kmo_overall:.4f}')
    if kmo_overall >= 0.9:
        print('  Interpretation: Marvelous')
    elif kmo_overall >= 0.8:
        print('  Interpretation: Meritorious')
    elif kmo_overall >= 0.7:
        print('  Interpretation: Middling (acceptable, like the paper)')
    elif kmo_overall >= 0.6:
        print('  Interpretation: Mediocre')
    elif kmo_overall >= 0.5:
        print('  Interpretation: Miserable')
    else:
        print('  Interpretation: Unacceptable')

    print(f'\n--- Bartlett Test ---')
    print(f'χ² = {chi2:.2f}, df = {df:.0f}, p = {p_val:.2e}')
    if p_val < 0.001:
        print('  Interpretation: Significant (p < 0.001) - data suitable for PCA')
    else:
        print('  Interpretation: NOT significant - data NOT suitable for PCA')

    print(f'\n--- Condition Number ---')
    cond_num = compute_condition_number(X_s)
    print(f'κ = {cond_num:.1f}')
    if cond_num < 100:
        print('  Interpretation: No serious multicollinearity')
    elif cond_num < 1000:
        print('  Interpretation: Moderate multicollinearity')
    else:
        print('  Interpretation: SEVERE multicollinearity')

    print(f'\n--- VIF Summary ---')
    vifs = compute_vif(X_s)
    n_high = np.sum(vifs > 10)
    n_moderate = np.sum((vifs > 5) & (vifs <= 10))
    print(f'VIF > 10 (severe): {n_high}/{p} features')
    print(f'VIF 5-10 (moderate): {n_moderate}/{p} features')
    print(f'VIF < 5 (acceptable): {p - n_high - n_moderate}/{p} features')

    top_vif_idx = np.argsort(vifs)[::-1][:10]
    print(f'\nTop 10 features by VIF:')
    for rank, idx in enumerate(top_vif_idx, 1):
        flag = '⚠️' if vifs[idx] > 10 else ('⚡' if vifs[idx] > 5 else '  ')
        feat_name = valid_feats[idx] if idx < len(valid_feats) else f'feat_{idx}'
        print(f'  {rank:2d}. VIF={vifs[idx]:8.1f} {flag} {feat_name}')

    report_path = os.path.join(FIGURES_DIR, 'kmo_bartlett_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f'KMO & Bartlett Test Report\n')
        f.write(f'{"=" * 50}\n')
        f.write(f'Samples: {n}, Features: {p}, d/n: {p/n:.2f}\n')
        f.write(f'Overall KMO: {kmo_overall:.4f}\n')
        f.write(f'Bartlett χ²: {chi2:.2f}, df: {df:.0f}, p: {p_val:.2e}\n')
        f.write(f'Condition Number κ: {cond_num:.1f}\n')
        f.write(f'VIF > 10: {n_high}/{p}, VIF 5-10: {n_moderate}/{p}\n')
        f.write(f'\nTop 20 features by VIF:\n')
        for rank, idx in enumerate(top_vif_idx[:20], 1):
            feat_name = valid_feats[idx] if idx < len(valid_feats) else f'feat_{idx}'
            f.write(f'  {rank:2d}. VIF={vifs[idx]:8.1f}  {feat_name}\n')
    print(f'\nReport saved: {report_path}')

    return {
        'kmo_overall': kmo_overall,
        'bartlett_chi2': chi2,
        'bartlett_p': p_val,
        'condition_number': cond_num,
        'vifs': vifs,
        'n_high_vif': n_high,
    }


def plot_correlation_heatmap(fm, feature_names):
    print('\n' + '=' * 70)
    print('ANALYSIS 2: Feature Correlation Heatmap')
    print('=' * 70)

    X, valid_feats, _ = _filter_and_impute(fm, feature_names, min_completeness=0.5)
    corr = np.corrcoef(X, rowvar=False)
    feature_names = valid_feats

    short_names = _shorten_names(feature_names)

    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    n_redundant = np.sum(np.abs(corr[mask]) > 0.9)
    n_high_corr = np.sum(np.abs(corr[mask]) > 0.8)
    print(f'Feature pairs with |r| > 0.9: {n_redundant}')
    print(f'Feature pairs with |r| > 0.8: {n_high_corr}')

    top_pairs = []
    for i in range(len(feature_names)):
        for j in range(i + 1, len(feature_names)):
            if abs(corr[i, j]) > 0.85:
                top_pairs.append((abs(corr[i, j]), feature_names[i], feature_names[j]))
    top_pairs.sort(reverse=True)
    print(f'\nTop 15 highly correlated pairs (|r| > 0.85):')
    for rank, (r_val, f1, f2) in enumerate(top_pairs[:15], 1):
        flag = '⚠️' if r_val > 0.95 else '  '
        print(f'  {rank:2d}. r={r_val:+.4f} {flag} {f1[:50]}')
        print(f'       {" " * 8}  ↔ {f2[:50]}')

    n_features = len(feature_names)
    fig_size = max(14, n_features * 0.35)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.9))

    cmap = plt.cm.RdBu_r
    im = ax.imshow(corr, cmap=cmap, vmin=-1, vmax=1, aspect='auto')

    ax.set_xticks(range(n_features))
    ax.set_yticks(range(n_features))
    ax.set_xticklabels(short_names, rotation=90, ha='center', fontsize=5)
    ax.set_yticklabels(short_names, fontsize=5)

    for i in range(n_features):
        for j in range(n_features):
            if i != j and abs(corr[i, j]) > 0.7:
                color = 'white' if abs(corr[i, j]) > 0.6 else 'black'
                ax.text(j, i, f'{corr[i, j]:.2f}', ha='center', va='center',
                        fontsize=3.5, color=color, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Pearson r', fontsize=10)

    ax.set_title(f'Feature Correlation Matrix\n'
                 f'({n_features} features, |r|>0.9: {n_redundant} pairs, |r|>0.8: {n_high_corr} pairs)',
                 fontweight='bold', fontsize=12)

    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, 'feature_correlation_heatmap.png')
    fig.savefig(save_path)
    plt.close()
    print(f'Saved: {save_path}')

    return corr, top_pairs


def build_feature_subsets(fm, extended_sel):
    d5_feats = [f for f in extended_sel if '_D5' in f and not f.startswith('Delta_') and not f.startswith('RelChange_')]
    delta_feats = [f for f in extended_sel if f.startswith('Delta_')]
    relchange_feats = [f for f in extended_sel if f.startswith('RelChange_')]
    frac_feats = [f for f in extended_sel if 'Frac' in f and not f.startswith('Delta_') and not f.startswith('RelChange_')]
    delta_frac_feats = [f for f in extended_sel if f.startswith('Delta_') and 'Frac' in f]

    subsets = {
        'A: D5 Absolute Only': {
            'features': d5_feats,
            'description': f'{len(d5_feats)}D - Static endpoint values',
            'color': '#3498DB',
        },
        'B: Delta Only': {
            'features': delta_feats,
            'description': f'{len(delta_feats)}D - Absolute changes (D5-D3)',
            'color': '#E74C3C',
        },
        'C: RelChange Only': {
            'features': relchange_feats,
            'description': f'{len(relchange_feats)}D - Relative changes ((D5-D3)/|D3|)',
            'color': '#F39C12',
        },
        'D: D5 + Delta': {
            'features': d5_feats + delta_feats,
            'description': f'{len(d5_feats) + len(delta_feats)}D - Combined static + delta',
            'color': '#2ECC71',
        },
        'E: Delta + Fractions': {
            'features': delta_feats + delta_frac_feats,
            'description': f'{len(delta_feats) + len(delta_frac_feats)}D - Delta + cluster fractions',
            'color': '#9B59B6',
        },
        'F: All Features': {
            'features': extended_sel,
            'description': f'{len(extended_sel)}D - Full extended set',
            'color': '#E67E22',
        },
    }

    print('\n' + '=' * 70)
    print('Feature Subset Definitions')
    print('=' * 70)
    for name, cfg in subsets.items():
        n_valid = len([f for f in cfg['features'] if f in fm.columns])
        print(f'  {name}: {cfg["description"]} (valid: {n_valid})')

    return subsets


def run_subset_pca_benchmark(fm, subsets):
    print('\n' + '=' * 70)
    print('ANALYSIS 3: PCA Performance Across Feature Subsets')
    print('=' * 70)

    results = []

    for name, cfg in subsets.items():
        feats = [f for f in cfg['features'] if f in fm.columns]
        if len(feats) < 3:
            print(f'\n  {name}: SKIP (only {len(feats)} valid features)')
            continue

        valid = fm.dropna(subset=feats + ['ATP'])
        if len(valid) < 3:
            X, used_feats, valid_rows = _filter_and_impute(fm, feats, min_completeness=0.5)
            if X.shape[0] < 3:
                print(f'\n  {name}: SKIP (only {X.shape[0]} rows)')
                continue
            fm_sub = fm.loc[valid_rows.values]
            y = fm_sub['ATP'].values
        else:
            X = valid[feats].values
            y = valid['ATP'].values
        n, p = X.shape

        X_s = StandardScaler().fit_transform(X)

        _, kmo = compute_kmo(X_s)
        cond_num = compute_condition_number(X_s)

        max_comp = min(p, n - 1)
        pca_full = PCA(n_components=max_comp, random_state=42)
        pca_full.fit(X_s)

        eigenvalues = pca_full.explained_variance_
        evr = pca_full.explained_variance_ratio_
        cumvar = np.cumsum(evr)

        n_kaiser = max(1, int(np.sum(eigenvalues > 1.0)))
        n_70 = int(np.searchsorted(cumvar, 0.70)) + 1
        n_85 = int(np.searchsorted(cumvar, 0.85)) + 1

        for n_pc in [4, n_kaiser, n_70, n_85]:
            if n_pc > max_comp:
                continue
            if n_pc == 4 and 4 > max_comp:
                continue

            pca = PCA(n_components=n_pc, random_state=42)
            X_p = pca.fit_transform(X_s)
            vr = pca.explained_variance_ratio_
            wts = vr / vr.sum()
            score = np.dot(X_p, wts)

            r_p, p_val = pearsonr(score, y)
            if r_p < 0:
                r_p = -r_p

            results.append({
                'Subset': name,
                'N_PC': n_pc,
                'PC_Strategy': 'Kaiser' if n_pc == n_kaiser else ('70%' if n_pc == n_70 else ('85%' if n_pc == n_85 else 'Fixed4')),
                'N_Features': p,
                'N_Samples': n,
                'd_n_Ratio': p / n,
                'KMO': kmo,
                'Cond_Num': cond_num,
                'CumVar': np.sum(vr),
                'Pearson_r': r_p,
                'P_Value': p_val,
                'Eigenvalues': eigenvalues,
                'Cumulative_Var': cumvar,
                'Color': cfg['color'],
            })

        print(f'\n  {name} ({p}D, n={n}):')
        print(f'    KMO={kmo:.3f}, κ={cond_num:.1f}')
        for n_pc, tag in [(4, 'Fixed4'), (n_kaiser, 'Kaiser'), (n_70, '70%'), (n_85, '85%')]:
            if n_pc > max_comp:
                continue
            subset_results = [r for r in results if r['Subset'] == name and r['PC_Strategy'] == tag]
            if subset_results:
                r = subset_results[-1]
                print(f'    {tag} ({n_pc}PCs): r={r["Pearson_r"]:.4f} (d/n={p/n:.2f})')

    return results


def plot_subset_benchmark(results):
    print('\n' + '=' * 70)
    print('Generating Benchmark Figures')
    print('=' * 70)

    strategies = ['Fixed4', 'Kaiser', '70%', '85%']
    all_subsets = sorted(set(r['Subset'] for r in results))

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    for ax_idx, strategy in enumerate(strategies):
        ax = axes[ax_idx // 2][ax_idx % 2]
        strat_results = [r for r in results if r['PC_Strategy'] == strategy]

        for s in all_subsets:
            sr = [r for r in strat_results if r['Subset'] == s]
            if sr:
                r = sr[0]
                ax.bar(s, r['Pearson_r'], color=r['Color'], alpha=0.8, edgecolor='white', linewidth=0.5)
                ax.text(s, r['Pearson_r'] + 0.01, f'{r["Pearson_r"]:.3f}',
                        ha='center', va='bottom', fontsize=8, fontweight='bold')

        ax.axhline(y=0.938, color='#E74C3C', linestyle='--', linewidth=1.5, alpha=0.7, label='Paper r=0.938 (OAC)')
        ax.axhline(y=0.906, color='#3498DB', linestyle=':', linewidth=1.5, alpha=0.7, label='Paper r=0.906 (morph only)')
        ax.set_ylim(0.5, 1.0)
        ax.set_ylabel('Pearson r (Score vs ATP)', fontweight='bold')
        ax.set_title(f'PCA Strategy: {strategy} PCs', fontweight='bold', fontsize=11)
        ax.tick_params(axis='x', rotation=45, labelsize=7)
        ax.legend(fontsize=7, loc='lower right')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle('PCA Performance: Feature Subset × PC Strategy Benchmark',
                 fontweight='bold', fontsize=14, y=1.01)
    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, 'feature_subset_benchmark.png')
    fig.savefig(save_path)
    plt.close()
    print(f'Saved: {save_path}')

    fig2, axes2 = plt.subplots(2, 3, figsize=(18, 12))
    axes2 = axes2.flatten()

    for idx, s in enumerate(all_subsets):
        ax = axes2[idx]
        sr_list = [r for r in results if r['Subset'] == s]
        if not sr_list:
            continue
        sr = sr_list[0]
        eigenvalues = sr['Eigenvalues']
        cumvar = sr['Cumulative_Var']
        n_comp = len(eigenvalues)
        x = np.arange(1, n_comp + 1)

        ax.bar(x, eigenvalues / eigenvalues.sum() * 100, color=sr['Color'], alpha=0.7,
               edgecolor='white', linewidth=0.5, label='Variance %')
        ax.plot(x, cumvar * 100, 'o-', color='black', lw=2, ms=4, label='Cumulative %')

        ax.axhline(y=70, color='#E74C3C', linestyle='--', lw=1, alpha=0.5)
        ax.axhline(y=85, color='#F39C12', linestyle='--', lw=1, alpha=0.5)

        n_70 = int(np.searchsorted(cumvar, 0.70)) + 1
        n_85 = int(np.searchsorted(cumvar, 0.85)) + 1
        n_kaiser = max(1, int(np.sum(eigenvalues > 1.0)))

        ax.set_title(f'{s}\n'
                     f'(KMO={sr["KMO"]:.2f}, κ={sr["Cond_Num"]:.0f}, '
                     f'Kaiser={n_kaiser}PCs, 70%={n_70}, 85%={n_85})',
                     fontsize=9, fontweight='bold')
        ax.set_xlabel('Principal Component')
        ax.set_ylabel('Variance (%)')
        ax.legend(fontsize=7, loc='center right')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlim(0.5, min(15, n_comp + 0.5))

    for idx in range(len(all_subsets), len(axes2)):
        axes2[idx].set_visible(False)

    fig2.suptitle('PCA Scree Plots: Feature Subset Comparison',
                  fontweight='bold', fontsize=14)
    plt.tight_layout()
    save_path2 = os.path.join(FIGURES_DIR, 'subset_pca_scree_grid.png')
    fig2.savefig(save_path2)
    plt.close()
    print(f'Saved: {save_path2}')

    print('\n' + '=' * 70)
    print('BENCHMARK SUMMARY TABLE')
    print('=' * 70)
    print(f'{"Subset":<22s} {"d":>3s} {"n":>3s} {"d/n":>5s} {"KMO":>6s} {"κ":>7s} {"Kaiser":>6s} {"r_K":>6s} {"r_F4":>6s} {"r_85":>6s}')
    print('-' * 85)
    for s in all_subsets:
        sr_list = [r for r in results if r['Subset'] == s]
        if not sr_list:
            continue
        sr = sr_list[0]
        r_K = next((r['Pearson_r'] for r in results if r['Subset'] == s and r['PC_Strategy'] == 'Kaiser'), 0)
        r_F4 = next((r['Pearson_r'] for r in results if r['Subset'] == s and r['PC_Strategy'] == 'Fixed4'), 0)
        r_85 = next((r['Pearson_r'] for r in results if r['Subset'] == s and r['PC_Strategy'] == '85%'), 0)
        n_kaiser = max(1, int(np.sum(sr['Eigenvalues'] > 1.0)))
        print(f'{s:<22s} {sr["N_Features"]:>3d} {sr["N_Samples"]:>3d} '
              f'{sr["d_n_Ratio"]:>5.2f} {sr["KMO"]:>6.3f} {sr["Cond_Num"]:>7.1f} '
              f'{n_kaiser:>6d} {r_K:>6.4f} {r_F4:>6.4f} {r_85:>6.4f}')

    return results


def _shorten_names(feature_names):
    short = []
    for f in feature_names:
        s = f
        s = s.replace('Healthy_', 'H_')
        s = s.replace('RelChange_', 'RC_')
        s = s.replace('Delta_', 'Δ_')
        s = s.replace('Fraction', 'Frac')
        s = s.replace('Sphericity', 'Sph')
        s = s.replace('Volume', 'Vol')
        s = s.replace('Cavity', 'Cav')
        s = s.replace('Scatt', 'Sc')
        s = s.replace('Mean', 'M')
        s = s.replace('STD', 'S')
        s = s.replace('Surface', 'Surf')
        s = s.replace('Long', 'L')
        s = s.replace('Short', 'Sh')
        s = s.replace('Axis', 'Ax')
        s = s.replace('Number', 'Num')
        s = s.replace('Red_', 'R_')
        s = s.replace('Yellow', 'Yel')
        s = s.replace('_D5', '')
        s = s.replace('_D3', '')
        s = s.replace('_Avg', '')
        s = s.replace('_All', '')
        short.append(s[:35])
    return short


def main():
    print('\n' + '=' * 70)
    print(' ' * 15 + 'PCA QUALITY ASSESSMENT')
    print(' ' * 20 + 'KMO | Bartlett | VIF | Subset Benchmark')
    print('=' * 70)

    fm, extended_sel, d3_feats, d5_feats = load_extended_feature_matrix()

    valid_feats = [f for f in extended_sel if f in fm.columns]
    print(f'\nExtended features: {len(extended_sel)} total, {len(valid_feats)} valid in matrix')

    results = {}

    results['kmo_bartlett'] = run_kmo_bartlett(fm, valid_feats)

    results['corr_matrix'], results['top_pairs'] = plot_correlation_heatmap(fm, valid_feats)

    subsets = build_feature_subsets(fm, valid_feats)
    results['benchmark'] = run_subset_pca_benchmark(fm, subsets)

    plot_subset_benchmark(results['benchmark'])

    print('\n' + '=' * 70)
    print('ANALYSIS COMPLETE')
    print(f'Output: {FIGURES_DIR}/')
    print('=' * 70)

    return results


if __name__ == '__main__':
    main()