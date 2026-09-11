import os

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from cluster_utils import CV_FEATURES, FEATURE_SHORT_NAMES, GR_FEATURES, IQR_FEATURES, load_model_package
from .config import (
    ATP_DATABASE, CLUSTER_COLORS, CLUSTER_NAMES,
    DAMAGED_CLUSTERS, HEALTHY_CLUSTERS,
    MIN_HEALTHY_SAMPLES, MIN_SAMPLES_FOR_CV, MIN_SAMPLES_FOR_SKEW, MODEL_PATH,
)


def apply_clustering(df, features):
    print('\n' + '=' * 70)
    print('Step 2: KMeans Clustering -> Phenotype Statistics')
    print('=' * 70)

    pkg = load_model_package(MODEL_PATH)
    preprocessor = pkg.get('preprocessor')
    model = pkg.get('model') or pkg.get('kmeans') or pkg.get('gmm')
    if model is None:
        raise RuntimeError('Model package missing model object (expected one of: model/kmeans/gmm)')

    raw_to_final = pkg.get('raw_to_final', {})

    # Prefer model-declared feature order when available.
    model_features = pkg.get('feature_names') or features
    missing = [c for c in model_features if c not in df.columns]
    if missing:
        raise RuntimeError(f'Missing feature columns required by model: {missing}')

    X = df[model_features].values
    if preprocessor is not None:
        if not hasattr(preprocessor, 'mode'):
            preprocessor.mode = 'full'
        X_proc = preprocessor.transform(df[model_features])
    else:
        scaler = pkg.get('scaler')
        if scaler is None:
            raise RuntimeError('Model package missing both preprocessor and scaler')
        X_proc = scaler.transform(X)

    labels = np.array([raw_to_final.get(l, l) for l in model.predict(X_proc)])
    df = df.copy()
    df['Cluster'] = labels

    print('\nCluster distribution:')
    for c in sorted(CLUSTER_COLORS.keys()):
        cnt = (labels == c).sum()
        print(f'  {CLUSTER_NAMES[c]:12s}: {cnt:>6,} ({cnt / len(labels) * 100:>5.1f}%)')

    stats_list = []
    for (_w, _d), g in df.groupby(['_well', '_day']):
        tot = len(g)
        vc = g['Cluster'].value_counts()
        r, y, g_, b = [vc.get(i, 0) for i in range(4)]
        h = r + y
        stats_list.append(
            {
                '_well': _w,
                '_day': _d,
                '_well_id': _w.split('_')[0] if '_' in _w else _w,
                'Total': tot,
                'Red': r,
                'Yellow': y,
                'Green': g_,
                'Blue': b,
                'Healthy': h,
                'Red_Fraction': r / tot,
                'Yellow_Fraction': y / tot,
                'Healthy_Fraction': h / tot,
                'Blue_Fraction': b / tot,
            }
        )

    ws = pd.DataFrame(stats_list)
    print(f'\nComputed stats for {len(ws)} well-day combinations')
    return df, ws


def stratified_median_aggregation(df, ws, feats, wells):
    print('\n' + '=' * 70)
    print('Step 3: Stratified Feature Extraction (Median + CV + IQR + Ratio + GR + Roughness)')
    print('=' * 70)

    short_names = {f: FEATURE_SHORT_NAMES.get(f, f) for f in feats}
    has_roughness = 'Roughness' in df.columns

    rows = []

    for wid in wells:
        row = {'Well_ID': wid}

        d5 = df[(df['_well_id'] == wid) & (df['_day'] == '0703')]
        d3 = df[(df['_well_id'] == wid) & (df['_day'] == '0701')]

        c01_5 = d5[d5['Cluster'].isin(HEALTHY_CLUSTERS)]
        c01_3 = d3[d3['Cluster'].isin(HEALTHY_CLUSTERS)]
        c23_5 = d5[d5['Cluster'].isin(DAMAGED_CLUSTERS)]
        c23_3 = d3[d3['Cluster'].isin(DAMAGED_CLUSTERS)]

        n_c01_5, n_c01_3 = len(c01_5), len(c01_3)
        n_c23_5, n_c23_3 = len(c23_5), len(c23_3)
        n_d5, n_d3 = len(d5), len(d3)

        for f in feats:
            sn = short_names[f]

            # === Group A: Median (per-cluster + global) ===
            if n_c01_5 >= MIN_HEALTHY_SAMPLES:
                row[f'C01_Med_{sn}_D5'] = c01_5[f].median()
            else:
                row[f'C01_Med_{sn}_D5'] = np.nan

            if n_c01_3 >= MIN_HEALTHY_SAMPLES:
                row[f'C01_Med_{sn}_D3'] = c01_3[f].median()
            else:
                row[f'C01_Med_{sn}_D3'] = np.nan

            if n_d5 > 0:
                row[f'All_Med_{sn}_D5'] = d5[f].median()
            else:
                row[f'All_Med_{sn}_D5'] = np.nan

            if n_d3 > 0:
                row[f'All_Med_{sn}_D3'] = d3[f].median()
            else:
                row[f'All_Med_{sn}_D3'] = np.nan

            # === Group B: CV (heterogeneity, subset only) ===
            if sn in CV_FEATURES:
                if n_c01_5 >= MIN_SAMPLES_FOR_CV:
                    med = c01_5[f].median()
                    row[f'C01_CV_{sn}_D5'] = (c01_5[f].std() / med) if (med and med != 0) else np.nan
                else:
                    row[f'C01_CV_{sn}_D5'] = np.nan

                if n_c01_3 >= MIN_SAMPLES_FOR_CV:
                    med = c01_3[f].median()
                    row[f'C01_CV_{sn}_D3'] = (c01_3[f].std() / med) if (med and med != 0) else np.nan
                else:
                    row[f'C01_CV_{sn}_D3'] = np.nan

            # === Group C: Growth Rate (relative D5/D3, subset only) ===
            if sn in GR_FEATURES:
                v5 = row.get(f'C01_Med_{sn}_D5')
                v3 = row.get(f'C01_Med_{sn}_D3')
                if pd.notna(v5) and pd.notna(v3) and v3 != 0:
                    row[f'C01_GR_{sn}'] = v5 / v3
                else:
                    row[f'C01_GR_{sn}'] = np.nan

            # === Group D: IQR (robust dispersion, subset only) ===
            if sn in IQR_FEATURES:
                if n_c01_5 >= MIN_SAMPLES_FOR_CV:
                    row[f'C01_IQR_{sn}_D5'] = c01_5[f].quantile(0.75) - c01_5[f].quantile(0.25)
                else:
                    row[f'C01_IQR_{sn}_D5'] = np.nan

        # === Group F: Cross-cluster ratios (C01 vs C23) ===
        if n_c01_5 >= MIN_HEALTHY_SAMPLES and n_c23_5 >= MIN_HEALTHY_SAMPLES:
            c01_fill = c01_5['Organoids_Volume_Fill'].median()
            c23_fill = c23_5['Organoids_Volume_Fill'].median()
            row['C01_C23_VolRatio_D5'] = c01_fill / c23_fill if c23_fill != 0 else np.nan
        else:
            row['C01_C23_VolRatio_D5'] = np.nan

        if n_c01_5 >= MIN_HEALTHY_SAMPLES and n_c23_5 >= MIN_HEALTHY_SAMPLES:
            row['C01_C23_CountRatio_D5'] = n_c01_5 / n_c23_5
        else:
            row['C01_C23_CountRatio_D5'] = np.nan

        if n_c01_5 >= MIN_HEALTHY_SAMPLES and n_c23_5 >= MIN_HEALTHY_SAMPLES:
            c01_oac = c01_5['Scatt_Mean'].median()
            c23_oac = c23_5['Scatt_Mean'].median()
            row['C01_C23_OACmRatio_D5'] = c01_oac / c23_oac if c23_oac != 0 else np.nan
        else:
            row['C01_C23_OACmRatio_D5'] = np.nan

        if n_c01_3 >= MIN_HEALTHY_SAMPLES and n_c23_3 >= MIN_HEALTHY_SAMPLES:
            c01_fill_3 = c01_3['Organoids_Volume_Fill'].median()
            c23_fill_3 = c23_3['Organoids_Volume_Fill'].median()
            row['C01_C23_VolRatio_D3'] = c01_fill_3 / c23_fill_3 if c23_fill_3 != 0 else np.nan
        else:
            row['C01_C23_VolRatio_D3'] = np.nan

        if n_c01_3 >= MIN_HEALTHY_SAMPLES and n_c23_3 >= MIN_HEALTHY_SAMPLES:
            row['C01_C23_CountRatio_D3'] = n_c01_3 / n_c23_3
        else:
            row['C01_C23_CountRatio_D3'] = np.nan

        if n_c01_3 >= MIN_HEALTHY_SAMPLES and n_c23_3 >= MIN_HEALTHY_SAMPLES:
            c01_oac_3 = c01_3['Scatt_Mean'].median()
            c23_oac_3 = c23_3['Scatt_Mean'].median()
            row['C01_C23_OACmRatio_D3'] = c01_oac_3 / c23_oac_3 if c23_oac_3 != 0 else np.nan
        else:
            row['C01_C23_OACmRatio_D3'] = np.nan

        # === Group G: Fraction features (cluster proportions) ===
        w5 = ws[(ws['_well_id'] == wid) & (ws['_day'] == '0703')]
        w3 = ws[(ws['_well_id'] == wid) & (ws['_day'] == '0701')]

        if len(w5) > 0:
            row['C0_Frac_D5'] = w5.iloc[0]['Red_Fraction']
            row['C1_Frac_D5'] = w5.iloc[0]['Yellow_Fraction']
            row['C01_Frac_D5'] = w5.iloc[0]['Healthy_Fraction']
            row['C23_Frac_D5'] = 1.0 - w5.iloc[0]['Healthy_Fraction']

        if len(w3) > 0:
            row['C0_Frac_D3'] = w3.iloc[0]['Red_Fraction']
            row['C1_Frac_D3'] = w3.iloc[0]['Yellow_Fraction']
            row['C01_Frac_D3'] = w3.iloc[0]['Healthy_Fraction']
            row['C23_Frac_D3'] = 1.0 - w3.iloc[0]['Healthy_Fraction']

        # === Group H: Roughness features ===
        if has_roughness:
            if n_c01_5 >= MIN_HEALTHY_SAMPLES:
                rough_c01 = c01_5['Roughness'].dropna()
                row['C01_Med_Rough_D5'] = rough_c01.median() if len(rough_c01) >= MIN_HEALTHY_SAMPLES else np.nan
            else:
                row['C01_Med_Rough_D5'] = np.nan

            if n_d5 > 0:
                rough_all = d5['Roughness'].dropna()
                row['All_Med_Rough_D5'] = rough_all.median() if len(rough_all) >= MIN_HEALTHY_SAMPLES else np.nan
            else:
                row['All_Med_Rough_D5'] = np.nan

        rows.append(row)

    fm = pd.DataFrame(rows)

    # Collect feature groups for reporting
    c01_med_d5 = [f'C01_Med_{short_names[f]}_D5' for f in feats]
    c01_med_d3 = [f'C01_Med_{short_names[f]}_D3' for f in feats]
    all_med_d5 = [f'All_Med_{short_names[f]}_D5' for f in feats]
    all_med_d3 = [f'All_Med_{short_names[f]}_D3' for f in feats]

    c01_cv_d5 = [f'C01_CV_{sn}_D5' for sn in CV_FEATURES]
    c01_cv_d3 = [f'C01_CV_{sn}_D3' for sn in CV_FEATURES]

    c01_gr = [f'C01_GR_{sn}' for sn in GR_FEATURES]

    c01_iqr = [f'C01_IQR_{sn}_D5' for sn in IQR_FEATURES]

    ratio_d5_feats = ['C01_C23_VolRatio_D5', 'C01_C23_CountRatio_D5', 'C01_C23_OACmRatio_D5']
    ratio_d3_feats = ['C01_C23_VolRatio_D3', 'C01_C23_CountRatio_D3', 'C01_C23_OACmRatio_D3']

    frac_d5 = ['C0_Frac_D5', 'C1_Frac_D5', 'C01_Frac_D5', 'C23_Frac_D5']
    frac_d3 = ['C0_Frac_D3', 'C1_Frac_D3', 'C01_Frac_D3', 'C23_Frac_D3']

    rough_feats = ['C01_Med_Rough_D5', 'All_Med_Rough_D5'] if has_roughness else []

    all_feat_groups = (
        c01_med_d5 + c01_med_d3 + all_med_d5 + all_med_d3
        + c01_cv_d5 + c01_cv_d3 + c01_gr + c01_iqr
        + ratio_d5_feats + ratio_d3_feats + frac_d5 + frac_d3 + rough_feats
    )

    seen = set()
    extended_feats = []
    for f in all_feat_groups:
        if f not in seen and f in fm.columns:
            extended_feats.append(f)
            seen.add(f)

    d3_feats = [c for c in extended_feats if c.endswith('_D3')]
    d5_feats = [c for c in extended_feats if c.endswith('_D5') or c.startswith('C01_GR_')]

    print(f'\nFeature matrix: {fm.shape[0]} wells x {len(extended_feats)} features')
    print(f'  A1: C01_Med_D5   = {len(c01_med_d5)}')
    print(f'  A2: C01_Med_D3   = {len(c01_med_d3)}')
    print(f'  A3: All_Med_D5   = {len(all_med_d5)}')
    print(f'  A4: All_Med_D3   = {len(all_med_d3)}')
    print(f'  B:  C01_CV_D5    = {len(c01_cv_d5)} ({len(CV_FEATURES)} key features)')
    print(f'  B2: C01_CV_D3    = {len(c01_cv_d3)} ({len(CV_FEATURES)} key features)')
    print(f'  C:  C01_GR       = {len(c01_gr)} ({len(GR_FEATURES)} key features)')
    print(f'  D:  C01_IQR_D5   = {len(c01_iqr)} ({len(IQR_FEATURES)} key features)')
    print(f'  F:  Cross-Cluster D5 = {len(ratio_d5_feats)}')
    print(f'  F2: Cross-Cluster D3 = {len(ratio_d3_feats)}')
    print(f'  G:  Fractions    = {len(frac_d5 + frac_d3)}')
    print(f'  H:  Roughness    = {len(rough_feats)}')

    def _print_group(title, feat_list):
        present = [f for f in feat_list if f in fm.columns]
        if not present:
            return
        print(f'\n  {title} ({len(present)} features):')
        for i, f in enumerate(present, 1):
            valid = fm[f].notna().sum()
            print(f'    {i:2d}. {f:30s} valid={valid}/{len(fm)}')

    _print_group('Group A1: C01 Median D5', c01_med_d5)
    _print_group('Group A2: C01 Median D3', c01_med_d3)
    _print_group('Group A3: All Median D5', all_med_d5)
    _print_group('Group A4: All Median D3', all_med_d3)
    _print_group('Group B: C01 CV D5 (heterogeneity)', c01_cv_d5)
    _print_group('Group B2: C01 CV D3 (heterogeneity)', c01_cv_d3)
    _print_group('Group C: C01 Growth Rate (D5/D3)', c01_gr)
    _print_group('Group D: C01 IQR D5 (robust dispersion)', c01_iqr)
    _print_group('Group F: Cross-Cluster Ratios D5', ratio_d5_feats)
    _print_group('Group F2: Cross-Cluster Ratios D3', ratio_d3_feats)
    _print_group('Group G: Fraction D5', frac_d5)
    _print_group('Group G: Fraction D3', frac_d3)
    if rough_feats:
        _print_group('Group H: Roughness', rough_feats)

    return fm, extended_feats, d3_feats, d5_feats


def feature_selection(fm, sel_feats, nan_threshold=0.5, var_threshold=1e-6, corr_threshold=0.95):
    print('\n' + '=' * 70)
    print('Step 3b: Feature Selection (No ATP - Matrix Quality Only)')
    print('=' * 70)

    valid_feats = []
    drop_nan = []
    drop_var = []
    
    for f in sel_feats:
        if f not in fm.columns:
            continue
        nan_ratio = fm[f].isna().mean()
        if nan_ratio > nan_threshold:
            drop_nan.append((f, nan_ratio))
            continue
        valid_count = fm[f].notna().sum()
        if valid_count > 1:
            var_val = fm[f].dropna().var()
            if var_val < var_threshold:
                drop_var.append((f, var_val))
                continue
        valid_feats.append(f)
    
    print(f'\n  Step 1: NaN filter (>{nan_threshold:.0%} missing)')
    print(f'    Input: {len(sel_feats)} -> Kept: {len(valid_feats)} (Dropped: {len(drop_nan)})')
    for f, r in drop_nan:
        print(f'      DROP {f}: NaN={r:.1%}')
    
    print(f'\n  Step 2: Variance filter (<{var_threshold})')
    print(f'    Kept: {len(valid_feats)} (Dropped: {len(drop_var)})')
    for f, v in drop_var:
        print(f'      DROP {f}: var={v:.2e}')
    
    remaining = valid_feats[:]
    drop_corr = []
    
    fm_valid = fm[remaining].copy()
    for col in remaining:
        fm_valid[col] = fm_valid[col].fillna(fm_valid[col].median())
    
    corr_matrix = fm_valid.corr().abs()
    
    checked = set()
    for i, f1 in enumerate(remaining):
        if f1 in checked:
            continue
        for j, f2 in enumerate(remaining):
            if j <= i or f2 in checked:
                continue
            if corr_matrix.loc[f1, f2] > corr_threshold:
                v1 = fm_valid[f1].var()
                v2 = fm_valid[f2].var()
                drop_f = f2 if v1 >= v2 else f1
                keep_f = f1 if v1 >= v2 else f2
                drop_corr.append((drop_f, keep_f, corr_matrix.loc[f1, f2]))
                checked.add(drop_f)
    
    selected = [f for f in remaining if f not in checked]
    
    print(f'\n  Step 3: Redundancy filter (|r| > {corr_threshold})')
    print(f'    Kept: {len(selected)} (Dropped: {len(drop_corr)})')
    for drop_f, keep_f, r_val in drop_corr:
        print(f'      DROP {drop_f}: r={r_val:.4f} with {keep_f}')
    
    print(f'\n  Summary: {len(sel_feats)} -> {len(selected)} features')
    print(f'  Dropped: {len(drop_nan)} (NaN) + {len(drop_var)} (zeroVar) + {len(drop_corr)} (redundant)')
    
    corr_df = pd.DataFrame([
        {'Feature': f, 'NaN_ratio': fm[f].isna().mean(), 'Variance': fm[f].dropna().var()}
        for f in selected
    ])
    
    return selected, corr_df


def compute_relative_score(fm, d3_feats, d5_feats):
    print('\n' + '=' * 70)
    print('Step 4b: Delta-F Self-Control Score (dF = F_D5 - F_D3)')
    print('=' * 70)

    valid_d3 = ~fm[d3_feats].isnull().any(axis=1)
    valid_d5 = ~fm[d5_feats].isnull().any(axis=1)
    valid = valid_d3 & valid_d5
    
    ids = fm.loc[valid, 'Well_ID'].values
    X_d3 = fm.loc[valid, d3_feats].values
    X_d5 = fm.loc[valid, d5_feats].values

    print(f'D3 features: {X_d3.shape[1]}, D5 features: {X_d5.shape[1]}')
    print(f'Complete cases (both timepoints): {len(ids)}/{len(fm)} wells')

    X_combined = np.vstack([X_d3, X_d5])
    
    scaler_shared = StandardScaler()
    X_combined_s = scaler_shared.fit_transform(X_combined)

    max_comp = min(X_combined_s.shape[1], X_combined_s.shape[0] - 1)
    
    pca_full = PCA(n_components=max_comp, random_state=42)
    pca_full.fit(X_combined_s)
    
    eigenvalues = pca_full.explained_variance_
    n_comp_kaiser = max(2, int(np.sum(eigenvalues > 1.0)))
    
    cumvar_full = np.cumsum(pca_full.explained_variance_ratio_)
    n_comp_cumvar = int(np.searchsorted(cumvar_full, 0.85)) + 1
    n_comp = max(n_comp_kaiser, n_comp_cumvar)
    n_comp = min(n_comp, max_comp)
    
    print(f'\nShared PCA Selection:')
    print(f'  Kaiser (>1.0):      {n_comp_kaiser} PCs')
    print(f'  CumVar (>85%):       {n_comp_cumvar} PCs')
    print(f'  Selected:            {n_comp} PCs')

    pca_final = PCA(n_components=n_comp, random_state=42)
    X_all_p = pca_final.fit_transform(X_combined_s)
    
    vr = pca_final.explained_variance_ratio_
    wts = vr / vr.sum()
    
    print(f'\n  PC weights:')
    for i in range(n_comp):
        print(f'    PC{i+1}: var={vr[i]:.1%}, weight={wts[i]:.3f}')
    
    n_samples = len(ids)
    X_d3_p = X_all_p[:n_samples]
    X_d5_p = X_all_p[n_samples:]
    
    F_d3 = np.dot(X_d3_p, wts)
    F_d5 = np.dot(X_d5_p, wts)

    delta_score = F_d5 - F_d3

    print(f'\n  Score Statistics:')
    print(f'    F_D3: mean={F_d3.mean():.4f}, std={F_d3.std():.4f}')
    print(f'    F_D5: mean={F_d5.mean():.4f}, std={F_d5.std():.4f}')
    print(f'    dF (D5-D3): mean={delta_score.mean():.4f}, std={delta_score.std():.4f}')

    coef = np.dot(pca_final.components_.T, wts)
    cdf_rel = pd.DataFrame({
        'Feature': d3_feats,
        'Coef': coef[:len(d3_feats)],
        'AbsCoef': np.abs(coef[:len(d3_feats)])
    }).sort_values('AbsCoef', ascending=False)
    
    print('\n  Top 10 features by dF weight:')
    for _, r in cdf_rel.head(10).iterrows():
        print(f"    {r['Coef']:+.3f}  {r['Feature']}")

    score_df = pd.DataFrame({
        'Well_ID': ids,
        'Score': delta_score,
        'F_D3': F_d3,
        'F_D5': F_d5,
    })
    
    return score_df, pca_final, wts, scaler_shared, cdf_rel


def pca_analysis(fm, sel_feats, cumvar_threshold=0.85):
    print('\n' + '=' * 70)
    print(f'Step 4: PCA -> Composite Score (CumVar > {cumvar_threshold:.0%})')
    print('=' * 70)

    X = fm[sel_feats].values
    valid = ~np.isnan(X).any(axis=1)
    X_clean = X[valid]
    ids = fm.loc[valid, 'Well_ID'].values

    print(f'Input: {X.shape}, Complete cases: {X_clean.shape[0]}')
    if X_clean.shape[0] < 3:
        raise RuntimeError('Too few complete-case wells for PCA')

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X_clean)

    max_comp = min(X_clean.shape[1], X_clean.shape[0] - 1)
    pca_full = PCA(n_components=max_comp, random_state=42)
    pca_full.fit(X_s)

    eigenvalues = pca_full.explained_variance_
    explained_var_ratio = pca_full.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var_ratio)
    
    n_comp_cumvar = int(np.searchsorted(cumulative_var, cumvar_threshold)) + 1
    n_comp_kaiser = max(2, int(np.sum(eigenvalues > 1.0)))
    
    n_comp = max(n_comp_cumvar, n_comp_kaiser)
    n_comp = min(n_comp, max_comp)

    print(f'\nPC Selection Criteria:')
    print(f'  Kaiser (>1.0):      {n_comp_kaiser} PCs')
    print(f'  CumVar (>{cumvar_threshold:.0%}): {n_comp_cumvar} PCs')
    print(f'  Selected:           {n_comp} PCs (max of both criteria)')

    pca = PCA(n_components=n_comp, random_state=42)
    X_p = pca.fit_transform(X_s)

    vr = pca.explained_variance_ratio_
    cv = np.cumsum(vr)
    wts = vr / vr.sum()
    score = np.dot(X_p, wts)

    atp_values = fm.loc[fm['Well_ID'].isin(ids), 'ATP'].values
    valid_atp = ~np.isnan(atp_values)
    
    coef = np.dot(pca.components_.T, wts)
    
    if valid_atp.sum() > 5:
        temp_r, _ = pearsonr(score[valid_atp], atp_values[valid_atp])
        if temp_r < 0:
            score = -score
            wts = -wts
            coef = -coef
            print(f'\n[!] Score direction flipped (original r={temp_r:.4f}<0) -> now positive')

    print(f'\nPCA Results ({n_comp} components):')
    for i in range(n_comp):
        kaiser_mark = ' [K]' if eigenvalues[i] > 1.0 else ''
        cumvar_mark = f' <- CumVar>{cumvar_threshold:.0%}' if i == n_comp - 1 and cv[i] >= cumvar_threshold else ''
        print(f'  PC{i + 1}: var={eigenvalues[i]:.4f}, ratio={vr[i]:.1%}, cum={cv[i]:.1%}{kaiser_mark}{cumvar_mark}')

    cdf = pd.DataFrame({'Feature': sel_feats, 'Coef': coef, 'AbsCoef': np.abs(coef)}).sort_values(
        'AbsCoef', ascending=False
    )

    print('\nTop 10 features by weight:')
    for _, r in cdf.head(10).iterrows():
        print(f"  {r['Coef']:+.3f}  {r['Feature']}")

    score_df = pd.DataFrame({'Well_ID': ids, 'Score': score})
    return score_df, pca, wts, scaler, cdf


def atp_correlation(score_df):
    print('\n' + '=' * 70)
    print('Step 5: ATP Matching & Correlation')
    print('=' * 70)

    merged = score_df.copy()
    merged['ATP'] = merged['Well_ID'].map(ATP_DATABASE)
    valid = merged.dropna(subset=['Score', 'ATP'])

    if len(valid) < 5:
        print(f'ERROR Only {len(valid)} wells matched!')
        return None, None

    print(f'\nMatched: {len(valid)} wells')

    X, Y = valid['Score'].values, valid['ATP'].values
    pr, pp = pearsonr(X, Y)
    sp, sp_ = spearmanr(X, Y)

    print(f'\n{"=" * 50}')
    print('PEARSON CORRELATION: Score vs ATP')
    print(f'{"=" * 50}')
    print(f'N = {len(valid)}')
    print(f'Pearson r = {pr:.4f} (p={pp:.2e})')
    print(f'Spearman rho = {sp:.4f} (p={sp_:.2e})')

    res = {
        'pearson_r': pr,
        'pearson_p': pp,
        'spearman_rho': sp,
        'spearman_p': sp_,
    }

    return valid, res


def leave_one_patient_out_cv(fm, sel_feats, cumvar_threshold=0.85):
    print('\n' + '=' * 70)
    print('Step 5b: Leave-One-Patient-Out Cross-Validation')
    print('=' * 70)

    valid_data = fm.dropna(subset=sel_feats + ['ATP']).copy()
    
    def extract_patient_id(well_id):
        return well_id[0] if isinstance(well_id, str) and len(well_id) > 0 else well_id
    
    valid_data['Patient'] = valid_data['Well_ID'].apply(extract_patient_id)
    patients = sorted(valid_data['Patient'].unique())
    
    print(f'\nTotal wells: {len(valid_data)}')
    print(f'Patients (by Well_ID prefix): {patients}')
    print(f'Number of patients: {len(patients)}')

    cv_results = []
    
    for test_patient in patients:
        train_data = valid_data[valid_data['Patient'] != test_patient]
        test_data = valid_data[valid_data['Patient'] == test_patient]
        
        if len(train_data) < 5 or len(test_data) < 2:
            print(f'\n  Skip patient "{test_patient}": train={len(train_data)}, test={len(test_data)} (insufficient data)')
            continue
        
        X_train = train_data[sel_feats].values
        y_train = train_data['ATP'].values
        X_test = test_data[sel_feats].values
        y_test = test_data['ATP'].values
        
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        max_comp = min(X_train_s.shape[1], X_train_s.shape[0] - 1)
        pca_full = PCA(n_components=max_comp, random_state=42)
        pca_full.fit(X_train_s)
        
        eigenvalues = pca_full.explained_variance_
        explained_var_ratio = pca_full.explained_variance_ratio_
        cumulative_var = np.cumsum(explained_var_ratio)
        
        n_comp_cumvar = int(np.searchsorted(cumulative_var, cumvar_threshold)) + 1
        n_comp_kaiser = max(2, int(np.sum(eigenvalues > 1.0)))
        n_comp = min(max(n_comp_cumvar, n_comp_kaiser), max_comp)
        
        pca = PCA(n_components=n_comp, random_state=42)
        X_train_p = pca.fit_transform(X_train_s)
        X_test_p = pca.transform(X_test_s)
        
        vr = pca.explained_variance_ratio_
        wts = vr / vr.sum()
        
        train_score = np.dot(X_train_p, wts)
        test_score = np.dot(X_test_p, wts)
        
        train_r, _ = pearsonr(train_score, y_train)
        if train_r < 0:
            test_score = -test_score
            train_r = -train_r
        
        test_r, test_p = pearsonr(test_score, y_test)
        test_sp, test_sp_p = spearmanr(test_score, y_test)
        
        cv_results.append({
            'Test_Patient': test_patient,
            'Train_Size': len(train_data),
            'Test_Size': len(test_data),
            'N_Components': n_comp,
            'Train_R': train_r,
            'Test_R': test_r,
            'Test_P': test_p,
            'Test_Spearman': test_sp,
            'Test_Spearman_P': test_sp_p,
        })
        
        print(f'\n  Fold: Leave-Out Patient "{test_patient}"')
        print(f'    Train: {len(train_data)} wells | Test: {len(test_data)} wells | PCs: {n_comp}')
        print(f'    Test Pearson r = {test_r:.4f} (p={test_p:.2e})')
        print(f'    Test Spearman rho = {test_sp:.4f} (p={test_sp_p:.2e})')

    if not cv_results:
        print('\nERROR: No valid CV folds completed!')
        return None
    
    cv_df = pd.DataFrame(cv_results)
    
    mean_r = cv_df['Test_R'].mean()
    std_r = cv_df['Test_R'].std()
    mean_sp = cv_df['Test_Spearman'].mean()
    
    print(f'\n{"=" * 50}')
    print('LOPOCV SUMMARY')
    print(f'{"=" * 50}')
    print(f'Folds completed: {len(cv_df)} / {len(patients)}')
    print(f'Mean Pearson r  = {mean_r:.4f} +/- {std_r:.4f}')
    print(f'Mean Spearman rho = {mean_sp:.4f}')
    print(f'\nPer-fold results:')
    for _, row in cv_df.iterrows():
        print(f'  Patient "{row["Test_Patient"]}": r={row["Test_R"]:+.4f} (n={row["Test_Size"]})')
    
    return cv_df


def external_validation(test_data_path, test_atp_dict, trained_scaler, trained_pca, trained_wts, trained_sel_feats):
    print('\n' + '=' * 70)
    print('Step 6: External Validation (True Out-of-Sample Test)')
    print('=' * 70)
    
    print(f'\nTest dataset: {test_data_path}')
    print(f'Trained model features: {len(trained_sel_feats)}')
    
    try:
        df_test, feats_test, wells_test = load_organoid_data_custom(test_data_path)
        
        if len(wells_test) == 0:
            print('\nERROR: No valid wells found in test dataset!')
            return None
        
        df_test, ws_test = apply_clustering(df_test, feats_test)
        fm_test, extended_sel_test, d3_feats_test, d5_feats_test = stratified_median_aggregation(
            df_test, ws_test, feats_test, wells_test
        )
        
        fm_test['ATP'] = fm_test['Well_ID'].map(test_atp_dict)
        
        common_feats = [f for f in trained_sel_feats if f in fm_test.columns]
        if len(common_feats) < len(trained_sel_feats):
            missing = set(trained_sel_feats) - set(common_feats)
            print(f'\nWARNING: {len(missing)} training features missing in test data:')
            for m in missing:
                print(f'  - {m}')
            print(f'Using {len(common_feats)} common features')
        
        if len(common_feats) < 3:
            print(f'\nERROR: Too few common features ({len(common_feats)}) for validation!')
            return None
        
        X_test = fm_test[common_feats].values
        valid_test = ~np.isnan(X_test).any(axis=1) & fm_test['ATP'].notna()
        
        if valid_test.sum() < 3:
            print(f'\nERROR: Too few complete cases in test set ({valid_test.sum()})!')
            return None
        
        X_test_clean = X_test[valid_test]
        ids_test = fm_test.loc[valid_test, 'Well_ID'].values
        atp_test = fm_test.loc[valid_test, 'ATP'].values
        
        X_test_scaled = trained_scaler.transform(X_test_clean)
        X_test_pcs = trained_pca.transform(X_test_scaled)
        score_test = np.dot(X_test_pcs, trained_wts)
        
        r_test, p_test = pearsonr(score_test, atp_test)
        sp_test, sp_test_p = spearmanr(score_test, atp_test)
        
        print(f'\n{"=" * 50}')
        print('EXTERNAL VALIDATION RESULTS')
        print(f'{"=" * 50}')
        print(f'Test samples: {valid_test.sum()} wells')
        print(f'Features used: {len(common_feats)} / {len(trained_sel_feats)} (training)')
        print(f'\nPearson r  = {r_test:.4f} (p={p_test:.2e})')
        print(f'Spearman rho = {sp_test:.4f} (p={sp_test_p:.2e})')
        
        result_df = pd.DataFrame({
            'Well_ID': ids_test,
            'Score': score_test,
            'ATP': atp_test,
            'Dataset': 'External_Test'
        })
        
        res_external = {
            'pearson_r': r_test,
            'pearson_p': p_test,
            'spearman_rho': sp_test,
            'spearman_p': sp_test_p,
            'n_samples': valid_test.sum(),
            'n_features': len(common_feats),
        }
        
        return result_df, res_external
        
    except Exception as e:
        print(f'\nERROR during external validation: {str(e)}')
        import traceback
        traceback.print_exc()
        return None


def load_organoid_data_custom(data_base_dir):
    from cluster_utils import PROCESSED_FEATURES
    import glob as glob_module

    morph_feats = list(PROCESSED_FEATURES)

    data_folders_custom = {}

    dir_0701 = os.path.join(data_base_dir, 'FXN_0701', 'measure_excel')
    if not os.path.exists(dir_0701):
        dir_0701 = os.path.join(data_base_dir, 'FXN_20230701', 'measure_excel')

    if os.path.exists(dir_0701):
        data_folders_custom['0701'] = dir_0701

    possible_0703_names = ['FXN_0703', 'FXN_20230703']
    dir_0703 = None
    for name in possible_0703_names:
        candidate = os.path.join(data_base_dir, name, 'measure_excel')
        if os.path.exists(candidate):
            dir_0703 = candidate
            break

    if dir_0703 is not None:
        data_folders_custom['0703'] = dir_0703

    all_dfs = []
    wells_seen = set()

    for day, folder in data_folders_custom.items():
        if not os.path.exists(folder):
            continue

        xlsx_files = glob_module.glob(os.path.join(folder, '*.xlsx'))

        for fpath in sorted(xlsx_files):
            fname = os.path.basename(fpath)

            well_id = fname.replace('_0701.xlsx', '').replace('_0703.xlsx', '').replace('.xlsx', '')

            if not well_id:
                continue

            well_day_key = (well_id, day)
            if well_day_key in wells_seen:
                continue

            try:
                tmp = pd.read_excel(fpath)

                if 'Cavity_Ratio' not in tmp.columns and 'Cavity_Volume' in tmp.columns and 'Organoids_Volume_Fill' in tmp.columns:
                    tmp['Cavity_Ratio'] = tmp['Cavity_Volume'] / (tmp['Organoids_Volume_Fill'] + 1e-10)

                expected_cols = {'Index'} | set(morph_feats)
                actual_cols = set(tmp.columns)

                if not expected_cols.issubset(actual_cols):
                    missing = expected_cols - actual_cols
                    print(f'  SKIP {fname}: missing columns {missing}')
                    continue

                tmp = tmp.dropna(subset=morph_feats)
                if len(tmp) == 0:
                    continue

                tmp['_well'] = fname.replace('.xlsx', '')
                tmp['_well_id'] = well_id
                tmp['_day'] = day
                all_dfs.append(tmp)
                wells_seen.add(well_day_key)

            except Exception as e:
                print(f'  ERROR reading {fname}: {e}')

    if not all_dfs:
        raise RuntimeError(f'No valid data files found in {data_base_dir}')

    df = pd.concat(all_dfs, ignore_index=True)
    wells_sorted = sorted(set(w for w, _ in wells_seen))

    roughness_dfs = []
    for day in ['0701', '0703']:
        roughness_dir = os.path.join(data_base_dir, f'FXN_202307{day}', 'roughness')
        if not os.path.exists(roughness_dir):
            continue
        for fp in sorted(glob_module.glob(os.path.join(roughness_dir, '*.xlsx'))):
            rdf = pd.read_excel(fp)
            if 'Index' not in rdf.columns or 'Roughness' not in rdf.columns:
                continue
            well_name = os.path.splitext(os.path.basename(fp))[0]
            rdf['_well'] = well_name
            rdf['_day'] = day
            rdf['_well_id'] = well_name.split('_')[0] if '_' in well_name else well_name
            roughness_dfs.append(rdf[['Index', 'Roughness', '_well', '_day', '_well_id']])

    if roughness_dfs:
        roughness_all = pd.concat(roughness_dfs, ignore_index=True)
        df = df.merge(roughness_all, on=['Index', '_well', '_day', '_well_id'], how='left')
        print(f'  Roughness merged: {df["Roughness"].notna().sum():,}/{len(df):,} organoids')

    print(f'Loaded custom dataset: {data_base_dir}')
    print(f'  Wells: {len(wells_sorted)}, Objects: {len(df)}')
    print(f'  Days available: {list(data_folders_custom.keys())}')

    return df, morph_feats, wells_sorted