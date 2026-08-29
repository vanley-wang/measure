import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from cluster_utils import load_model_package
from .config import ATP_DATABASE, CLUSTER_COLORS, CLUSTER_NAMES, HEALTHY_CLUSTERS, MIN_HEALTHY_SAMPLES, MODEL_PATH


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
    print('Step 3: Stratified Median Aggregation (Extended to ~24D)')
    print('=' * 70)

    rows = []
    warnings_list = []

    for wid in wells:
        row = {'Well_ID': wid}

        d5 = df[(df['_well_id'] == wid) & (df['_day'] == '0703')]
        d3 = df[(df['_well_id'] == wid) & (df['_day'] == '0701')]

        h5 = d5[d5['Cluster'].isin(HEALTHY_CLUSTERS)]
        h3 = d3[d3['Cluster'].isin(HEALTHY_CLUSTERS)]

        n_h5, n_h3 = len(h5), len(h3)

        for f in feats:
            if n_h5 >= MIN_HEALTHY_SAMPLES:
                row[f'Healthy_{f}_D5'] = h5[f].median()
            else:
                row[f'Healthy_{f}_D5'] = np.nan
                warnings_list.append((wid, 'D5', n_h5))

            if n_h3 >= MIN_HEALTHY_SAMPLES:
                row[f'Healthy_{f}_D3'] = h3[f].median()
            else:
                row[f'Healthy_{f}_D3'] = np.nan
                warnings_list.append((wid, 'D3', n_h3))

            v5, v3 = row.get(f'Healthy_{f}_D5'), row.get(f'Healthy_{f}_D3')
            row[f'Delta_Healthy_{f}'] = (v5 - v3) if (pd.notna(v5) and pd.notna(v3)) else np.nan
            
            if pd.notna(v3) and v3 != 0:
                row[f'RelChange_Healthy_{f}'] = ((v5 - v3) / abs(v3)) if pd.notna(v5) else np.nan
            else:
                row[f'RelChange_Healthy_{f}'] = np.nan

        w5 = ws[(ws['_well_id'] == wid) & (ws['_day'] == '0703')]
        w3 = ws[(ws['_well_id'] == wid) & (ws['_day'] == '0701')]

        if len(w5) > 0:
            row['Red_Frac_D5'] = w5.iloc[0]['Red_Fraction']
            row['Yel_Frac_D5'] = w5.iloc[0]['Yellow_Fraction']
            row['Healthy_Frac_D5'] = w5.iloc[0]['Healthy_Fraction']

        if len(w3) > 0:
            row['Red_Frac_D3'] = w3.iloc[0]['Red_Fraction']
            row['Yel_Frac_D3'] = w3.iloc[0]['Yellow_Fraction']
            row['Healthy_Frac_D3'] = w3.iloc[0]['Healthy_Fraction']

        hf5, hf3 = row.get('Healthy_Frac_D5'), row.get('Healthy_Frac_D3')
        row['Delta_Healthy_Frac'] = (hf5 - hf3) if (pd.notna(hf5) and pd.notna(hf3)) else np.nan
        
        rf5, rf3 = row.get('Red_Frac_D5', 0), row.get('Red_Frac_D3', 0)
        yf5, yf3 = row.get('Yel_Frac_D5', 0), row.get('Yel_Frac_D3', 0)
        
        row['Delta_Red_Frac'] = rf5 - rf3
        row['Delta_Yel_Frac'] = yf5 - yf3
        
        if rf3 != 0:
            row['RelChange_Red_Frac'] = (rf5 - rf3) / abs(rf3)
        else:
            row['RelChange_Red_Frac'] = np.nan
            
        if yf3 != 0:
            row['RelChange_Yel_Frac'] = (yf5 - yf3) / abs(yf3)
        else:
            row['RelChange_Yel_Frac'] = np.nan
            
        if pd.notna(hf3) and hf3 != 0:
            row['RelChange_Healthy_Frac'] = (hf5 - hf3) / abs(hf3)
        else:
            row['RelChange_Healthy_Frac'] = np.nan

        rows.append(row)

    fm = pd.DataFrame(rows)
    
    delta_feats = [f'Delta_Healthy_{f}' for f in feats] + ['Delta_Healthy_Frac', 'Delta_Red_Frac', 'Delta_Yel_Frac']
    
    d5_abs_feats = [f'Healthy_{f}_D5' for f in feats] + ['Red_Frac_D5', 'Yel_Frac_D5', 'Healthy_Frac_D5']
    
    rel_change_feats = [f'RelChange_Healthy_{f}' for f in feats] + ['RelChange_Red_Frac', 'RelChange_Yel_Frac', 'RelChange_Healthy_Frac']
    
    all_feats = d5_abs_feats + delta_feats + rel_change_feats
    seen = set()
    extended_feats = []
    for f in all_feats:
        if f not in seen:
            extended_feats.append(f)
            seen.add(f)
    
    d3_feats = [f'Healthy_{f}_D3' for f in feats] + ['Red_Frac_D3', 'Yel_Frac_D3', 'Healthy_Frac_D3']
    d5_feats = [f'Healthy_{f}_D5' for f in feats] + ['Red_Frac_D5', 'Yel_Frac_D5', 'Healthy_Frac_D5']

    print(f'\nFeature matrix: {fm.shape}')
    print(f'\nExtended Features ({len(extended_feats)}D):')
    print(f'  Group 1 - D5 Absolute Values ({len(d5_abs_feats)} features):')
    for i, f in enumerate(d5_abs_feats, 1):
        valid = fm[f].notna().sum()
        print(f'    {i:2d}. {f:35s} (valid: {valid}/{len(fm)})')
    
    print(f'  Group 2 - Delta (D5-D3) ({len(delta_feats)} features):')
    for i, f in enumerate(delta_feats, 1):
        valid = fm[f].notna().sum()
        print(f'    {i:2d}. {f:35s} (valid: {valid}/{len(fm)})')
        
    print(f'  Group 3 - Relative Change ((D5-D3)/|D3|) ({len(rel_change_feats)} features):')
    for i, f in enumerate(rel_change_feats, 1):
        valid = fm[f].notna().sum()
        print(f'    {i:2d}. {f:35s} (valid: {valid}/{len(fm)})')

    if warnings_list:
        print(f'\nWARN Wells with <{MIN_HEALTHY_SAMPLES} Healthy organoids:')
        seen = set()
        for w, d, n in warnings_list:
            k = (w, d, n)
            if k in seen:
                continue
            seen.add(k)
            print(f'     {w}({d}): n={n}')

    return fm, extended_feats, d3_feats, d5_feats


def feature_selection(fm, sel_feats, threshold=0.7):
    print('\n' + '=' * 70)
    print(f'Step 3b: Feature Selection (Pearson r > {threshold})')
    print('=' * 70)

    corr_list = []
    for f in sel_feats:
        if f in fm.columns and fm[f].notna().any():
            valid = fm[[f, 'ATP']].dropna()
            if len(valid) >= 5:
                r, p = pearsonr(valid[f], valid['ATP'])
                corr_list.append((f, r, p))

    corr_df = pd.DataFrame(corr_list, columns=['Feature', 'Pearson_r', 'p_value'])
    corr_df = corr_df.sort_values('Pearson_r', key=abs, ascending=False)
    
    selected = corr_df[corr_df['Pearson_r'].abs() > threshold]['Feature'].tolist()
    
    print(f'\nFeature Selection Results:')
    print(f'  Input features: {len(sel_feats)}')
    print(f'  Selected: {len(selected)} (|r| > {threshold})')
    print(f'  Dropped: {len(sel_feats) - len(selected)}')
    
    print(f'\nTop 15 Features by |Pearson r|:')
    for i, (_, row) in enumerate(corr_df.head(15).iterrows(), 1):
        mark = ' ✓' if abs(row['Pearson_r']) > threshold else ''
        print(f'  {i:2d}. {row["Pearson_r"]:+.4f}  {row["Feature"]:35s} p={row["p_value"]:.2e}{mark}')
    
    return selected, corr_df


def compute_relative_score(fm, d3_feats, d5_feats):
    print('\n' + '=' * 70)
    print('Step 4b: Relative Growth Score (ΔF = F_D5 - F_D3)')
    print('=' * 70)

    valid_d3 = ~fm[d3_feats].isnull().any(axis=1)
    valid_d5 = ~fm[d5_feats].isnull().any(axis=1)
    valid = valid_d3 & valid_d5
    
    ids = fm.loc[valid, 'Well_ID'].values
    X_d3 = fm.loc[valid, d3_feats].values
    X_d5 = fm.loc[valid, d5_feats].values

    print(f'Day3 features: {X_d3.shape}, Day5 features: {X_d5.shape}')
    print(f'Complete cases (both timepoints): {len(ids)}')

    X_combined = np.vstack([X_d3, X_d5])
    
    scaler_shared = StandardScaler()
    X_combined_s = scaler_shared.fit_transform(X_combined)

    max_comp = min(X_combined_s.shape[1], X_combined_s.shape[0] - 1)
    
    pca_full = PCA(n_components=max_comp, random_state=42)
    pca_full.fit(X_combined_s)
    
    eigenvalues = pca_full.explained_variance_
    n_comp_kaiser = max(2, int(np.sum(eigenvalues > 1.0)))
    
    print(f'\nShared PCA (Kaiser): {n_comp_kaiser} PCs with eigenvalue > 1.0')
    for i in range(min(len(eigenvalues), n_comp_kaiser)):
        mark = ' ✓ (Kaiser)' if eigenvalues[i] > 1.0 else ''
        print(f'  PC{i+1}: var={eigenvalues[i]:.4f}{mark}')

    pca_final = PCA(n_components=n_comp_kaiser, random_state=42)
    X_all_p = pca_final.fit_transform(X_combined_s)
    
    vr = pca_final.explained_variance_ratio_
    wts = vr / vr.sum()
    
    n_samples = len(ids)
    X_d3_p = X_all_p[:n_samples]
    X_d5_p = X_all_p[n_samples:]
    
    F_d3 = np.dot(X_d3_p, wts)
    F_d5 = np.dot(X_d5_p, wts)

    delta_score = F_d5 - F_d3

    print(f'\nRelative Score Statistics:')
    print(f'  F_D3: mean={F_d3.mean():.4f}, std={F_d3.std():.4f}')
    print(f'  F_D5: mean={F_d5.mean():.4f}, std={F_d5.std():.4f}')
    print(f'  ΔF (D5-D3): mean={delta_score.mean():.4f}, std={delta_score.std():.4f}')

    coef = np.dot(pca_final.components_.T, wts)
    cdf_rel = pd.DataFrame({
        'Feature': d3_feats,
        'Coef': coef[:len(d3_feats)],
        'AbsCoef': np.abs(coef[:len(d3_feats)])
    }).sort_values('AbsCoef', ascending=False)

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
            print(f'\n⚠ Score direction flipped (original r={temp_r:.4f}<0) → now positive')

    print(f'\nPCA Results ({n_comp} components):')
    for i in range(n_comp):
        kaiser_mark = ' ✓ (Kaiser)' if eigenvalues[i] > 1.0 else ''
        cumvar_mark = f' ← CumVar>{cumvar_threshold:.0%}' if i == n_comp - 1 and cv[i] >= cumvar_threshold else ''
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
        print(f'    Test Spearman ρ = {test_sp:.4f} (p={test_sp_p:.2e})')

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
    print(f'Mean Pearson r  = {mean_r:.4f} ± {std_r:.4f}')
    print(f'Mean Spearman ρ = {mean_sp:.4f}')
    print(f'\nPer-fold results:')
    for _, row in cv_df.iterrows():
        print(f'  Patient "{row["Test_Patient"]}": r={row["Test_R"]:+.4f} (n={row["Test_Size"]})')
    
    return cv_df