import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from cluster_utils import load_model_package
from .config import CLUSTER_COLORS, CLUSTER_NAMES, HEALTHY_CLUSTERS, MIN_HEALTHY_SAMPLES, MODEL_PATH


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
    print('Step 3: Stratified Median Aggregation (12D)')
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

        w5 = ws[(ws['_well_id'] == wid) & (ws['_day'] == '0703')]
        w3 = ws[(ws['_well_id'] == wid) & (ws['_day'] == '0701')]

        if len(w5) > 0:
            row['Red_Frac_D5'] = w5.iloc[0]['Red_Fraction']
            row['Yel_Frac_D5'] = w5.iloc[0]['Yellow_Fraction']
            row['Healthy_Frac_D5'] = w5.iloc[0]['Healthy_Fraction']

        if len(w3) > 0:
            row['Healthy_Frac_D3'] = w3.iloc[0]['Healthy_Fraction']

        hf5, hf3 = row.get('Healthy_Frac_D5'), row.get('Healthy_Frac_D3')
        row['Delta_Healthy_Frac'] = (hf5 - hf3) if (pd.notna(hf5) and pd.notna(hf3)) else np.nan

        rows.append(row)

    fm = pd.DataFrame(rows)
    sel_feats = [f'Delta_Healthy_{f}' for f in feats] + ['Red_Frac_D5', 'Yel_Frac_D5', 'Delta_Healthy_Frac']

    print(f'\nFeature matrix: {fm.shape}')
    print('\n12D Features:')
    for i, f in enumerate(sel_feats, 1):
        valid = fm[f].notna().sum()
        print(f'  {i:2d}. {f:30s} (valid: {valid}/{len(fm)})')

    if warnings_list:
        print(f'\nWARN Wells with <{MIN_HEALTHY_SAMPLES} Healthy organoids:')
        seen = set()
        for w, d, n in warnings_list:
            k = (w, d, n)
            if k in seen:
                continue
            seen.add(k)
            print(f'     {w}({d}): n={n}')

    return fm, sel_feats


def pca_analysis(fm, sel_feats):
    print('\n' + '=' * 70)
    print('Step 4: PCA -> Composite Score')
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

    n_comp = min(4, X_clean.shape[1], X_clean.shape[0] - 1)
    pca = PCA(n_components=n_comp, random_state=42)
    X_p = pca.fit_transform(X_s)

    vr = pca.explained_variance_ratio_
    cv = np.cumsum(vr)
    wts = vr / vr.sum()
    score = np.dot(X_p, wts)

    print(f'\nPCA Results ({n_comp} components):')
    for i in range(n_comp):
        print(f'  PC{i + 1}: var={pca.explained_variance_[i]:.4f}, ratio={vr[i]:.1%}, cum={cv[i]:.1%}')

    coef = np.dot(pca.components_.T, wts)
    cdf = pd.DataFrame({'Feature': sel_feats, 'Coef': coef, 'AbsCoef': np.abs(coef)}).sort_values(
        'AbsCoef', ascending=False
    )

    print('\nTop 10 features by weight:')
    for _, r in cdf.head(10).iterrows():
        print(f"  {r['Coef']:+.3f}  {r['Feature']}")

    score_df = pd.DataFrame({'Well_ID': ids, 'Score': score})
    return score_df, pca, wts, scaler, cdf


def atp_regression(score_df, atp_df):
    print('\n' + '=' * 70)
    print('Step 5: ATP Matching & Regression')
    print('=' * 70)

    if atp_df is None:
        print('ERROR Cannot load ATP data!')
        return None, None

    our_ids = set(score_df['Well_ID'])

    m1 = atp_df[atp_df['_key'].isin(our_ids)].copy()
    atp_df = atp_df.copy()
    atp_df['_pre'] = atp_df['_key'].apply(lambda x: str(x).split('_')[0] if '_' in str(x) else str(x))
    m2 = atp_df[atp_df['_pre'].isin(our_ids)].copy()

    matched = m1 if len(m1) >= len(m2) else m2
    matched['Well_ID'] = matched['_key'] if len(m1) >= len(m2) else matched['_pre']

    merged = score_df.merge(matched[['Well_ID', 'ATP']], on='Well_ID', how='inner')

    if len(merged) < 5:
        print(f'ERROR Only {len(merged)} wells matched!')
        return None, None

    print(f'\nMatched: {len(merged)} wells')

    X, Y = merged['Score'].values, merged['ATP'].values
    sl, ic, r, p, se = stats.linregress(X, Y)
    pr, pp = pearsonr(X, Y)
    sp, sp_ = spearmanr(X, Y)

    print(f'\n{"=" * 50}')
    print('REGRESSION: Score -> ATP')
    print(f'{"=" * 50}')
    print(f'N = {len(merged)}')
    print(f'ATP = {ic:,.0f} + {sl:,.0f} * Score')
    print(f'R2 = {r ** 2:.4f}')
    print(f'Pearson r = {pr:.4f} (p={pp:.2e})')
    print(f'Spearman rho = {sp:.4f} (p={sp_:.2e})')

    res = {
        'slope': sl,
        'intercept': ic,
        'r2': r ** 2,
        'pearson_r': pr,
        'pearson_p': pp,
        'spearman_rho': sp,
        'spearman_p': sp_,
        'eq': f'ATP = {ic:,.0f} + {sl:,.0f} * Score',
    }

    return merged, res
