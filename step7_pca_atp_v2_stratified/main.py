from datetime import datetime

from .config import ATP_DATABASE, FIGURES_DIR
from .data_io import load_organoid_data, validate_data_structure
from .modeling import (
    apply_clustering,
    atp_correlation,
    compute_relative_score,
    feature_selection,
    leave_one_patient_out_cv,
    pca_analysis,
    stratified_median_aggregation,
)
from .plotting import generate_figures


def run_pipeline():
    print('\n' + '=' * 70)
    print(' ' * 15 + 'PCA-ATP BRIDGE V2 (STRATIFIED MEDIAN)')
    print(' ' * 25 + 'Data: FXN_2023_new（ICC）')
    print(' ' * 25 + f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print('=' * 70)

    t0 = datetime.now()

    validate_data_structure()
    df, feats, wells = load_organoid_data()
    df, ws = apply_clustering(df, feats)
    fm, extended_sel, d3_feats, d5_feats = stratified_median_aggregation(df, ws, feats, wells)
    
    fm['ATP'] = fm['Well_ID'].map(ATP_DATABASE)
    
    selected_feats, corr_df = feature_selection(fm, extended_sel, threshold=0.7)
    
    print('\n' + '=' * 70)
    print('  METHOD COMPARISON: Extended-PCA (with Feature Selection)')
    print('=' * 70)
    
    sdf_ext, pca_ext, wts_ext, scaler_ext, cdf_ext = pca_analysis(fm, selected_feats)
    merged_ext, res_ext = atp_correlation(sdf_ext)
    
    print(f'\n{"=" * 50}')
    print('RESULT: Extended PCA (with Feature Selection)')
    print(f'{"=" * 50}')
    if res_ext:
        print(f'Extended-PCA (new):   Pearson r = {res_ext["pearson_r"]:.4f} (p={res_ext["pearson_p"]:.2e})')
    
    cv_df = leave_one_patient_out_cv(fm, selected_feats)
    
    sdf, pca, wts, scaler = sdf_ext, pca_ext, wts_ext, scaler_ext
    merged, res = merged_ext, res_ext
    sel = selected_feats
    cdf = cdf_ext
    print('\nUsing EXTENDED-PCA Score for figures and output')

    generate_figures(df, ws, fm, merged, res, cdf, pca, sel)

    dt = (datetime.now() - t0).total_seconds()
    print('\n' + '=' * 70)
    print('OK ANALYSIS COMPLETE')
    print(f'  Time: {dt:.1f}s | Wells: {len(wells)} | Features: {len(sel)}D')
    if res:
        print(f"  Result: Pearson r={res['pearson_r']:.4f} (p={res['pearson_p']:.2e}) | Spearman rho={res['spearman_rho']:.4f}")
    print(f'  Output: {FIGURES_DIR}/')
    print('=' * 70 + '\n')

    return {
        'df': df,
        'ws': ws,
        'fm': fm,
        'score_df': sdf,
        'merged': merged,
        'result': res,
        'coef_df': cdf,
    }


def main():
    try:
        run_pipeline()
    except Exception as e:
        print(f'\nFATAL ERROR: {type(e).__name__}: {e}')
        import traceback

        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()