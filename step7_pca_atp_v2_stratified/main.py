from datetime import datetime

from .config import FIGURES_DIR
from .data_io import load_organoid_data, validate_data_structure
from .modeling import (
    apply_clustering,
    atp_correlation,
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
    fm, sel = stratified_median_aggregation(df, ws, feats, wells)
    sdf, pca, wts, scaler, cdf = pca_analysis(fm, sel)

    merged, res = atp_correlation(sdf)

    generate_figures(df, ws, fm, merged, res, cdf)

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