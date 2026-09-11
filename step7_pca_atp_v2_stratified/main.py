from datetime import datetime
import os
import pandas as pd

from .config import ATP_DATABASE, FIGURES_DIR
from .data_io import load_organoid_data, validate_data_structure
from .modeling import (
    apply_clustering,
    atp_correlation,
    compute_relative_score,
    external_validation,
    feature_selection,
    leave_one_patient_out_cv,
    pca_analysis,
    stratified_median_aggregation,
)
from .plotting import generate_figures


def run_pipeline():
    print('\n' + '=' * 70)
    print(' ' * 15 + 'PCA-ATP BRIDGE V2 (STRATIFIED MEDIAN)')
    print(' ' * 25 + 'Data: FXN_2023_new (ICC)')
    print(' ' * 25 + f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print('=' * 70)

    t0 = datetime.now()

    validate_data_structure()
    df, feats, wells = load_organoid_data()
    df, ws = apply_clustering(df, feats)
    fm, extended_sel, d3_feats, d5_feats = stratified_median_aggregation(df, ws, feats, wells)
    
    fm['ATP'] = fm['Well_ID'].map(ATP_DATABASE)
    
    selected_feats, corr_df = feature_selection(fm, extended_sel)
    
    d3_sel = [c for c in selected_feats if c.endswith('_D3')]
    d5_sel = [c for c in selected_feats if c.endswith('_D5')]
    
    paired_d3 = []
    paired_d5 = []
    for f5 in d5_sel:
        f3 = f5.replace('_D5', '_D3')
        if f3 in d3_sel:
            paired_d3.append(f3)
            paired_d5.append(f5)
    
    unpaired_d5 = [f for f in d5_sel if f not in paired_d5]
    unpaired_d3 = [f for f in d3_sel if f not in paired_d3]
    
    print(f'\nDelta-F feature pairing:')
    print(f'  Paired D3/D5: {len(paired_d3)} features')
    if unpaired_d5:
        print(f'  D5-only (excluded): {unpaired_d5}')
    if unpaired_d3:
        print(f'  D3-only (excluded): {unpaired_d3}')
    
    print('\n' + '=' * 70)
    print('  METHOD 1: Endpoint F (D5 only)')
    print('=' * 70)
    
    sdf_ext, pca_ext, wts_ext, scaler_ext, cdf_ext = pca_analysis(fm, selected_feats)
    merged_ext, res_ext = atp_correlation(sdf_ext)
    
    print(f'\n{"=" * 50}')
    print('RESULT: Endpoint PCA (D5)')
    print(f'{"=" * 50}')
    if res_ext:
        print(f'Endpoint F (D5):     Pearson r = {res_ext["pearson_r"]:.4f} (p={res_ext["pearson_p"]:.2e})')
    
    print('\n' + '=' * 70)
    print('  METHOD 2: Delta-F Self-Control (dF = F_D5 - F_D3)')
    print('=' * 70)
    
    sdf_delta, pca_delta, wts_delta, scaler_delta, cdf_delta = compute_relative_score(fm, paired_d3, paired_d5)
    merged_delta, res_delta = atp_correlation(sdf_delta)
    
    print(f'\n{"=" * 50}')
    print('RESULT: Delta-F Self-Control (dF)')
    print(f'{"=" * 50}')
    if res_delta:
        print(f'dF (D5-D3):          Pearson r = {res_delta["pearson_r"]:.4f} (p={res_delta["pearson_p"]:.2e})')
        print(f'                      Spearman rho = {res_delta["spearman_rho"]:.4f}')
        
        if res_ext:
            delta_gain = res_delta["pearson_r"] - res_ext["pearson_r"]
            print(f'\n  dF vs Endpoint: dr = {delta_gain:+.4f} ({"GAIN" if delta_gain > 0 else "LOSS"})')
    
    try:
        import pickle as _pkl
        _pca_pkg = {
            'scaler': scaler_delta,
            'pca': pca_delta,
            'weights': wts_delta,
            'selected_features': selected_feats,
            'd3_features': d3_sel,
            'd5_features': d5_sel,
        }
        _pca_path = os.path.join('model', 'PCA_ATP_v2_delta.pkl')
        os.makedirs('model', exist_ok=True)
        with open(_pca_path, 'wb') as _f:
            _pkl.dump(_pca_pkg, _f)
        print(f'\nPCA model saved -> {_pca_path}')
    except Exception as _e:
        print(f'\nWarning: PCA model save failed: {_e}')
    
    cv_df = leave_one_patient_out_cv(fm, selected_feats)
    
    print('\n' + '=' * 70)
    print('  EXTERNAL VALIDATION ON INDEPENDENT DATASETS')
    print('=' * 70)
    
    external_results = {}
    
    try:
        gc_atp_path = os.path.join('Data', 'FXN_2023_new（GC）', 'ATP.xlsx')
        if os.path.exists(gc_atp_path):
            gc_atp_df = pd.read_excel(gc_atp_path)
            gc_atp_dict = dict(zip(gc_atp_df.iloc[:, 0], gc_atp_df.iloc[:, 1]))
            
            gc_data_path = os.path.join('Data', 'FXN_2023_new（GC）')
            ext_gc, res_gc = external_validation(
                gc_data_path, gc_atp_dict, 
                scaler_ext, pca_ext, wts_ext, selected_feats
            )
            if ext_gc is not None:
                external_results['GC_GastricCancer'] = res_gc
                print(f'\n[OK] GC (Gastric Cancer) External Validation: r={res_gc["pearson_r"]:.4f}')
            else:
                print(f'\n[FAIL] GC validation failed')
        else:
            print(f'\n[WARN] ATP file not found: {gc_atp_path}')
    except Exception as e:
        print(f'\n[ERROR] Error during GC external validation: {e}')
    
    try:
        icc005_data_path = os.path.join('Data', 'ICC005_20240424')
        if os.path.exists(icc005_data_path):
            icc005_analysis_file = os.path.join(icc005_data_path, 'ICC005_20240424_Analysis.xlsx')
            if os.path.exists(icc005_analysis_file):
                icc005_df = pd.read_excel(icc005_analysis_file)
                if len(icc005_df.columns) >= 2:
                    icc005_atp_dict = dict(zip(icc005_df.iloc[:, 0], icc005_df.iloc[:, 1]))
                    
                    ext_icc005, res_icc005 = external_validation(
                        icc005_data_path, icc005_atp_dict,
                        scaler_ext, pca_ext, wts_ext, selected_feats
                    )
                    if ext_icc005 is not None:
                        external_results['ICC005_Patient'] = res_icc005
                        print(f'\n[OK] ICC005 (Different Patient) External Validation: r={res_icc005["pearson_r"]:.4f}')
                    else:
                        print(f'\n[FAIL] ICC005 validation failed')
                else:
                    print(f'\n[WARN] ICC005 Analysis file format unclear')
            else:
                print(f'\n[WARN] ICC005 Analysis file not found')
    except Exception as e:
        print(f'\n[ERROR] Error during ICC005 external validation: {e}')
    
    if external_results:
        print(f'\n{"=" * 50}')
        print('EXTERNAL VALIDATION SUMMARY')
        print(f'{"=" * 50}')
        print(f'Training (ICC) Endpoint F: r = {res_ext["pearson_r"]:.4f}')
        print(f'Training (ICC) Delta-F dF: r = {res_delta["pearson_r"]:.4f}')
        for name, res in external_results.items():
            drop = res_ext["pearson_r"] - res["pearson_r"]
            print(f'{name}: r = {res["pearson_r"]:.4f} (dr = {drop:+.4f})')
    
    print('\n' + '=' * 70)
    print('  nnUNet SEGMENTATION ROBUSTNESS TEST')
    print('=' * 70)
    
    try:
        nnunet_data_path = os.path.join('Data', 'nnUNet_FXN_2023')
        if os.path.exists(nnunet_data_path):
            nnunet_result = external_validation(
                nnunet_data_path, ATP_DATABASE,
                scaler_ext, pca_ext, wts_ext, selected_feats
            )
            if nnunet_result is not None:
                nnunet_df, nnunet_res = nnunet_result
                print(f'\n[OK] nnUNet Robustness Test: r={nnunet_res["pearson_r"]:.4f} (p={nnunet_res["pearson_p"]:.2e})')
                print(f'   Training (ICC): r={res_ext["pearson_r"]:.4f}')
                delta_r = res_ext["pearson_r"] - nnunet_res["pearson_r"]
                print(f'   dr = {delta_r:+.4f} ({"MINOR" if abs(delta_r) < 0.1 else "MODERATE" if abs(delta_r) < 0.2 else "LARGE"} degradation)')
            else:
                print(f'\n[FAIL] nnUNet validation failed')
        else:
            print(f'\n[WARN] nnUNet data path not found: {nnunet_data_path}')
    except Exception as e:
        print(f'\n[ERROR] Error during nnUNet robustness test: {e}')
        import traceback
        traceback.print_exc()
    
    sdf, pca, wts, scaler = sdf_delta, pca_delta, wts_delta, scaler_delta
    merged, res = merged_delta, res_delta
    sel = selected_feats
    cdf = cdf_delta
    print('\nUsing DELTA-F (Self-Control) Score for figures and output')

    generate_figures(df, ws, fm, merged, res, cdf, pca, sel)

    dt = (datetime.now() - t0).total_seconds()
    print('\n' + '=' * 70)
    print('OK ANALYSIS COMPLETE')
    print(f'  Time: {dt:.1f}s | Wells: {len(wells)} | Features: {len(sel)}D')
    if res:
        print(f"  Delta-F (dF):  Pearson r={res['pearson_r']:.4f} (p={res['pearson_p']:.2e}) | Spearman rho={res['spearman_rho']:.4f}")
    if res_ext:
        print(f"  Endpoint (F):  Pearson r={res_ext['pearson_r']:.4f} (p={res_ext['pearson_p']:.2e}) | Spearman rho={res_ext['spearman_rho']:.4f}")
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