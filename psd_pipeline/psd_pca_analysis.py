# ============================================================
# PSD-Pipeline 步骤7：PCA 降维 + ATP 相关性验证
# 用分表型 Delta 特征做 PCA 降维，构建综合指标
# 计算与 ATP 金标准的 Pearson 相关性
# ============================================================

import os
import sys
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from psd_config import (
    OUTPUT_DIR, MODEL_DIR, ATP_DATABASE,
    PCA_MAX_COMPONENTS, PCA_RANDOM_STATE,
    NAN_THRESHOLD, VAR_THRESHOLD, CORR_THRESHOLD
)


def load_delta_features():
    delta_path = os.path.join(OUTPUT_DIR, 'psd_delta_features.xlsx')
    if not os.path.exists(delta_path):
        raise FileNotFoundError(f"请先运行 psd_delta_features.py: {delta_path}")
    df = pd.read_excel(delta_path)
    print(f"加载 Delta 特征: {len(df)} 个孔位, {len(df.columns)-1} 个特征")
    return df


def feature_selection(fm, feature_cols):
    print("\n" + "=" * 60)
    print("特征选择 (无ATP泄露: NaN + 方差 + 冗余)")
    print("=" * 60)

    drop_nan = []
    drop_var = []
    selected = []

    for f in feature_cols:
        if f not in fm.columns:
            continue
        nan_ratio = fm[f].isna().mean()
        if nan_ratio > NAN_THRESHOLD:
            drop_nan.append((f, nan_ratio))
            continue
        valid_vals = fm[f].dropna()
        if len(valid_vals) < 2:
            drop_var.append((f, 0.0))
            continue
        var_val = valid_vals.var()
        if var_val < VAR_THRESHOLD:
            drop_var.append((f, var_val))
            continue
        selected.append(f)

    print(f"\n  Step 1: NaN过滤 (>{NAN_THRESHOLD:.0%}缺失)")
    print(f"    输入: {len(feature_cols)} -> 保留: {len(selected)} (删除: {len(drop_nan)})")
    for f, r in drop_nan[:5]:
        print(f"      DROP {f}: NaN={r:.1%}")
    if len(drop_nan) > 5:
        print(f"      ... 还有 {len(drop_nan)-5} 个")

    print(f"\n  Step 2: 方差过滤 (<{VAR_THRESHOLD})")
    print(f"    保留: {len(selected)} (删除: {len(drop_var)})")
    for f, v in drop_var[:5]:
        print(f"      DROP {f}: var={v:.2e}")
    if len(drop_var) > 5:
        print(f"      ... 还有 {len(drop_var)-5} 个")

    remaining = selected[:]
    drop_corr = []

    fm_imputed = fm[remaining].copy()
    for col in remaining:
        med = fm_imputed[col].median()
        if pd.notna(med):
            fm_imputed[col] = fm_imputed[col].fillna(med)
        else:
            fm_imputed[col] = fm_imputed[col].fillna(0)

    corr_matrix = fm_imputed.corr().abs()
    checked = set()
    for i, f1 in enumerate(remaining):
        if f1 in checked:
            continue
        for j, f2 in enumerate(remaining):
            if j <= i or f2 in checked:
                continue
            if corr_matrix.loc[f1, f2] > CORR_THRESHOLD:
                v1 = fm_imputed[f1].var()
                v2 = fm_imputed[f2].var()
                drop_f = f2 if v1 >= v2 else f1
                keep_f = f1 if v1 >= v2 else f2
                drop_corr.append((drop_f, keep_f, corr_matrix.loc[f1, f2]))
                checked.add(drop_f)

    final_selected = [f for f in remaining if f not in checked]

    print(f"\n  Step 3: 冗余过滤 (|r| > {CORR_THRESHOLD})")
    print(f"    保留: {len(final_selected)} (删除: {len(drop_corr)})")
    for drop_f, keep_f, r_val in drop_corr[:5]:
        print(f"      DROP {drop_f}: r={r_val:.4f} with {keep_f}")
    if len(drop_corr) > 5:
        print(f"      ... 还有 {len(drop_corr)-5} 对")

    print(f"\n  总结: {len(feature_cols)} -> {len(final_selected)} 特征")
    print(f"  删除: {len(drop_nan)}(NaN) + {len(drop_var)}(零方差) + {len(drop_corr)}(冗余)")

    return final_selected, fm_imputed


def main():
    print("=" * 60)
    print("PSD 步骤7：PCA 降维 + ATP 相关性验证")
    print("=" * 60)

    # 1. 加载 Delta 特征
    print("\n[1/6] 加载分表型 Delta 特征...")
    df_delta = load_delta_features()

    feature_cols = [c for c in df_delta.columns if c != 'Well_ID']
    n_samples = len(df_delta)

    # 2. 特征选择
    print(f"\n[2/6] 特征选择 (输入: {len(feature_cols)} 特征)...")
    selected_feats, fm_imputed = feature_selection(df_delta, feature_cols)

    X = fm_imputed[selected_feats].values
    well_ids = df_delta['Well_ID'].values
    n_features = X.shape[1]
    print(f"\n  样本数: {n_samples}, 特征数: {n_features}, d/n={n_samples/n_features:.2f}")

    # 3. 标准化
    print("\n[3/6] 标准化...")
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    # 4. PCA 动态选择主成分数
    print(f"\n[4/6] PCA 动态主成分选择...")
    max_comp = min(n_features, n_samples - 1, PCA_MAX_COMPONENTS)
    pca_full = PCA(n_components=max_comp, random_state=PCA_RANDOM_STATE)
    pca_full.fit(X_std)

    eigenvalues = pca_full.explained_variance_
    n_comp_kaiser = max(2, int(np.sum(eigenvalues > 1.0)))
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    n_comp_cumvar = int(np.searchsorted(cumvar, 0.85)) + 1
    n_comp = max(n_comp_kaiser, n_comp_cumvar)
    n_comp = min(n_comp, max_comp)

    print(f"  Kaiser (>1.0):   {n_comp_kaiser} PCs")
    print(f"  CumVar (>85%):   {n_comp_cumvar} PCs")
    print(f"  选定:            {n_comp} PCs")

    pca = PCA(n_components=n_comp, random_state=PCA_RANDOM_STATE)
    X_pca = pca.fit_transform(X_std)

    variance_ratio = pca.explained_variance_ratio_
    weights = variance_ratio / np.sum(variance_ratio)

    print(f"\n各主成分解释方差比:")
    for i, vr in enumerate(variance_ratio):
        print(f"  PC{i+1}: {vr:.4f} ({vr*100:.1f}%), weight={weights[i]:.3f}")
    print(f"  累计: {np.sum(variance_ratio):.4f} ({np.sum(variance_ratio)*100:.1f}%)")

    # 综合得分
    composite_score = np.dot(X_pca, weights)

    # 5. 构建结果表
    print("\n[5/6] 构建结果表...")
    df_result = df_delta[['Well_ID']].copy()
    for i in range(n_comp):
        df_result[f'PC{i+1}'] = X_pca[:, i]
    df_result['PSD_Score'] = composite_score

    # 映射 ATP
    df_result['ATP'] = df_result['Well_ID'].map(ATP_DATABASE)
    df_result['ATP_Log'] = np.log10(df_result['ATP'])

    # 6. 相关性计算
    print("\n[6/6] 计算 ATP 相关性...")
    from scipy.stats import spearmanr
    valid = df_result.dropna(subset=['PSD_Score', 'ATP'])

    if len(valid) >= 3:
        r_psd, p_psd = pearsonr(valid['PSD_Score'], valid['ATP'])
        r_psd_s, p_psd_s = spearmanr(valid['PSD_Score'], valid['ATP'])
        r_psd_log, p_psd_log = pearsonr(valid['PSD_Score'], valid['ATP_Log'])
        r_psd_log_s, p_psd_log_s = spearmanr(valid['PSD_Score'], valid['ATP_Log'])

        print(f"\n{'='*60}")
        print(f"PSD 综合得分 vs ATP:")
        print(f"  Pearson r  = {r_psd:.4f}, p = {p_psd:.4f}")
        print(f"  Spearman   = {r_psd_s:.4f}, p = {p_psd_s:.4f}")
        print(f"  vs log10   r = {r_psd_log:.4f}, p = {p_psd_log:.4f}")
        print(f"  vs log10 rho = {r_psd_log_s:.4f}, p = {p_psd_log_s:.4f}")
        print(f"{'='*60}")

        print(f"\n各主成分 vs ATP:")
        for i in range(n_comp):
            r_pc, p_pc = pearsonr(valid[f'PC{i+1}'], valid['ATP'])
            r_pc_s, p_pc_s = spearmanr(valid[f'PC{i+1}'], valid['ATP'])
            sig = "*" if p_pc < 0.05 else ""
            print(f"  PC{i+1}: r={r_pc:.4f}, rho={r_pc_s:.4f}, p={p_pc:.4f} {sig}")

        # 特征贡献分析
        print(f"\nTop 10 特征对 PSD Score 的贡献:")
        coef = np.dot(pca.components_.T, weights)
        contrib_df = pd.DataFrame({
            'Feature': selected_feats,
            'Coef': coef,
            'AbsCoef': np.abs(coef)
        }).sort_values('AbsCoef', ascending=False)
        for _, row in contrib_df.head(10).iterrows():
            print(f"  {row['Coef']:+.4f}  {row['Feature']}")

        df_result.loc[0, 'PSD_vs_ATP_r'] = r_psd
        df_result.loc[0, 'PSD_vs_ATP_p'] = p_psd
        df_result.loc[0, 'PSD_vs_ATP_rho'] = r_psd_s
        df_result.loc[0, 'PSD_vs_logATP_r'] = r_psd_log
        df_result.loc[0, 'PSD_n_features'] = n_features
        df_result.loc[0, 'PSD_n_components'] = n_comp
    else:
        print(f"\n有效样本数不足: {len(valid)}")

    # 保存结果
    result_path = os.path.join(OUTPUT_DIR, 'psd_pca_result.xlsx')
    df_result.to_excel(result_path, index=False)
    print(f"\n结果已保存: {result_path}")

    # 保存 PCA 模型
    import joblib
    model_package = {
        'features_list': selected_feats,
        'scaler': scaler,
        'pca': pca,
        'weights': weights,
        'variance_ratio': variance_ratio,
        'n_components': n_comp,
        'description': 'PSD-Features 综合评分模型 (含特征选择+动态PC)'
    }
    model_save_path = os.path.join(MODEL_DIR, 'psd_pca_model.pkl')
    joblib.dump(model_package, model_save_path)
    print(f"PCA 模型已保存: {model_save_path}")

    print("\n步骤7 完成！")


if __name__ == '__main__':
    main()