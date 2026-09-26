# ============================================================
# PSD-Pipeline: 验证分析脚本
# P1: 排除"是否给药"混淆
# P3: 特征重要性分析
# ============================================================

import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from psd_config import OUTPUT_DIR, MODEL_DIR, ATP_DATABASE, DRUG_MAPPING

OUT_DIR = os.path.join(OUTPUT_DIR, 'validation')
os.makedirs(OUT_DIR, exist_ok=True)


def load_data():
    delta_path = os.path.join(OUTPUT_DIR, 'psd_delta_features.xlsx')
    pca_path = os.path.join(OUTPUT_DIR, 'psd_pca_result.xlsx')

    if not os.path.exists(pca_path):
        raise FileNotFoundError(f"请先运行 run_all.py: {pca_path}")

    df_pca = pd.read_excel(pca_path)
    df_delta = pd.read_excel(delta_path)

    print(f"加载 PCA 结果: {len(df_pca)} wells")
    print(f"加载 Delta 特征: {len(df_delta)} wells, {len(df_delta.columns)-1} 特征")
    return df_pca, df_delta


def add_labels(df):
    df['Drug'] = df['Well_ID'].apply(lambda w: DRUG_MAPPING.get(w, ('?','?','?',))[0])
    df['Mechanism'] = df['Well_ID'].apply(lambda w: DRUG_MAPPING.get(w, ('?','?','?'))[1])
    df['Group'] = df['Well_ID'].apply(lambda w: DRUG_MAPPING.get(w, ('?','?','?'))[2])
    df['Is_Control'] = df['Well_ID'].apply(lambda w: 1 if DRUG_MAPPING.get(w, ('?',))[0] == '对照' else 0)
    return df


# ================================================================
# P1: 排除"是否给药"混淆
# ================================================================
def p1_confound_check(df_pca):
    print("\n" + "=" * 70)
    print("  P1: 排除\"是否给药\"混淆")
    print("=" * 70)

    df = df_pca.copy()
    df = add_labels(df)
    valid = df.dropna(subset=['PSD_Score', 'ATP'])
    valid = valid.copy()
    valid['ATP_Log'] = np.log10(valid['ATP'])
    valid['Is_Treated'] = 1 - valid['Is_Control']

    # Step 1: 计算 Control vs Treated 的二分类 AUC
    print("\n--- Step 1: PSD Score 能否区分 Control vs Treated? ---")
    if valid['Is_Treated'].nunique() >= 2:
        auc = roc_auc_score(valid['Is_Treated'], valid['PSD_Score'])
        print(f"  AUC(Control vs Treated) = {auc:.4f}")

        if auc > 0.90:
            print(f"  >>> 高度可区分! PCA 严重受给药状态主导")
            verdict = "PCA 主要由给药状态驱动"
        elif auc > 0.70:
            print(f"  >>> 中度可区分, PCA 部分受给药状态影响")
            verdict = "PCA 部分受给药状态影响"
        else:
            print(f"  >>> 难以区分, 给药状态不是 PCA 主要驱动")
            verdict = "PCA 不受给药状态主导"

        control_mean = valid[valid['Is_Control']==1]['PSD_Score'].mean()
        treated_mean = valid[valid['Is_Treated']==1]['PSD_Score'].mean()
        print(f"  Control PSD Score 均值: {control_mean:.4f}")
        print(f"  Treated PSD Score 均值: {treated_mean:.4f}")

        result_rows = [
            ["--- P1: 排除\"是否给药\"混淆 ---", "", "", ""],
            ["AUC(Control vs Treated)", f"{auc:.4f}", verdict, ""],
            [f"Control 均值 PSD={control_mean:.4f}, Treated 均值 PSD={treated_mean:.4f}", "", "", ""],
        ]
    else:
        result_rows = [["--- P1: 无对照组数据 ---", "", "", ""]]
        auc = None

    # Step 2: 给药组内部 PSD Score vs ATP
    print("\n--- Step 2: 给药组内部 PSD Score vs ATP ---")
    treated = valid[valid['Is_Treated'] == 1]

    r_full, _ = pearsonr(valid['PSD_Score'], valid['ATP_Log'])
    rho_full, _ = spearmanr(valid['PSD_Score'], valid['ATP_Log'])

    if len(treated) >= 3:
        r_treated, p_treated = pearsonr(treated['PSD_Score'], treated['ATP_Log'])
        rho_treated, p_treated_s = spearmanr(treated['PSD_Score'], treated['ATP_Log'])

        print(f"  全量 (n={len(valid)}):    r={r_full:.4f}, rho={rho_full:.4f}")
        print(f"  给药组 (n={len(treated)}): r={r_treated:.4f}, rho={rho_treated:.4f}, p={p_treated:.6f}")

        drop = r_full - r_treated
        if abs(drop) > 0.15:
            print(f"  >>> 相关性从 {abs(r_full):.3f} 降至 {abs(r_treated):.3f}, 下降 {drop:.2f}")
            print(f"  >>> PCA 严重受给药状态混淆!")
            confound_verdict = f"严重混淆: 全量|r|={abs(r_full):.3f} -> 给药组|r|={abs(r_treated):.3f}"
        elif abs(drop) > 0.05:
            print(f"  >>> 相关性轻微下降, 部分混淆")
            confound_verdict = f"轻微混淆: 全量|r|={abs(r_full):.3f} -> 给药组|r|={abs(r_treated):.3f}"
        else:
            print(f"  >>> 相关性基本稳定, 无混淆")
            confound_verdict = f"无混淆: 全量|r|={abs(r_full):.3f} ≈ 给药组|r|={abs(r_treated):.3f}"

        result_rows += [
            [f"全量(n={len(valid)}) r/rho", f"{r_full:.4f}", f"{rho_full:.4f}", ""],
            [f"给药组(n={len(treated)}) r/rho", f"{r_treated:.4f}", f"{rho_treated:.4f}", f"p={p_treated:.6f}"],
            ["混淆判定", confound_verdict, "", ""],
        ]
    else:
        print(f"  给药组样本不足: {len(treated)}")
        confound_verdict = "样本不足无法判断"

    # Step 3: 按药物分组相关性
    print("\n--- Step 3: 各药物组 PSD Score vs ATP ---")
    result_rows += [["--- 各药物组相关性 ---", "", "", ""]]

    drug_groups = valid.groupby('Group')
    for group_name, subgroup in drug_groups:
        if len(subgroup) >= 3:
            r_g, p_g = pearsonr(subgroup['PSD_Score'], subgroup['ATP_Log'])
            rho_g, _ = spearmanr(subgroup['PSD_Score'], subgroup['ATP_Log'])
            sig = "*" if p_g < 0.05 else ""
            print(f"  {group_name} (n={len(subgroup)}): r={r_g:.4f}, rho={rho_g:.4f}, p={p_g:.4f} {sig}")
            result_rows.append([
                group_name, f"n={len(subgroup)}", f"r={r_g:.4f}", f"rho={rho_g:.4f} {sig}"
            ])
        else:
            print(f"  {group_name} (n={len(subgroup)}): 样本不足")

    return result_rows, confound_verdict


# ================================================================
# P3: 特征重要性分析
# ================================================================
def p3_feature_importance(df_delta, df_pca):
    print("\n\n" + "=" * 70)
    print("  P3: 特征重要性分析")
    print("=" * 70)

    df = df_delta.copy()
    df['ATP'] = df['Well_ID'].map(ATP_DATABASE)
    df['ATP_Log'] = np.log10(df['ATP'])
    df = add_labels(df)

    delta_cols = [c for c in df_delta.columns if c != 'Well_ID' and 'Delta' in c]
    result_rows = []

    # Step 1: 每个 Delta 特征与 ATP 的单独相关性
    print("\n--- Step 1: 单特征 vs log10 ATP ---")
    print(f"  共 {len(delta_cols)} 个 Delta 特征")

    feat_corrs = []
    for f in delta_cols:
        valid = df[[f, 'ATP_Log']].dropna()
        if len(valid) >= 5:
            r, p = pearsonr(valid[f], valid['ATP_Log'])
            rho, _ = spearmanr(valid[f], valid['ATP_Log'])
            feat_corrs.append({
                'Feature': f, 'r': r, 'rho': rho, 'p': p,
                'abs_r': abs(r),
                'Cluster': int(f.split('_Delta_')[1]) if '_Delta_' in f else -1,
                'Param': f.split('_Delta_')[0] if '_Delta_' in f else f
            })

    feat_corrs.sort(key=lambda x: x['abs_r'], reverse=True)

    print(f"\n  Top 15 单特征 ATP 相关性:")
    print(f"  {'Rank':<5} {'Feature':<30} {'r':>8} {'rho':>8} {'p':>10}")
    print(f"  {'-'*65}")

    for i, fc in enumerate(feat_corrs[:15]):
        sig = "***" if fc['p'] < 0.001 else "**" if fc['p'] < 0.01 else "*" if fc['p'] < 0.05 else ""
        print(f"  {i+1:<5} {fc['Feature']:<30} {fc['r']:>8.4f} {fc['rho']:>8.4f} {fc['p']:>10.6f} {sig}")
        result_rows.append([
            f"单特征 #{i+1}", fc['Feature'], f"r={fc['r']:.4f}", f"rho={fc['rho']:.4f}, p={fc['p']:.6f} {sig}"
        ])

    # Step 2: 按聚类汇总特征贡献
    print(f"\n--- Step 2: 按聚类汇总特征贡献 ---")
    cluster_summary = {}
    for fc in feat_corrs:
        c = fc['Cluster']
        if c not in cluster_summary:
            cluster_summary[c] = []
        cluster_summary[c].append(fc['abs_r'])

    result_rows += [["--- 按聚类汇总 ---", "", "", ""]]
    for c in sorted(cluster_summary.keys()):
        vals = cluster_summary[c]
        mean_abs_r = np.mean(vals)
        top_feat = max(
            [fc for fc in feat_corrs if fc['Cluster'] == c],
            key=lambda x: x['abs_r']
        )
        print(f"  Cluster {c}: 均值|r|={mean_abs_r:.4f}, 最佳特征: {top_feat['Feature']} (|r|={top_feat['abs_r']:.4f})")
        result_rows.append([
            f"Cluster {c}", f"均值|r|={mean_abs_r:.4f}",
            f"最佳: {top_feat['Feature']}", f"|r|={top_feat['abs_r']:.4f}"
        ])

    # Step 3: 按参数类型汇总
    print(f"\n--- Step 3: 按参数类型汇总 ---")
    param_summary = {}
    for fc in feat_corrs:
        p = fc['Param']
        if p not in param_summary:
            param_summary[p] = []
        param_summary[p].append(fc['abs_r'])

    result_rows += [["--- 按参数类型汇总 ---", "", "", ""]]
    for p in sorted(param_summary.keys(), key=lambda x: np.mean(param_summary[x]), reverse=True):
        mean_abs_r = np.mean(param_summary[p])
        print(f"  {p:<20}: 均值|r|={mean_abs_r:.4f} ({len(param_summary[p])} 聚类)")
        result_rows.append([
            p, f"均值|r|={mean_abs_r:.4f}", f"({len(param_summary[p])} 聚类)", ""
        ])

    # Step 4: PCA 载荷贡献 (从 PCA 模型读取)
    print(f"\n--- Step 4: PCA 载荷贡献 ---")
    import joblib
    model_path = os.path.join(MODEL_DIR, 'psd_pca_model.pkl')
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        pca = model['pca']
        weights = model['weights']
        features_list = model['features_list']

        coef = np.dot(pca.components_.T, weights)
        contrib_df = pd.DataFrame({
            'Feature': features_list,
            'Coef': coef,
            'AbsCoef': np.abs(coef)
        }).sort_values('AbsCoef', ascending=False)

        print(f"\n  Top 15 PCA 载荷贡献:")
        print(f"  {'Rank':<5} {'Feature':<30} {'Coef':>10}")

        result_rows += [["--- PCA载荷贡献 ---", "", "", ""]]
        for i, (_, row) in enumerate(contrib_df.head(15).iterrows()):
            print(f"  {i+1:<5} {row['Feature']:<30} {row['Coef']:>10.4f}")
            result_rows.append([
                f"载荷 #{i+1}", row['Feature'], f"Coef={row['Coef']:.4f}", ""
            ])

        cluster_coef = {}
        for _, row in contrib_df.iterrows():
            feat = row['Feature']
            if '_Delta_' in feat:
                c = int(feat.split('_Delta_')[1])
                if c not in cluster_coef:
                    cluster_coef[c] = []
                cluster_coef[c].append(abs(row['Coef']))

        print(f"\n  按聚类 PCA 载荷均值:")
        result_rows += [["--- 按聚类 PCA 载荷 ---", "", "", ""]]
        for c in sorted(cluster_coef.keys()):
            mean_c = np.mean(cluster_coef[c])
            print(f"    Cluster {c}: 均值|Coef|={mean_c:.4f}")
            result_rows.append([
                f"Cluster {c}", f"均值|Coef|={mean_c:.4f}", "", ""
            ])

    # Step 5: 给药组内的特征重要性 (去混淆)
    print(f"\n--- Step 5: 给药组内特征相关性 (去混淆) ---")
    treated = df[df['Is_Control'] == 0]

    treated_feat_corrs = []
    for f in delta_cols:
        valid_t = treated[[f, 'ATP_Log']].dropna()
        if len(valid_t) >= 5:
            r, p = pearsonr(valid_t[f], valid_t['ATP_Log'])
            rho, _ = spearmanr(valid_t[f], valid_t['ATP_Log'])
            treated_feat_corrs.append({
                'Feature': f, 'r': r, 'rho': rho, 'p': p, 'abs_r': abs(r)
            })

    treated_feat_corrs.sort(key=lambda x: x['abs_r'], reverse=True)

    print(f"  Top 10 给药组内特征:")
    result_rows += [["--- 给药组内 Top 10 ---", "", "", ""]]
    for i, fc in enumerate(treated_feat_corrs[:10]):
        sig = "***" if fc['p'] < 0.001 else "**" if fc['p'] < 0.01 else "*" if fc['p'] < 0.05 else ""
        print(f"  {i+1:<5} {fc['Feature']:<30} r={fc['r']:>8.4f} rho={fc['rho']:>8.4f} p={fc['p']:.6f} {sig}")
        result_rows.append([
            f"给药组 #{i+1}", fc['Feature'], f"r={fc['r']:.4f}", f"rho={fc['rho']:.4f} {sig}"
        ])

    return result_rows, feat_corrs


# ================================================================
# 主函数
# ================================================================
def main():
    print("=" * 70)
    print("  PSD-Pipeline 验证分析: P1 (给药混淆) + P3 (特征重要性)")
    print("=" * 70)

    df_pca, df_delta = load_data()

    # P1
    p1_rows, confound_verdict = p1_confound_check(df_pca)
    p3_rows, feat_corrs = p3_feature_importance(df_delta, df_pca)

    # 综合报告
    print("\n\n" + "=" * 70)
    print("  综合诊断")
    print("=" * 70)
    print(f"  P1: {confound_verdict}")

    # 保存报告
    report_path = os.path.join(OUT_DIR, 'validation_report.xlsx')
    all_rows = (p1_rows + [["", "", "", ""]] + p3_rows)
    df_report = pd.DataFrame(all_rows, columns=['Section', 'Item', 'Value1', 'Value2'])
    df_report.to_excel(report_path, index=False)
    print(f"\n  报告已保存: {report_path}")

    print("\n 验证分析完成！")


if __name__ == '__main__':
    main()