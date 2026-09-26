# ============================================================
# 论文 Delta (K6→4 mean, 先差后PCA, r=0.957) — 完整分析
# 1. 特征选择结果
# 2. 聚类中心 + 生物学意义
# 3. PCA 主成分公式 (每个 PC 的特征组成)
# 4. 模型导出 (可用于新数据预测)
# ============================================================
import os, sys, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, spearmanr
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model')
os.makedirs(OUTPUT_DIR, exist_ok=True)

ATP_DB = {
    'B10': 601300, 'B11': 11180000, 'B2': 5391000, 'B3': 6538000,
    'B4': 7103000, 'B5': 511900, 'B6': 404500, 'B7': 403900,
    'B8': 312700, 'B9': 140300, 'C10': 211800, 'C11': 13930000,
    'C2': 6336000, 'C3': 8336000, 'C4': 6800000, 'C5': 330900,
    'C6': 238900, 'C7': 682100, 'C8': 211300, 'C9': 465900,
    'D11': 11240000, 'E11': 14700000, 'F10': 21910000, 'F11': 11180000,
    'F2': 18240000, 'F3': 14110000, 'F4': 13740000, 'F5': 17250000,
    'F6': 20320000, 'F7': 20000000, 'F8': 17170000, 'F9': 15830000,
}

NUMERIC_MAP = {3: 0, 5: 0, 1: 1, 0: 2, 4: 2, 2: 3}
PHENOTYPE_NAMES = {0: '巨大囊泡型', 1: '中等过渡型', 2: '小体积基准型', 3: '高散射实心型'}
N_PC = 4
N_MERGED = 4

FEATURE_LONG_NAMES = {
    'Organoids_Volume': 'O_Volume',
    'Organoids_Volume_Fill': 'Volume_Fill',
    'Organoids_Surface': 'Surface',
    'Cavity_Volume': 'Cavity_Vol',
    'CavityNum': 'CavityNum',
    'LongAxis': 'Long_Axis',
    'ShortAxis': 'Short_Axis',
    'Wall_Thickness': 'Wall_Thick',
    'Sphericity': 'Sphericity',
    'Scatt_Mean': 'Scatt_Mean',
    'Scatt_STD': 'Scatt_STD',
}
CLUSTER_FEATURES = list(FEATURE_LONG_NAMES.keys())


def load_analysis_table():
    path = os.path.join(OUTPUT_DIR, 'FXN_2023_Analysis.xlsx')
    df = pd.read_excel(path)
    df['Well_ID'] = df['Name'].apply(lambda x: str(x).split('_')[0])
    df['TimePoint'] = df['Name'].apply(lambda x: str(x).split('_')[1])
    return df


# ============================================================
# Part 1: 特征选择结果
# ============================================================
def analyze_selected_features():
    print('='*70)
    print('  Part 1: 最优特征组合分析 (论文 Delta 方法)')
    print('='*70)

    # 从搜索结果中读取最优特征
    search_path = os.path.join(OUTPUT_DIR, 'delta_search_Paper_Delta_(K6\u21924_mean).xlsx')
    df_search = pd.read_excel(search_path)
    best = df_search.iloc[0]
    features_str = best['features']
    feature_list = [f.strip() for f in features_str.split(',')]
    print(f'\n  最优组合: {len(feature_list)} 个特征, |r| = {best["|r|"]:.4f}')
    print(f'  Spearman rho = {best.get("rho", "N/A")}')

    # 按功能分组
    groups = {}
    for f in feature_list:
        # 提取参数名和聚类编号
        # 格式: Cyst_Thick_Avg_3, Number_2, etc.
        parts = f.rsplit('_', 1)
        if len(parts) == 2 and parts[-1].isdigit():
            cluster_id = int(parts[-1])
            param = parts[0]
        else:
            cluster_id = 'global'
            param = f

        key = f'Cluster {cluster_id}' if isinstance(cluster_id, int) else cluster_id
        groups.setdefault(key, []).append(param)

    print('\n  按表型分组:')
    clus_count = {}
    for k, v in groups.items():
        print(f'    {k}: {v}')
        for vi in v:
            ci = int(k.split()[-1]) if 'Cluster' in k else -1
            clus_count[ci] = clus_count.get(ci, 0) + 1

    print('\n  各表型贡献特征数:')
    for c in range(1, N_MERGED + 1):
        cnt = clus_count.get(c, 0)
        print(f'    表型 {c} ({PHENOTYPE_NAMES[c-1]}): {cnt} 个特征')

    # 按参数类型分组
    print('\n  按参数类型分组:')
    param_count = {}
    for f in feature_list:
        parts = f.rsplit('_', 1)
        param = parts[0]
        # simplify: remove trailing _Avg etc
        base = param.replace('_Avg', '').replace('_All', '')
        param_count[base] = param_count.get(base, 0) + 1
    for k, v in sorted(param_count.items(), key=lambda x: -x[1]):
        print(f'    {k}: {v}')

    return feature_list


# ============================================================
# Part 2: 聚类中心 + 生物学意义
# ============================================================
def analyze_clusters():
    print('\n' + '='*70)
    print('  Part 2: K=6 聚类中心 → 合并为 4 表型')
    print('='*70)

    with open(os.path.join(MODEL_DIR, 'kmeans_k6.pkl'), 'rb') as f:
        kmeans = pickle.load(f)
    with open(os.path.join(MODEL_DIR, 'scaler_k6.pkl'), 'rb') as f:
        scaler = pickle.load(f)

    centers_std = kmeans.cluster_centers_
    centers_raw = scaler.inverse_transform(centers_std)

    # K=6 → 合并为 4
    df_c = pd.DataFrame(centers_raw, columns=CLUSTER_FEATURES)
    df_c['K6_Label'] = range(6)
    df_c['Merged_Label'] = [NUMERIC_MAP[l] for l in range(6)]
    df_c['Phenotype'] = [PHENOTYPE_NAMES[NUMERIC_MAP[l]] for l in range(6)]
    df_c = df_c.sort_values('Merged_Label')

    print('\n  K6 → K4 合并映射:')
    for k6, k4 in sorted(NUMERIC_MAP.items()):
        print(f'    K6={k6} → K4={k4} ({PHENOTYPE_NAMES[k4]})')

    # 合并后的聚类中心
    print('\n  合并后 K=4 聚类中心 (原始尺度):')
    merge_centers = {}
    for c in range(N_MERGED):
        mask = df_c['Merged_Label'] == c
        sub = df_c[mask]
        # 权重平均（按聚类大小）
        center = sub[CLUSTER_FEATURES].mean()
        merge_centers[c] = center

    df_mc = pd.DataFrame(merge_centers).T
    df_mc['Phenotype'] = [PHENOTYPE_NAMES[c] for c in df_mc.index]

    # 格式化
    for f in CLUSTER_FEATURES:
        short = FEATURE_LONG_NAMES[f]
        vals = df_mc[f].values
        print(f'\n    {short}:')
        for c in range(N_MERGED):
            print(f'      表型{c} ({PHENOTYPE_NAMES[c]}): {vals[c]:.4f}')

    # 生物学解读
    print('\n' + '='*70)
    print('  表型生物学解读')
    print('='*70)

    interpretations = {
        0: {
            '特征': '体积最大(>440万μm³), 空腔容积最高(>180万μm³), 长轴最长',
            '意义': '囊泡化最严重 — 内部有大空腔, 可能是分化成熟或退化早期',
            '药物响应预期': '对化疗敏感 — 囊泡易破裂, 体积收缩空间大',
        },
        1: {
            '特征': '中等体积(~130万μm³), 中等散射, 壁厚适中',
            '意义': '过渡态 — 既有正常结构又有轻度异常',
            '药物响应预期': '中等响应 — 取决于药物机制',
        },
        2: {
            '特征': '体积最小(~64万μm³), 散射最低, 实心无空腔',
            '意义': '基准正常型 — 增殖活跃的实心小球体',
            '药物响应预期': '相对耐药 — 实心结构药物渗透难',
        },
        3: {
            '特征': '散射均值最高(~105), 体积中等, 球形度低',
            '意义': '高散射实心型 — 细胞密度大/坏死, 形态不规则',
            '药物响应预期': '耐药核心 — 高散射暗示高密度/坏死',
        },
    }

    for c in range(N_MERGED):
        print(f'\n  表型 {c} ({PHENOTYPE_NAMES[c]}):')
        print(f'    特征: {interpretations[c]["特征"]}')
        print(f'    生物学: {interpretations[c]["意义"]}')
        print(f'    药敏预期: {interpretations[c]["药物响应预期"]}')

    # 保存
    df_mc.to_excel(os.path.join(OUTPUT_DIR, 'cluster_centers_biology.xlsx'), index=True)

    # 雷达图
    from math import pi
    feats_radar = ['Organoids_Volume_Fill', 'Organoids_Surface',
                   'Cavity_Volume', 'LongAxis', 'Wall_Thickness',
                   'Sphericity', 'Scatt_Mean', 'Scatt_STD']
    n_f = len(feats_radar)
    angles = [n / n_f * 2 * pi for n in range(n_f)]
    angles += angles[:1]

    # 标准化到 [0,1]
    all_vals = df_mc[feats_radar].values
    vmin, vmax = all_vals.min(axis=0), all_vals.max(axis=0)
    norms = (all_vals - vmin) / (vmax - vmin + 1e-10)

    colors = ['#E74C3C', '#F1C40F', '#2ECC71', '#3498DB']
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for c in range(N_MERGED):
        vals = list(norms[c]) + [norms[c][0]]
        ax.fill(angles, vals, alpha=0.15, color=colors[c])
        ax.plot(angles, vals, 'o-', lw=2, color=colors[c],
                label=f'{PHENOTYPE_NAMES[c]} (n={int(df_mc.index[c])})')

    ax.set_xticks(angles[:-1])
    short_names = [FEATURE_LONG_NAMES[f] for f in feats_radar]
    ax.set_xticklabels(short_names, fontsize=9)
    ax.set_title('K=4 Phenotype Radar (Merged from K=6)', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'phenotype_radar.png'), dpi=300)
    plt.close(fig)

    return df_mc


# ============================================================
# Part 3: PCA 主成分公式
# ============================================================
def analyze_pca(feature_list):
    print('\n' + '='*70)
    print('  Part 3: PCA 主成分公式 (Delta 特征)')
    print('='*70)

    df = load_analysis_table()
    # 只取数值列
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_list_num = [f for f in feature_list if f in num_cols]
    print(f'  Numeric features available: {len(feature_list_num)}/{len(feature_list)}')

    d3 = df[df['TimePoint'] == '0701'].set_index('Well_ID')
    d5 = df[df['TimePoint'] == '0703'].set_index('Well_ID')
    common = sorted(set(d3.index) & set(d5.index) & set(ATP_DB.keys()))
    delta = d5.loc[common][feature_list_num] - d3.loc[common][feature_list_num]
    feature_list = feature_list_num

    X = delta[feature_list].fillna(0).values
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    pca = PCA(n_components=N_PC, random_state=42)
    scores = pca.fit_transform(X_std)

    var = pca.explained_variance_ratio_
    cumvar = np.cumsum(var)
    total_var = np.sum(var)
    weights = var / total_var

    print(f'\n  PCA 方差解释:')
    for i in range(N_PC):
        print(f'    PC{i+1}: {var[i]:.1%} (累计 {cumvar[i]:.1%}), 权重={weights[i]:.3f}')
    print(f'    累计: {total_var:.1%}')

    # Result_Score 公式
    print(f'\n  Result_Score = ΔX · W')
    print(f'  W = Σ(weight_i × PC_i_loading)')
    print(f'  即: Result_Score = {weights[0]:.4f}·PC1 + {weights[1]:.4f}·PC2 + {weights[2]:.4f}·PC3 + {weights[3]:.4f}·PC4')

    # 每个 PC 的 Loading
    print(f'\n  各 PC 的 Loading (特征组合):')
    for i in range(N_PC):
        loadings = pca.components_[i]
        idx = np.argsort(np.abs(loadings))[::-1]
        print(f'\n  PC{i+1} (方差 {var[i]:.1%}, 权重 {weights[i]:.3f}):')
        print(f'    {"特征":<35s} {"Loading":>10s}  {"贡献%":>8s}')
        print(f'    {"-"*55}')
        for j in idx[:10]:
            contrib = (loadings[j] ** 2) / np.sum(pca.components_[i] ** 2) * 100
            print(f'    {feature_list[j]:<35s} {loadings[j]:>+10.4f}  {contrib:>7.1f}%')

    # Result_Score 综合公式：每个原始特征的净贡献
    print(f'\n' + '='*70)
    print('  Result_Score = ΔF · β   (原始特征 → 综合得分)')
    print('='*70)
    print(f'\n  综合权重 β = Σ(weight_i × PC_i):')
    beta = np.dot(pca.components_.T, weights)
    idx_beta = np.argsort(np.abs(beta))[::-1]

    print(f'  {"排名":<5s} {"特征":<35s} {"β权重":>12s} {"|贡献|%":>10s}')
    print(f'  {"-"*62}')
    total_abs = np.sum(np.abs(beta))
    for rank, j in enumerate(idx_beta, 1):
        print(f'  {rank:<5d} {feature_list[j]:<35s} {beta[j]:>+12.6f} {abs(beta[j])/total_abs*100:>9.1f}%')

    # 保存 PCA 模型
    pca_pkg = {
        'scaler': scaler,
        'pca': pca,
        'weights': weights,
        'beta': beta,
        'feature_list': feature_list,
        'var_ratio': var,
        'cumvar': cumvar,
        'total_var': total_var,
    }
    pca_path = os.path.join(MODEL_DIR, 'paper_delta_pca_model.pkl')
    with open(pca_path, 'wb') as f:
        pickle.dump(pca_pkg, f)
    print(f'\n  PCA 模型已保存: {pca_path}')

    # PC1 vs PC2 双标图
    fig, ax = plt.subplots(figsize=(10, 8))
    score_x = scores[:, 0]
    score_y = scores[:, 1]
    ax.scatter(score_x, score_y, c='steelblue', edgecolors='white', s=80, zorder=3)
    for i, wid in enumerate(common):
        ax.annotate(wid, (score_x[i], score_y[i]), fontsize=7, alpha=0.7)

    # 特征向量
    scale = max(abs(score_x).max(), abs(score_y).max()) * 0.8
    ld_x = pca.components_[0] * scale / max(abs(pca.components_[0]))
    ld_y = pca.components_[1] * scale / max(abs(pca.components_[1]))
    for j, fname in enumerate(feature_list):
        if abs(ld_x[j]) > 0.01 or abs(ld_y[j]) > 0.01:
            ax.arrow(0, 0, ld_x[j], ld_y[j], head_width=0.02*scale, head_length=0.03*scale,
                     fc='red', ec='red', alpha=0.6)
            ax.annotate(fname, (ld_x[j]*1.1, ld_y[j]*1.1), fontsize=6, color='darkred', alpha=0.8)

    ax.set_xlabel(f'PC1 ({var[0]:.1%})', fontsize=12)
    ax.set_ylabel(f'PC2 ({var[1]:.1%})', fontsize=12)
    ax.set_title('PCA Biplot: Paper Delta Features', fontsize=14, fontweight='bold')
    ax.axhline(y=0, color='gray', lw=0.5)
    ax.axvline(x=0, color='gray', lw=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'pca_biplot.png'), dpi=300)
    plt.close(fig)

    # 保存 loadings
    df_load = pd.DataFrame({
        'Feature': feature_list,
        'PC1': pca.components_[0],
        'PC2': pca.components_[1],
        'PC3': pca.components_[2],
        'PC4': pca.components_[3],
        'Beta': beta,
        'AbsBeta': np.abs(beta),
    }).sort_values('AbsBeta', ascending=False)
    df_load.to_excel(os.path.join(OUTPUT_DIR, 'pca_loadings.xlsx'), index=False)

    return scaler, pca, weights, beta, common, delta, scores, feature_list


# ============================================================
# Part 4: 模型 F 用于新数据验证
# ============================================================
def analyze_model_export(scaler, pca, weights, beta, feature_list, common, delta):
    print('\n' + '='*70)
    print('  Part 4: 模型 F 导出 & 新数据适用性分析')
    print('='*70)

    # 在当前数据上重新计算 Score
    X = delta[feature_list].fillna(0).values
    X_std = scaler.transform(X)
    scores = pca.transform(X_std)
    result_score = np.dot(scores, weights)

    # ATP 相关性验证
    atp_vals = np.array([ATP_DB.get(w, np.nan) for w in common])
    valid = ~np.isnan(atp_vals)
    r, p = pearsonr(result_score[valid], atp_vals[valid])
    rho, _ = spearmanr(result_score[valid], atp_vals[valid])

    print(f'\n  当前数据验证: r={r:.4f}, rho={rho:.4f}, p={p:.6f}')

    # 保存预测结果
    df_pred = pd.DataFrame({
        'Well_ID': list(common),
        'ATP': atp_vals,
        'Result_Score': result_score,
    })
    df_pred.to_excel(os.path.join(OUTPUT_DIR, 'paper_delta_predictions.xlsx'), index=False)

    # 公式总结
    print(f'\n  模型 F 公式:')
    print(f'  ┌────────────────────────────────────────────────────┐')
    print(f'  │  Step 1: X = [Delta特征]  (本文本 len={len(feature_list)})         │')
    print(f'  │  Step 2: X_std = scaler.transform(X)              │')
    print(f'  │  Step 3: PC = pca.transform(X_std)                │')
    print(f'  │  Step 4: Score = PC · weights                     │')
    print(f'  │  Step 5: 对每个孔 Score = Σ(β_j × Delta_j_std)    │')
    print(f'  └────────────────────────────────────────────────────┘')

    # 线性形式
    print(f'\n  线性展开: Score = β₁·X¹_std + β₂·X²_std + ... + β{len(feature_list)}·X{len(feature_list)}_std')
    print(f'  (X_j_std 是第 j 个 Delta 特征的标准化值)')

    # 对新数据的适用性
    print(f'\n  新数据适用性分析:')
    print(f'  ┌────────────────────────────────────────────────────┐')
    print(f'  │ 必要条件:                                           │')
    print(f'  │  1. 同样的 11 个聚类特征可用                        │')
    print(f'  │  2. 同样的 K6→4 聚类模型 + Scaler                  │')
    print(f'  │  3. 同样的聚合方法 (逐表型均值)                     │')
    print(f'  │  4. Day3 + Day5 配对数据                           │')
    print(f'  │                                                     │')
    print(f'  │ 预测流程:                                           │')
    print(f'  │  a. 对 Day3 数据用 K6→4 聚类                        │')
    print(f'  │  b. 逐孔逐表型计算均值聚合 (与训练数据一致)         │')
    print(f'  │  c. 计算 Delta = Day5_feats − Day3_feats            │')
    print(f'  │  d. 提取相同 {len(feature_list)} 个特征              │')
    print(f'  │  e. 用本文本的 Scaler + PCA + Weights 计算 Score     │')
    print(f'  │  f. Score vs 新 ATP 算相关性                       │')
    print(f'  └────────────────────────────────────────────────────┘')

    # 灵敏度分析：drop 一个特征对 r 的影响
    print(f'\n  单特征剔除灵敏度 (drop one):')
    base_r = r
    for j, fname in enumerate(feature_list):
        mask = np.ones(len(feature_list), dtype=bool)
        mask[j] = False
        sub_feats = [feature_list[k] for k in range(len(feature_list)) if mask[k]]
        X_sub = delta[sub_feats].fillna(0).values
        sc_sub = StandardScaler()
        Xs_sub = sc_sub.fit_transform(X_sub)
        pca_sub = PCA(n_components=N_PC, random_state=42)
        sc_sub2 = pca_sub.fit_transform(Xs_sub)
        vr_sub = pca_sub.explained_variance_ratio_
        tv_sub = np.sum(vr_sub)
        if tv_sub == 0: continue
        w_sub = vr_sub / tv_sub
        r_sub, _ = pearsonr(np.dot(sc_sub2, w_sub)[valid], atp_vals[valid])
        delta_r = base_r - abs(r_sub)
        print(f'    -{fname:40s}  → |r|={abs(r_sub):.4f}  (Δ={delta_r:+.4f})')

    # 导出可部署模型
    deploy = {
        'model_type': 'Paper_Delta_PCA',
        'version': '1.0',
        'features': feature_list,
        'n_features': len(feature_list),
        'n_pc': N_PC,
        'scaler': scaler,
        'pca': pca,
        'weights': weights,
        'beta': beta,
        'training_r': r,
        'training_p': p,
    }
    deploy_path = os.path.join(MODEL_DIR, 'paper_delta_deploy.pkl')
    with open(deploy_path, 'wb') as f:
        pickle.dump(deploy, f)
    print(f'\n  可部署模型已保存: {deploy_path}')

    return r


def main():
    # Part 1
    feature_list = analyze_selected_features()

    # Part 2
    analyze_clusters()

    # Part 3
    scaler, pca, weights, beta, common, delta, scores, feature_list = analyze_pca(feature_list)

    # Part 4
    analyze_model_export(scaler, pca, weights, beta, feature_list, common, delta)

    print('\nDone. 所有输出已保存到 output/ 和 model/ 目录.')


if __name__ == '__main__':
    main()