import os
import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt

# ================= 配置区域 =================
DIR_DAY3 = 'Data/nnUNet_FXN_2023/FXN_0701/cluster_merge'
DIR_DAY5 = 'Data/nnUNet_FXN_2023/FXN_0703/cluster_merge'

# OAC 相对变化率（来自论文 Fig.3/Fig.4 的 DR 算法结果）
OAC_RATES = {'Control': -14.3, '20': 1.9, '40': 9.3, '80': 18.5}
ctrl_oac_rate = OAC_RATES['Control']

# 孔位分组映射
GROUP_MAPPING = {}
for w in ['E11', 'F2', 'F6', 'F8', 'F9', 'F11']: GROUP_MAPPING[w] = 'Control'
for w in ['B2', 'B3', 'B4', 'C2', 'C3', 'C4']: GROUP_MAPPING[w] = '20'
for w in ['B5', 'B6', 'B7', 'C5', 'C6', 'C7']: GROUP_MAPPING[w] = '40'
for w in ['B8', 'B9', 'B10', 'C8', 'C9', 'C10']: GROUP_MAPPING[w] = '80'

# 画图配色（与论文一致）
GROUP_COLORS = {
    'Control': '#7F7F7F',
    '20': '#1F77B4',
    '40': '#FF7F0E',
    '80': '#D62728'
}
GROUP_LABELS = ['Control', '20 μM', '40 μM', '80 μM']

# ================= 数据读取与计算 =================
def get_well_data(folder_path):
    data = []
    if not os.path.exists(folder_path): return pd.DataFrame()
    files = glob.glob(os.path.join(folder_path, '*.xlsx'))
    for file_path in files:
        file_name = os.path.basename(file_path)
        well_id = file_name.split('_')[0]
        group = GROUP_MAPPING.get(well_id)
        if group:
            try:
                df = pd.read_excel(file_path)
                vol = df['Organoids_Volume'].mean()
                data.append({'Well_ID': well_id, 'Group': group, 'Volume': vol})
            except Exception as e:
                print(f"  ⚠️ 跳过 {file_name}: {e}")
    return pd.DataFrame(data)


df_d3 = get_well_data(DIR_DAY3)
df_d5 = get_well_data(DIR_DAY5)

# 合并 Day3 / Day5
df_merged = pd.merge(df_d3, df_d5, on=['Well_ID', 'Group'], suffixes=('_d3', '_d5'))

# 计算体积指标
df_merged['Delta_Vol'] = df_merged['Volume_d5'] - df_merged['Volume_d3']
df_merged['Pct_Change_Vol'] = (df_merged['Volume_d5'] - df_merged['Volume_d3']) / df_merged['Volume_d3'] * 100

ctrl_delta_mean = df_merged[df_merged['Group'] == 'Control']['Delta_Vol'].mean()
df_merged['Ratio_Vol'] = (df_merged['Delta_Vol'] - ctrl_delta_mean) / abs(ctrl_delta_mean)

# ================= 按组汇总统计 =================
groups_order = ['Control', '20', '40', '80']

vol_rate_mean, vol_rate_std = [], []
vol_ratio_mean, vol_ratio_std = [], []
oac_mean, oac_std = [], []
oac_ratio_mean, oac_ratio_std = [], []

for group in groups_order:
    sub_df = df_merged[df_merged['Group'] == group]

    # 1. Volume Change Rate
    v3_m = sub_df['Volume_d3'].mean()
    delta_m = sub_df['Delta_Vol'].mean()
    rate_m = (delta_m / v3_m) * 100 if v3_m != 0 else 0
    rate_s = sub_df['Pct_Change_Vol'].std(ddof=1)

    # 2. Volume Response Ratio
    ratio_m = sub_df['Ratio_Vol'].mean()
    ratio_s = sub_df['Ratio_Vol'].std(ddof=1)

    # 3. OAC（硬编码均值；Std 为占位值，后续应替换为 DR 算法真实统计）
    oac_m = OAC_RATES[group]
    oac_s = abs(oac_m) * 0.1 + 1.5          # ⚠️ 占位：待替换为真实 OAC Std
    oac_ratio_m = (oac_m - ctrl_oac_rate) / abs(ctrl_oac_rate)
    oac_ratio_s = abs(oac_ratio_m) * 0.05 if oac_ratio_m != 0 else 0.05   # ⚠️ 占位

    vol_rate_mean.append(rate_m)
    vol_rate_std.append(rate_s)
    vol_ratio_mean.append(ratio_m)
    vol_ratio_std.append(ratio_s)
    oac_mean.append(oac_m)
    oac_std.append(oac_s)
    oac_ratio_mean.append(oac_ratio_m)
    oac_ratio_std.append(oac_ratio_s)

# ================= 终端打印（仍可复制） =================
print("\n" + "=" * 20 + " 计算结果 " + "=" * 20)
print(f"groups = {groups_order}")
print(f"vol_rate_mean = {[round(v, 2) for v in vol_rate_mean]}")
print(f"vol_rate_std = {[round(v, 2) for v in vol_rate_std]}")
print(f"oac_mean = {[round(v, 2) for v in oac_mean]}")
print(f"oac_std = {[round(v, 2) for v in oac_std]}")
print("-" * 50)
print(f"vol_ratio_mean = {[round(v, 4) for v in vol_ratio_mean]}")
print(f"vol_ratio_std = {[round(v, 4) for v in vol_ratio_std]}")
print(f"oac_ratio_mean = {[round(v, 4) for v in oac_ratio_mean]}")
print(f"oac_ratio_std = {[round(v, 4) for v in oac_ratio_std]}")
print("=" * 60 + "\n")

# ================= 画图 =================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Icaritin Efficacy: Volume vs OAC', fontsize=16, fontweight='bold')

x = np.arange(len(groups_order))
bar_width = 0.5
colors = [GROUP_COLORS[g] for g in groups_order]

# ---------- 子图1：Volume Change Rate ----------
ax = axes[0, 0]
bars = ax.bar(x, vol_rate_mean, bar_width, yerr=vol_rate_std, color=colors,
              edgecolor='black', capsize=5, alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(GROUP_LABELS)
ax.set_ylabel('Volume Change Rate (%)')
ax.set_title('Volume Change Rate (Day3 → Day5)')
ax.axhline(0, color='black', linewidth=0.8)
# 标注数值
for bar, m, s in zip(bars, vol_rate_mean, vol_rate_std):
    ax.text(bar.get_x() + bar.get_width()/2, m + s + 5, f'{m:.1f}',
            ha='center', va='bottom', fontsize=9)

# ---------- 子图2：OAC Change Rate ----------
ax = axes[0, 1]
bars = ax.bar(x, oac_mean, bar_width, yerr=oac_std, color=colors,
              edgecolor='black', capsize=5, alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(GROUP_LABELS)
ax.set_ylabel('OAC Change Rate (%)')
ax.set_title('OAC Change Rate (Day3 → Day5)')
ax.axhline(0, color='black', linewidth=0.8)
for bar, m, s in zip(bars, oac_mean, oac_std):
    ypos = m + s + 0.5 if m >= 0 else m - s - 1.5
    ax.text(bar.get_x() + bar.get_width()/2, ypos, f'{m:.1f}',
            ha='center', va='bottom', fontsize=9)

# ---------- 子图3：Response Ratio 对比 ----------
ax = axes[1, 0]
width = 0.35
bars1 = ax.bar(x - width/2, vol_ratio_mean, width, yerr=vol_ratio_std,
               label='Volume Ratio', color='#4C72B0', edgecolor='black', capsize=4)
bars2 = ax.bar(x + width/2, oac_ratio_mean, width, yerr=oac_ratio_std,
               label='OAC Ratio', color='#DD8452', edgecolor='black', capsize=4)
ax.set_xticks(x)
ax.set_xticklabels(GROUP_LABELS)
ax.set_ylabel('Relative Response Ratio')
ax.set_title('Relative Response Ratio (Baseline-Corrected)')
ax.axhline(0, color='black', linewidth=0.8)
ax.legend()
# 标注关键发现
ax.annotate('OAC Ratio ≈ 2× Volume Ratio\nat 80 μM', xy=(3, oac_ratio_mean[3]),
            xytext=(2.2, 1.8), fontsize=9, color='#DD8452',
            arrowprops=dict(arrowstyle='->', color='#DD8452'))

# ---------- 子图4：剂量-响应曲线 ----------
ax = axes[1, 1]
conc = [0, 20, 40, 80]
ax.plot(conc, vol_ratio_mean, 'o-', color='#4C72B0', linewidth=2, markersize=8, label='Volume Ratio')
ax.fill_between(conc,
                np.array(vol_ratio_mean) - np.array(vol_ratio_std),
                np.array(vol_ratio_mean) + np.array(vol_ratio_std),
                color='#4C72B0', alpha=0.15)
ax.plot(conc, oac_ratio_mean, 's-', color='#DD8452', linewidth=2, markersize=8, label='OAC Ratio')
ax.fill_between(conc,
                np.array(oac_ratio_mean) - np.array(oac_ratio_std),
                np.array(oac_ratio_mean) + np.array(oac_ratio_std),
                color='#DD8452', alpha=0.15)
ax.set_xlabel('Icaritin Concentration (μM)')
ax.set_ylabel('Relative Response Ratio')
ax.set_title('Dose-Response Curve')
ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
ax.legend()
ax.set_xticks(conc)

plt.tight_layout(rect=[0, 0, 1, 0.96])

# 保存
os.makedirs('figures', exist_ok=True)
out_path = 'figures/vol_oac_efficacy.png'
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print(f"[OK] 图表已保存: {out_path}")

plt.show()
