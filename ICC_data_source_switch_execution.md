# FXN_2023_new（ICC）数据源切换执行文档

## 1. 目标

将分析数据源从 `Data/nnUNet_FXN_2023` 切换到 `Data/FXN_2023_new（ICC）`，并给出可直接执行的流程。

## 2. 已完成的结构探查结论

### 2.1 顶层结构

`Data/FXN_2023_new（ICC）` 下存在：

- `FXN_20230701/`
- `FXN_20230703/`
- `FXN_2023_Analysis.xlsx`
- `FXN_2023_PCA.xlsx`
- `FXN_2023_PCA_Result.xlsx`
- `FXN_2023_PCA全部.xlsx`
- `FXN_2023_PCA去散射系数.xlsx`
- `Feature_Optimization_Sorted.xlsx`
- `PCA_Model_Details.xlsx`
- 其他历史结果文件

### 2.2 两个时间点目录结构

`FXN_20230701/` 与 `FXN_20230703/` 结构一致，均包含：

- `measure_excel/`
- `cluster_excel/`
- `cluster_merge/`
- `roughness/`
- `scatt/`
- `scatttt/`
- `scatt_mat/`
- `scatt_seg/`
- `seg_fill/`
- `seg_label/`
- `seg_mat_test/`

### 2.3 文件数量与格式

每个时间点（0701/0703）统计如下：

- `.xlsx`: 193
- `.mat`: 128
- `.tif`: 32

`measure_excel/` 关键结论：

- 0701 文件数: 32
- 0703 文件数: 32
- 孔位一一配对完整: 32 对
- `B2_0701`、`B2_0703` 均存在（未删除 B2）
- 全部文件列结构一致（1 种 schema）

## 3. 数据格式理解

### 3.1 单孔 measure_excel 文件格式

抽样文件：`B2_0701.xlsx`、`B2_0703.xlsx`。

列名共 14 列：

1. `Index`
2. `Organoids_Volume`
3. `Organoids_Volume_Fill`
4. `Organoids_Surface`
5. `Cavity_Volume`
6. `CavityNum`
7. `CavityVolumeMax`
8. `CavityVolumeMin`
9. `LongAxis`
10. `ShortAxis`
11. `Wall_Thickness`
12. `Sphericity`
13. `Scatt_Mean`
14. `Scatt_STD`

行数范围：

- 0701: 119 到 485
- 0703: 93 到 387

### 3.2 汇总主表格式

`FXN_2023_Analysis.xlsx`：

- 形状: 64 行 x 44 列
- `Name` 唯一值: 64（无重复）
- 时间点: `0701` 与 `0703` 各 32 行

该文件是后续 PCA/ATP 分析的主输入。

### 3.3 已有 PCA 结果格式

`FXN_2023_PCA全部.xlsx`：

- 形状: 32 行 x 17 列
- 含列: `Name`, `Result`, `Name.1`, `Result1`, `得分差值`, `ATP`, `ATP_郭靖修改` 等
- 这是“配对后展示”格式（不是 64 行长表）

`FXN_2023_PCA_Result.xlsx`：

- 形状: 64 行 x 9 列
- 列含 `Name`, `PC1-4`, `Result`, `ATP`, `差值相关`, `终点相关`
- 适合按 0701/0703 重算差值相关

## 4. 与旧数据源差异（结论）

旧源 `Data/nnUNet_FXN_2023`：多数脚本默认硬编码此路径。

新源 `Data/FXN_2023_new（ICC）`：

- 已具备完整中间结果（`measure_excel`, `seg_*`, `scatt_mat` 等）
- 已具备汇总结果（`FXN_2023_Analysis.xlsx`、多版 PCA 输出）
- 目录名与时间点命名从 `FXN_0701/0703` 变为 `FXN_20230701/20230703`

因此切换重点是“路径与批次名适配”，而不是重建全部上游处理。

## 5. 新数据源执行步骤（推荐）

以下命令均在仓库根目录执行：`d:/Desktop/music/measure`

### Step 0: 先做完整性检查（必做）

```powershell
python -c "from pathlib import Path; import pandas as pd; b=Path(r'Data/FXN_2023_new（ICC）');
f1=list((b/'FXN_20230701'/'measure_excel').glob('*.xlsx')); f2=list((b/'FXN_20230703'/'measure_excel').glob('*.xlsx'));
print('0701=',len(f1),'0703=',len(f2));
w1={p.stem.split('_')[0] for p in f1}; w2={p.stem.split('_')[0] for p in f2};
print('paired=',len(w1&w2),'only_0701=',sorted(w1-w2),'only_0703=',sorted(w2-w1));
ana=pd.read_excel(b/'FXN_2023_Analysis.xlsx');
print('analysis shape=',ana.shape,'name_dup=',int(ana['Name'].duplicated().sum()))"
```

期望输出：

- 0701=32, 0703=32
- paired=32, only 列表均为空
- `analysis shape=(64, 44)` 且 `name_dup=0`

### Step 1: 聚类合并（若需重跑 cluster_merge）

`cluster-merge.py` 支持命令行覆盖 roots，可直接切到 ICC。

```powershell
python src/04_clustering/cluster-merge.py --model model/Kmeans-scatt.pickle --method kmeans --roots "Data/FXN_2023_new（ICC）/FXN_20230701" "Data/FXN_2023_new（ICC）/FXN_20230703"
```

输出目录：

- `Data/FXN_2023_new（ICC）/FXN_20230701/cluster_merge/*.xlsx`
- `Data/FXN_2023_new（ICC）/FXN_20230703/cluster_merge/*.xlsx`

### Step 2: 生成孔级统计 Sheet2（若需重跑）

注意：

- `src/04_clustering/cluster_analysis_1.py`
- `src/04_clustering/cluster_analysis_2.py`

当前默认硬编码的是旧路径 `Data/nnUNet_FXN_2023/...`。

两种方式：

1. 临时改脚本常量后运行。
2. 保持脚本不改，使用你已有的 `Data/FXN_2023_new（ICC）/FXN_2023_Analysis.xlsx`（推荐）。

如果直接复用现成主表，可跳过 Step 2。

### Step 3: PCA 与 ATP 相关分析（ICC）

使用已适配 ICC 路径的脚本：

```powershell
python src/07_analysis/PCA-ICC.py
```

默认读取：

- `Data/FXN_2023_new（ICC）/FXN_2023_Analysis.xlsx`

默认输出：

- `Data/FXN_2023_new（ICC）/FXN_PCA_Scores_lunwen.xlsx`

### Step 4: 三类相关性复核（起点/终点/差值）

```powershell
python src/07_analysis/pca_atp_three_way_correlation.py --file "Data/FXN_2023_new（ICC）/FXN_2023_PCA全部.xlsx"
```

### Step 5: 双版本结果对比（可选）

```powershell
python src/07_analysis/pca_atp_batch_compare.py
```

默认对比：

- `FXN_2023_PCA去散射系数.xlsx`
- `FXN_2023_PCA全部.xlsx`

### Step 6: 出图（可选）

```powershell
python src/07_analysis/plot_pca_atp_scatter.py --file "Data/FXN_2023_new（ICC）/FXN_2023_PCA全部.xlsx"
```

输出目录：

- `reports/figures/pca_atp_scatter/`

## 6. 需要你特别注意的坑

1. `0701/0703` 与 `20230701/20230703` 命名差异。
2. 多脚本硬编码旧路径，切源时优先用命令行参数覆盖；无参数脚本需要改常量。
3. `FXN_2023_PCA全部.xlsx` 是 32 行配对展示，不是 64 行长表。
4. B2 在新源里仍存在，若论文版本要去 B2，需在下游统一过滤规则。

## 7. 快速切换检查清单

- [ ] 根目录执行命令
- [ ] 0701/0703 各 32 个 `measure_excel`
- [ ] 32 对孔位配对完整
- [ ] `FXN_2023_Analysis.xlsx` 为 64 行
- [ ] 使用 ICC 路径脚本或参数
- [ ] 明确是否保留 B2

## 8. 本次结论摘要

新数据源 `Data/FXN_2023_new（ICC）` 结构完整、格式统一、可直接进入 PCA/ATP 分析。
你当前最省事路径是：

1. 复用已存在的 `FXN_2023_Analysis.xlsx`。
2. 直接运行 `src/07_analysis/PCA-ICC.py`。
3. 用 `pca_atp_three_way_correlation.py` 和散点图脚本做结果复核与展示。
