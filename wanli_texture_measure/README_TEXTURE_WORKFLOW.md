# OCT 类器官三维纹理特征提取流程

## 1. 你的问题怎么做

是的，通常是 **一个 `imagesTr` 文件 + 一个 `predictions_imageTr` 掩膜文件配对提取一次**，然后每个类器官输出 **一行特征**。

也就是说：
- 1 个类器官体数据 → 1 行特征表
- 100 个类器官 → 100 行特征表
- 最终得到一个 `CSV`，后续所有统计分析都基于这张表

你不需要手工一个一个点；脚本会自动遍历目录并匹配文件名。

## 2. 建议的输入目录

```text
nnUNet_data/nnUNet_raw/Dataset507_organoid/imagesTr
nnUNet_results/Dataset507_organoid/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres/fold_0/predictions_imageTr
```

通常要求：
- `imagesTr` 中是图像文件，常见为 `xxx_0000.nii.gz`
- `predictions_imageTr` 中是对应的预测掩膜，常见为 `xxx.nii.gz`
- 脚本会自动去掉 ` _0000` 后缀再配对

## 3. 直接运行批处理

```bash
cd /mnt/cache_ssd/wangwanli/mnt_new/sambashare/Exersice/nnUNet/OCT_texture
conda run --live-stream --name transunet_py311 python batch_extract_oct_texture.py \
  --images-dir /mnt/cache_ssd/wangwanli/mnt_new/sambashare/Exersice/nnUNet/nnUNet_data/nnUNet_raw/Dataset507_organoid/imagesTr \
  --masks-dir /mnt/cache_ssd/wangwanli/mnt_new/sambashare/Exersice/nnUNet/nnUNet_results/Dataset507_organoid/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres/fold_0/predictions_imageTr \
  --output-csv /mnt/cache_ssd/wangwanli/mnt_new/sambashare/Exersice/nnUNet/OCT_texture_outputs/texture_features.csv \
  --output-summary-csv /mnt/cache_ssd/wangwanli/mnt_new/sambashare/Exersice/nnUNet/OCT_texture_outputs/texture_features_summary.csv \
  --output-heatmap /mnt/cache_ssd/wangwanli/mnt_new/sambashare/Exersice/nnUNet/OCT_texture_outputs/texture_heatmap.png
```

## 4. 输出文件怎么用

建议至少保存三类文件：

- `texture_features.csv`
  - 每行一个类器官
  - 每列一个纹理特征
  - 这是后续建模、统计分析的主文件

- `texture_features_summary.csv`
  - 整个数据集的均值、标准差、最小值、最大值、中位数、缺失率
  - 用于快速理解整体分布

- `texture_heatmap.png`
  - 直观看各类器官的特征分布
  - 看哪些类器官纹理更强、更异质

## 5. 如何分析全部数据

推荐按下面顺序做：

1. **基础统计**
   - 先看每个特征的均值、标准差、分位数
   - 判断有没有异常值、极端值、空值

2. **标准化**
   - 建模前对特征做 `z-score`
   - 避免不同量纲影响模型

3. **相关性分析**
   - 看特征之间是否高度相关
   - 去掉冗余特征，减少共线性

4. **降维**
   - 用 `PCA` 看样本在低维空间的分布
   - 如果有药物组别/处理组，可以观察分离趋势

5. **分组比较**
   - 例如对照组 vs 处理组
   - 计算 `t-test`、`Mann-Whitney U`、效应量

6. **聚类/可视化**
   - `k-means`、层次聚类、UMAP/t-SNE
   - 看是否存在纹理亚群

7. **和表型/疗效关联**
   - 用特征去预测活性、体积变化、存活率等
   - 做回归或分类模型

## 6. 代码层面的推荐调用

### 目录批量提取

```python
from oct_texture_feature_extraction import extract_texture_features_from_nifti_directories

feature_df = extract_texture_features_from_nifti_directories(
    images_dir=".../imagesTr",
    masks_dir=".../predictions_imageTr",
    output_csv=".../texture_features.csv",
    output_summary_csv=".../texture_features_summary.csv",
    output_heatmap=".../texture_heatmap.png",
)
```

### 读取后做总体分析

```python
import pandas as pd
from oct_texture_feature_extraction import summarize_feature_dataframe, plot_feature_correlation_heatmap

df = pd.read_csv("texture_features.csv")
summary = summarize_feature_dataframe(df)
summary.to_csv("texture_features_summary.csv", index=False)
plot_feature_correlation_heatmap(df, save_path="texture_corr.png", show=False)
```

## 7. 参数建议

- `levels=32`：与你的需求一致，通常够用
- `use_bbox_crop=True`：大 3D 数据强烈建议开启
- `include_wavelet=False`：如果先想快速跑通，可以先关掉小波
- `include_glrlm=True`：若数据量大、速度慢，可先关掉只做 GLCM

## 8. 一个关键建议

如果你的原始 `imagesTr` 非常大，建议不要一次性全部放进内存。
更稳妥的方式是：
- 按文件逐个读入
- 提取一例，保存一例
- 最后统一合并成一个 CSV

如果你愿意，我可以下一步直接给你补一个：
- **逐文件流式读取、逐样本保存结果的版本**
- 以及一个 **PCA + 聚类 + 相关性分析** 的后处理脚本
