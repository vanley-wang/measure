基于OCT纹理分析的类器官异质性量化

命令行
 conda run --live-stream --name transunet_py311 python batch_extract_oct_texture.py   --images-dir /mnt/cache_ssd/wangwanli/mnt_new/sambashare/Exersice/nnUNet/nnUNet_data/nnUNet_raw/Dataset507_organoid/imagesTr   --masks-dir /mnt/cache_ssd/wangwanli/mnt_new/sambashare/Exersice/nnUNet/nnUNet_results/Dataset507_organoid/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres/fold_0/predictions_imageTr   --output-csv /mnt/cache_ssd/wangwanli/mnt_new/sambashare/Exersice/nnUNet/OCT_texture_outputs/texture_features.csv   --output-summary-csv /mnt/cache_ssd/wangwanli/mnt_new/sambashare/Exersice/nnUNet/OCT_texture_outputs/texture_features_summary.csv   --output-heatmap /mnt/cache_ssd/wangwanli/mnt_new/sambashare/Exersice/nnUNet/OCT_texture_outputs/texture_heatmap.png

- 除了OAC均值和标准差，还可以提取**灰度共生矩阵、小波变换**等纹理特征，描述类器官内部结构的空间异质性。
- **创新点**：证明纹理特征比OAC均值能更早预测耐药性。

第3章 类器官三维纹理特征提取与筛选
3.1 纹理特征的数学定义与计算方法
3.1.1 基于灰度共生矩阵（GLCM）的特征

对比度（contrast）、相关性（correlation）、能量（energy）、同质性（homogeneity）、熵（entropy）

三维实现：在XY、XZ、YZ三个正交方向分别计算后取均值

3.1.2 基于灰度游程矩阵（GLRLM）的特征

短游程强调、长游程强调、游程不均匀度等

3.1.3 基于小波变换的多尺度纹理

使用Daubechies小波对ROI进行三层分解，提取各子带能量和标准差

3.1.4 其他补充特征（可选）

局部二值模式（LBP）、Gabor滤波响应

3.2 纹理特征计算流程
输入：单个类器官的三维分割掩膜内的OCT强度体数据（已归一化）

计算区域：整个实体区域（不包括空腔）

输出：每个类器官的一个纹理特征向量（例如20~30维）

3.3 特征可重复性与稳定性评估
对同一类器官在不同扫描时刻（无药物变化时）计算纹理特征变异系数

筛选出高稳定性（低变异）且对生物变化敏感的特征

3.4 典型类器官的纹理可视化
选取囊状、实心、受损等典型类器官，展示其GLCM热图、小波子带图像

初步观察药物作用前后的纹理变化趋势

第4章 药物作用下纹理特征的动态演变规律
4.1 群体水平的纹理参数统计
对照组 vs 各浓度组在加药前后纹理特征的均值、标准差变化（柱状图/箱线图）

筛选出变化最显著的纹理参数（例如熵显著上升、能量显著下降）

4.2 单个类器官的纹理演变追踪
选取3~5个典型类器官个体，展示其在Day3和Day5的纹理参数变化

结合三维渲染图，定性与定量结合

4.3 纹理特征与常规指标的灵敏度对比
计算每个指标的相对响应比（参照原文3.4节公式）

比较：体积变化率、平均OAC变化率、熵变化率、对比度变化率

验证纹理特征是否在低浓度/早期表现出更显著变化

4.4 纹理参数与OAC的相关性分析
绘制散点图矩阵，计算Pearson相关系数


判断纹理是否提供了与OAC互补的信息（相关性中等或较低则为互补）