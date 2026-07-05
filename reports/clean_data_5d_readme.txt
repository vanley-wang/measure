============================================================
5维精简特征清洗数据集说明
============================================================

生成时间: 2026-07-05 02:16:38.147069
原始样本数: 21170
清洗后样本数: 20111
剔除异常值比例: 5.00%

5维特征设计:
  - Organoids_Volume_Fill: 体型大小（log1p变换）
  - Sphericity: 形状球度
  - Scatt_Mean: OAC均值（核心药效指标）
  - Scatt_STD: OAC标准差（内部异质性）
  - Cavity_Ratio: 空腔化程度（0=实心）

辅助列:
  _well: 来源 well 名称
  _conc: 药物浓度 (0=Control, 20, 40, 80)
  _outlier_flag: Isolation Forest 判定异常
  _pc1/_pc2/_pc3: PCA 前 3 主成分坐标

预处理器保存路径: reports\preprocessor_5d_fitted.pickle
