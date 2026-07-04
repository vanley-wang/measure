============================================================
清洗数据集说明
============================================================

生成时间: 2026-07-04 23:07:09.102473
原始样本数: 21170
清洗后样本数: 20111
剔除异常值比例: 5.00%

列说明:
  _well: 来源 well 名称
  _conc: 药物浓度 (0=Control, 20, 40, 80)
  _outlier_flag: Isolation Forest 判定异常 (True=异常)
  _pc1/_pc2/_pc3: PCA 前 3 主成分坐标
  其余列: 原始测量特征

特征列表 (预处理后):
  - Organoids_Volume_Fill
  - Organoids_Surface
  - LongAxis
  - ShortAxis
  - Cavity_Volume
  - Sphericity
  - Scatt_Mean
  - Scatt_STD
  - CavityNum
  - Cavity_Ratio

预处理器保存路径: reports\preprocessor_fitted.pickle
