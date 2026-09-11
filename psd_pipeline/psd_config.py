# ============================================================
# PSD-Pipeline 统一配置文件
# 表型分层时间差特征 (Phenotype-Stratified Delta Features)
# ============================================================

import os

# ================= 项目根目录 =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

# ================= 数据路径 =================
DATA_DIR = os.path.join(PROJECT_DIR, 'Data', 'FXN_2023_new（ICC）')
DAY3_DIR = os.path.join(DATA_DIR, 'FXN_20230701')
DAY5_DIR = os.path.join(DATA_DIR, 'FXN_20230703')
NET_EFFECT_PATH = os.path.join(DATA_DIR, '11', 'Stats_NetEffect.xlsx')

# Day3/Day5 的 measure_excel 目录
DAY3_MEASURE_DIR = os.path.join(DAY3_DIR, 'measure_excel')
DAY5_MEASURE_DIR = os.path.join(DAY5_DIR, 'measure_excel')

# 输出目录
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
MODEL_DIR = os.path.join(BASE_DIR, 'model')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ================= 聚类特征列（11个，与郭经一致） =================
CLUSTER_FEATURES = [
    'Organoids_Volume',
    'Organoids_Volume_Fill',
    'Organoids_Surface',
    'Cavity_Volume',
    'CavityNum',
    'LongAxis',
    'ShortAxis',
    'Wall_Thickness',
    'Sphericity',
    'Scatt_Mean',
    'Scatt_STD'
]

# ================= 聚合统计用的参数列（不含 CavityNum） =================
AGGREGATION_PARAMS = [
    ('Organoids_Volume',        'Volume'),
    ('Organoids_Volume_Fill',   'Volume_Fill'),
    ('Organoids_Surface',       'Surface'),
    ('Cavity_Volume',           'Cavity_Volume'),
    ('LongAxis',                'Long_Axis'),
    ('ShortAxis',               'Short_Axis'),
    ('Wall_Thickness',          'Cyst_Thick'),
    ('Sphericity',              'Sphericity'),
    ('Scatt_Mean',              'Scatt_Mean'),
    ('Scatt_STD',               'Scatt_STD'),
]

# 聚类数（4类表型）
N_CLUSTERS = 3

# 表型中文标签
PHENOTYPE_LABELS = {
    0: '巨大囊泡型',
    1: '中等过渡型',
    2: '小体积基准型',
    3: '高散射实心型'
}

# ================= ATP 数据库（ICC实验） =================
ATP_DATABASE = {
    'B10': 601300, 'B11': 11180000, 'B2': 5391000, 'B3': 6538000,
    'B4': 7103000, 'B5': 1264000, 'B6': 2548000, 'B7': 1579000,
    'B8': 3637000, 'B9': 140300, 'C10': 211800, 'C11': 13930000,
    'C2': 4460000, 'C3': 8336000, 'C4': 6800000, 'C5': 330900,
    'C6': 238900, 'C7': 682100, 'C8': 211300, 'C9': 465900,
    'D11': 11240000, 'E11': 12840000, 'F10': 21910000, 'F11': 14700000,
    'F2': 26980000, 'F3': 14110000, 'F4': 13740000, 'F5': 17250000,
    'F6': 20320000, 'F7': 20000000, 'F8': 17170000, 'F9': 15830000
}

# ================= 药物-孔位-机制映射（请根据实际实验填写） =================
# 格式: {孔位: (药物名称, 机制类别, 详细分组)}
# 机制类别: '对照', '抗代谢单药', '抗代谢联合', '一代铂联合', '三代铂联合'
DRUG_MAPPING = {
    # --- 对照组 ---
    'D11': ('对照', '对照', '空白对照'),
    'E11': ('对照', '对照', '空白对照'),
    'F2':  ('对照', '对照', '空白对照'),
    'F3':  ('对照', '对照', '空白对照'),
    'F4':  ('对照', '对照', '空白对照'),
    'F5':  ('对照', '对照', '空白对照'),
    'F6':  ('对照', '对照', '空白对照'),
    'F7':  ('对照', '对照', '空白对照'),
    'F8':  ('对照', '对照', '空白对照'),
    'F9':  ('对照', '对照', '空白对照'),
    'F10': ('对照', '对照', '空白对照'),
    'F11': ('对照', '对照', '空白对照'),

    # --- 给药组（需根据实际实验修改） ---
    'B2':  ('5-FU', '抗代谢单药', '单药抗代谢组'),
    'B3':  ('5-FU', '抗代谢单药', '单药抗代谢组'),
    'B4':  ('吉西他滨+5FU', '抗代谢联合', '双抗代谢联合组'),
    'B5':  ('吉西他滨+5FU', '抗代谢联合', '双抗代谢联合组'),
    'B6':  ('顺铂+吉西他滨', '一代铂联合', '一代铂联合组'),
    'B7':  ('顺铂+吉西他滨', '一代铂联合', '一代铂联合组'),
    'B8':  ('奥沙利铂+5FU', '三代铂联合', '三代铂联合方案1'),
    'B9':  ('奥沙利铂+5FU', '三代铂联合', '三代铂联合方案1'),
    'B10': ('奥沙利铂+吉西他滨', '三代铂联合', '三代铂联合方案2'),
    'B11': ('奥沙利铂+吉西他滨', '三代铂联合', '三代铂联合方案2'),
    'C2':  ('5-FU', '抗代谢单药', '单药抗代谢组'),
    'C3':  ('吉西他滨+5FU', '抗代谢联合', '双抗代谢联合组'),
    'C4':  ('吉西他滨+5FU', '抗代谢联合', '双抗代谢联合组'),
    'C5':  ('顺铂+吉西他滨', '一代铂联合', '一代铂联合组'),
    'C6':  ('顺铂+吉西他滨', '一代铂联合', '一代铂联合组'),
    'C7':  ('奥沙利铂+5FU', '三代铂联合', '三代铂联合方案1'),
    'C8':  ('奥沙利铂+5FU', '三代铂联合', '三代铂联合方案1'),
    'C9':  ('奥沙利铂+吉西他滨', '三代铂联合', '三代铂联合方案2'),
    'C10': ('奥沙利铂+吉西他滨', '三代铂联合', '三代铂联合方案2'),
    'C11': ('奥沙利铂+吉西他滨', '三代铂联合', '三代铂联合方案2'),
}

# ================= Delta 特征计算参数 =================
MIN_CLUSTER_SAMPLES = 3  # 聚类内最少类器官数，低于此数不计算中位数

# ================= 特征选择参数 =================
NAN_THRESHOLD = 0.5    # NaN 比例超过此值则删除特征
VAR_THRESHOLD = 1e-6   # 方差低于此值则删除特征
CORR_THRESHOLD = 0.95  # |r| 超过此值则删除冗余特征

# ================= PCA 参数 =================
PCA_MAX_COMPONENTS = 8  # 最大主成分数（上限）
PCA_RANDOM_STATE = 42