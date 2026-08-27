import os

BASE_DIR = os.path.join('Data', 'FXN_2023_new（ICC）')

DATA_FOLDERS = {
    '0701': os.path.join(BASE_DIR, 'FXN_20230701', 'measure_excel'),
    '0703': os.path.join(BASE_DIR, 'FXN_20230703', 'measure_excel'),
}

ATP_FILE_OPTIONS = [
    os.path.join(BASE_DIR, 'FXN_2023_PCA全部.xlsx'),
    os.path.join(BASE_DIR, 'FXN_2023_Analysis.xlsx'),
]

MODEL_PATH = 'model/Kmeans-scatt.pickle'
REPORTS_DIR = 'reports'
FIGURES_DIR = os.path.join(REPORTS_DIR, 'figures', 'v2_stratified')

CONC_MAP = {
    'E11': 0, 'F2': 0, 'F3': 0, 'F4': 0, 'F5': 0, 'F6': 0, 'F7': 0, 'F8': 0, 'F9': 0, 'F10': 0, 'F11': 0,
    'B11': 0, 'C11': 0, 'D11': 0,
    'B2': 20, 'B3': 20, 'B4': 20, 'C2': 20, 'C3': 20, 'C4': 20,
    'B5': 40, 'B6': 40, 'B7': 40, 'C5': 40, 'C6': 40, 'C7': 40,
    'B8': 80, 'B9': 80, 'B10': 80, 'C8': 80, 'C9': 80, 'C10': 80,
}

CLUSTER_COLORS = {0: '#E74C3C', 1: '#F1C40F', 2: '#2ECC71', 3: '#3498DB'}
CLUSTER_NAMES = {0: 'Red(H)', 1: 'Yellow(H)', 2: 'Green(I)', 3: 'Blue(D)'}

HEALTHY_CLUSTERS = [0, 1]
MIN_HEALTHY_SAMPLES = 5

ATP_DATABASE = {
    'B10': 601300, 'B11': 11180000, 'B2': 5391000, 'B3': 6538000,
    'B4': 7103000, 'B5': 511900, 'B6': 404500, 'B7': 403900,
    'B8': 312700, 'B9': 140300, 'C10': 211800, 'C11': 13930000,
    'C2': 6336000, 'C3': 8336000, 'C4': 6800000, 'C5': 330900,
    'C6': 238900, 'C7': 682100, 'C8': 211300, 'C9': 465900,
    'D11': 11240000, 'E11': 14700000, 'F10': 21910000, 'F11': 11180000,
    'F2': 18240000, 'F3': 14110000, 'F4': 13740000, 'F5': 17250000,
    'F6': 20320000, 'F7': 20000000, 'F8': 17170000, 'F9': 15830000,
}