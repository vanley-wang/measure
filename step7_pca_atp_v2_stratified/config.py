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
