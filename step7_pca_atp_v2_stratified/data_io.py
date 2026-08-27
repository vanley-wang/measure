import glob
import os

import pandas as pd

from cluster_utils import PROCESSED_FEATURES
from .config import ATP_FILE_OPTIONS, BASE_DIR, DATA_FOLDERS


def validate_data_structure():
    print('=' * 70)
    print('Step 0: ICC Data Structure Validation')
    print('=' * 70)

    if not os.path.exists(BASE_DIR):
        print(f'\nERROR: Base directory not found:\n   {BASE_DIR}')
        return False

    print(f'\nOK Base directory exists: {BASE_DIR}')

    for day, folder in DATA_FOLDERS.items():
        if os.path.exists(folder):
            files = glob.glob(os.path.join(folder, '*.xlsx'))
            print(f'OK Day {day}: {len(files)} files in {folder}')
        else:
            print(f'ERROR Day {day} NOT FOUND: {folder}')

    found_atp = False
    for atp_file in ATP_FILE_OPTIONS:
        if os.path.exists(atp_file):
            df = pd.read_excel(atp_file)
            print(f'OK ATP file: {os.path.basename(atp_file)} ({df.shape[0]} rows)')
            found_atp = True

    if not found_atp:
        print('WARN No ATP file found!')

    return True


def load_organoid_data():
    print('\n' + '=' * 70)
    print('Step 1: Load Organoid Features')
    print('=' * 70)

    all_dfs = []
    for day, folder in DATA_FOLDERS.items():
        if not os.path.exists(folder):
            continue

        for fp in sorted(glob.glob(os.path.join(folder, '*.xlsx'))):
            df = pd.read_excel(fp)
            well_name = os.path.splitext(os.path.basename(fp))[0]
            df['_well'] = well_name
            df['_day'] = day
            df['_well_id'] = well_name.split('_')[0] if '_' in well_name else well_name
            all_dfs.append(df)

    if not all_dfs:
        raise RuntimeError('No data loaded!')

    df_all = pd.concat(all_dfs, ignore_index=True)
    feats = [f for f in PROCESSED_FEATURES if f in df_all.columns]

    before = len(df_all)
    df_all = df_all.dropna(subset=feats)
    removed = before - len(df_all)
    print(f'Loaded {len(df_all):,} organoids ({removed:,} NaN removed)')
    print(f'Features: {len(feats)}/{len(PROCESSED_FEATURES)} available')

    d3 = set(df_all[df_all['_day'] == '0701']['_well_id'].unique())
    d5 = set(df_all[df_all['_day'] == '0703']['_well_id'].unique())
    common = sorted(d3 & d5)
    print(f'Well pairing: Day3={len(d3)}, Day5={len(d5)}, Common={len(common)}')

    return df_all, feats, common


def load_atp_table():
    """Load ATP table with robust ATP column detection."""
    for af in ATP_FILE_OPTIONS:
        if not os.path.exists(af):
            continue

        tmp = pd.read_excel(af)
        print(f'\nTrying: {os.path.basename(af)} ({tmp.shape})')

        id_col = next((c for c in ['Name', 'name', 'Well_ID', 'ID'] if c in tmp.columns), None)

        # Prefer explicit ATP-named columns first.
        atp_col = next((c for c in tmp.columns if str(c).strip().lower().startswith('atp')), None)
        if atp_col is None:
            atp_col = next(
                (c for c in tmp.columns if c != id_col and pd.api.types.is_numeric_dtype(tmp[c])),
                None,
            )

        if id_col and atp_col:
            print(f'  ID={id_col}, ATP={atp_col}')
            atp_df = tmp[[id_col, atp_col]].copy()
            atp_df.columns = ['ID', 'ATP']
            atp_df['_key'] = atp_df['ID'].astype(str).str.strip()
            return atp_df

    return None
