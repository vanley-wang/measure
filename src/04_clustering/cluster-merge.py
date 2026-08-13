import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', '..')))

import pandas as pd
import os
import glob
import argparse

from cluster_utils import (
    load_model_package,
    rule_classify_df,
    PHENOTYPE_NAMES,
)

# ================= 配置 =================
DEFAULT_MODEL_PATH = 'model/Kmeans-scatt.pickle'

ROOT_FOLDERS = [
    'Data/nnUNet_FXN_2023/FXN_0701',
    'Data/nnUNet_FXN_2023/FXN_0703'
]

OUTPUT_FOLDER_NAME = 'cluster_merge'

# ============================================================================

def process_file(file_path: str, model_pkg: dict, method: str) -> pd.DataFrame:
    """
    Apply clustering model to a single well's measurement file.

    Args:
        file_path: path to measure_excel/*.xlsx
        model_pkg: loaded model package dict
        method: 'kmeans' | 'gmm' | 'rule'

    Returns:
        DataFrame with added 'Cluster' and 'Phenotype_Desc' columns
    """
    df = pd.read_excel(file_path)
    features = model_pkg['feature_names']

    if not all(col in df.columns for col in features):
        missing = [c for c in features if c not in df.columns]
        raise ValueError(f"Missing feature columns: {missing}")

    if method == 'rule':
        # Rule-based: per-well adaptive thresholds, no scaler needed
        labels = rule_classify_df(df[features])
        df['Cluster'] = labels.values

    elif method in ('kmeans', 'gmm'):
        # Model-based: preprocess + predict
        preprocessor = model_pkg.get('preprocessor')
        scaler = model_pkg.get('scaler')
        model = model_pkg['model']
        raw_to_final = model_pkg.get('raw_to_final', {})

        # If new unified preprocessor exists, use it; else fallback to legacy scaler
        if preprocessor is not None:
            X = preprocessor.transform(df[features])
        elif scaler is not None:
            X = scaler.transform(df[features])
        else:
            raise ValueError("Model package missing both preprocessor and scaler")

        raw_labels = model.predict(X)

        # Map raw cluster IDs to phenotype labels
        if raw_to_final:
            final_labels = [raw_to_final.get(l, l) for l in raw_labels]
        else:
            final_labels = raw_labels  # identity mapping

        df['Cluster'] = final_labels

        # For GMM, also save probability for QC
        if method == 'gmm' and hasattr(model, 'predict_proba'):
            probs = model.predict_proba(X)
            max_prob = probs.max(axis=1)
            df['Cluster_Prob'] = max_prob.round(4)

    else:
        raise ValueError(f"Unknown method: {method}")

    df['Phenotype_Desc'] = [PHENOTYPE_NAMES.get(int(c), 'Unknown') for c in df['Cluster']]
    return df


def main():
    parser = argparse.ArgumentParser(description='Apply clustering model to all wells')
    parser.add_argument('--model', default=DEFAULT_MODEL_PATH, help='Path to model pickle')
    parser.add_argument('--method', choices=['kmeans', 'gmm', 'rule'], default='kmeans',
                        help='Clustering method to use')
    parser.add_argument('--roots', nargs='+', default=ROOT_FOLDERS,
                        help='Root data folders to process')
    args = parser.parse_args()

    print(f">>> Loading model: {args.model} (method={args.method})")
    model_pkg = load_model_package(args.model)
    print(f"    model_type: {model_pkg.get('model_type', 'unknown')}")

    for root in args.roots:
        input_dir = os.path.join(root, 'measure_excel')
        output_dir = os.path.join(root, OUTPUT_FOLDER_NAME)

        if not os.path.exists(input_dir):
            print(f"[WARN] Skipping {root}: measure_excel not found")
            continue

        os.makedirs(output_dir, exist_ok=True)
        files = sorted(glob.glob(os.path.join(input_dir, '*.xlsx')))
        print(f"\n>>> Processing {os.path.basename(root)}: {len(files)} files")

        for fp in files:
            try:
                df = process_file(fp, model_pkg, args.method)

                # Save
                fname = os.path.basename(fp).replace('.xlsx', '_merge.xlsx')
                out_path = os.path.join(output_dir, fname)
                df.to_excel(out_path, index=False)

            except Exception as e:
                print(f"  [ERR] {os.path.basename(fp)}: {e}")

        print(f"  -> Saved to: {output_dir}")

    print("\n[Done] All wells processed.")


if __name__ == "__main__":
    main()
