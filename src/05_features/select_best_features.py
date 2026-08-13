"""
Greedy feature selection: add one new feature at a time,
keep only if it improves Calinski-Harabasz.
"""
import os
import glob
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import calinski_harabasz_score

# ================= 配置 =================
DATA_FOLDERS = [
    'Data/nnUNet_FXN_2023/FXN_0701/measure_excel',
    'Data/nnUNet_FXN_2023/FXN_0703/measure_excel',
]

BASE_5D = ['Organoids_Volume_Fill', 'Sphericity', 'Scatt_Mean', 'Scatt_STD', 'Cavity_Volume']
NEW_FEATURES = ['Scatt_Skewness', 'Scatt_Kurtosis', 'Scatt_IQR', 'Scatt_Range', 'Scatt_10pct', 'Scatt_90pct', 'Compactness']

# ==========================================


def load_all_data():
    df_list = []
    for folder in DATA_FOLDERS:
        if not os.path.exists(folder):
            continue
        for fp in glob.glob(os.path.join(folder, '*.xlsx')):
            df = pd.read_excel(fp)
            df_list.append(df)
    return pd.concat(df_list, ignore_index=True)


def preprocess_and_cluster(df, features):
    """Preprocess and train KMeans, return Calinski-Harabasz."""
    df_proc = df[features].copy()
    if 'Organoids_Volume_Fill' in df_proc.columns:
        df_proc['Organoids_Volume_Fill'] = np.log1p(df_proc['Organoids_Volume_Fill'])
    if 'Cavity_Volume' in df_proc.columns and 'Organoids_Volume_Fill' in df_proc.columns:
        df_proc['Cavity_Ratio'] = df_proc['Cavity_Volume'] / df_proc['Organoids_Volume_Fill'].clip(lower=1)
        df_proc.drop(columns=['Cavity_Volume'], inplace=True)

    scaler = StandardScaler()
    X = scaler.fit_transform(df_proc.fillna(0))

    km = KMeans(n_clusters=4, init='k-means++', random_state=42, n_init=10)
    labels = km.fit_predict(X)
    return calinski_harabasz_score(X, labels)


def main():
    print("=" * 60)
    print("Greedy Feature Selection")
    print("=" * 60)

    df = load_all_data()
    print(f"Total organoids: {len(df)}")

    # Baseline
    baseline_ch = preprocess_and_cluster(df, BASE_5D)
    print(f"\n5D Baseline Calinski-Harabasz: {baseline_ch:.1f}")

    # Test each new feature individually
    print("\n--- Individual Feature Gain ---")
    gains = {}
    for feat in NEW_FEATURES:
        trial_features = BASE_5D + [feat]
        ch = preprocess_and_cluster(df, trial_features)
        gain = ch - baseline_ch
        gains[feat] = gain
        status = "KEEP" if gain > 0 else "DROP"
        print(f"  +{feat}: CH={ch:.1f}, gain={gain:+.1f} [{status}]")

    # Greedy forward selection
    print("\n--- Greedy Forward Selection ---")
    selected = BASE_5D.copy()
    current_ch = baseline_ch

    while True:
        best_gain = 0
        best_feat = None
        for feat, gain in gains.items():
            if feat in selected:
                continue
            trial = selected + [feat]
            ch = preprocess_and_cluster(df, trial)
            gain = ch - current_ch
            if gain > best_gain:
                best_gain = gain
                best_feat = feat

        if best_feat is None:
            break

        selected.append(best_feat)
        current_ch = preprocess_and_cluster(df, selected)
        print(f"  Add {best_feat}: CH={current_ch:.1f} (+{best_gain:.1f})")

    print(f"\nFinal selected ({len(selected)} features): {selected}")
    print(f"Final Calinski-Harabasz: {current_ch:.1f} (baseline: {baseline_ch:.1f})")

    # Save
    pd.DataFrame({'Selected_Features': selected}).to_excel(
        'reports/selected_features_greedy.xlsx', index=False)
    print("\n[Done]")


if __name__ == "__main__":
    main()
