"""
Re-run steps 5-10 of the MedNext_GC032 pipeline with the new KMeans model.

Steps 1-3 (nnunet_bridge, measure, scatt) are unchanged.
Step 4 (cluster_merge) has already been re-run with new model.
Re-run steps 5-10: cluster_analysis_1/2, roughness, PCA, feature_opt.
Also re-generate cluster_png.
"""
import os
import sys

# Add repo root for imports
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)

# Pipeline config
BASE_DIR = "MedNext_GC032"
BATCHES = ["20230701", "20230703"]
MIN_VOLUME = 50

# Import pipeline functions
import importlib.util
spec = importlib.util.spec_from_file_location(
    "pipeline", os.path.join(REPO_ROOT, "run_mednext_gc032_pipeline.py")
)
pipeline = importlib.util.module_from_spec(spec)
# Set globals before exec
pipeline.BASE_DIR = BASE_DIR
pipeline.BATCHES = BATCHES
pipeline.MIN_VOLUME = MIN_VOLUME
spec.loader.exec_module(pipeline)

import pandas as pd
import numpy as np


def main():
    print("=" * 60)
    print("MedNext_GC032 - Re-run pipeline (steps 5-10) with new KMeans")
    print("=" * 60)

    # Step 5: cluster_analysis_1 (孔板级汇总)
    pipeline.step5_cluster_analysis_1()

    # Step 6: cluster_analysis_2 (主表汇总)
    pipeline.step6_cluster_analysis_2()

    # Step 7: roughness calculation
    # Note: roughness depends on cluster identity (Cluster 0 and 1), so must re-run
    pipeline.step7_roughness()

    # Step 8: roughness analysis (汇总到主表)
    pipeline.step8_roughness_analysis()

    # Step 9: PCA
    pipeline.step9_pca()

    # Step 10: Feature optimization
    pipeline.step10_feature_optimization()

    print("\n" + "=" * 60)
    print("All steps completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
