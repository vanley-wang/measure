"""Command line entry point for batch OCT texture extraction.

Example
-------
python batch_extract_oct_texture.py \
  --images-dir /path/to/imagesTr \
  --masks-dir /path/to/predictions_imageTr \
  --output-csv /path/to/output/texture_features.csv \
  --output-summary-csv /path/to/output/texture_features_summary.csv \
  --output-heatmap /path/to/output/texture_heatmap.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

from oct_texture_feature_extraction import extract_texture_features_from_nifti_directories


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Batch extract OCT organoid texture features from NIfTI files.")
    parser.add_argument("--images-dir", required=True, help="Directory containing nnUNet image files, e.g. imagesTr.")
    parser.add_argument("--masks-dir", required=True, help="Directory containing predicted mask files.")
    parser.add_argument("--output-csv", required=True, help="Output CSV path for per-organoid features.")
    parser.add_argument("--output-summary-csv", default=None, help="Optional output CSV path for cohort summary statistics.")
    parser.add_argument("--output-heatmap", default=None, help="Optional output image path for feature heatmap.")
    parser.add_argument("--levels", type=int, default=32, help="Gray levels after quantization.")
    parser.add_argument("--no-glcm", action="store_true", help="Disable GLCM features.")
    parser.add_argument("--no-glrlm", action="store_true", help="Disable GLRLM features.")
    parser.add_argument("--no-wavelet", action="store_true", help="Disable wavelet features.")
    return parser.parse_args()


def main() -> None:
    """Run the batch extraction workflow."""

    args = parse_args()
    extract_texture_features_from_nifti_directories(
        images_dir=Path(args.images_dir),
        masks_dir=Path(args.masks_dir),
        output_csv=Path(args.output_csv),
        output_summary_csv=Path(args.output_summary_csv) if args.output_summary_csv else None,
        output_heatmap=Path(args.output_heatmap) if args.output_heatmap else None,
        levels=args.levels,
        include_glcm=not args.no_glcm,
        include_glrlm=not args.no_glrlm,
        include_wavelet=not args.no_wavelet,
    )


if __name__ == "__main__":
    main()
