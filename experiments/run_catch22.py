from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.catch22_analysis import analyze_catch22, extract_catch22_features
from src.data import build_selected_series, load_clean_data

RAW_FEATURES_PATH = PROJECT_ROOT / "reports" / "tables" / "catch22_features.csv"
SCALED_FEATURES_PATH = PROJECT_ROOT / "reports" / "tables" / "catch22_features_standardized.csv"


def main() -> None:
    print("Loading data...")
    df = load_clean_data()

    print("Building series...")
    series_dict = build_selected_series(df)

    print("Extracting catch22 features...")
    df_features = extract_catch22_features(series_dict)

    RAW_FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_features.to_csv(RAW_FEATURES_PATH)
    print(f"[saved] {RAW_FEATURES_PATH}")

    print("Running catch22 analysis (PCA, clustering, heatmaps, distances)...")
    df_scaled = analyze_catch22(df_features, output_dir=PROJECT_ROOT)

    df_scaled.to_csv(SCALED_FEATURES_PATH)
    print(f"[saved] {SCALED_FEATURES_PATH}")


if __name__ == "__main__":
    main()
