from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .config import PROJECT_ROOT, RANDOM_SEED

# Canonical catch22 feature order, as returned by both supported backends.
FEATURE_NAMES = [
    "DN_HistogramMode_5",
    "DN_HistogramMode_10",
    "CO_f1ecac",
    "CO_FirstMin_ac",
    "CO_HistogramAMI_even_2_5",
    "CO_trev_1_num",
    "MD_hrv_classic_pnn40",
    "SB_BinaryStats_mean_longstretch1",
    "SB_TransitionMatrix_3ac_sumdiagcov",
    "PD_PeriodicityWang_th0_01",
    "CO_Embed2_Dist_tau_d_expfit_meandiff",
    "IN_AutoMutualInfoStats_40_gaussian_fmmi",
    "FC_LocalSimple_mean1_tauresrat",
    "DN_OutlierInclude_p_001_mdrmd",
    "DN_OutlierInclude_n_001_mdrmd",
    "SP_Summaries_welch_rect_area_5_1",
    "SB_BinaryStats_diff_longstretch0",
    "SB_MotifThree_quantile_hh",
    "SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1",
    "SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1",
    "SP_Summaries_welch_rect_centroid",
    "FC_LocalSimple_mean3_stderr",
]

# catch22 features such as CO_FirstMin_ac need enough history to be meaningful.
MIN_SERIES_LENGTH = 24

_BACKEND_ERROR = (
    "No catch22 implementation is available.\n"
    "Install one of the supported backends:\n"
    "  pip install pycatch22   (reference C implementation; needs a prebuilt wheel "
    "or a C compiler)\n"
    "  pip install aeon        (pure-Python/numba implementation; works on Python 3.11+)\n"
    "The analysis is NOT run with substitute features: catch22 values must come from a "
    "real implementation for the results to mean anything."
)


def _extract_with_pycatch22(values: np.ndarray) -> np.ndarray:
    import pycatch22

    result = pycatch22.catch22_all(values.tolist())
    return np.asarray(result["values"], dtype=float)


def _extract_with_aeon(values: np.ndarray) -> np.ndarray:
    from aeon.transformations.collection.feature_based import Catch22

    transformer = Catch22(catch24=False)
    out = transformer.fit_transform(values.reshape(1, -1))
    return np.asarray(out, dtype=float).ravel()


def resolve_backend() -> tuple[str, callable]:
    """Return (backend_name, extractor). Raise if no real catch22 is installed."""
    from importlib.util import find_spec

    if find_spec("pycatch22") is not None:
        return "pycatch22", _extract_with_pycatch22
    if find_spec("aeon") is not None:
        return "aeon", _extract_with_aeon
    raise RuntimeError(_BACKEND_ERROR)


def _is_usable(series: pd.Series) -> tuple[bool, str]:
    clean = series.dropna()
    if len(clean) == 0:
        return False, "series is empty"
    if len(clean) < MIN_SERIES_LENGTH:
        return False, f"only {len(clean)} observations (minimum {MIN_SERIES_LENGTH})"
    if clean.nunique() <= 1:
        return False, "series is constant"
    return True, ""


def extract_catch22_features(series_dict: dict[str, pd.Series]) -> pd.DataFrame:
    """Extract the 22 catch22 features per series (rows = series, columns = features)."""
    backend_name, extract = resolve_backend()
    print(f"[catch22] Backend: {backend_name}")

    rows: list[np.ndarray] = []
    names: list[str] = []
    skipped: list[tuple[str, str]] = []

    for name, series in series_dict.items():
        usable, reason = _is_usable(series)
        if not usable:
            skipped.append((name, reason))
            continue

        values = series.dropna().to_numpy(dtype=float)
        features = extract(values)

        if features.shape[0] != len(FEATURE_NAMES):
            raise RuntimeError(
                f"Backend '{backend_name}' returned {features.shape[0]} features for "
                f"'{name}', expected {len(FEATURE_NAMES)}."
            )

        rows.append(features)
        names.append(name)

    if skipped:
        print("[catch22] Series excluded from the analysis:")
        for name, reason in skipped:
            print(f"  - {name}: {reason}")

    if not rows:
        raise RuntimeError("No series had enough data to compute catch22 features.")

    df_features = pd.DataFrame(rows, columns=FEATURE_NAMES, index=names)
    df_features.index.name = "series_name"

    all_nan = df_features.columns[df_features.isna().all()].tolist()
    if all_nan:
        print(f"[catch22] Dropping features that are NaN for every series: {all_nan}")
        df_features = df_features.drop(columns=all_nan)

    partial_nan = df_features.columns[df_features.isna().any()].tolist()
    if partial_nan:
        print(f"[catch22] Filling isolated NaN with the feature mean: {partial_nan}")
        df_features = df_features.fillna(df_features.mean())

    print(f"[catch22] Feature matrix: {df_features.shape[0]} series x {df_features.shape[1]} features")
    return df_features


def standardize_features(df_features: pd.DataFrame) -> pd.DataFrame:
    """Exercise 2.4: z-score the feature matrix before any comparative analysis."""
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df_features)
    return pd.DataFrame(scaled, columns=df_features.columns, index=df_features.index)


def analyze_catch22(
    df_features: pd.DataFrame,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """Exercise 2.5: PCA, clustering, heatmap, correlations and distance map."""
    output_dir = output_dir or PROJECT_ROOT
    figures_dir = output_dir / "reports" / "figures"
    tables_dir = output_dir / "reports" / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    df_scaled = standardize_features(df_features)
    n_series = len(df_scaled)

    # 2.5.1 PCA
    n_components = min(2, n_series, df_scaled.shape[1])
    pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
    pca_res = pca.fit_transform(df_scaled)

    plt.figure(figsize=(10, 6))
    plt.scatter(pca_res[:, 0], pca_res[:, 1])
    for i, name in enumerate(df_scaled.index):
        plt.annotate(name, (pca_res[i, 0] + 0.1, pca_res[i, 1] + 0.1), fontsize=9)
    plt.title("catch22 PCA - 2 Components")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%})")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / "catch22_pca.png", dpi=140)
    plt.close()

    # 2.8: which features drive the separation between series.
    loadings = pd.DataFrame(
        pca.components_.T,
        index=df_scaled.columns,
        columns=[f"PC{i + 1}" for i in range(n_components)],
    )
    loadings["abs_PC1"] = loadings["PC1"].abs()
    loadings = loadings.sort_values("abs_PC1", ascending=False)
    loadings.to_csv(tables_dir / "catch22_pca_loadings.csv")
    print(f"[saved] {tables_dir / 'catch22_pca_loadings.csv'}")

    # 2.5.2 Clustering (KMeans)
    n_clusters = min(3, n_series)
    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED, n_init=10)
    clusters = kmeans.fit_predict(df_scaled)

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=pca_res[:, 0], y=pca_res[:, 1], hue=clusters, palette="Set1", s=100)
    for i, name in enumerate(df_scaled.index):
        plt.annotate(name, (pca_res[i, 0] + 0.1, pca_res[i, 1] + 0.1), fontsize=9)
    plt.title("catch22 K-Means Clustering on PCs")
    plt.tight_layout()
    plt.savefig(figures_dir / "catch22_kmeans.png", dpi=140)
    plt.close()

    cluster_table = pd.DataFrame(
        {"cluster": clusters, "PC1": pca_res[:, 0], "PC2": pca_res[:, 1]},
        index=df_scaled.index,
    )
    cluster_table.to_csv(tables_dir / "catch22_clusters.csv")
    print(f"[saved] {tables_dir / 'catch22_clusters.csv'}")

    # 2.5.3 Heatmap of standardized features
    plt.figure(figsize=(12, 8))
    sns.heatmap(df_scaled, cmap="coolwarm", center=0)
    plt.title("Heatmap of Standardized catch22 Features")
    plt.tight_layout()
    plt.savefig(figures_dir / "catch22_heatmap_features.png", dpi=140)
    plt.close()

    # 2.5.4 Correlation matrix between features
    plt.figure(figsize=(14, 12))
    sns.heatmap(df_scaled.corr(), cmap="vlag", center=0, annot=False)
    plt.title("Correlation Matrix of catch22 Features")
    plt.tight_layout()
    plt.savefig(figures_dir / "catch22_correlation.png", dpi=140)
    plt.close()

    # 2.5.5 Distance map between series
    dist_matrix = squareform(pdist(df_scaled, metric="euclidean"))
    df_dist = pd.DataFrame(dist_matrix, index=df_scaled.index, columns=df_scaled.index)

    plt.figure(figsize=(10, 8))
    sns.heatmap(df_dist, cmap="YlGnBu", annot=True, fmt=".2f")
    plt.title("Euclidean Distance between Time Series (catch22 space)")
    plt.tight_layout()
    plt.savefig(figures_dir / "catch22_distances.png", dpi=140)
    plt.close()

    df_dist.to_csv(tables_dir / "catch22_distances.csv")
    print(f"[saved] {tables_dir / 'catch22_distances.csv'}")

    print(f"[catch22] Analysis completed. Figures saved to {figures_dir}")
    return df_scaled
