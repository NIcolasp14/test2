"""
umap_clustering.py

Main UMAP and clustering analysis of SDoH features.
Includes sophisticated imputation, dimensionality reduction, and clustering.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from umap import UMAP
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge, LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

from utils import pick_col, SENTINELS
from visualization import (
    plot_umap_categorical, plot_umap_continuous
)


# -----------------------------
# Configuration
# -----------------------------

ROW_NAN_FRAC_MAX = 0.40
COL_NAN_FRAC_MAX = 0.95
UMAP_N_NEIGHBORS = 30
UMAP_MIN_DIST = 0.10
UMAP_METRIC = "cosine"
RANDOM_STATE = 42
IMPUTER_MODE = "iterative"  # "iterative" or "knn"


def run_umap_clustering(sdoh_all, imputer_mode=IMPUTER_MODE, 
                        row_nan_frac_max=ROW_NAN_FRAC_MAX,
                        col_nan_frac_max=COL_NAN_FRAC_MAX,
                        n_clusters=8):
    """
    Run complete UMAP and clustering analysis on SDoH data.
    
    Args:
        sdoh_all: DataFrame with 'flag' column marking start of SDoH features
        imputer_mode: "iterative" for MICE-like or "knn" for KNN imputation
        row_nan_frac_max: Maximum fraction of NaN per row for QC
        col_nan_frac_max: Maximum fraction of NaN per column for QC
        n_clusters: Number of clusters for KMeans
    
    Returns:
        DataFrame with UMAP coordinates, clusters, and metadata
    """
    # 1) Extract SDoH features = all columns AFTER 'flag' (excluding 'flag')
    if "flag" not in sdoh_all.columns:
        raise ValueError("Column 'flag' not found in sdoh_all.")

    flag_idx = sdoh_all.columns.get_loc("flag")
    sdoh_cols = list(sdoh_all.columns[flag_idx + 1:])

    if len(sdoh_cols) == 0:
        raise ValueError("No columns found after 'flag'.")

    X = sdoh_all[sdoh_cols].copy()

    # numeric coercion
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    # sentinel-to-missing (optional)
    X = X.mask(X.isin(SENTINELS))

    # 2) QC: drop columns + rows with lots of NaNs
    col_nan_frac = X.isna().mean()
    keep_cols = col_nan_frac[col_nan_frac <= col_nan_frac_max].index
    X = X[keep_cols]

    row_nan_frac = X.isna().mean(axis=1)
    keep_rows = row_nan_frac <= row_nan_frac_max

    print(f"QC: kept {keep_rows.sum():,} / {len(keep_rows):,} rows "
          f"(dropped {(~keep_rows).sum():,} rows with >{row_nan_frac_max:.0%} NaNs)")
    print(f"QC: kept {X.shape[1]:,} / {len(sdoh_cols):,} SDoH cols "
          f"(dropped {(len(sdoh_cols)-X.shape[1]):,} cols with >{col_nan_frac_max:.0%} NaNs)")

    X = X.loc[keep_rows].copy()
    df_qc = sdoh_all.loc[keep_rows].copy()

    if X.shape[0] < 10:
        raise ValueError(f"Too few rows left after QC ({X.shape[0]}). Loosen ROW_NAN_FRAC_MAX.")

    # 3) Sophisticated imputation + scaling
    X_np = X.to_numpy(dtype=float)

    if imputer_mode == "iterative":
        # MICE-style: model each feature from others
        imputer = IterativeImputer(
            estimator=BayesianRidge(),
            max_iter=10,
            tol=1e-3,
            n_nearest_features=min(50, X.shape[1]),
            imputation_order="random",
            skip_complete=True,
            random_state=RANDOM_STATE
        )
        X_imp = imputer.fit_transform(X_np)
    elif imputer_mode == "knn":
        # KNNImputer: often much faster
        imputer = KNNImputer(n_neighbors=10, weights="distance")
        X_imp = imputer.fit_transform(X_np)
    else:
        raise ValueError("IMPUTER_MODE must be 'iterative' or 'knn'.")

    # Drop zero-variance columns after imputation
    var = X_imp.var(axis=0)
    nonzero = var > 0
    X_imp = X_imp[:, nonzero]
    kept_features = np.array(X.columns)[nonzero]
    print(f"Features used after zero-variance drop: {X_imp.shape[1]:,}")

    scaler = StandardScaler(with_mean=True, with_std=True)
    X_scaled = scaler.fit_transform(X_imp)

    # 4) UMAP per row
    umap = UMAP(
        n_components=2,
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        metric=UMAP_METRIC,
        random_state=RANDOM_STATE,
    )
    emb = umap.fit_transform(X_scaled)

    df_qc["umap_x"] = emb[:, 0]
    df_qc["umap_y"] = emb[:, 1]

    # 5) Geo columns + extra colorings
    zip_col = pick_col(df_qc, ["memberzipcode", "memberzip", "zip", "zipcode", "member_zipcode"])
    city_col = pick_col(df_qc, ["city", "membercity", "member_city"])
    state_col = pick_col(df_qc, ["state", "memberstate", "member_state"])

    if zip_col is not None:
        df_qc["zip_str"] = df_qc[zip_col].astype("string").str.extract(r"(\d{5})", expand=False)
        df_qc["zip3"] = df_qc["zip_str"].str[:3]
    if city_col is not None:
        df_qc["city_str"] = df_qc[city_col].astype("string")
    if state_col is not None:
        df_qc["state_str"] = df_qc[state_col].astype("string")

    # additional useful colorings
    df_qc["sdoh_missing_frac"] = row_nan_frac.loc[keep_rows].values
    df_qc["sdoh_nonzero_count"] = (X.fillna(0).to_numpy() != 0).sum(axis=1)

    # quick clustering
    km = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init="auto")
    df_qc[f"cluster_kmeans{n_clusters}"] = km.fit_predict(X_scaled)

    # Return both df_qc and X_scaled for downstream analysis
    return df_qc, X_scaled, X


def visualize_umap(df_qc):
    """
    Generate standard UMAP visualizations.
    
    Args:
        df_qc: DataFrame with UMAP coordinates and metadata
    """
    # Basic UMAP
    plt.figure()
    plt.scatter(df_qc["umap_x"], df_qc["umap_y"], s=6, alpha=0.6)
    plt.title("UMAP of SDoH features (after sophisticated imputation)")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.tight_layout()
    plt.show()

    # Geographic colorings
    if "zip_str" in df_qc.columns:
        plot_umap_categorical(df_qc, "zip_str", 
                             title="UMAP colored by ZIP (top + Other)", top_n=15)

    if "zip3" in df_qc.columns:
        plot_umap_categorical(df_qc, "zip3", title="UMAP colored by ZIP3", top_n=25)

    if "city_str" in df_qc.columns:
        plot_umap_categorical(df_qc, "city_str", 
                             title="UMAP colored by City (top + Other)", top_n=15)

    if "state_str" in df_qc.columns:
        plot_umap_categorical(df_qc, "state_str", 
                             title="UMAP colored by State", top_n=60)

    # Missingness and burden
    plot_umap_continuous(df_qc, "sdoh_missing_frac", 
                        title="UMAP colored by SDoH missingness (row-wise)")
    plot_umap_continuous(df_qc, "sdoh_nonzero_count", 
                        title="UMAP colored by SDoH non-zero count (proxy burden)")

    # Clusters
    cluster_col = [c for c in df_qc.columns if c.startswith("cluster_kmeans")]
    if cluster_col:
        plot_umap_categorical(df_qc, cluster_col[0], 
                             title="UMAP colored by KMeans clusters", top_n=20)


def analyze_cluster(df_qc, X, cluster_id, cluster_col="cluster_kmeans8"):
    """
    Analyze a specific cluster in detail.
    
    Args:
        df_qc: DataFrame with UMAP coordinates and clusters
        X: Raw SDoH feature matrix
        cluster_id: Cluster ID to analyze
        cluster_col: Name of cluster column
    """
    sub = df_qc[df_qc[cluster_col] == cluster_id].copy()

    # Highlight plot
    plt.figure()
    plt.scatter(df_qc["umap_x"], df_qc["umap_y"], s=4, alpha=0.15)
    plt.scatter(sub["umap_x"], sub["umap_y"], s=10, alpha=0.8)
    plt.title(f"Zoom highlight: {cluster_col} = {cluster_id}")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.tight_layout()
    plt.show()

    # True zoom
    pad = 0.5
    xmin, xmax = sub["umap_x"].min() - pad, sub["umap_x"].max() + pad
    ymin, ymax = sub["umap_y"].min() - pad, sub["umap_y"].max() + pad

    plt.figure()
    plt.scatter(sub["umap_x"], sub["umap_y"], s=10, alpha=0.8)
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.title(f"Zoomed view: cluster {cluster_id}")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.tight_layout()
    plt.show()

    # Feature enrichment
    sub_idx = sub.index
    rest_idx = df_qc.index.difference(sub_idx)

    sub_mean = X.loc[sub_idx].mean(numeric_only=True)
    rest_mean = X.loc[rest_idx].mean(numeric_only=True)

    diff = (sub_mean - rest_mean).sort_values(ascending=False)
    print(f"\nTop features enriched in cluster {cluster_id}:")
    print(diff.head(20))

    print(f"\nTop features depleted in cluster {cluster_id}:")
    print(diff.tail(20))


def run_proxy_classification(df_qc, X_scaled, quantile=0.8):
    """
    Run proxy classification task: predict high burden patients.
    
    Args:
        df_qc: DataFrame with metadata
        X_scaled: Scaled feature matrix
        quantile: Quantile threshold for defining high burden
    
    Returns:
        Dictionary with classification metrics
    """
    # define proxy label: top quantile burden = 1
    y = (df_qc["sdoh_nonzero_count"] >= df_qc["sdoh_nonzero_count"].quantile(quantile)).astype(int).to_numpy()

    clf = LogisticRegression(max_iter=2000, n_jobs=-1, class_weight="balanced")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs, prs, f1s = [], [], []

    for tr, te in skf.split(X_scaled, y):
        clf.fit(X_scaled[tr], y[tr])
        p = clf.predict_proba(X_scaled[te])[:, 1]
        yhat = (p >= 0.5).astype(int)

        aucs.append(roc_auc_score(y[te], p))
        prs.append(average_precision_score(y[te], p))
        f1s.append(f1_score(y[te], yhat))

    print(f"Proxy task (high burden, top {int((1-quantile)*100)}%):")
    print(f"ROC-AUC: {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")
    print(f"PR-AUC : {np.mean(prs):.3f} ± {np.std(prs):.3f}")
    print(f"F1     : {np.mean(f1s):.3f} ± {np.std(f1s):.3f}")

    return {
        "roc_auc": aucs,
        "pr_auc": prs,
        "f1": f1s
    }









