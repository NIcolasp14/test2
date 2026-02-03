"""
diagnosis_analysis.py

UMAP analysis of SDoH features colored by diagnosis codes and geography.
Includes functions for creating diagnosis overlays and geographic visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from umap import UMAP

from visualization import plot_dx_overlay, plot_topN_categories


def create_sdoh_umap(df, sdoh_cols, row_nan_frac_max=0.40, col_nan_frac_max=0.95,
                     sentinels={9, 99, 999, 9999}, n_neighbors=30, min_dist=0.1,
                     metric="cosine", random_state=42):
    """
    Create UMAP embedding from SDoH features.
    
    Args:
        df: DataFrame containing SDoH columns
        sdoh_cols: List of SDoH column names
        row_nan_frac_max: Maximum fraction of NaN per row for QC
        col_nan_frac_max: Maximum fraction of NaN per column for QC
        sentinels: Set of sentinel values to treat as missing
        n_neighbors: UMAP n_neighbors parameter
        min_dist: UMAP min_dist parameter
        metric: UMAP distance metric
        random_state: Random seed
    
    Returns:
        DataFrame with added umap_x and umap_y columns
    """
    # Build SDoH feature matrix (numeric → sentinels→NaN)
    X = df[sdoh_cols].apply(pd.to_numeric, errors="coerce")
    X = X.mask(X.isin(sentinels))

    # Optional QC: drop very-empty rows/cols
    X = X.loc[:, X.isna().mean() <= col_nan_frac_max]
    row_mask = X.isna().mean(axis=1) <= row_nan_frac_max
    X = X.loc[row_mask]
    df_qc = df.loc[row_mask].reset_index(drop=True)

    print(f"After QC: {X.shape[0]:,} rows, {X.shape[1]:,} SDoH columns")

    # Impute (KNN) + Scale
    X_imp = KNNImputer(n_neighbors=10, weights="distance").fit_transform(X)
    X_scaled = StandardScaler().fit_transform(X_imp)

    # UMAP (2-D)
    umap = UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist,
                metric=metric, random_state=random_state)
    emb = umap.fit_transform(X_scaled)
    df_qc["umap_x"], df_qc["umap_y"] = emb[:, 0], emb[:, 1]

    return df_qc


def analyze_diagnosis_umap(df_path, flag_col="flag", top_n_dx=10):
    """
    Load diagnosis+acxiom merged data and create UMAP colored by diagnoses.
    
    Args:
        df_path: Path to merged diagnosis CSV file
        flag_col: Column marking start of SDoH features
        top_n_dx: Number of top diagnosis codes to visualize
    
    Returns:
        DataFrame with UMAP coordinates
    """
    print("Loading data...")
    df = pd.read_csv(df_path, low_memory=False)

    # Identify column groups
    flag_idx = df.columns.get_loc(flag_col)
    sdoh_cols = list(df.columns[flag_idx + 1:])
    dx_cols = [c for c in df.columns if c.startswith("dx_") and c != "dx_other_count"]

    print(f"Found {len(sdoh_cols)} SDoH columns and {len(dx_cols)} diagnosis columns")

    # Create UMAP
    df_qc = create_sdoh_umap(df, sdoh_cols)

    # Pick top diagnosis codes
    dx_prevalence = df_qc[dx_cols].sum().sort_values(ascending=False)
    top_dx = list(dx_prevalence.head(top_n_dx).index)
    print(f"Top-{top_n_dx} diagnosis columns:", top_dx)

    # Plot — one UMAP per Dx code
    plt.style.use("default")
    for dx in top_dx:
        values = pd.to_numeric(df_qc[dx], errors="coerce").fillna(0)

        plt.figure(figsize=(6, 5))
        
        # grey background = patients without that diagnosis
        no_mask = values == 0
        plt.scatter(df_qc.loc[no_mask, "umap_x"],
                    df_qc.loc[no_mask, "umap_y"],
                    c="lightgray", s=6, alpha=0.2, label="Dx = 0")

        # coloured foreground = patients with the diagnosis
        yes_mask = values != 0
        plt.scatter(df_qc.loc[yes_mask, "umap_x"],
                    df_qc.loc[yes_mask, "umap_y"],
                    c="crimson", s=6, alpha=0.8, label="Dx = 1")

        plt.xlabel("UMAP-1")
        plt.ylabel("UMAP-2")
        plt.title(f"UMAP of SDoH – colored by {dx}")
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.show()

    return df_qc


def create_diagnosis_geography_overlays(df_qc, top_n_dx=10, top_n_geo=20):
    """
    Create UMAP overlays for diagnosis codes and geography.
    
    Assumes df_qc already contains:
        - umap_x, umap_y (2-D UMAP coordinates)
        - dx_* columns (binary diagnosis flags)
        - state_str, city_str, zip_str, zip3 (any subset is okay)
    
    Args:
        df_qc: DataFrame with UMAP coordinates and metadata
        top_n_dx: Number of top diagnosis codes to visualize
        top_n_geo: Number of top geographic categories to show
    """
    # Add zip5 alias if missing
    if "zip5" not in df_qc.columns and "zip_str" in df_qc.columns:
        df_qc["zip5"] = df_qc["zip_str"]

    # 1) Ten most-common diagnosis flags
    dx_cols = [c for c in df_qc.columns 
               if c.startswith("dx_") and c != "dx_other_count"]
    top_dx = df_qc[dx_cols].sum().sort_values(ascending=False).head(top_n_dx).index
    print(f"Top-{top_n_dx} Dx columns:", list(top_dx))

    for dx in top_dx:
        plot_dx_overlay(df_qc, dx)

    # 2) Geography overlays (run only if column exists)
    for col in ["state_str", "zip5", "zip3", "city_str"]:
        if col in df_qc.columns:
            plot_topN_categories(df_qc, col, n=top_n_geo)
        else:
            print(f"⧗ Skipping {col}: column not found.")


def load_and_clean_merged_data(df_path, max_nan_frac=0.40):
    """
    Load merged diagnosis+acxiom data and clean rows with too many NaNs.
    
    Args:
        df_path: Path to merged CSV file
        max_nan_frac: Maximum fraction of NaN allowed per row
    
    Returns:
        Cleaned DataFrame
    """
    print("Loading data...")
    df = pd.read_csv(df_path, low_memory=False)

    row_nan_frac = df.isna().mean(axis=1)
    keep_mask = row_nan_frac <= max_nan_frac

    clean_df = df.loc[keep_mask].copy()

    print(f"Kept {keep_mask.sum():,} / {len(keep_mask):,} rows "
          f"(dropped {(~keep_mask).sum():,} rows with > {max_nan_frac:.0%} NaNs)")

    return clean_df









