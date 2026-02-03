"""plot_sdoh_umaps.py

Create UMAP visualizations of SDoH features from the merged
`diagnosis_with_acxiom.csv` file. The script:

- Detects SDoH columns (prefer columns after a `flag` column when present),
- Drops rows with > `row_nan_frac_max` missing SDoH values,
- Uses KNN imputation for remaining missing values,
- Scales features, runs UMAP, and
- Produces UMAP scatter plots colored by the presence/absence of the
  top diagnosis codes (one diagnosis per plot). Saves `n_umaps` PNG files.

Usage:
    python plot_sdoh_umaps.py --input diagnosis_with_acxiom.csv --outdir umap_plots

"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from umap import UMAP


def detect_sdoh_columns(df: pd.DataFrame) -> list:
    cols = list(df.columns)
    if "flag" in cols:
        flag_idx = cols.index("flag")
        sdoh = cols[flag_idx + 1 :]
        # exclude dx_ columns if present
        sdoh = [c for c in sdoh if not c.startswith("dx_")]
        return sdoh

    # fallback: choose columns that look numeric (coerceable) and are not obvious metadata
    exclude = {"member_id", "full_name", "address", "city", "state", "memberzipcode", "zip", "zip5", "flag"}
    cand = [c for c in cols if c not in exclude and not c.startswith("dx_")]
    numeric_cols = []
    for c in cand:
        # sample a few non-null values and test numeric coercion
        sample = df[c].dropna().astype(str).head(100)
        if sample.empty:
            continue
        # if more than half the non-null sample are numeric-like, keep
        num_like = sample.str.replace(r"[^0-9.+-eE]", "", regex=True).str.len() > 0
        if num_like.sum() >= max(1, int(len(sample) * 0.5)):
            numeric_cols.append(c)
    return numeric_cols


def build_umap(X_scaled: np.ndarray, n_neighbors=30, min_dist=0.1, metric="cosine", random_state=42):
    umap = UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist, metric=metric, random_state=random_state)
    emb = umap.fit_transform(X_scaled)
    return emb


def main(args: argparse.Namespace):
    inp = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Reading merged CSV (this may use memory)...")
    df = pd.read_csv(inp, low_memory=False)

    # detect diagnosis one-hot columns
    dx_cols = [c for c in df.columns if c.startswith("dx_")]
    if not dx_cols:
        print("No diagnosis one-hot columns found (dx_...). Exiting.")
        return

    # convert dx cols to numeric 0/1 if possible
    for c in dx_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    # detect sdoh columns
    sdoh_cols = detect_sdoh_columns(df)
    if not sdoh_cols:
        print("No SDoH columns detected. Exiting.")
        return

    print(f"Detected {len(sdoh_cols)} candidate SDoH columns; applying QC")

    # coerce sdoh columns to numeric where possible
    X = df[sdoh_cols].apply(pd.to_numeric, errors="coerce")

    # drop columns with very high missingness
    col_nan_frac = X.isna().mean()
    keep_cols = col_nan_frac[col_nan_frac <= args.col_nan_frac_max].index.tolist()
    X = X[keep_cols]
    print(f"Kept {len(keep_cols)} SDoH features after col missingness filter")

    # drop rows with too much missingness
    row_nan_frac = X.isna().mean(axis=1)
    keep_rows = row_nan_frac <= args.row_nan_frac_max
    print(f"Keeping {keep_rows.sum():,} / {len(keep_rows):,} rows after row missingness filter")
    X = X.loc[keep_rows].copy()
    df_keep = df.loc[keep_rows].copy()

    if X.shape[0] < 10:
        raise RuntimeError("Too few rows left after QC — loosen row_nan_frac_max")

    # impute using KNN for remaining missing values
    imputer = KNNImputer(n_neighbors=10, weights="distance")
    X_imp = imputer.fit_transform(X)

    # scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)

    # UMAP embedding
    print("Computing UMAP embedding...")
    emb = build_umap(X_scaled, n_neighbors=args.n_neighbors, min_dist=args.min_dist, metric=args.metric, random_state=args.random_state)
    df_keep["umap_x"] = emb[:, 0]
    df_keep["umap_y"] = emb[:, 1]

    # compute counts and separability (silhouette) for each diagnosis in UMAP space
    n_samples = emb.shape[0]
    dx_counts = df_keep[dx_cols].sum().sort_values(ascending=False)
    scores = {}
    for dx in dx_cols:
        cnt = int(df_keep[dx].sum())
        if cnt < args.min_dx_count or cnt == 0 or cnt == n_samples:
            continue
        try:
            s = silhouette_score(emb, df_keep[dx].astype(int).values, metric="euclidean")
            scores[dx] = float(s)
        except Exception:
            continue

    if scores:
        # pick top diagnoses by silhouette score
        top_dx = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)[: args.n_umaps]
        print("Top diagnosis columns by silhouette separability:")
        for k in top_dx:
            print(f"  {k}: count={int(df_keep[k].sum())}, sil={scores.get(k):.3f}")
    else:
        # fallback: choose by frequency
        dx_counts = df_keep[dx_cols].sum().sort_values(ascending=False)
        top_dx = list(dx_counts.head(args.n_umaps).index)
        print(f"No valid separability scores; falling back to top-by-count: {top_dx}")

    # Generate plots
    for i, dx in enumerate(top_dx, start=1):
        outpng = outdir / f"umap_{i:02d}_{dx}.png"
        plt.figure(figsize=(6, 5))
        # background (no-dx) in light gray
        mask_no = df_keep[dx] == 0
        plt.scatter(df_keep.loc[mask_no, "umap_x"], df_keep.loc[mask_no, "umap_y"], c=args.bg_color, s=6, alpha=0.6, label="no")
        # dx present in color
        mask_yes = df_keep[dx] == 1
        plt.scatter(df_keep.loc[mask_yes, "umap_x"], df_keep.loc[mask_yes, "umap_y"], c="C1", s=8, alpha=0.8, label="has_dx")
        plt.title(f"UMAP colored by {dx} (top-{i}) — {dx_counts.get(dx,0)} patients")
        plt.xlabel("UMAP-1"); plt.ylabel("UMAP-2")
        plt.legend(loc="best", markerscale=2)
        plt.tight_layout()
        plt.savefig(outpng, dpi=150)
        plt.close()
        print(f"Saved: {outpng}")

    print("All done — plots saved to", outdir)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Path to diagnosis_with_acxiom.csv")
    p.add_argument("--outdir", default="umap_plots", help="Directory to save umap PNGs")
    p.add_argument("--n_umaps", type=int, default=10, help="Number of diagnosis UMAPs to produce")
    p.add_argument("--min_dx_count", type=int, default=20, help="Minimum patients with dx to consider for separability")
    p.add_argument("--bg_color", default="red", help="Color for points without the diagnosis (default: red)")
    p.add_argument("--row_nan_frac_max", type=float, default=0.40, help="Max fraction missing per row to keep")
    p.add_argument("--col_nan_frac_max", type=float, default=0.95, help="Max fraction missing per column to keep")
    p.add_argument("--n_neighbors", type=int, default=30)
    p.add_argument("--min_dist", type=float, default=0.1)
    p.add_argument("--metric", default="cosine")
    p.add_argument("--random_state", type=int, default=42)
    args = p.parse_args()
    main(args)
