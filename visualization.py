"""
visualization.py

Plotting functions for UMAP visualizations with categorical and continuous coloring.
Includes small-dot overlays, facet grids, and hexbin density plots.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns


# -----------------------------
# Basic UMAP Plotting
# -----------------------------

def plot_umap_categorical(df_plot, color_col, title=None, top_n=15, point_size=6, alpha=0.6):
    """
    Plot UMAP with categorical coloring, showing top N categories.
    
    Args:
        df_plot: DataFrame with 'umap_x' and 'umap_y' columns
        color_col: Column name to color by
        title: Plot title (optional)
        top_n: Number of top categories to show distinctly
        point_size: Size of scatter points
        alpha: Transparency of points
    """
    d = df_plot.copy()
    s = d[color_col].astype("string").fillna("Missing")
    vc = s.value_counts(dropna=False)

    if vc.shape[0] > top_n:
        top = set(vc.head(top_n).index)
        s = s.where(s.isin(top), other="Other")

    cats = pd.Categorical(s)
    d["_cat_codes"] = cats.codes

    plt.figure()
    plt.scatter(d["umap_x"], d["umap_y"], c=d["_cat_codes"], s=point_size, alpha=alpha)
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.title(title or f"UMAP colored by {color_col}")
    plt.tight_layout()
    plt.show()


def plot_umap_continuous(df_plot, color_col, title=None, point_size=6, alpha=0.6):
    """
    Plot UMAP with continuous coloring.
    
    Args:
        df_plot: DataFrame with 'umap_x' and 'umap_y' columns
        color_col: Column name to color by
        title: Plot title (optional)
        point_size: Size of scatter points
        alpha: Transparency of points
    """
    plt.figure()
    sc = plt.scatter(df_plot["umap_x"], df_plot["umap_y"], 
                     c=df_plot[color_col], s=point_size, alpha=alpha)
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.title(title or f"UMAP colored by {color_col}")
    plt.colorbar(sc, label=color_col)
    plt.tight_layout()
    plt.show()


# -----------------------------
# Advanced Visualization
# -----------------------------

def _topN(series: pd.Series, n=20):
    """
    Helper: Keep top-n categories; others → 'Other'.
    
    Args:
        series: Series of categorical values
        n: Number of top categories to keep
    
    Returns:
        Series with rare categories replaced by 'Other'
    """
    vc = series.value_counts()
    top = set(vc.head(n).index)
    return series.where(series.isin(top), other="Other").astype("string")


def umap_small_alpha(df, label_col, top_n=20, point_size=4):
    """
    UMAP scatter with small dots + transparency, gray background for 'Other'.
    
    Args:
        df: DataFrame with 'umap_x' and 'umap_y' columns
        label_col: Column name to color by
        top_n: Number of top categories to show
        point_size: Size of scatter points
    """
    cats = _topN(df[label_col], top_n)
    uniq = pd.unique(cats)
    cmap = plt.get_cmap("tab20", len(uniq))

    plt.figure(figsize=(6, 5))
    
    # gray background
    m_other = cats == "Other"
    plt.scatter(df.loc[m_other, "umap_x"],
                df.loc[m_other, "umap_y"],
                c="lightgray", s=point_size, alpha=0.15, label="Other")

    # overlay each top category
    for i, cat in enumerate(uniq):
        if cat == "Other":
            continue
        m = cats == cat
        plt.scatter(df.loc[m, "umap_x"],
                    df.loc[m, "umap_y"],
                    c=[cmap(i)], s=point_size, alpha=0.85, label=cat)

    plt.title(f"UMAP – top {top_n} {label_col} (others gray)")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left",
               markerscale=2, fontsize="small")
    plt.tight_layout()
    plt.show()


def umap_facet(df, label_col, n_facets=6, point_size=6):
    """
    Facet grid (small-multiples) for the top N categories.
    
    Args:
        df: DataFrame with 'umap_x' and 'umap_y' columns
        label_col: Column name to facet by
        n_facets: Number of categories to show
        point_size: Size of scatter points
    """
    top = df[label_col].value_counts().head(n_facets).index
    sub = df[df[label_col].isin(top)].copy()

    g = sns.FacetGrid(sub, col=label_col, col_wrap=3, height=3,
                      sharex=False, sharey=False)
    g.map_dataframe(sns.scatterplot, "umap_x", "umap_y",
                    s=point_size, alpha=.8, color="steelblue")
    
    for ax in g.axes.flatten():
        ax.set_xlabel("")
        ax.set_ylabel("")
    
    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle(f"UMAP – each panel = one of top {n_facets} {label_col}")
    plt.show()


def umap_hexbin_single(df, label_col, label_value, gridsize=60):
    """
    Hexbin / density plot for ONE category of interest.
    
    Args:
        df: DataFrame with 'umap_x' and 'umap_y' columns
        label_col: Column name to filter by
        label_value: Specific value to show density for
        gridsize: Hexbin grid size
    """
    m = df[label_col] == label_value
    if m.sum() == 0:
        print(f"No rows where {label_col} == {label_value!r}")
        return

    plt.figure(figsize=(6, 5))
    hb = plt.hexbin(df.loc[m, "umap_x"], df.loc[m, "umap_y"],
                    gridsize=gridsize, cmap="viridis", mincnt=3)
    plt.colorbar(hb, label="count")
    plt.title(f"{label_value} density in UMAP space")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.tight_layout()
    plt.show()


def plot_top20(df, col, title):
    """
    Plot UMAP with top 20 categories colored, others in gray.
    
    Args:
        df: DataFrame with 'umap_x' and 'umap_y' columns
        col: Column name to color by
        title: Plot title
    """
    vc = df[col].value_counts()
    top = set(vc.head(20).index)

    cats = df[col].where(df[col].isin(top), other="Other").astype("string").fillna("Missing")
    codes = pd.Categorical(cats).codes
    cmap = plt.get_cmap("tab20", len(pd.unique(cats)))

    plt.figure(figsize=(6, 5))
    
    # plot Other first in gray
    plt.scatter(df.loc[cats == "Other", "umap_x"],
                df.loc[cats == "Other", "umap_y"],
                c="lightgray", s=6, alpha=0.4, label="Other")

    for i, cat in enumerate(pd.unique(cats)):
        if cat == "Other":
            continue
        plt.scatter(df.loc[cats == cat, "umap_x"],
                    df.loc[cats == cat, "umap_y"],
                    c=[cmap(i)], s=6, alpha=0.8, label=cat)

    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", 
               markerscale=2, fontsize="small")
    plt.tight_layout()
    plt.show()


# -----------------------------
# Diagnosis-Specific Overlays
# -----------------------------

def plot_dx_overlay(df: pd.DataFrame, dx_col: str,
                    yes_colour="#003366",    # navy
                    no_colour="#8c8c8c",     # mid-gray
                    pt_size=6, alpha_yes=.85, alpha_no=.35):
    """
    Two-tone overlay for diagnosis codes: navy = Dx present, gray = Dx absent.
    
    Args:
        df: DataFrame with 'umap_x', 'umap_y', and diagnosis columns
        dx_col: Diagnosis column name
        yes_colour: Color for patients with diagnosis
        no_colour: Color for patients without diagnosis
        pt_size: Point size
        alpha_yes: Alpha for positive cases
        alpha_no: Alpha for negative cases
    """
    yes = df[dx_col].astype(float).fillna(0) != 0

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(df.loc[~yes, "umap_x"], df.loc[~yes, "umap_y"],
               c=no_colour, s=pt_size, alpha=alpha_no, label="Dx = 0")
    ax.scatter(df.loc[yes, "umap_x"], df.loc[yes, "umap_y"],
               c=yes_colour, s=pt_size, alpha=alpha_yes, label="Dx = 1")

    ax.set_title(f"UMAP of SDoH  –  {dx_col}")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.show()


def plot_topN_categories(df: pd.DataFrame, col: str, n=20,
                         pt_size=6, alpha_bg=.12, alpha_fg=.85):
    """
    Color top-n categories with Tableau palette; others gray.
    
    Args:
        df: DataFrame with 'umap_x' and 'umap_y' columns
        col: Column to color by
        n: Number of top categories to show
        pt_size: Point size
        alpha_bg: Alpha for background (Other)
        alpha_fg: Alpha for foreground categories
    """
    cats = _topN(df[col], n).fillna("Missing")
    uniq = pd.unique(cats)

    palette = list(mcolors.TABLEAU_COLORS.values())
    while len(palette) < len(uniq):
        palette.extend(palette)  # repeat if >20 needed

    fig, ax = plt.subplots(figsize=(6, 5))
    bg = cats == "Other"
    ax.scatter(df.loc[bg, "umap_x"], df.loc[bg, "umap_y"],
               c="#d0d0d0", s=pt_size, alpha=alpha_bg, label="Other")

    for i, cat in enumerate(uniq):
        if cat == "Other":
            continue
        m = cats == cat
        ax.scatter(df.loc[m, "umap_x"], df.loc[m, "umap_y"],
                   c=[palette[i]], s=pt_size, alpha=alpha_fg, label=str(cat))

    ax.set_title(f"UMAP – top {n} {col}  (others gray)")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left",
              fontsize="small", markerscale=2)
    plt.tight_layout()
    plt.show()









