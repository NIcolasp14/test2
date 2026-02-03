"""
utils.py

Utility functions for SDOH analysis, including column detection, 
data preprocessing helpers, and common configurations.
"""

import pandas as pd
import numpy as np


# -----------------------------
# Configuration Constants
# -----------------------------

CANDIDATE_KEYS = [
    "empi",
    "lumeris_empi",
    "empi_id",
    "member_id",
    "sys_mbr_sk",
    "clm_sys_mbr_sk",
    "memberid",
    "acxiom_id",
    "acx_id",
    "subscriber_id",
]

DIAG_CODE_CANDIDATES = [
    "diagnosis_code",
    "dx_code",
    "diag_code",
    "icd_code",
    "icd10",
    "icd9",
    "dx",
    "diagnosis",
    "code",
]

PATIENT_ID_CANDIDATES = [
    "empi",
    "lumeris_empi",
    "empi_id",
    "member_id",
    "sys_mbr_sk",
    "clm_sys_mbr_sk",
    "memberid",
    "subscriber_id",
]

SENTINELS = {9, 99, 999, 9999}


# -----------------------------
# Column Detection Helpers
# -----------------------------

def pick_col(df, candidates):
    """
    Find the first matching column name from a list of candidates.
    
    Args:
        df: DataFrame to search
        candidates: List of candidate column names
    
    Returns:
        First matching column name, or None if not found
    """
    for c in candidates:
        if c in df.columns:
            return c
    return None


def find_patient_id_col(cols):
    """
    Detect patient identifier column from dataframe columns.
    
    Args:
        cols: List or Index of column names
    
    Returns:
        Name of patient ID column, or None if not found
    """
    for c in PATIENT_ID_CANDIDATES:
        if c in cols:
            return c
    # fallback: return first column named like an id
    for c in cols:
        if c.lower().endswith("id") or c.lower().endswith("_id"):
            return c
    return None


def find_diag_code_col(cols):
    """
    Detect diagnosis code column from dataframe columns.
    
    Args:
        cols: List or Index of column names
    
    Returns:
        Name of diagnosis code column, or None if not found
    """
    # Exact matches first
    for c in DIAG_CODE_CANDIDATES:
        if c in cols:
            return c

    # Case-insensitive exact
    lower = {c.lower(): c for c in cols}
    for cand in DIAG_CODE_CANDIDATES:
        if cand.lower() in lower:
            return lower[cand.lower()]

    # Substring match
    cand_tokens = [t.lower() for t in DIAG_CODE_CANDIDATES]
    for col in cols:
        cl = col.lower()
        for tok in cand_tokens:
            if tok and tok in cl:
                return col

    return None


def find_merge_keys(diag_cols, acx_cols):
    """
    Automatically detect matching keys between two dataframes.
    
    Args:
        diag_cols: Column names from diagnosis/left dataframe
        acx_cols: Column names from acxiom/right dataframe
    
    Returns:
        Tuple of (left_key, right_key) or (None, None) if not found
    """
    # 1) exact match
    for c in acx_cols:
        if c in diag_cols:
            return c, c
    
    # 2) candidate-based: diag contains candidate and acx has case-insensitive match
    acx_lower = {c.lower(): c for c in acx_cols}
    for cand in CANDIDATE_KEYS:
        if cand in diag_cols and cand.lower() in acx_lower:
            return cand, acx_lower[cand.lower()]
    
    # 3) case-insensitive exact between the two
    diag_lower = {c.lower(): c for c in diag_cols}
    for c_lower, c_orig in diag_lower.items():
        if c_lower in acx_lower:
            return c_orig, acx_lower[c_lower]
    
    return None, None


# -----------------------------
# Data Processing Helpers
# -----------------------------

def build_sdoh_X(sdoh_all, row_nan_frac_max=0.40, col_nan_frac_max=0.95, sentinels=SENTINELS):
    """
    Extract and QC SDoH features from a dataframe.
    
    Assumes dataframe has a 'flag' column, after which all columns are SDoH features.
    
    Args:
        sdoh_all: DataFrame with 'flag' column marking start of SDoH features
        row_nan_frac_max: Maximum fraction of NaN allowed per row
        col_nan_frac_max: Maximum fraction of NaN allowed per column
        sentinels: Set of values to treat as missing
    
    Returns:
        Tuple of (df_qc, X_qc): QC'd full dataframe and SDoH feature matrix
    """
    if "flag" not in sdoh_all.columns:
        raise ValueError("Column 'flag' not found in sdoh_all.")

    flag_idx = sdoh_all.columns.get_loc("flag")
    sdoh_cols = list(sdoh_all.columns[flag_idx + 1:])
    
    if not sdoh_cols:
        raise ValueError("No columns found after 'flag'.")

    X = sdoh_all[sdoh_cols].copy()

    # numeric coercion
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    # sentinel-to-missing (optional)
    X = X.mask(X.isin(set(sentinels)))

    # drop super-missing columns
    col_nan_frac = X.isna().mean()
    keep_cols = col_nan_frac[col_nan_frac <= col_nan_frac_max].index
    X = X[keep_cols]

    # drop rows with too much missingness
    row_nan_frac = X.isna().mean(axis=1)
    keep_rows = row_nan_frac <= row_nan_frac_max

    df_qc = sdoh_all.loc[keep_rows].copy()
    X_qc = X.loc[keep_rows].copy()

    print(f"QC rows kept: {keep_rows.sum():,}/{len(keep_rows):,}  (row_nan_frac_max={row_nan_frac_max})")
    print(f"QC cols kept: {X_qc.shape[1]:,}/{len(sdoh_cols):,}  (col_nan_frac_max={col_nan_frac_max})")

    return df_qc, X_qc


def collapse_rare_classes(y, min_count=50, top_n=None):
    """
    Keep classes with count >= min_count (or keep only top_n classes),
    map the rest to 'Other'.
    
    Args:
        y: Series of class labels
        min_count: Minimum count to keep a class
        top_n: If specified, keep only top N classes by frequency
    
    Returns:
        Series with rare classes mapped to 'Other'
    """
    y = y.astype("string").fillna("Missing")
    vc = y.value_counts(dropna=False)

    if top_n is not None:
        keep = set(vc.head(top_n).index)
    else:
        keep = set(vc[vc >= min_count].index)

    return y.where(y.isin(keep), other="Other")


def choose_n_splits(y_codes, desired=5):
    """
    Choose appropriate number of cross-validation splits based on class counts.
    
    Args:
        y_codes: Array or Series of integer class labels
        desired: Desired number of splits
    
    Returns:
        Number of splits (between 2 and desired)
    """
    counts = pd.Series(y_codes).value_counts()
    min_count = counts.min()
    return max(2, min(desired, int(min_count)))









