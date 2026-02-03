"""
data_merge.py

Robust merging of diagnosis.csv with acxiom_full.csv, using demographics.csv
as a bridge when needed. Includes automatic key detection and memory-friendly
chunked merging for large files.
"""

import os
from pathlib import Path
import pandas as pd
from utils import (
    CANDIDATE_KEYS, DIAG_CODE_CANDIDATES, PATIENT_ID_CANDIDATES,
    find_patient_id_col, find_diag_code_col, find_merge_keys
)


def aggregate_diagnoses_to_patient(diagnosis_df, patient_col, diag_col, 
                                   top_n=100, min_count=None):
    """
    Aggregate diagnosis rows to patient-level and one-hot encode top diagnosis codes.

    Returns a DataFrame indexed by patient id with binary columns for each selected
    diagnosis code (1 = patient has that code at least once).
    
    Args:
        diagnosis_df: DataFrame with diagnosis data
        patient_col: Column name for patient identifier
        diag_col: Column name for diagnosis code
        top_n: Number of top diagnosis codes to encode
        min_count: Minimum count for a diagnosis code to be included
    
    Returns:
        DataFrame with one row per patient and one-hot encoded diagnosis columns
    """
    if patient_col not in diagnosis_df.columns:
        raise ValueError(f"Patient id column {patient_col!r} not found in diagnosis dataframe")
    if diag_col not in diagnosis_df.columns:
        raise ValueError(f"Diagnosis code column {diag_col!r} not found in diagnosis dataframe")

    # drop rows missing patient id or diagnosis code
    d = diagnosis_df[[patient_col, diag_col]].dropna(subset=[patient_col])

    # Use string form for codes
    d[diag_col] = d[diag_col].astype(str)

    # choose codes to keep
    vc = d[diag_col].value_counts()
    if min_count is not None:
        keep = set(vc[vc >= min_count].index)
    else:
        keep = set(vc.head(top_n).index)

    # mark presence via dummies grouped by patient
    d_keep = d[d[diag_col].isin(keep)].copy()
    if d_keep.empty:
        # If nothing in keep, return an empty frame with patient ids
        patients = pd.DataFrame({patient_col: d[patient_col].unique()})
        patients = patients.set_index(patient_col)
        return patients

    dummies = pd.get_dummies(d_keep[diag_col], prefix="dx")
    d_idx = d_keep[[patient_col]].reset_index(drop=True)
    d_onehot = pd.concat([d_idx, dummies], axis=1)
    patient_onehot = d_onehot.groupby(patient_col).max()

    # optionally add a column counting how many rare/other diagnosis codes the patient has
    other_mask = ~d[diag_col].isin(keep)
    if other_mask.any():
        other_counts = d.loc[other_mask].groupby(patient_col).size()
        patient_onehot["dx_other_count"] = other_counts
        patient_onehot["dx_other_count"] = patient_onehot["dx_other_count"].fillna(0).astype(int)

    # ensure index is a column for downstream merges
    patient_onehot = patient_onehot.reset_index()
    return patient_onehot


def chunked_bridge_merge(diagnosis, acx_path, diag_key, acx_key, chunksize=200_000):
    """
    Memory-friendly chunked merge for large acxiom files.
    
    Reads acxiom file in chunks and maps values for matching keys into the 
    diagnosis dataframe. Keeps first occurrence for duplicate keys.
    
    Args:
        diagnosis: DataFrame (left side of merge)
        acx_path: Path to acxiom CSV file
        diag_key: Key column name in diagnosis dataframe
        acx_key: Key column name in acxiom file
        chunksize: Number of rows to read per chunk
    
    Returns:
        Merged DataFrame
    """
    # normalize diag values (strip whitespace) and use a set for fast lookup
    diag_vals = set(v.strip() for v in diagnosis[diag_key].dropna().astype(str).unique())
    print(f"Chunked merge: {len(diag_vals):,} unique diag keys to match")

    # Read acxiom header once to know available columns
    acx_cols = pd.read_csv(acx_path, nrows=0, dtype=str).columns.tolist()
    if acx_key not in acx_cols:
        raise RuntimeError(f"Expected acx key {acx_key} not in acxiom columns")

    # Add any missing acxiom columns to `diagnosis` in one concat operation
    add_cols = [c for c in acx_cols if c != acx_key and c not in diagnosis.columns]
    if add_cols:
        new_cols = pd.DataFrame({c: pd.NA for c in add_cols}, index=diagnosis.index)
        diagnosis = pd.concat([diagnosis, new_cols], axis=1)

    # Iterate file-by-chunk, mapping first-seen acxiom values into diagnosis
    matched_keys = set()
    for i, chunk in enumerate(pd.read_csv(acx_path, chunksize=chunksize, 
                                          dtype=str, low_memory=False)):
        # normalize acx_key values and filter by diag_vals
        chunk[acx_key] = chunk[acx_key].astype(str).str.strip()
        chunk_subset = chunk[chunk[acx_key].isin(diag_vals)]
        if chunk_subset.empty:
            continue

        # drop duplicate keys keeping first occurrence
        chunk_subset = chunk_subset.drop_duplicates(subset=[acx_key])

        # build mapping dicts for columns and apply with vectorized fillna
        matched_in_chunk = set(chunk_subset[acx_key].astype(str).unique())
        matched_keys.update(matched_in_chunk)
        
        for col in chunk_subset.columns:
            if col == acx_key:
                continue
            mapping = dict(zip(chunk_subset[acx_key].astype(str), chunk_subset[col]))
            mask = diagnosis[diag_key].notna()
            mapped_series = diagnosis.loc[mask, diag_key].astype(str).str.strip().map(mapping)
            diagnosis.loc[mask, col] = diagnosis.loc[mask, col].fillna(mapped_series)

        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1} chunks — matched so far: {len(matched_keys):,} unique keys")

    # report match statistics
    match_rate = len(matched_keys) / max(1, len(diag_vals))
    print(f"Chunked merge finished — matched {len(matched_keys):,} unique keys ({match_rate:.2%} of diag keys)")
    
    # Defragment frame after many assignments
    diagnosis = diagnosis.copy()
    return diagnosis


def simple_merge(acx_path, diagnosis, left_key, right_key):
    """
    Simple full-file merge (for smaller acxiom files).
    
    Args:
        acx_path: Path to acxiom CSV file
        diagnosis: DataFrame (left side of merge)
        left_key: Key column in diagnosis
        right_key: Key column in acxiom
    
    Returns:
        Merged DataFrame
    """
    acx_full = pd.read_csv(acx_path, dtype=str, low_memory=False)
    merged = diagnosis.merge(acx_full, left_on=left_key, right_on=right_key, how="left")
    return merged


def merge_diagnosis_with_acxiom(acx_path, diag_path, dem_path=None, 
                                output_path=None, chunksize=200_000,
                                aggregate_diagnoses=True, top_n_diagnoses=200):
    """
    Main function to merge diagnosis data with acxiom data.
    
    Args:
        acx_path: Path to acxiom CSV file
        diag_path: Path to diagnosis CSV file
        dem_path: Optional path to demographics CSV file (for bridging)
        output_path: Where to save merged output
        chunksize: Chunk size for large file processing
        aggregate_diagnoses: Whether to aggregate diagnosis rows to patient level
        top_n_diagnoses: Number of top diagnosis codes to one-hot encode
    
    Returns:
        Merged DataFrame
    """
    print("Reading diagnosis (as strings for robust joins)...")
    if not Path(diag_path).exists():
        raise FileNotFoundError(f"Diagnosis file not found: {diag_path}")
    diagnosis = pd.read_csv(diag_path, dtype=str, low_memory=False)

    # Read demographics if provided
    demographics = None
    if dem_path and Path(dem_path).exists():
        print("Reading demographics (for bridging)...")
        demographics = pd.read_csv(dem_path, dtype=str, low_memory=False)

    # normalize: if clm_sys_mbr_sk exists, rename to sys_mbr_sk for consistency
    if 'clm_sys_mbr_sk' in diagnosis.columns and 'sys_mbr_sk' not in diagnosis.columns:
        diagnosis = diagnosis.rename(columns={'clm_sys_mbr_sk': 'sys_mbr_sk'})

    # Aggregate diagnosis rows to patient-level if requested
    if aggregate_diagnoses:
        patient_id_col = find_patient_id_col(diagnosis.columns)
        if patient_id_col is None:
            raise RuntimeError("Could not detect patient identifier. Add one of: " + 
                             ",".join(PATIENT_ID_CANDIDATES))

        diag_code_col = find_diag_code_col(diagnosis.columns)
        if diag_code_col is None:
            raise RuntimeError("Could not detect diagnosis code column. Add one of: " + 
                             ",".join(DIAG_CODE_CANDIDATES))

        print(f"Aggregating diagnoses to patient-level using '{patient_id_col}' and '{diag_code_col}'")
        diagnosis = aggregate_diagnoses_to_patient(diagnosis, patient_id_col, 
                                                   diag_code_col, top_n=top_n_diagnoses)
        print(f"Patient-level diagnoses: {diagnosis.shape[0]:,} patients, {diagnosis.shape[1]-1:,} diagnosis features")

    # Get acxiom columns
    try:
        acx_cols = pd.read_csv(acx_path, nrows=0).columns.tolist()
        print("Acxiom columns (preview):", acx_cols[:20])
    except Exception as e:
        print("Failed to read acxiom columns:", e)
        acx_cols = []

    # Find merge keys
    diag_cols = set(diagnosis.columns)
    diag_key, acx_key = find_merge_keys(diag_cols, acx_cols)

    merged = None
    acx_size = os.path.getsize(acx_path)
    MB = 1024 * 1024

    if diag_key and acx_key:
        print(f"Found merge keys: diagnosis.{diag_key} <-> acxiom.{acx_key}")
        if acx_size > 200 * MB:
            print("Large acxiom file detected — performing chunked merge")
            merged = chunked_bridge_merge(diagnosis.copy(), acx_path, diag_key, 
                                         acx_key, chunksize=chunksize)
        else:
            print("Acxiom file small enough — reading fully for simple merge")
            merged = simple_merge(acx_path, diagnosis, diag_key, acx_key)
    else:
        print("No direct key found. Trying demographics bridge...")
        if demographics is None:
            raise RuntimeError("No demographics provided — cannot bridge automatically.")

        if 'sys_mbr_sk' not in demographics.columns or 'empi' not in demographics.columns:
            raise RuntimeError("Demographics missing 'sys_mbr_sk' or 'empi' columns.")

        dem_map = demographics[['sys_mbr_sk', 'empi']].drop_duplicates()
        diag_with_empi = diagnosis.merge(dem_map, on='sys_mbr_sk', how='left')

        # Find empi column in acxiom
        empi_col = None
        for c in acx_cols:
            if 'empi' in c.lower():
                empi_col = c
                break

        if empi_col:
            print(f"Found acxiom empi column: {empi_col} — merging via demographics bridge")
            if acx_size > 200 * MB:
                merged = chunked_bridge_merge(diag_with_empi.copy(), acx_path, 
                                             'empi', empi_col, chunksize=chunksize)
            else:
                merged = simple_merge(acx_path, diag_with_empi, 'empi', empi_col)
        else:
            raise RuntimeError('No merge key found. Inspect acxiom columns manually.')

    # Clean up dtypes
    if merged is not None:
        # Coerce diagnosis one-hot columns to integer 0/1
        dx_cols = [c for c in merged.columns if c.startswith("dx_")]
        if dx_cols:
            merged[dx_cols] = merged[dx_cols].fillna(0).astype("int8")

        # Coerce Acxiom columns to string dtype
        acx_cols_safe = [c for c in acx_cols if c in merged.columns]
        for c in acx_cols_safe:
            try:
                merged[c] = merged[c].astype("string")
            except Exception:
                merged[c] = merged[c].astype(str).replace("nan", "")

        # Defragment before save
        merged = merged.copy()

    # Save if output path provided
    if output_path and merged is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(output_path, index=False)
        print(f"✅ Saved merged dataset to {output_path} — shape: {merged.shape}")

    return merged









