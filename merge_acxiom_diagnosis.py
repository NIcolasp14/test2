"""
merge_acxiom_diagnosis.py

Robustly merge `diagnosis.csv` with `acxiom_full.csv`, using `demographics.csv`
as a bridge when needed. The script attempts to automatically detect matching
ID columns and will perform a memory-friendly chunked merge if `acxiom_full.csv`
is large.

Usage (defaults point to the Files V2 used in the notebook):
python merge_acxiom_diagnosis.py \
    --acxiom acxiom_full.csv \
    --diagnosis "C:\\Users\\npigadas\\OneDrive - Lumeris Solutions Company, LLC\\Desktop\\Files V2\\diagnosis.csv" \
    --demographics "C:\\Users\\npigadas\\OneDrive - Lumeris Solutions Company, LLC\\Desktop\\Files V2\\demographics.csv" \
    --output diagnosis_with_acxiom.csv

The script saves `--output` with acxiom columns appended (first-match behavior
for duplicated keys in acxiom).
"""
import os
import sys
from pathlib import Path

import pandas as pd


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

# Columns to look for when finding a diagnosis code column
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

# Patient identifier candidates (in order of preference)
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


def find_patient_id_col(cols):
    for c in PATIENT_ID_CANDIDATES:
        if c in cols:
            return c
    # fallback: return first column named like an id
    for c in cols:
        if c.lower().endswith("id") or c.lower().endswith("_id"):
            return c
    return None


def find_diag_code_col(cols):
    # Exact matches first
    for c in DIAG_CODE_CANDIDATES:
        if c in cols:
            return c

    # Case-insensitive exact
    lower = {c.lower(): c for c in cols}
    for cand in DIAG_CODE_CANDIDATES:
        if cand.lower() in lower:
            return lower[cand.lower()]

    # Substring match: allow candidates like 'icd9' to match 'icd9_diagnosis_cd'
    cand_tokens = [t.lower() for t in DIAG_CODE_CANDIDATES]
    for col in cols:
        cl = col.lower()
        for tok in cand_tokens:
            if tok and tok in cl:
                return col

    return None


def aggregate_diagnoses_to_patient(diagnosis_df, patient_col, diag_col, top_n=100, min_count=None):
    """Aggregate diagnosis rows to patient-level and one-hot encode top diagnosis codes.

    Returns a DataFrame indexed by patient id with binary columns for each selected
    diagnosis code (1 = patient has that code at least once).
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



def find_merge_keys(diag_cols, acx_cols):
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


def chunked_bridge_merge(diagnosis, acx_path, diag_key, acx_key, chunksize=200_000):
    # diagnosis: dataframe (left), diag_key column present
    # We'll perform an outer-leaning approach: for each chunk of acxiom,
    # map values for matching keys into the diagnosis dataframe. If acx has
    # multiple rows per key, we keep the first occurrence encountered.
    # Collect distinct diag values (strings) to look up in acxiom chunks
    # normalize diag values (strip whitespace) and use a set for fast lookup
    diag_vals = set(v.strip() for v in diagnosis[diag_key].dropna().astype(str).unique())
    print(f"Chunked merge: {len(diag_vals):,} unique diag keys to match (preview)")

    # Read acxiom header once to know available columns (cheap operation)
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
    for i, chunk in enumerate(pd.read_csv(acx_path, chunksize=chunksize, dtype=str, low_memory=False)):
        # keep only rows for keys present in diagnosis
        # normalize acx_key values and filter by diag_vals (strip strings)
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
    print(f"Chunked merge finished — matched {len(matched_keys):,} unique keys ({len(matched_keys)/max(1,len(diag_vals)):.2%} of diag keys)")
    # Defragment frame after many assignments
    diagnosis = diagnosis.copy()
    return diagnosis


def simple_merge(acx_path, diagnosis, left_key, right_key):
    acx_full = pd.read_csv(acx_path, dtype=str, low_memory=False)
    merged = diagnosis.merge(acx_full, left_on=left_key, right_on=right_key, how="left")
    return merged


# Notebook-style top-level execution (hardcoded paths)
# Paths (adjust if needed)
ACX_PATH = Path(r"acxiom_full.csv")
DIAG_PATH = Path(r"C:\Users\npigadas\OneDrive - Lumeris Solutions Company, LLC\Desktop\Files V2\diagnosis.csv")
DEM_PATH = Path(r"C:\Users\npigadas\OneDrive - Lumeris Solutions Company, LLC\Desktop\Files V2\demographics.csv")
OUT_PATH = Path(r"diagnosis_with_acxiom.csv")
CHUNKSIZE = 200_000

print("Reading diagnosis (as strings for robust joins)...")
if not DIAG_PATH.exists():
    raise FileNotFoundError(f"Diagnosis file not found: {DIAG_PATH}")
diagnosis = pd.read_csv(DIAG_PATH, dtype=str, low_memory=False)

print("Reading demographics (for bridging)...")
if not DEM_PATH.exists():
    print(f"Warning: demographics file not found at {DEM_PATH}; bridge options will be limited")
    demographics = None
else:
    demographics = pd.read_csv(DEM_PATH, dtype=str, low_memory=False)

# normalize: if clm_sys_mbr_sk exists, rename to sys_mbr_sk for consistency
if 'clm_sys_mbr_sk' in diagnosis.columns and 'sys_mbr_sk' not in diagnosis.columns:
    diagnosis = diagnosis.rename(columns={'clm_sys_mbr_sk': 'sys_mbr_sk'})

# -----------------------------
# Aggregate diagnosis rows to patient-level (one row per patient)
# -----------------------------
patient_id_col = find_patient_id_col(diagnosis.columns)
if patient_id_col is None:
    raise RuntimeError("Could not detect a patient identifier column in diagnosis dataframe. Add one of: " + ",".join(PATIENT_ID_CANDIDATES))

diag_code_col = find_diag_code_col(diagnosis.columns)
if diag_code_col is None:
    raise RuntimeError("Could not detect a diagnosis code column in diagnosis dataframe. Add one of: " + ",".join(DIAG_CODE_CANDIDATES))

print(f"Aggregating diagnoses to patient-level using patient id '{patient_id_col}' and diagnosis code '{diag_code_col}'")
patient_df = aggregate_diagnoses_to_patient(diagnosis, patient_id_col, diag_code_col, top_n=200, min_count=None)

print(f"Patient-level diagnoses: {patient_df.shape[0]:,} patients, {patient_df.shape[1]-1:,} diagnosis features")


try:
    acx_cols = pd.read_csv(ACX_PATH, nrows=0).columns.tolist()
except Exception as e:
    print("Failed to read acxiom columns preview:", e)
    acx_cols = []

print("Acxiom columns (preview):", acx_cols[:20])

# Use patient-level columns when choosing merge keys
diag_cols = set(patient_df.columns)
diag_key, acx_key = find_merge_keys(diag_cols, acx_cols)

merged = None

if diag_key and acx_key:
    print(f"Found merge keys: diagnosis.{diag_key} <-> acxiom.{acx_key}")
    acx_size = os.path.getsize(ACX_PATH)
    MB = 1024 * 1024
    if acx_size > 200 * MB:
        print("Large acxiom file detected — performing chunked, memory-friendly merge")
        merged = chunked_bridge_merge(patient_df.copy(), ACX_PATH, diag_key, acx_key, chunksize=CHUNKSIZE)
    else:
        print("Acxiom file small enough — reading fully for simple merge")
        merged = simple_merge(ACX_PATH, patient_df, diag_key, acx_key)
else:
    print("No direct key found between diagnosis and acxiom. Trying demographics bridge...")
    if demographics is None:
        raise RuntimeError("No demographics provided — cannot bridge automatically.")

    if 'sys_mbr_sk' not in demographics.columns or 'empi' not in demographics.columns:
        raise RuntimeError("Demographics missing 'sys_mbr_sk' or 'empi' columns — cannot bridge.")

    dem_map = demographics[['sys_mbr_sk', 'empi']].drop_duplicates()
    diag_with_empi = patient_df.merge(dem_map, left_on='sys_mbr_sk', right_on='sys_mbr_sk', how='left')

    empi_col = None
    for c in acx_cols:
        if c.lower() == 'empi' or 'empi' in c.lower() or c.lower() == 'lumeris_empi':
            empi_col = c
            break

    if empi_col:
        print(f"Found acxiom empi-like column: {empi_col} — merging via demographics bridge")
        acx_size = os.path.getsize(ACX_PATH)
        if acx_size > 200 * 1024 * 1024:
            merged = chunked_bridge_merge(diag_with_empi.copy(), ACX_PATH, 'empi', empi_col, chunksize=CHUNKSIZE)
        else:
            merged = simple_merge(ACX_PATH, diag_with_empi, 'empi', empi_col)
    else:
        bridge_col = None
        for c in acx_cols:
            if demographics is not None and c in demographics.columns:
                bridge_col = c
                break
        if bridge_col:
            print(f"Using demographics bridge via column: {bridge_col}")
            if os.path.getsize(ACX_PATH) > 200 * 1024 * 1024:
                augmented = patient_df.merge(demographics[['sys_mbr_sk', bridge_col]].drop_duplicates(), on='sys_mbr_sk', how='left')
                merged = chunked_bridge_merge(augmented.copy(), ACX_PATH, bridge_col, bridge_col, chunksize=CHUNKSIZE)
            else:
                acx_full = pd.read_csv(ACX_PATH, dtype=str, low_memory=False)
                merged = patient_df.merge(demographics[['sys_mbr_sk', bridge_col]].drop_duplicates(), on='sys_mbr_sk', how='left').merge(acx_full, left_on=bridge_col, right_on=bridge_col, how='left')
        else:
            print("No suitable bridge column found automatically.")
            print("Sample of acxiom columns (first 5):", acx_cols[:5])
            raise RuntimeError('No merge key found automatically. Inspect acxiom columns and choose a key to merge on.')

# final save: coerce reasonable dtypes to avoid mixed-type columns on read
if merged is None:
    raise RuntimeError("No merged dataframe produced — aborting save")

# 1) coerce diagnosis one-hot columns (prefix 'dx_') to integer 0/1
dx_cols = [c for c in merged.columns if c.startswith("dx_")]
if dx_cols:
    merged[dx_cols] = merged[dx_cols].fillna(0).astype("int8")

# 2) coerce known Acxiom columns to pandas 'string' dtype for consistency
acx_cols_safe = [c for c in acx_cols if c in merged.columns]
for c in acx_cols_safe:
    try:
        merged[c] = merged[c].astype("string")
    except Exception:
        # best-effort: fallback to generic string conversion
        merged[c] = merged[c].astype(str).replace("nan", "")

# Defragment before write
merged = merged.copy()

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
merged.to_csv(OUT_PATH, index=False)
print(f"✅ Saved merged dataset to {OUT_PATH} — shape: {merged.shape}")


if __name__ == '__main__':
    # Hardcoded settings (as you'd run from a notebook cell)
    class Args:
        pass

    args = Args()
    args.acxiom = r"acxiom_full.csv"
    args.diagnosis = r"C:\Users\npigadas\OneDrive - Lumeris Solutions Company, LLC\Desktop\Files V2\diagnosis.csv"
    args.demographics = r"C:\Users\npigadas\OneDrive - Lumeris Solutions Company, LLC\Desktop\Files V2\demographics.csv"
    args.output = r"diagnosis_with_acxiom.csv"
    args.chunksize = 200_000

    print('Running merge with hardcoded paths:')
    print(' acxiom ->', args.acxiom)
    print(' diagnosis ->', args.diagnosis)
    print(' demographics ->', args.demographics)
    print(' output ->', args.output)

    main(args)
