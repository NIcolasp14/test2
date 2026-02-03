"""
Minimal script: create grouped diagnosis labels for members with diagnosis
and attach SDoH features (no zeros assigned to members without diagnosis).

Output:
 - `sdoh_with_diag_groups.csv`: one row per member present in diagnosis file,
    with SDoH features (numeric, NaNs preserved) and group_* binary labels.
 - `diag_groups_mapping.csv`: mapping of group_i -> dx codes in that group.

Usage: run from project root where `full_acxiom.csv`, `diagnosis_with_acxiom3.csv`,
and optionally `demographics.csv` are present.
"""

import pandas as pd
import numpy as np
import re
import os
import warnings
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
warnings.filterwarnings('ignore')

# Paths (adjust if needed)
ACXIOM_PATH = "full_acxiom.csv"
DIAG_PATH = "diagnosis_with_acxiom3.csv"
DEM_PATH = "demographics.csv"

N_GROUPS = 10
TARGET_FRAC = 0.5
RANDOM_STATE = 42

# Defaults used by some optional ML helpers later in the file
MISSING_THRESHOLD = 0.95
VARIANCE_THRESHOLD = 1e-5


def identify_sdoh_columns(df):
    pattern = re.compile(r'^[A-Za-z]{2}\d+$')
    return [c for c in df.columns if isinstance(c, str) and pattern.match(c)]


def find_join_key(df_acx, df_diag):
    candidates = ['sys_mbr_sk', 'clm_sys_mbr_sk', 'empi', 'member_id', 'acxiom_id', 'member_sk']
    # case-insensitive column lookup
    acx_cols_lower = {c.lower(): c for c in df_acx.columns}
    diag_cols_lower = {c.lower(): c for c in df_diag.columns}

    # try direct candidate matches (case-insensitive)
    best_key = None
    best_overlap = 0
    for cand in candidates:
        if cand.lower() in acx_cols_lower and cand.lower() in diag_cols_lower:
            acx_col = acx_cols_lower[cand.lower()]
            diag_col = diag_cols_lower[cand.lower()]
            try:
                s1 = set(df_acx[acx_col].dropna().astype(str).str.strip().unique())
                s2 = set(df_diag[diag_col].dropna().astype(str).str.strip().unique())
                ov = len(s1 & s2)
            except Exception:
                ov = 0
            if ov > best_overlap:
                best_overlap = ov
                best_key = acx_col

    if best_key and best_overlap > 0:
        return best_key

    # try to bridge via demographics.empi -> sys_mbr_sk
    try:
        dem = pd.read_csv(DEM_PATH, usecols=['sys_mbr_sk', 'empi'])
        dem['empi'] = dem['empi'].astype(str).str.strip()
        dem['sys_mbr_sk'] = dem['sys_mbr_sk'].astype(str).str.strip()
        dem_map = dem.drop_duplicates('empi').set_index('empi')['sys_mbr_sk'].to_dict()

        if 'empi' in diag_cols_lower:
            diag_empi_col = diag_cols_lower['empi']
            diag_empis = set(df_diag[diag_empi_col].dropna().astype(str).str.strip().unique())
            mapped_sys = {dem_map.get(e) for e in diag_empis if dem_map.get(e) is not None}
            mapped_sys = {s for s in mapped_sys if s}
            if mapped_sys:
                # see if any Acxiom column contains these sys_mbr_sk values
                for acx_col in df_acx.columns:
                    try:
                        vals = set(df_acx[acx_col].dropna().astype(str).str.strip().unique())
                        if len(vals & mapped_sys) > 0:
                            return acx_col
                    except Exception:
                        continue
    except Exception:
    """

    import pandas as pd
    import numpy as np
    import re
    import os
    import warnings
    warnings.filterwarnings('ignore')

    # Input paths (edit if different)
    ACXIOM_PATH = "full_acxiom.csv"
    DIAG_PATH = "diagnosis_with_acxiom3.csv"
    DEM_PATH = "demographics.csv"

    # Parameters
    N_GROUPS = 10
    TARGET_FRAC = 0.5


    def safe_read_csv(path, **kwargs):
        try:
            return pd.read_csv(path, **kwargs)
        except Exception:
            try:
                return pd.read_csv(path, engine='python', **kwargs)
            except Exception:
                chunks = []
                for chunk in pd.read_csv(path, engine='python', chunksize=200000, **kwargs):
                    chunks.append(chunk)
                return pd.concat(chunks, ignore_index=True)


    def identify_sdoh_columns(df):
        pattern = re.compile(r'^[A-Za-z]{2}\d+$')
        return [c for c in df.columns if isinstance(c, str) and pattern.match(c)]


    def find_join_key(df_acx, df_diag):
        # try common keys by overlap
        candidates = ['sys_mbr_sk', 'clm_sys_mbr_sk', 'empi', 'member_id', 'acxiom_id', 'member_sk']
        acx_lower = {c.lower(): c for c in df_acx.columns}
        diag_lower = {c.lower(): c for c in df_diag.columns}

        best = (None, 0)
        for cand in candidates:
            if cand.lower() in acx_lower and cand.lower() in diag_lower:
                a = acx_lower[cand.lower()]
                d = diag_lower[cand.lower()]
                try:
                    s1 = set(df_acx[a].dropna().astype(str).str.strip().unique())
                    s2 = set(df_diag[d].dropna().astype(str).str.strip().unique())
                    ov = len(s1 & s2)
                except Exception:
                    ov = 0
                if ov > best[1]:
                    best = (a, ov)

        if best[0] and best[1] > 0:
            return best[0]

        # try demographics bridge if available
        try:
            dem = pd.read_csv(DEM_PATH, usecols=['sys_mbr_sk', 'empi'])
            dem['empi'] = dem['empi'].astype(str).str.strip()
            dem['sys_mbr_sk'] = dem['sys_mbr_sk'].astype(str).str.strip()
            dem_map = dem.drop_duplicates('empi').set_index('empi')['sys_mbr_sk'].to_dict()
            if 'empi' in diag_lower:
                diag_empi = set(df_diag[diag_lower['empi']].dropna().astype(str).str.strip().unique())
                mapped = {dem_map.get(e) for e in diag_empi if dem_map.get(e)}
                for col in df_acx.columns:
                    try:
                        vals = set(df_acx[col].dropna().astype(str).str.strip().unique())
                        if len(vals & mapped) > 0:
                            return col
                    except Exception:
                        continue
        except Exception:
            pass

        return None


    def filter_acxiom_to_diag(df_acx, df_diag, key=None):
        if key and key in df_acx.columns and key in df_diag.columns:
            diag_ids = set(df_diag[key].dropna().astype(str).unique())
            mask = df_acx[key].astype(str).isin(diag_ids)
            return df_acx.loc[mask].copy(), key

        # try demographics bridge
        if 'empi' in df_diag.columns and 'sys_mbr_sk' in df_acx.columns:
            try:
                dem = pd.read_csv(DEM_PATH, usecols=['sys_mbr_sk', 'empi'])
                dem['empi'] = dem['empi'].astype(str).str.strip()
                mapping = dem.drop_duplicates('empi').set_index('empi')['sys_mbr_sk'].to_dict()
                diag_empis = df_diag['empi'].dropna().astype(str).unique()
                mapped_sys = [mapping.get(e) for e in diag_empis if mapping.get(e) is not None]
                if mapped_sys:
                    mask = df_acx['sys_mbr_sk'].astype(str).isin(set(mapped_sys))
                    return df_acx.loc[mask].copy(), 'sys_mbr_sk'
            except Exception:
                pass

        raise KeyError('Could not determine join key to filter Acxiom to diagnosis cohort')


    def aggregate_diagnoses(df_diag, join_key):
        df_diag[join_key] = df_diag[join_key].astype(str).str.strip()
        dx_cols = [c for c in df_diag.columns if isinstance(c, str) and c.startswith('dx_')]
        if not dx_cols:
            raise ValueError('No dx_ columns found in diagnosis file')
        for c in dx_cols:
            df_diag[c] = pd.to_numeric(df_diag[c], errors='coerce').fillna(0).astype(int)
        df_agg = df_diag.groupby(join_key)[dx_cols].max().reset_index()
        return df_agg, dx_cols


    def create_balanced_groups(df, dx_cols, n_groups=10, target_frac=0.5, unique=True):
        N = len(df)
        target = int(round(target_frac * N))
        remaining = list(dx_cols)
        groups = []

        dx_bool = {c: (pd.to_numeric(df[c], errors='coerce').fillna(0) > 0).values for c in dx_cols}
        all_union = np.zeros(N, dtype=bool)
        for arr in dx_bool.values():
            all_union = all_union | arr
        max_achievable = int(all_union.sum())
        if max_achievable < target:
            target = int(round(max_achievable / max(1, n_groups)))

        for _ in range(n_groups):
            if not remaining:
                break
            group = []
            mask = np.zeros(N, dtype=bool)
            current_pos = 0
            while current_pos < target and remaining:
                best_dx = None
                best_dist = None
                best_new = None
                for dx in remaining:
                    cand = mask | dx_bool[dx]
                    new_count = int(cand.sum())
                    if new_count == current_pos:
                        continue
                    dist = abs(new_count - target)
                    if best_dist is None or dist < best_dist or (dist == best_dist and new_count > best_new):
                        best_dist = dist
                        best_dx = dx
                        best_new = new_count
                if best_dx is None:
                    break
                group.append(best_dx)
                mask = mask | dx_bool[best_dx]
                current_pos = int(mask.sum())
                if unique:
                    remaining.remove(best_dx)
            if group:
                groups.append(group)
        return groups


    def make_group_labels(df_agg, groups):
        out = df_agg.copy()
        for i, grp in enumerate(groups):
            col = f'group_{i+1}'
            out[col] = out[grp].apply(lambda row: (pd.to_numeric(row, errors='coerce').fillna(0) > 0).any(), axis=1).astype(int)
        return out


    def main():
        if not os.path.exists(ACXIOM_PATH):
            raise FileNotFoundError(f'{ACXIOM_PATH} not found')
        if not os.path.exists(DIAG_PATH):
            raise FileNotFoundError(f'{DIAG_PATH} not found')

        print('Loading files...')
        df_acx = safe_read_csv(ACXIOM_PATH, low_memory=False)
        df_diag = safe_read_csv(DIAG_PATH, low_memory=False)

        print('Determining join key and filtering Acxiom to diagnosis cohort...')
        join_key = find_join_key(df_acx, df_diag)
        if join_key is None:
            raise KeyError('No join key found. Provide demographics.csv for bridge.')
        df_acx_sub, used_key = filter_acxiom_to_diag(df_acx, df_diag, key=join_key)

        print(f'Filtered Acxiom rows: {len(df_acx)} -> {len(df_acx_sub)} using key {used_key}')

        print('Identifying SDoH columns...')
        sdoh_cols = identify_sdoh_columns(df_acx_sub)
        print(f'Found {len(sdoh_cols)} SDoH columns')

        df_sdoh = df_acx_sub[[used_key] + sdoh_cols].copy()
        for c in sdoh_cols:
            df_sdoh[c] = pd.to_numeric(df_sdoh[c], errors='coerce')

        print('Aggregating diagnoses per member...')
        df_diag_agg, dx_cols = aggregate_diagnoses(df_diag, used_key)

        df_diag_agg = df_diag_agg.set_index(used_key)
        df_sdoh[used_key] = df_sdoh[used_key].astype(str)
        paired_ids = df_sdoh[used_key].astype(str).values
        df_paired = df_diag_agg.reindex(paired_ids).reset_index()

        has_diag_mask = df_paired[dx_cols].notna().any(axis=1)
        df_sdoh = df_sdoh.loc[has_diag_mask.values].reset_index(drop=True)
        df_paired = df_paired.loc[has_diag_mask.values].reset_index(drop=True)

        print(f'Members with diagnosis and SDoH data: {len(df_sdoh)}')

        print('Creating diagnosis groups...')
        groups = create_balanced_groups(df_paired, dx_cols, n_groups=N_GROUPS, target_frac=TARGET_FRAC)
        print(f'Built {len(groups)} groups')

        print('Creating group labels...')
        df_with_groups = make_group_labels(df_paired[[used_key] + dx_cols], groups)

        out = df_sdoh.copy()
        for i in range(len(groups)):
            col = f'group_{i+1}'
            out[col] = df_with_groups[col].values

        out_path = 'sdoh_with_diag_groups.csv'
        out.to_csv(out_path, index=False)
        print(f'Saved: {out_path}')

        mapping_rows = []
        for i, grp in enumerate(groups):
            mapping_rows.append({'group': f'group_{i+1}', 'dx_codes': ';'.join(grp)})
        pd.DataFrame(mapping_rows).to_csv('diag_groups_mapping.csv', index=False)
        print('Saved diag_groups_mapping.csv')


    if __name__ == '__main__':
        main()
    """
    print("\n" + "=" * 70)
    print("Step 3: Merging ED Labels with Acxiom Data")
    print("=" * 70)
    
    # Load Acxiom data
    df_acxiom = safe_read_csv(acxiom_path, low_memory=False)
    print(f"Loaded Acxiom data: {df_acxiom.shape}")

    # Normalize/identify member ID column in Acxiom (common variants)
    possible_keys = ['sys_mbr_sk', 'clm_sys_mbr_sk', 'acxiom_id', 'empi', 'member_id', 'member_sk']
    acx_key = None
    for k in possible_keys:
        if k in df_acxiom.columns:
            acx_key = k
            break

    if acx_key is None:
        # try case-insensitive search
        cols_lower = {c.lower(): c for c in df_acxiom.columns}
        for want in possible_keys:
            if want.lower() in cols_lower:
                acx_key = cols_lower[want.lower()]
                break

    if acx_key is None:
        print("Acxiom columns (sample):", list(df_acxiom.columns)[:20])
        raise KeyError("No recognized member identifier column found in Acxiom data to merge on.\n" \
                       "Expected one of: {}".format(possible_keys))

    # If Acxiom uses a member-like id (e.g., 'member_id' or 'empi'), try bridging via demographics
    if acx_key in ('member_id', 'acxiom_id') or acx_key.lower() == 'empi' or acx_key.lower().startswith('member'):
        try:
            dem_path = "demographics.csv"
            dem = pd.read_csv(dem_path, usecols=['sys_mbr_sk', 'empi'])
            dem['sys_mbr_sk'] = dem['sys_mbr_sk'].astype(str).str.strip()
            dem['empi'] = dem['empi'].astype(str).str.strip()

            ed_labels_df = ed_labels_df.copy()
            ed_labels_df['sys_mbr_sk'] = ed_labels_df['sys_mbr_sk'].astype(str).str.strip()

            # attach EMPI to ED labels
            ed_with_empi = ed_labels_df.merge(dem, on='sys_mbr_sk', how='left')

            # Normalize Acxiom key and EMPI to strings
            df_acxiom[acx_key] = df_acxiom[acx_key].astype(str).str.strip()
            ed_with_empi['empi'] = ed_with_empi['empi'].astype(str).str.strip()

            # Merge Acxiom (by its member-like id) with ED labels via EMPI
            df_merged = df_acxiom.merge(
                ed_with_empi[['empi', 'total_ed_visits', 'ed_utilization_class']],
                left_on=acx_key,
                right_on='empi',
                how='left'
            )
            # Drop auxiliary empi column if present
            if 'empi' in df_merged.columns:
                df_merged = df_merged.drop(columns=['empi'])

            print(f"Merged via demographics bridge: acxiom.{acx_key} <-> demographics.empi <-> ed_labels.sys_mbr_sk")
        except Exception as e:
            print(f"Bridge via demographics failed: {e}. Falling back to direct sys_mbr_sk rename.")
            df_acxiom = df_acxiom.rename(columns={acx_key: 'sys_mbr_sk'})
            df_acxiom['sys_mbr_sk'] = df_acxiom['sys_mbr_sk'].astype(str).str.strip()
            ed_labels_df = ed_labels_df.copy()
            ed_labels_df['sys_mbr_sk'] = ed_labels_df['sys_mbr_sk'].astype(str).str.strip()
            df_merged = df_acxiom.merge(
                ed_labels_df[['sys_mbr_sk', 'total_ed_visits', 'ed_utilization_class']],
                on='sys_mbr_sk',
                how='left'
            )
    else:
        if acx_key != 'sys_mbr_sk':
            # rename the detected key to sys_mbr_sk for consistent merging
            df_acxiom = df_acxiom.rename(columns={acx_key: 'sys_mbr_sk'})
            print(f"Renamed Acxiom column '{acx_key}' to 'sys_mbr_sk' for merging.")

        # Ensure both keys are comparable strings to avoid dtype mismatches
        df_acxiom['sys_mbr_sk'] = df_acxiom['sys_mbr_sk'].astype(str).str.strip()
        ed_labels_df = ed_labels_df.copy()
        ed_labels_df['sys_mbr_sk'] = ed_labels_df['sys_mbr_sk'].astype(str).str.strip()

        # Merge on sys_mbr_sk
        df_merged = df_acxiom.merge(
            ed_labels_df[['sys_mbr_sk', 'total_ed_visits', 'ed_utilization_class']],
            on='sys_mbr_sk',
            how='left'
        )
    
    # Do NOT assign class 0 to all unmatched Acxiom rows.
    # Keep NaN for 'ed_utilization_class' for patients without ED matches so
    # downstream steps can exclude them. Create a flag to indicate merges.
    df_merged['__ed_matched'] = df_merged['total_ed_visits'].notna()

    # For matched rows ensure integer type for total_ed_visits
    matched_idx = df_merged['total_ed_visits'].notna()
    if matched_idx.any():
        df_merged.loc[matched_idx, 'total_ed_visits'] = df_merged.loc[matched_idx, 'total_ed_visits'].astype(int)

    print(f"Merged data shape: {df_merged.shape}")
    print(f"\nPatients with ED match: {df_merged['__ed_matched'].sum():,}")
    print(f"Patients without ED match: {(~df_merged['__ed_matched']).sum():,}")

    # If demographics is available, consider members present in demographics but not
    # in ED as having 0 ED visits (so they become class 0). This avoids treating
    # the entire Acxiom population as class 0 when we only have ED data for a subset.
    try:
        dem_path = "demographics.csv"
        dem = pd.read_csv(dem_path, usecols=['sys_mbr_sk'])
        dem['sys_mbr_sk'] = dem['sys_mbr_sk'].astype(str).str.strip()

        # Identify Acxiom rows whose sys_mbr_sk exists in demographics but had no ED match
        if 'sys_mbr_sk' in df_merged.columns:
            dem_set = set(dem['sys_mbr_sk'].unique())
            mask_no_match = (~df_merged['__ed_matched']) & (df_merged['sys_mbr_sk'].astype(str).str.strip().isin(dem_set))

            if mask_no_match.any():
                df_merged.loc[mask_no_match, 'total_ed_visits'] = 0
                df_merged.loc[mask_no_match, 'ed_utilization_class'] = 0
                df_merged.loc[mask_no_match, '__ed_matched'] = True
                print(f"Assigned class 0 (0 visits) to {mask_no_match.sum():,} Acxiom rows present in demographics.")
    except Exception:
        # demographics not available or mismatch — skip this step silently
        pass
    
    # Save if output path provided
    if output_path:
        df_merged.to_csv(output_path, index=False)
        print(f"\n✅ Saved merged data to: {output_path}")
    
    return df_merged


# =============================================================================
# Step 4: Identify SDoH Columns
# =============================================================================

def identify_sdoh_columns(df):
    """
    Identify SDoH columns (2 letters followed by numbers, e.g., ap006775).
    
    Args:
        df: DataFrame to search
    
    Returns:
        List of SDoH column names
    """
    print("\n" + "=" * 70)
    print("Step 4: Identifying SDoH Columns")
    print("=" * 70)
    
    # Pattern: 2 letters followed by digits
    pattern = re.compile(r'^[a-zA-Z]{2}\d+$')
    
    sdoh_cols = [col for col in df.columns if pattern.match(col)]
    
    print(f"Found {len(sdoh_cols)} SDoH columns")
    print(f"Examples: {sdoh_cols[:10]}")
    
    return sdoh_cols


# =============================================================================
# Custom Transformers (NO DATA LEAKAGE)
# =============================================================================

class ColumnDropper(BaseEstimator, TransformerMixin):
    """
    Remove columns with too many missing values or zero variance.
    Fit on training data only to prevent leakage.
    """
    def __init__(self, missing_threshold=0.95, variance_threshold=1e-5):
        self.missing_threshold = missing_threshold
        self.variance_threshold = variance_threshold
        self.cols_to_keep_ = None
        
    def fit(self, X, y=None):
        df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        
        # Identify columns to drop based on training data only
        missing_frac = df.isna().mean()
        variance = df.var()
        
        # Keep columns that pass both criteria
        keep_missing = missing_frac <= self.missing_threshold
        keep_variance = (variance > self.variance_threshold) | variance.isna()
        
        self.cols_to_keep_ = df.columns[keep_missing & keep_variance].tolist()
        
        if len(self.cols_to_keep_) == 0:
            print("⚠️ WARNING: No columns passed filtering criteria!")
            self.cols_to_keep_ = df.columns.tolist()[:10]  # Keep at least 10
        
        return self
    
    def transform(self, X):
        df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        return df[self.cols_to_keep_].values
    
    def get_feature_names_out(self, input_features=None):
        return np.array(self.cols_to_keep_)


class FeatureNameTracker(BaseEstimator, TransformerMixin):
    """Track feature names through the pipeline."""
    def __init__(self, feature_names):
        self.feature_names = feature_names
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X
    
    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names)


# =============================================================================
# Step 5: Prepare Data for Machine Learning (NO LEAKAGE VERSION)
# =============================================================================

def prepare_ml_data(df, sdoh_cols, include_diagnosis=True, exclude_dx_for_temporal=False):
    """
    Prepare feature matrix and target for machine learning.
    NO PREPROCESSING - keeps raw data with NaNs for pipeline.
    
    Args:
        df: Merged DataFrame
        sdoh_cols: List of SDoH column names
        include_diagnosis: Whether to include dx_* columns as features
        exclude_dx_for_temporal: If True, exclude dx_* to avoid temporal leakage
    
    Returns:
        X_raw (features with NaNs), y (target), feature_names, class_mapping
    """
    print("\n" + "=" * 70)
    print("Step 5: Preparing Data for Machine Learning (NO LEAKAGE)")
    print("=" * 70)
    
    # Get diagnosis columns if requested
    dx_cols = []
    if include_diagnosis and not exclude_dx_for_temporal:
        dx_cols = [col for col in df.columns if col.startswith('dx_') and col != 'dx_other_count']
        print(f"Found {len(dx_cols)} diagnosis columns")
        print("⚠️ WARNING: Including dx_* features may cause TEMPORAL LEAKAGE if")
        print("   diagnoses are from same period as ED visits!")
        print("   Consider setting exclude_dx_for_temporal=True for robust evaluation.")
    elif exclude_dx_for_temporal:
        print("🔒 Excluding dx_* features to prevent temporal leakage")
    
    # Combine features
    feature_cols = sdoh_cols + dx_cols
    print(f"Total features: {len(feature_cols)} ({len(sdoh_cols)} SDoH + {len(dx_cols)} diagnoses)")
    
    # Extract features and target - KEEP RAW DATA WITH NaNs
    X_raw = df[feature_cols].copy()
    y = df['ed_utilization_class'].copy()
    
    # Remove rows with missing target
    valid_mask = y.notna()
    X_raw = X_raw[valid_mask]
    y = y[valid_mask]
    
    print(f"\nSamples: {len(X_raw):,}")
    print(f"Original target distribution:")
    print(y.value_counts().sort_index())
    
    # IMPORTANT: Remap classes to 0-indexed for XGBoost compatibility
    unique_classes = sorted(y.unique())
    class_mapping = {orig: new for new, orig in enumerate(unique_classes)}
    reverse_mapping = {new: orig for orig, new in class_mapping.items()}
    
    y_remapped = y.map(class_mapping)
    
    print(f"\nRemapped classes for model compatibility:")
    print(f"  Original → New")
    for orig, new in class_mapping.items():
        print(f"  Class {int(orig)} → Class {new}")
    
    print(f"\nRemapped target distribution:")
    print(y_remapped.value_counts().sort_index())
    
    # Convert to numeric BUT KEEP NaNs - no imputation here!
    print("\nConverting features to numeric (keeping NaNs for pipeline)...")
    for col in X_raw.columns:
        X_raw[col] = pd.to_numeric(X_raw[col], errors='coerce')
    
    # Report missingness for awareness
    missing_pct = X_raw.isna().mean()
    print(f"\nMissingness statistics:")
    print(f"  Features with >50% missing: {(missing_pct > 0.5).sum()}")
    print(f"  Features with >90% missing: {(missing_pct > 0.9).sum()}")
    print(f"  Average missingness: {missing_pct.mean():.2%}")
    
    # Remove ONLY completely empty columns (all NaN)
    all_nan_cols = X_raw.columns[X_raw.isna().all()]
    if len(all_nan_cols) > 0:
        print(f"\n⚠️ Removing {len(all_nan_cols)} completely empty features (100% NaN)")
        X_raw = X_raw.drop(columns=all_nan_cols)
        feature_cols = [c for c in feature_cols if c not in all_nan_cols]
    
    print(f"\n✅ Prepared RAW feature matrix: {X_raw.shape}")
    print("   (NaNs preserved - will be handled in CV pipeline)")
    
    return X_raw, y_remapped, feature_cols, reverse_mapping


def load_and_merge_diagnosis(df_merged, diagnosis_path):
    """
    Load diagnosis file and merge diagnosis indicator columns into merged Acxiom dataframe.
    Returns dataframe with diagnosis columns added.
    """
    print("\nLoading diagnosis file:", diagnosis_path)
    df_diag = pd.read_csv(diagnosis_path, low_memory=False)

    # Candidate ID columns to consider
    id_candidates = ['sys_mbr_sk', 'clm_sys_mbr_sk', 'empi', 'member_id', 'acxiom_id', 'member_sk']

    # Normalize string columns in both dataframes
    for df_obj in (df_diag, df_merged):
        for col in list(df_obj.columns):
            if col in id_candidates:
                try:
                    df_obj[col] = df_obj[col].astype(str).str.strip()
                except Exception:
                    pass

    # Find common id column names
    # Prefer a common key that actually has overlapping IDs; otherwise fall back to demographic bridge
    common_ids = [c for c in id_candidates if (c in df_diag.columns and c in df_merged.columns)]
    if common_ids:
        # compute overlap counts and pick the best key (with >0 overlap)
        overlaps = {}
        for c in common_ids:
            try:
                s1 = set(df_diag[c].astype(str).str.strip().unique())
                s2 = set(df_merged[c].astype(str).str.strip().unique())
                overlaps[c] = len(s1 & s2)
            except Exception:
                overlaps[c] = 0

        # select key with maximum overlap
        best_key = max(overlaps, key=overlaps.get)
        if overlaps.get(best_key, 0) > 0:
            key = best_key
            print(f"Merging on common key with overlap: {key} (overlap={overlaps[key]})")
            df_out = df_merged.merge(df_diag, on=key, how='left', suffixes=(None, '_diag'))
            print(f"After merging diagnosis on {key}: {df_out.shape}")
            return df_out
        else:
            print(f"Found common keys {common_ids} but no overlapping IDs; will attempt demographic bridge.")

            # Attempt to find if any column in df_merged corresponds to demographics.empi
            try:
                dem = pd.read_csv('demographics.csv', usecols=['sys_mbr_sk', 'empi'])
                dem['empi'] = dem['empi'].astype(str).str.strip()
                dem_set = set(dem['empi'].unique())
                mapped_col = None
                for col in df_merged.columns:
                    try:
                        col_set = set(df_merged[col].astype(str).str.strip().unique())
                        if len(col_set & dem_set) > 0:
                            mapped_col = col
                            break
                    except Exception:
                        continue

                if mapped_col is not None:
                    print(f"Found df_merged column '{mapped_col}' that overlaps demographics.empi; using it to bridge via EMPI.")
                    df_merged['_empi_bridge'] = df_merged[mapped_col].astype(str).str.strip()
                    # merge diagnosis via empi if available
                    if 'empi' in df_diag.columns:
                        df_out = df_merged.merge(df_diag, left_on='_empi_bridge', right_on='empi', how='left', suffixes=(None, '_diag'))
                        print(f"After bridging via {mapped_col}->empi: {df_out.shape}")
                        return df_out
                    else:
                        # map empi -> sys_mbr_sk via demographics then merge on sys_mbr_sk
                        df_map = dem.drop_duplicates('empi')
                        df_map = df_map.rename(columns={'empi': '_empi_bridge'})
                        df_tmp = df_merged.merge(df_map, on='_empi_bridge', how='left')
                        if 'sys_mbr_sk' in df_tmp.columns and 'sys_mbr_sk' in df_diag.columns:
                            df_out = df_tmp.merge(df_diag, on='sys_mbr_sk', how='left', suffixes=(None, '_diag'))
                            print(f"After bridging via {mapped_col}->empi->sys_mbr_sk: {df_out.shape}")
                            return df_out
            except FileNotFoundError:
                print('demographics.csv not found; cannot attempt empi bridge')
            except Exception as e:
                print('Error while attempting empi-based bridge:', e)

    # Try matching by mapping via demographics (sys_mbr_sk <-> empi)
    try:
        dem = pd.read_csv('demographics.csv', usecols=['sys_mbr_sk', 'empi'])
        dem['sys_mbr_sk'] = dem['sys_mbr_sk'].astype(str).str.strip()
        dem['empi'] = dem['empi'].astype(str).str.strip()

        # If diag has empi and merged has sys_mbr_sk -> map empi->sys_mbr_sk and merge
        if 'empi' in df_diag.columns and 'sys_mbr_sk' in df_merged.columns:
            print("Bridging via demographics: df_diag.empi -> demographics -> sys_mbr_sk -> df_merged")
            df_diag_map = df_diag.merge(dem, on='empi', how='left')
            if 'sys_mbr_sk' in df_diag_map.columns:
                df_out = df_merged.merge(df_diag_map, on='sys_mbr_sk', how='left', suffixes=(None, '_diag'))
                print(f"After bridging via empi: {df_out.shape}")
                return df_out

        # If diag has member_id and merged has member_id, handled above; else try map member_id -> empi via demographics
        if 'member_id' in df_diag.columns and 'empi' in dem.columns:
            print("Attempting to map diag.member_id -> demographics.empi -> sys_mbr_sk")
            # If demographics contains member_id field, use it; otherwise skip
            if 'member_id' in dem.columns:
                dem_map = dem
            else:
                dem_map = dem
            # try to merge via empi if diag contains empi-like member ids
            if 'empi' in df_diag.columns:
                df_diag_map = df_diag.merge(dem_map, on='empi', how='left')
                if 'sys_mbr_sk' in df_diag_map.columns:
                    df_out = df_merged.merge(df_diag_map, on='sys_mbr_sk', how='left', suffixes=(None, '_diag'))
                    print(f"After mapping member_id via demographics: {df_out.shape}")
                    return df_out
    except FileNotFoundError:
        print("demographics.csv not found; skipping bridge attempts")
    except Exception as e:
        print(f"Error while attempting demographic bridge: {e}")

    # As a last resort, attempt a Cartesian left-join on stringified IDs if a plausible pair exists
    # e.g., df_diag has a column that matches df_merged's member-like column with different name
    for left_col in ['sys_mbr_sk', 'empi', 'member_id', 'acxiom_id', 'member_sk']:
        for right_col in ['sys_mbr_sk', 'empi', 'member_id', 'acxiom_id', 'member_sk']:
            if left_col in df_diag.columns and right_col in df_merged.columns:
                try:
                    print(f"Attempting merge: df_merged.{right_col} <- df_diag.{left_col}")
                    df_diag_temp = df_diag.rename(columns={left_col: right_col})
                    df_out = df_merged.merge(df_diag_temp, on=right_col, how='left', suffixes=(None, '_diag'))
                    print(f"After attempting merge on {right_col}: {df_out.shape}")
                    return df_out
                except Exception:
                    continue

    # If we reach here, we couldn't find a join key
    raise KeyError("No matching identifier found between merged Acxiom dataframe and diagnosis file.\n"
                   "Look for columns like sys_mbr_sk, empi, member_id in both files or provide a bridge via demographics.csv.")


def filter_acxiom_to_diag_cohort(df_merged, diagnosis_path):
    """
    Restrict df_merged to only rows whose member identifier appears in the
    diagnosis file. Attempts to find the best join key by overlap.

    Returns filtered df_merged (may be unchanged if no join found).
    """
    try:
        df_diag = pd.read_csv(diagnosis_path, low_memory=False)
    except FileNotFoundError:
        print(f"Diagnosis file not found: {diagnosis_path}; skipping cohort filtering.")
        return df_merged

    id_candidates = ['sys_mbr_sk', 'clm_sys_mbr_sk', 'empi', 'member_id', 'acxiom_id', 'member_sk']

    # normalize candidate columns
    for col in id_candidates:
        if col in df_diag.columns:
            try:
                df_diag[col] = df_diag[col].astype(str).str.strip()
            except Exception:
                pass
        if col in df_merged.columns:
            try:
                df_merged[col] = df_merged[col].astype(str).str.strip()
            except Exception:
                pass

    # find best overlapping key
    best_key = None
    best_overlap = 0
    for c in id_candidates:
        if c in df_diag.columns and c in df_merged.columns:
            try:
                s1 = set(df_diag[c].dropna().unique())
                s2 = set(df_merged[c].dropna().unique())
                ov = len(s1 & s2)
                if ov > best_overlap:
                    best_overlap = ov
                    best_key = c
            except Exception:
                continue

    if best_key and best_overlap > 0:
        print(f"Filtering Acxiom cohort to diagnosis IDs using key '{best_key}' (overlap={best_overlap})")
        diag_ids = set(df_diag[best_key].dropna().unique())
        mask = df_merged[best_key].astype(str).isin(diag_ids)
        filtered = df_merged[mask].copy()
        print(f"Cohort reduced: {len(df_merged):,} -> {len(filtered):,} rows")
        return filtered

    # attempt bridge via demographics (empi)
    try:
        dem = pd.read_csv('demographics.csv', usecols=['sys_mbr_sk', 'empi'])
        dem['empi'] = dem['empi'].astype(str).str.strip()
        dem_map = dem.drop_duplicates('empi').set_index('empi')['sys_mbr_sk'].to_dict()

        if 'empi' in df_diag.columns and any(col in df_merged.columns for col in ['sys_mbr_sk', 'empi', 'member_id']):
            diag_empi = df_diag['empi'].dropna().astype(str).unique()
            mapped_sys = [dem_map.get(e) for e in diag_empi if dem_map.get(e) is not None]
            mapped_set = set(mapped_sys)
            if mapped_set:
                if 'sys_mbr_sk' in df_merged.columns:
                    filtered = df_merged[df_merged['sys_mbr_sk'].astype(str).isin(mapped_set)].copy()
                    print(f"Filtered via demographics bridge: {len(df_merged):,} -> {len(filtered):,} rows")
                    return filtered
    except Exception:
        pass

    print('Could not determine a join key to restrict the Acxiom cohort to diagnosis IDs; leaving cohort as-is.')
    return df_merged


def balance_dataset(X, y, method='undersample', random_state=RANDOM_STATE):
    """Return balanced X,y according to method. Currently supports undersample."""
    if method is None or method == 'none':
        return X.reset_index(drop=True), y.reset_index(drop=True)

    if method == 'undersample':
        y = y.reset_index(drop=True)
        X = X.reset_index(drop=True)
        pos_idx = y[y == 1].index.values
        neg_idx = y[y == 0].index.values
        if len(pos_idx) == 0:
            return X, y
        if len(neg_idx) <= len(pos_idx):
            return X, y
        rng = np.random.RandomState(random_state)
        neg_sample = rng.choice(neg_idx, size=len(pos_idx), replace=False)
        keep = np.concatenate([pos_idx, neg_sample])
        X_bal = X.loc[keep].reset_index(drop=True)
        y_bal = y.loc[keep].reset_index(drop=True)
        return X_bal, y_bal

    # unknown method
    return X.reset_index(drop=True), y.reset_index(drop=True)


def classify_diagnoses(df, feature_cols, diag_prefix='dx_', n_folds=5, min_pos=10):
    """
    For each diagnosis code column in `df` (columns starting with `diag_prefix`),
    run cross-validated evaluation for all pipelines returned by `create_model_pipelines()`.

    Prints per-model scores for each diagnosis and returns a list of result dicts.
    """
    # find diagnosis columns
    diag_cols = [c for c in df.columns if c.startswith(diag_prefix)]
    print(f"\nFound {len(diag_cols)} diagnosis columns to evaluate (prefix={diag_prefix})")

    models = create_model_pipelines()
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)

    all_results = []
    for diag in diag_cols:
        # Build binary label (presence > 0)
        s = df[diag]
        # If boolean dtype, cast directly
        if pd.api.types.is_bool_dtype(s):
            y = s.astype(int)
        else:
            # Try numeric conversion first (counts)
            y_num = pd.to_numeric(s, errors='coerce')
            if y_num.notna().sum() > 0:
                y = (y_num.fillna(0) > 0).astype(int)
            else:
                # Fallback: interpret common truthy strings
                s_str = s.astype(str).str.strip().str.lower()
                truthy = s_str.isin(['true', 't', '1', 'yes', 'y', 'yess'])
                y = truthy.astype(int)

        # Skip if too few positives or negatives
        pos = int(y.sum())
        neg = int((y == 0).sum())
        if pos < min_pos or neg < min_pos:
            print(f"Skipping {diag}: insufficient samples (pos={pos}, neg={neg})")
            continue

        print(f"\nEvaluating diagnosis {diag}: pos={pos:,}, neg={neg:,}, total={len(y):,}")

        X = df[feature_cols].copy()

        diag_result = {'diag': diag, 'pos': pos, 'neg': neg, 'models': {}}
        best_model = None
        best_score = -1
        best_folds = None

        for model_name, pipeline in models.items():
            print(f"  Running model: {model_name}")
            scoring = {'f1': 'f1', 'roc_auc': 'roc_auc', 'accuracy': 'accuracy'}
            try:
                cv_res = cross_validate(
                    pipeline, X, y, cv=skf, scoring=scoring,
                    n_jobs=1, return_train_score=False, error_score='raise'
                )
            except Exception as e:
                print(f"    Error for {model_name}: {e}")
                continue

            mean_f1 = float(np.mean(cv_res['test_f1']))
            std_f1 = float(np.std(cv_res['test_f1']))
            mean_roc = float(np.mean(cv_res['test_roc_auc'])) if 'test_roc_auc' in cv_res else np.nan

            print(f"    {model_name}: F1={mean_f1:.3f}±{std_f1:.3f}, ROC-AUC={mean_roc:.3f}")

            diag_result['models'][model_name] = {
                'mean_f1': mean_f1,
                'std_f1': std_f1,
                'mean_roc': mean_roc,
                'folds': list(cv_res['test_f1'])
            }

            if mean_f1 > best_score:
                best_score = mean_f1
                best_model = model_name
                best_folds = list(cv_res['test_f1'])

        diag_result['best_model'] = best_model
        diag_result['best_mean_f1'] = best_score
        diag_result['best_folds'] = best_folds or []
        all_results.append(diag_result)

    return all_results


def create_diagnosis_groups(df, dx_cols, n_groups=10):
    """
    Partition diagnosis columns into n_groups to balance positive counts across groups.

    Args:
        df: DataFrame containing diagnosis columns
        dx_cols: list of dx_ column names
        n_groups: number of groups to create

    Returns:
        List of groups, each a list of dx column names
    """
    # compute positive counts per diagnosis
    counts = []
    for c in dx_cols:
        try:
            s = pd.to_numeric(df[c], errors='coerce').fillna(0)
            pos = int((s > 0).sum())
        except Exception:
            pos = 0
        counts.append((c, pos))

    # sort diagnoses by descending positives
    counts.sort(key=lambda x: x[1], reverse=True)

    # greedy assign to group with smallest current sum
    groups = [[] for _ in range(n_groups)]
    group_sums = [0] * n_groups

    for dx, cnt in counts:
        # pick group with minimum sum
        i = int(min(range(n_groups), key=lambda k: group_sums[k]))
        groups[i].append(dx)
        group_sums[i] += cnt

    # Remove empty groups if any
    groups = [g for g in groups if len(g) > 0]
    return groups


def create_balanced_groups(df, dx_cols, n_groups=10, target_frac=0.5, tolerance=0.02, unique=True):
    """
    Create up to `n_groups` collections of diagnosis codes such that each group's
    positive label (having any diagnosis in the collection) is approximately
    `target_frac` of the cohort. Greedy selection: iteratively add the dx code
    that maximizes the gain in positive coverage until target reached.

    Args:
        df: DataFrame containing dx columns
        dx_cols: list of dx_ column names
        n_groups: requested number of groups to build
        target_frac: desired fraction of cohort labeled positive per group
        tolerance: acceptable fractional deviation from target (unused for stopping)
        unique: if True, do not reuse dx codes across groups

    Returns:
        List of groups (each a list of dx column names)
    """
    N = len(df)
    target = int(round(target_frac * N))
    remaining = list(dx_cols)
    groups = []

    # Precompute boolean arrays for dx positivity
    dx_bool = {}
    for c in dx_cols:
        try:
            dx_bool[c] = (pd.to_numeric(df[c], errors='coerce').fillna(0) > 0).values
        except Exception:
            dx_bool[c] = (df[c].astype(str).str.strip().isin(['1','True','true','Y','y'])).values

    # Compute union coverage across all dx codes (max achievable positives)
    all_union = np.zeros(N, dtype=bool)
    for arr in dx_bool.values():
        all_union = all_union | arr
    max_achievable = int(all_union.sum())
    if max_achievable < target:
        print(f"⚠️ Requested target {target} (~{target_frac*100:.0f}%) is not achievable;"
              f" max positives across all dx codes = {max_achievable} ({max_achievable/N:.2%})."
              " Will aim to maximize coverage per group instead.")
        # set a softer target: try to reach as much as possible per group, but cap by average
        target = int(round(max_achievable / max(1, n_groups)))

    for gi in range(n_groups):
        if not remaining:
            break
        group = []
        mask = np.zeros(N, dtype=bool)
        current_pos = 0

        # Greedily add dx codes until we reach or exceed the target (or until no gain possible)
        while current_pos < target and remaining:
            # choose dx that brings coverage closest to target (minimize abs(new_count - target))
            best_dx = None
            best_dist = None
            best_new_count = None
            for dx in remaining:
                candidate_mask = mask | dx_bool[dx]
                new_count = int(candidate_mask.sum())
                if new_count == current_pos:
                    continue
                dist = abs(new_count - target)
                if best_dist is None or dist < best_dist or (dist == best_dist and new_count > best_new_count):
                    best_dist = dist
                    best_dx = dx
                    best_new_count = new_count

            if best_dx is None:
                # no further gain possible
                break

            # add best_dx
            group.append(best_dx)
            mask = mask | dx_bool[best_dx]
            current_pos = int(mask.sum())
            if unique and best_dx in remaining:
                remaining.remove(best_dx)

        # If group is empty (no dx gives any positives) break
        if len(group) == 0:
            break

        groups.append(group)

    return groups


def evaluate_diagnosis_groups(df, feature_cols, groups, n_folds=5, min_pos=5, balance_method='undersample'):
    """
    For each diagnosis group (list of dx columns), build a binary label (any dx present),
    optionally balance classes by undersampling negatives, run cross-validated evaluation
    across available pipelines, and return structured results.
    """
    models = create_model_pipelines()
    results = []

    for i, grp in enumerate(groups):
        label_name = f'group_{i+1}'
        # create binary label for group
        try:
            y_series = df[grp].apply(lambda row: (pd.to_numeric(row, errors='coerce').fillna(0) > 0).any(), axis=1).astype(int)
        except Exception:
            # fallback slower path
            y_series = (df[grp] > 0).any(axis=1).astype(int)

        pos = int(y_series.sum())
        neg = int((y_series == 0).sum())
        total = len(y_series)
        print(f"\nEvaluating group {i+1}/{len(groups)}: {len(grp)} dx codes, pos={pos}, neg={neg}, total={total}")

        if pos < min_pos or neg < min_pos:
            print(f"Skipping group_{i+1}: insufficient positives or negatives (pos={pos}, neg={neg})")
            continue

        X = df[feature_cols].copy()
        y = y_series.copy()

        # Balance via undersampling negatives to match positives
        if balance_method == 'undersample' and pos > 0:
            # get positive and negative indices
            pos_idx = y[y == 1].index
            neg_idx = y[y == 0].index
            if len(neg_idx) > len(pos_idx):
                neg_sample = np.random.RandomState(RANDOM_STATE).choice(neg_idx, size=len(pos_idx), replace=False)
                keep_idx = np.concatenate([pos_idx, neg_sample])
                X_bal = X.loc[keep_idx].reset_index(drop=True)
                y_bal = y.loc[keep_idx].reset_index(drop=True)
            else:
                X_bal = X.reset_index(drop=True)
                y_bal = y.reset_index(drop=True)
        else:
            X_bal = X.reset_index(drop=True)
            y_bal = y.reset_index(drop=True)

        # adjust folds
        valid_pos = int((y_bal == 1).sum())
        folds = min(n_folds, max(2, valid_pos))
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)

        group_result = {'group': label_name, 'dx_codes': grp, 'pos': int(pos), 'neg': int(neg), 'models': {}}
        best_model = None
        best_score = -1

        for model_name, pipeline in models.items():
            print(f"  Running model: {model_name}")
            scoring = {'f1': 'f1', 'roc_auc': 'roc_auc', 'accuracy': 'accuracy'}
            try:
                cv_res = cross_validate(pipeline, X_bal, y_bal, cv=skf, scoring=scoring, n_jobs=1, return_train_score=False, error_score='raise')
            except Exception as e:
                print(f"    Error for {model_name}: {e}")
                continue

            mean_f1 = float(np.mean(cv_res['test_f1']))
            std_f1 = float(np.std(cv_res['test_f1']))
            mean_roc = float(np.mean(cv_res['test_roc_auc'])) if 'test_roc_auc' in cv_res else np.nan

            print(f"    {model_name}: F1={mean_f1:.3f}±{std_f1:.3f}, ROC-AUC={mean_roc:.3f}")

            group_result['models'][model_name] = {
                'mean_f1': mean_f1,
                'std_f1': std_f1,
                'mean_roc': mean_roc,
                'folds': list(cv_res['test_f1'])
            }

            if mean_f1 > best_score:
                best_score = mean_f1
                best_model = model_name

        group_result['best_model'] = best_model
        group_result['best_mean_f1'] = best_score
        results.append(group_result)

    return results


def save_group_results(results, out_csv='diagnosis_group_results.csv'):
    rows = []
    for r in results:
        grp = r['group']
        dxs = ';'.join(r.get('dx_codes', []))
        for mname, m in r.get('models', {}).items():
            rows.append({
                'group': grp,
                'dx_codes': dxs,
                'model': mname,
                'mean_f1': m.get('mean_f1'),
                'std_f1': m.get('std_f1'),
                'mean_roc': m.get('mean_roc'),
                'folds': str(m.get('folds', [])),
                'best_model': (mname == r.get('best_model'))
            })
    if rows:
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(f"Saved diagnosis group results to: {out_csv}")
    else:
        print("No group results to save (empty).")


def plot_group_distributions(results, out_png='diagnosis_group_best_model_scores.png'):
    # Create boxplot of best-model fold scores per group
    labels = []
    data = []
    for r in results:
        bm = r.get('best_model')
        if not bm:
            continue
        folds = r['models'].get(bm, {}).get('folds', [])
        if folds:
            labels.append(r['group'])
            data.append(folds)

    if not data:
        print('No per-group fold scores to plot.')
        return

    plt.figure(figsize=(max(10, len(data) * 0.6), 6))
    plt.boxplot(data, labels=labels, showmeans=True)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('F1 (per CV fold)')
    plt.title('Best-model CV fold F1 distribution per diagnosis group')
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    print(f"Saved: {out_png}")



def save_top20_diagnoses(results, out_csv='top20_diagnosis_classification.csv'):
    # results: list of diag_result dicts
    df = pd.DataFrame([{
        'diag': r['diag'],
        'best_model': r['best_model'],
        'best_mean_f1': r['best_mean_f1'],
        'pos': r['pos'],
        'neg': r['neg']
    } for r in results])
    df_sorted = df.sort_values('best_mean_f1', ascending=False)
    top20 = df_sorted.head(20)
    top20.to_csv(out_csv, index=False)
    print(f"\nSaved top 20 diagnosis performances to: {out_csv}")
    return top20


def plot_best_model_distributions(results, out_png='diagnosis_best_model_scores.png'):
    # For each diagnosis, plot distribution (boxplot) of best model fold scores
    labels = [r['diag'] for r in results]
    data = [r['best_folds'] for r in results]
    # filter out empty
    labels, data = zip(*[(l, d) for l, d in zip(labels, data) if len(d) > 0]) if len(results) > 0 else ([], [])
    if len(data) == 0:
        print("No per-diagnosis fold scores to plot.")
        return

    plt.figure(figsize=(max(12, len(data) * 0.25), 6))
    plt.boxplot(data, labels=labels, showmeans=True)
    plt.xticks(rotation=90)
    plt.ylabel('F1 (per CV fold)')
    plt.title('Best model CV fold F1 distribution per diagnosis')
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    print(f"Saved diagnosis best-model score distributions to: {out_png}")


# =============================================================================
# Step 6: Cross-Validation Model Comparison
# =============================================================================

def create_model_pipelines():
    """
    Create LEAKAGE-FREE pipelines with all preprocessing inside.
    
    Returns:
        Dictionary of model names and their pipeline objects
    """
    models = {}
    
    print("\n🔒 Creating leakage-free pipelines...")
    print("   All preprocessing (column filtering, imputation, scaling)")
    print("   happens INSIDE cross-validation folds")
    
    # Random Forest
    models['Random Forest'] = Pipeline([
        ('dropper', ColumnDropper(missing_threshold=MISSING_THRESHOLD, 
                                   variance_threshold=VARIANCE_THRESHOLD)),
        ('imputer', KNNImputer(n_neighbors=5, weights='uniform')),
        ('scaler', StandardScaler()),
        ('model', RandomForestClassifier(**RF_PARAMS))
    ])
    
    # Gradient Boosting
    models['Gradient Boosting'] = Pipeline([
        ('dropper', ColumnDropper(missing_threshold=MISSING_THRESHOLD, 
                                   variance_threshold=VARIANCE_THRESHOLD)),
        ('imputer', KNNImputer(n_neighbors=5, weights='uniform')),
        ('scaler', StandardScaler()),
        ('model', GradientBoostingClassifier(**GB_PARAMS))
    ])
    
    # LightGBM (if available)
    if LIGHTGBM_AVAILABLE:
        models['LightGBM'] = Pipeline([
            ('dropper', ColumnDropper(missing_threshold=MISSING_THRESHOLD, 
                                       variance_threshold=VARIANCE_THRESHOLD)),
            ('imputer', KNNImputer(n_neighbors=5, weights='uniform')),
            ('scaler', StandardScaler()),
            ('model', lgb.LGBMClassifier(**LGBM_PARAMS))
        ])
    
    # XGBoost (if available)
    if XGBOOST_AVAILABLE:
        models['XGBoost'] = Pipeline([
            ('dropper', ColumnDropper(missing_threshold=MISSING_THRESHOLD, 
                                       variance_threshold=VARIANCE_THRESHOLD)),
            ('imputer', KNNImputer(n_neighbors=5, weights='uniform')),
            ('scaler', StandardScaler()),
            ('model', xgb.XGBClassifier(**XGB_PARAMS))
        ])
    
    # CatBoost (if available)
    if CATBOOST_AVAILABLE:
        models['CatBoost'] = Pipeline([
            ('dropper', ColumnDropper(missing_threshold=MISSING_THRESHOLD, 
                                       variance_threshold=VARIANCE_THRESHOLD)),
            ('imputer', KNNImputer(n_neighbors=5, weights='uniform')),
            ('scaler', StandardScaler()),
            ('model', CatBoostClassifier(**CATBOOST_PARAMS))
        ])
    
    return models


def cross_validate_models(X, y, models, n_folds=N_CV_FOLDS):
    """
    Perform stratified k-fold cross-validation for all models.
    
    Args:
        X: Feature matrix
        y: Target vector (0-indexed)
        models: Dictionary of model pipelines
        n_folds: Number of CV folds
    
    Returns:
        DataFrame with CV results for each model
    """
    print("\n" + "=" * 70)
    print("Step 6: Cross-Validation Model Comparison")
    print("=" * 70)
    
    # Check class distribution
    unique_classes = np.unique(y)
    print(f"\nClasses present (0-indexed): {unique_classes}")
    print(f"Class distribution:")
    for cls in unique_classes:
        count = (y == cls).sum()
        pct = count / len(y) * 100
        print(f"  Class {int(cls)}: {count:,} samples ({pct:.1f}%)")
    
    # Check if we have enough samples per class
    min_samples_per_class = min((y == cls).sum() for cls in unique_classes)
    if min_samples_per_class < n_folds:
        n_folds = max(2, min_samples_per_class)
        print(f"\n⚠️ Reducing n_folds to {n_folds} due to small class size")
    
    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    
    # Determine scoring based on number of classes
    if len(unique_classes) == 2:
        scoring = {
            'accuracy': 'accuracy',
            'precision': 'precision',
            'recall': 'recall',
            'f1': 'f1',
            'roc_auc': 'roc_auc'
        }
    else:
        scoring = {
            'accuracy': 'accuracy',
            'precision': 'precision_macro',
            'recall': 'recall_macro',
            'f1': 'f1_macro',
            'roc_auc': 'roc_auc_ovr'
        }
    
    results = []
    
    print(f"\n{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'ROC-AUC':<12}")
    print("=" * 88)
    
    for model_name, pipeline in models.items():
        print(f"\nTraining {model_name}...")
        
        try:
            # Perform cross-validation
            cv_results = cross_validate(
                pipeline, X, y,
                cv=skf,
                scoring=scoring,
                n_jobs=1,  # Changed from -1 to avoid multiprocessing issues
                return_train_score=True,
                error_score='raise',
                verbose=0
            )
            
            # Calculate mean and std for each metric
            result = {
                'Model': model_name,
                'CV_Accuracy_Mean': cv_results['test_accuracy'].mean(),
                'CV_Accuracy_Std': cv_results['test_accuracy'].std(),
                'CV_Precision_Mean': cv_results['test_precision'].mean(),
                'CV_Precision_Std': cv_results['test_precision'].std(),
                'CV_Recall_Mean': cv_results['test_recall'].mean(),
                'CV_Recall_Std': cv_results['test_recall'].std(),
                'CV_F1_Mean': cv_results['test_f1'].mean(),
                'CV_F1_Std': cv_results['test_f1'].std(),
                'CV_ROC_AUC_Mean': cv_results['test_roc_auc'].mean(),
                'CV_ROC_AUC_Std': cv_results['test_roc_auc'].std(),
                'Train_Accuracy_Mean': cv_results['train_accuracy'].mean(),
                'Train_Accuracy_Std': cv_results['train_accuracy'].std(),
                'Overfit_Gap': cv_results['train_accuracy'].mean() - cv_results['test_accuracy'].mean()
            }
            results.append(result)
            
            # Print summary
            print(f"{model_name:<20} "
                  f"{result['CV_Accuracy_Mean']:.3f}±{result['CV_Accuracy_Std']:.3f}  "
                  f"{result['CV_Precision_Mean']:.3f}±{result['CV_Precision_Std']:.3f}  "
                  f"{result['CV_Recall_Mean']:.3f}±{result['CV_Recall_Std']:.3f}  "
                  f"{result['CV_F1_Mean']:.3f}±{result['CV_F1_Std']:.3f}  "
                  f"{result['CV_ROC_AUC_Mean']:.3f}±{result['CV_ROC_AUC_Std']:.3f}")
            
            if result['Overfit_Gap'] > 0.15:
                print(f"  🔴 Severe overfitting (gap: {result['Overfit_Gap']:.3f})")
            elif result['Overfit_Gap'] > 0.10:
                print(f"  ⚠️ Moderate overfitting (gap: {result['Overfit_Gap']:.3f})")
            elif result['Overfit_Gap'] > 0.05:
                print(f"  ⚡ Slight overfitting (gap: {result['Overfit_Gap']:.3f})")
            else:
                print(f"  ✅ Good generalization (gap: {result['Overfit_Gap']:.3f})")
            
        except Exception as e:
            print(f"  ❌ Error training {model_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    if not results:
        print("\n❌ No models completed successfully!")
        return pd.DataFrame()
    
    results_df = pd.DataFrame(results)
    
    # Sort by F1 score
    results_df = results_df.sort_values('CV_F1_Mean', ascending=False)
    
    print("\n" + "=" * 70)
    print("Cross-Validation Results Summary")
    print("=" * 70)
    print(results_df[['Model', 'CV_Accuracy_Mean', 'CV_F1_Mean', 'CV_ROC_AUC_Mean', 'Overfit_Gap']].to_string(index=False))
    
    # Print interpretation
    print("\n📊 Overfitting Analysis:")
    print("  ✅ Gap < 0.05: Excellent generalization")
    print("  ⚡ Gap 0.05-0.10: Acceptable")
    print("  ⚠️ Gap 0.10-0.15: Concerning")
    print("  🔴 Gap > 0.15: Severe overfitting - model not reliable")
    
    return results_df


def train_best_model_and_get_importances(X, y, feature_names, cv_results_df):
    """
    Train the best model on full data and extract feature importances.
    
    Args:
        X: Feature matrix
        y: Target vector
        feature_names: List of feature names
        cv_results_df: DataFrame with CV results
    
    Returns:
        Feature importances, trained model, test metrics
    """
    print("\n" + "=" * 70)
    print("Step 7: Training Best Model and Extracting Feature Importances")
    print("=" * 70)
    
    # Get best model
    best_model_name = cv_results_df.iloc[0]['Model']
    print(f"\nBest model (by F1-score): {best_model_name}")
    print(f"CV F1-Score: {cv_results_df.iloc[0]['CV_F1_Mean']:.3f} ± {cv_results_df.iloc[0]['CV_F1_Std']:.3f}")
    
    # Create and train the best model pipeline
    models = create_model_pipelines()
    best_pipeline = models[best_model_name]
    
    # Fit on all data for feature importance extraction
    print("\nFitting best model on full dataset for feature importance extraction...")
    best_pipeline.fit(X, y)
    
    # Extract the trained model from pipeline
    trained_model = best_pipeline.named_steps['model']
    
    # Get feature importances (if available)
    importances_df = None
    if hasattr(trained_model, 'feature_importances_'):
        importances = trained_model.feature_importances_
        n_imp = len(importances)
        # Align feature names length with importances length
        if len(feature_names) != n_imp:
            print(f"\n⚠️ Warning: feature_names length ({len(feature_names)}) != importances length ({n_imp}). Aligning...")
            if len(feature_names) > n_imp:
                feature_list = feature_names[:n_imp]
            else:
                # pad missing names
                feature_list = list(feature_names) + [f'feature_{i}' for i in range(len(feature_names), n_imp)]
        else:
            feature_list = feature_names

        importances_df = pd.DataFrame({
            'feature': feature_list,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        print("\n" + "=" * 70)
        print(f"Top {TOP_N_FEATURES} Most Important Features")
        print("=" * 70)
        print(importances_df.head(TOP_N_FEATURES).to_string(index=False))
    else:
        print(f"\n⚠️ {best_model_name} does not provide feature importances.")
    
    return importances_df, best_pipeline, best_model_name


# =============================================================================
# Step 8: Visualize Model Comparison
# =============================================================================

def plot_model_comparison(cv_results_df):
    """
    Plot comparison of model performance.
    
    Args:
        cv_results_df: DataFrame with CV results
    """
    print("\n" + "=" * 70)
    print("Step 8: Visualizing Model Comparison")
    print("=" * 70)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: F1-Score comparison
    ax1 = axes[0]
    models = cv_results_df['Model']
    f1_means = cv_results_df['CV_F1_Mean']
    f1_stds = cv_results_df['CV_F1_Std']
    
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(models)))
    bars = ax1.barh(range(len(models)), f1_means, xerr=f1_stds, 
                     color=colors, capsize=5)
    ax1.set_yticks(range(len(models)))
    ax1.set_yticklabels(models)
    ax1.set_xlabel('F1-Score', fontsize=12)
    ax1.set_title('Model Comparison: F1-Score (Mean ± Std)', fontsize=13, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    ax1.invert_yaxis()
    
    # Add value labels
    for i, (mean, std) in enumerate(zip(f1_means, f1_stds)):
        ax1.text(mean + std + 0.01, i, f'{mean:.3f}', 
                va='center', fontsize=10, fontweight='bold')
    
    # Plot 2: Overfitting analysis
    ax2 = axes[1]
    overfit_gaps = cv_results_df['Overfit_Gap']
    colors2 = ['red' if gap > 0.1 else 'green' for gap in overfit_gaps]
    
    bars = ax2.barh(range(len(models)), overfit_gaps, color=colors2, alpha=0.7)
    ax2.set_yticks(range(len(models)))
    ax2.set_yticklabels(models)
    ax2.set_xlabel('Overfit Gap (Train - Test Accuracy)', fontsize=12)
    ax2.set_title('Overfitting Analysis', fontsize=13, fontweight='bold')
    ax2.axvline(x=0.1, color='orange', linestyle='--', linewidth=2, label='Warning threshold')
    ax2.grid(axis='x', alpha=0.3)
    ax2.legend()
    ax2.invert_yaxis()
    
    # Add value labels
    for i, gap in enumerate(overfit_gaps):
        ax2.text(gap + 0.005, i, f'{gap:.3f}', 
                va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: model_comparison.png")
    plt.show()


# =============================================================================
# Step 9: Visualize Feature Importances
# =============================================================================

def plot_feature_importances(importances_df, top_n=TOP_N_FEATURES):
    """
    Plot top N feature importances.
    
    Args:
        importances_df: DataFrame with feature and importance columns
        top_n: Number of top features to plot
    """
    if importances_df is None:
        print("\n⚠️ No feature importances available to plot.")
        return
    
    print("\n" + "=" * 70)
    print("Step 9: Visualizing Feature Importances")
    print("=" * 70)
    
    top_features = importances_df.head(top_n)
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(top_features)), top_features['importance'], color='steelblue')
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title(f'Top {top_n} Feature Importances for ED Utilization Prediction', 
              fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('feature_importances.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: feature_importances.png")
    plt.show()


# =============================================================================
# Step 10: Correlation Heatmap for Top Features
# =============================================================================

def plot_correlation_heatmap(df, top_features, target_col='ed_utilization_class', 
                             top_n=TOP_N_FEATURES):
    """
    Plot correlation heatmap for top features and ED utilization target.
    
    Args:
        df: Full DataFrame with features and target
        top_features: DataFrame with feature importances
        target_col: Name of target column
        top_n: Number of top features to include
    """
    if top_features is None:
        print("\n⚠️ No feature importances available for correlation heatmap.")
        return
    
    print("\n" + "=" * 70)
    print("Step 10: Creating Correlation Heatmap")
    print("=" * 70)
    
    # Get top N feature names
    top_feature_names = top_features.head(top_n)['feature'].tolist()
    
    # Add target column
    cols_to_correlate = top_feature_names + [target_col]
    
    # Extract relevant columns and convert to numeric
    df_corr = df[cols_to_correlate].copy()
    for col in df_corr.columns:
        df_corr[col] = pd.to_numeric(df_corr[col], errors='coerce')
    
    # Calculate correlation matrix
    corr_matrix = df_corr.corr()
    
    # Plot heatmap
    plt.figure(figsize=(14, 12))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                vmin=-1, vmax=1)
    plt.title(f'Correlation Heatmap: Top {top_n} Features vs ED Utilization', 
              fontsize=14, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: correlation_heatmap.png")
    plt.show()
    
    # Print correlations with target
    print("\n" + "=" * 70)
    print(f"Correlations with ED Utilization Class")
    print("=" * 70)
    target_corr = corr_matrix[target_col].drop(target_col).sort_values(ascending=False)
    print(target_corr.to_string())


# =============================================================================
# Step 11: Additional Visualizations
# =============================================================================

def plot_cv_metrics_boxplot(models, X, y, n_folds=N_CV_FOLDS):
    """
    Create boxplots showing distribution of CV metrics across folds.
    
    Args:
        models: Dictionary of model pipelines
        X: Feature matrix
        y: Target vector
        n_folds: Number of CV folds
    """
    print("\n" + "=" * 70)
    print("Step 11: Creating CV Metrics Distribution Plot")
    print("=" * 70)
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    
    # Collect scores for each model
    all_scores = []
    for model_name, pipeline in models.items():
        try:
            scores = cross_val_score(pipeline, X, y, cv=skf, scoring='f1_macro', n_jobs=1)
            for score in scores:
                all_scores.append({'Model': model_name, 'F1-Score': score})
        except:
            continue
    
    if not all_scores:
        print("⚠️ Could not generate CV metrics distribution.")
        return
    
    scores_df = pd.DataFrame(all_scores)
    
    plt.figure(figsize=(12, 6))
    models_list = scores_df['Model'].unique()
    positions = range(len(models_list))
    
    bp = plt.boxplot([scores_df[scores_df['Model'] == m]['F1-Score'].values 
                       for m in models_list],
                      positions=positions,
                      labels=models_list,
                      patch_artist=True,
                      notch=True)
    
    # Color the boxes
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(models_list)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    plt.ylabel('F1-Score', fontsize=12)
    plt.title(f'Distribution of F1-Scores Across {n_folds} CV Folds', 
              fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('cv_metrics_distribution.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: cv_metrics_distribution.png")
    plt.show()
    
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', square=True,
                xticklabels=['No visits', 'One visit', 'High (2+)'],
                yticklabels=['No visits', 'One visit', 'High (2+)'])
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix: ED Utilization Classification', 
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: confusion_matrix.png")
    plt.show()


def plot_ed_utilization_distribution(df):
    """
    Plot distribution of ED utilization classes.
    
    Args:
        df: DataFrame with ed_utilization_class column
    """
    print("\nCreating ED Utilization Distribution Plot...")
    
    class_counts = df['ed_utilization_class'].value_counts().sort_index()
    class_labels = ['No visits\n(Class 0)', 'One visit\n(Class 1)', 'High (2+) visits\n(Class 2)']
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(class_counts)), class_counts.values, color=['#2ecc71', '#f39c12', '#e74c3c'])
    plt.xticks(range(len(class_counts)), class_labels)
    plt.ylabel('Number of Patients', fontsize=12)
    plt.title('Distribution of ED Utilization Classes', fontsize=14, fontweight='bold')
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        pct = height / class_counts.sum() * 100
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}\n({pct:.1f}%)',
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('ed_utilization_distribution.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: ed_utilization_distribution.png")
    plt.show()


# =============================================================================
# Main Execution
# =============================================================================

def main():
    """
    Main execution function.
    """
    print("\n" + "=" * 70)
    print("ED UTILIZATION ANALYSIS WITH SDoH FEATURES")
    print("=" * 70)
    
    # Step 1: Calculate ED visits
    ed_visits = calculate_ed_visits(NYU_EDU_PATH)
    
    # Step 2: Create utilization labels
    ed_labels = create_ed_utilization_labels(ed_visits)
    
    # Step 3: Merge with Acxiom data
    df_merged = merge_ed_labels_with_acxiom(ed_labels, ACXIOM_PATH, OUTPUT_PATH)

    # Optionally restrict the Acxiom/SDoH cohort to only those members
    # present in the diagnosis file (prevents using full_acxiom population).
    if FILTER_TO_DIAG_IDS:
        df_merged = filter_acxiom_to_diag_cohort(df_merged, DIAG_PATH)
    
    # Step 4: Identify SDoH columns
    sdoh_cols = identify_sdoh_columns(df_merged)
    
    # Check if we have enough data
    if len(sdoh_cols) == 0:
        print("\n⚠️ WARNING: No SDoH columns found!")
        print("The analysis will proceed with diagnosis features only.")
    
    # Data quality warnings
    print("\n" + "=" * 70)
    print("⚠️  CRITICAL DATA QUALITY WARNINGS")
    print("=" * 70)
    print("\n1. TEMPORAL LEAKAGE RISK:")
    print("   Your dx_* features may be from the SAME TIME PERIOD as ED visits")
    print("   This causes the model to look great in CV but fail in production")
    print("   RECOMMENDATION: Use only PRE-INDEX features for prediction")
    print("   • Set exclude_dx_for_temporal=True to test without dx features")
    print("   • Better: create time windows (6mo before → predict next 6mo)")
    
    print("\n2. COHORT DEFINITION ISSUE:")
    print("   'No ED visits' may actually mean:")
    print("   • Incomplete ED file coverage")
    print("   • Member not enrolled during observation window")
    print("   • ID linkage failure")
    print("   RECOMMENDATION: Define 'fully observed' cohort")
    print("   • Require continuous enrollment")
    print("   • Verify ED data completeness")
    print("   • Label only members with known observation window")
    
    print("\n3. SAMPLE SIZE:")
    print(f"   You have ~{len(df_merged):,} samples for {len(sdoh_cols)} features")
    if len(df_merged) < 500:
        print("   ⚠️ SMALL DATASET - high risk of overfitting even with regularization")
    
    print("\n" + "=" * 70)
    
    # Step 5: Prepare ML data (TWO OPTIONS)
    use_temporal_safe_features = False  # Set to True to exclude dx_*
    
    X_raw, y, feature_names, class_mapping = prepare_ml_data(
        df_merged, sdoh_cols, 
        include_diagnosis=True,
        exclude_dx_for_temporal=use_temporal_safe_features
    )
    
    if len(X_raw) == 0 or len(feature_names) == 0:
        print("\n❌ ERROR: No valid features or samples for analysis.")
        print("Please check your data files.")
        return

    # ------------------------------------------------------------------
    # Optional: Use diagnosis file as labels and evaluate classifiers per dx
    # ------------------------------------------------------------------
    if RUN_PER_DIAG:
        try:
            df_with_dx = load_and_merge_diagnosis(df_merged, DIAG_PATH)

            # Build a single-row-per-member diagnosis table aggregated across records,
            # then reindex to match the ML subset (X_raw) so df_paired aligns with X_raw rows.
            dx_cols = [c for c in df_with_dx.columns if isinstance(c, str) and c.startswith('dx_')]

            # Choose join key present in both merged and diagnosis tables
            join_key = None
            for cand in ['member_id', 'sys_mbr_sk', 'empi', 'acxiom_id']:
                if cand in df_with_dx.columns and cand in df_merged.columns:
                    join_key = cand
                    break

            if join_key is None:
                # fallback: try sys_mbr_sk in diagnosis table mapping via demographics
                join_key = 'sys_mbr_sk' if 'sys_mbr_sk' in df_with_dx.columns else None

            if join_key is None:
                raise KeyError('No common join key found between merged Acxiom and diagnosis tables')

            # Normalize join key strings
            df_with_dx[join_key] = df_with_dx[join_key].astype(str).str.strip()
            df_merged[join_key] = df_merged[join_key].astype(str).str.strip() if join_key in df_merged.columns else None

            # Convert dx columns to 0/1 presence and aggregate per member (max across rows)
            if len(dx_cols) == 0:
                raise ValueError('No dx_ columns found in diagnosis file')

            for c in dx_cols:
                df_with_dx[c] = pd.to_numeric(df_with_dx[c], errors='coerce').fillna(0).astype(int)

            df_diag_agg = df_with_dx.groupby(join_key)[dx_cols].max().reset_index()

            # Build paired_ids from df_merged corresponding to X_raw rows
            try:
                paired_ids = df_merged.loc[X_raw.index, join_key].astype(str).str.strip()
            except Exception:
                paired_ids = df_merged[join_key].astype(str).str.strip()

            # Reindex aggregated diagnosis to match ML subset order; missing members -> zeros
            df_paired = df_diag_agg.set_index(join_key).reindex(paired_ids.values).reset_index()
            # fill missing dx cols with 0
            for c in dx_cols:
                if c in df_paired.columns:
                    df_paired[c] = df_paired[c].fillna(0).astype(int)

            # Ensure df_paired has same length as X_raw
            if len(df_paired) != len(X_raw):
                print(f"\nNote: df_paired length ({len(df_paired)}) != ML subset length ({len(X_raw)}).")

            print(f"\nUsing aggregated per-member diagnosis table: {df_paired.shape}")

            # Use the SDoH feature columns to predict diagnoses on the paired subset
            diagnosis_results = classify_diagnoses(df_paired, feature_names, diag_prefix='dx_', n_folds=N_CV_FOLDS, min_pos=5)

            # Save detailed per-diagnosis results (all model fold scores and summary)
            def save_detailed_diagnosis_results(results, out_csv='diagnosis_detailed_results.csv'):
                # Flatten results into rows: diag, model, mean_f1, std_f1, mean_roc, folds (as string), best_model_flag
                rows = []
                for r in results:
                    diag = r.get('diag')
                    best_m = r.get('best_model')
                    for mname, minfo in r.get('models', {}).items():
                        rows.append({
                            'diag': diag,
                            'model': mname,
                            'mean_f1': minfo.get('mean_f1'),
                            'std_f1': minfo.get('std_f1'),
                            'mean_roc': minfo.get('mean_roc'),
                            'folds': str(minfo.get('folds', [])),
                            'best_model': (mname == best_m)
                        })
                if rows:
                    pd.DataFrame(rows).to_csv(out_csv, index=False)
                    print(f"\nSaved detailed diagnosis results to: {out_csv}")
                else:
                    print("\nNo diagnosis results to save (empty).")

            save_detailed_diagnosis_results(diagnosis_results, out_csv='diagnosis_detailed_results.csv')

            # Save top 20 performing diagnoses (robust to empty results)
            try:
                top20 = save_top20_diagnoses(diagnosis_results, out_csv='top20_diagnosis_classification.csv')
            except Exception:
                print('\nNo top-20 diagnoses to save (insufficient results).')

            # Plot distributions of best-model fold scores per diagnosis
            try:
                plot_best_model_distributions(diagnosis_results, out_png='diagnosis_best_model_scores.png')
            except Exception:
                print('\nCould not plot diagnosis best-model score distributions (no data).')

            # --- Create and evaluate grouped-disease classification tasks (to reduce imbalance) ---
            try:
                dx_cols = [c for c in df_paired.columns if c.startswith('dx_')]
                if len(dx_cols) > 0:
                    print(f"\nCreating up to 10 balanced diagnosis groups (target ~50% positives)...")
                    groups = create_balanced_groups(df_paired, dx_cols, n_groups=10, target_frac=0.5)
                    print(f"Built {len(groups)} groups; evaluating group-level predictions (NO undersampling)...")

                    group_results = evaluate_diagnosis_groups(df_paired, feature_names, groups, n_folds=N_CV_FOLDS, min_pos=5, balance_method=BALANCE_METHOD)
                    save_group_results(group_results, out_csv='diagnosis_group_results.csv')
                    plot_group_distributions(group_results, out_png='diagnosis_group_best_model_scores.png')
                    # Optionally replace main ML target with a selected diagnosis-group label
                    if TRAIN_ON_GROUP and len(groups) > 0:
                        if TRAIN_GROUP_INDEX < 0 or TRAIN_GROUP_INDEX >= len(groups):
                            print(f"TRAIN_GROUP_INDEX {TRAIN_GROUP_INDEX} out of range (0..{len(groups)-1}); skipping train-on-group.")
                        else:
                            picked = groups[TRAIN_GROUP_INDEX]
                            print(f"Replacing ED target with diagnosis-group #{TRAIN_GROUP_INDEX} ({len(picked)} dx codes) for training.")
                            # build binary label aligned with X_raw (df_paired rows correspond to X_raw)
                            try:
                                y_group = df_paired[picked].apply(lambda row: (pd.to_numeric(row, errors='coerce').fillna(0) > 0).any(), axis=1).astype(int)
                            except Exception:
                                y_group = df_paired[picked].any(axis=1).astype(int)

                            print(f"Group positive count before balancing: {int(y_group.sum())}, total: {len(y_group)}")
                            X_bal, y_bal = balance_dataset(X_raw, y_group, method=BALANCE_METHOD)
                            print(f"After balancing ({BALANCE_METHOD}): pos={int((y_bal==1).sum())}, neg={int((y_bal==0).sum())}, total={len(y_bal)}")
                            # replace main training set
                            X_raw = X_bal
                            y = y_bal
                else:
                    print('No dx_ columns found for grouping.')
            except Exception as e:
                print(f'Error during grouped-disease evaluation: {e}')
        except FileNotFoundError:
            print("\nDiagnosis file not found; skipping per-diagnosis classification.")
        except Exception as e:
            print(f"\nError during per-diagnosis classification: {e}")
    else:
        print('\nSkipping per-diagnosis classification (RUN_PER_DIAG=False)')
    
    # Step 6: Create model pipelines and perform cross-validation
    models = create_model_pipelines()
    cv_results_df = cross_validate_models(X_raw, y, models, n_folds=N_CV_FOLDS)
    
    # Save CV results
    cv_results_df.to_csv('cv_results.csv', index=False)
    print("\n✅ Saved: cv_results.csv")
    
    # Step 7: Train best model and get feature importances
    importances, best_model, best_model_name = train_best_model_and_get_importances(
        X_raw, y, feature_names, cv_results_df
    )
    
    # Step 8: Model comparison visualization
    plot_model_comparison(cv_results_df)
    
    # Step 9: Plot feature importances
    plot_feature_importances(importances, top_n=TOP_N_FEATURES)
    
    # Step 10: Correlation heatmap
    plot_correlation_heatmap(df_merged, importances, top_n=TOP_N_FEATURES)
    
    # Step 11: CV metrics distribution
    plot_cv_metrics_boxplot(models, X_raw, y, n_folds=N_CV_FOLDS)
    
    # Step 12: ED utilization distribution
    plot_ed_utilization_distribution(df_merged)
    
    # Summary
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\n✅ Output files generated:")
    print(f"   - {OUTPUT_PATH}")
    print(f"   - cv_results.csv")
    print(f"   - model_comparison.png")
    print(f"   - feature_importances.png")
    print(f"   - correlation_heatmap.png")
    print(f"   - cv_metrics_distribution.png")
    print(f"   - ed_utilization_distribution.png")
    
    print(f"\n📊 Key Results:")
    print(f"   - Total patients: {len(df_merged):,}")
    print(f"   - Features analyzed: {len(feature_names)}")
    print(f"   - Best model: {best_model_name}")
    print(f"   - CV F1-Score: {cv_results_df.iloc[0]['CV_F1_Mean']:.3f} ± {cv_results_df.iloc[0]['CV_F1_Std']:.3f}")
    print(f"   - CV Accuracy: {cv_results_df.iloc[0]['CV_Accuracy_Mean']:.3f} ± {cv_results_df.iloc[0]['CV_Accuracy_Std']:.3f}")
    if importances is not None:
        print(f"   - Top feature: {importances.iloc[0]['feature']} (importance: {importances.iloc[0]['importance']:.4f})")
    
    print("\n💡 Interpretation:")
    best_overfit = cv_results_df.iloc[0]['Overfit_Gap']
    if best_overfit < 0.05:
        print(f"   ✅ Model shows good generalization (overfit gap: {best_overfit:.3f})")
    elif best_overfit < 0.1:
        print(f"   ⚠️ Model shows slight overfitting (overfit gap: {best_overfit:.3f})")
    else:
        print(f"   🔴 Model shows significant overfitting (overfit gap: {best_overfit:.3f})")
        print("      Consider: more regularization, more data, or simpler model")
    
    print("\n" + "=" * 70)
    print("🔬 NEXT STEPS FOR PRODUCTION-READY MODEL:")
    print("=" * 70)
    print("1. Implement temporal validation:")
    print("   • Define index date for each patient")
    print("   • Features: 6-12 months BEFORE index")
    print("   • Label: ED visits in 6 months AFTER index")
    print("   • Split: train on early time, test on later time")
    print("\n2. Define clean cohort:")
    print("   • Require continuous enrollment")
    print("   • Verify complete ED capture")
    print("   • Exclude members with data quality issues")
    print("\n3. Validate on truly held-out data:")
    print("   • Different time period")
    print("   • Different geographic area")
    print("   • Track performance drift over time")
    print("\n" + "=" * 70)
    

if __name__ == "__main__":
    main()

