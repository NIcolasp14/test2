"""
diagnosis_prediction.py

Predict Diagnosis Presence from SDoH Features
----------------------------------------------
Creates balanced prediction tasks using combinations of diagnoses.
NO temporal leakage - diagnoses are separate from features.

Strategy:
- Find patients in both full_acxiom.csv and diagnosis.csv
- Create 10 random diagnosis combinations for balanced labels (~50% prevalence)
- Run leakage-free CV on each combination
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
from pathlib import Path
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    accuracy_score, roc_auc_score
)
import warnings
warnings.filterwarnings('ignore')

# Try importing optional models
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("⚠️ LightGBM not available. Install with: pip install lightgbm")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️ XGBoost not available. Install with: pip install xgboost")

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("⚠️ CatBoost not available. Install with: pip install catboost")


# =============================================================================
# Configuration
# =============================================================================

ACXIOM_PATH = "full_acxiom.csv"
DIAGNOSIS_PATH = "diagnosis.csv"
OUTPUT_PATH = "diagnosis_prediction_results.csv"
BASE_VIZ_FOLDER = "all_visualisations"
COMPARATIVE_FOLDER = "comparative_visualisations"

# Create base folders
Path(BASE_VIZ_FOLDER).mkdir(exist_ok=True)
Path(COMPARATIVE_FOLDER).mkdir(exist_ok=True)

# Target diagnosis codes (from user's list)
TARGET_DIAGNOSES = [
    'I10', 'E78.5', 'Z23', 'Z00.00', 'E78.2', 'Z12.11', 'Z79.899', 'Z13.31',
    'K21.9', 'Z12.31', 'Z01.30', 'Z13.89', 'M25.13', 'M52.4', 'Z73.0',
    'Z13.9', 'T91.81', 'E78.00', 'Z87.891', 'E55.9'
]

# Model parameters (aggressive regularization)
RANDOM_STATE = 42
N_CV_FOLDS = 5

RF_PARAMS = {
    'n_estimators': 50,
    'max_depth': 4,
    'min_samples_split': 20,
    'min_samples_leaf': 10,
    'max_features': 'sqrt',
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
    'class_weight': 'balanced',
    'max_samples': 0.7
}

GB_PARAMS = {
    'n_estimators': 50,
    'max_depth': 3,
    'learning_rate': 0.01,
    'min_samples_split': 20,
    'min_samples_leaf': 10,
    'max_features': 'sqrt',
    'random_state': RANDOM_STATE,
    'subsample': 0.7
}

VARIANCE_THRESHOLD = 1e-5
MISSING_THRESHOLD = 0.95


# =============================================================================
# Custom Transformers (NO DATA LEAKAGE)
# =============================================================================

class ColumnDropper(BaseEstimator, TransformerMixin):
    """Remove columns with too many missing values or zero variance."""
    def __init__(self, missing_threshold=0.95, variance_threshold=1e-5):
        self.missing_threshold = missing_threshold
        self.variance_threshold = variance_threshold
        self.cols_to_keep_ = None
        
    def fit(self, X, y=None):
        df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        
        missing_frac = df.isna().mean()
        variance = df.var()
        
        keep_missing = missing_frac <= self.missing_threshold
        keep_variance = (variance > self.variance_threshold) | variance.isna()
        
        self.cols_to_keep_ = df.columns[keep_missing & keep_variance].tolist()
        
        if len(self.cols_to_keep_) == 0:
            self.cols_to_keep_ = df.columns.tolist()[:10]
        
        return self
    
    def transform(self, X):
        df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        return df[self.cols_to_keep_].values


class ConditionalImputer(BaseEstimator, TransformerMixin):
    """Choose KNNImputer for small feature sets, otherwise SimpleImputer.

    This avoids the O(n_samples^2 * n_features) cost of KNNImputer on
    high-dimensional data which can hang the pipeline.
    """
    def __init__(self, knn_neighbors=5, feature_threshold=500):
        self.knn_neighbors = knn_neighbors
        self.feature_threshold = feature_threshold
        self.imputer_ = None

    def fit(self, X, y=None):
        X_arr = X if not isinstance(X, pd.DataFrame) else X.values
        n_features = X_arr.shape[1]
        if n_features <= self.feature_threshold:
            self.imputer_ = KNNImputer(n_neighbors=self.knn_neighbors)
        else:
            self.imputer_ = SimpleImputer(strategy='median')
        self.imputer_.fit(X_arr)
        return self

    def transform(self, X):
        X_arr = X if not isinstance(X, pd.DataFrame) else X.values
        return self.imputer_.transform(X_arr)


def create_bridge_from_diagnosis_with_acxiom(bridge_path='diagnosis_with_acxiom.csv', out_path='id_bridge.csv'):
    """Create small bridge CSV mapping diagnosis-side member/sk IDs to Acxiom `member_id`.

    The function looks for common columns like `sys_mbr_sk` / `clm_sys_mbr_sk` on the
    diagnosis side and `member_id` / `empi` on the Acxiom side inside the
    `diagnosis_with_acxiom.csv` file. It writes `id_bridge.csv` with columns
    `diag_id,member_id` (and `empi` if present) and returns the bridge DataFrame.
    """
    try:
        print(f"\nLoading bridge file: {bridge_path}...")
        df = pd.read_csv(bridge_path, low_memory=False, dtype=str)
    except FileNotFoundError:
        print(f"   Bridge file not found: {bridge_path}")
        return None
    except Exception as e:
        print(f"   Failed to read bridge file: {e}")
        return None

    # candidate columns
    diag_cands = ['sys_mbr_sk', 'clm_sys_mbr_sk', 'sys_mbr_id', 'clm_mbr_sk']
    acx_cands = ['member_id', 'empi', 'cerner_empi']

    diag_col = next((c for c in diag_cands if c in df.columns), None)
    acx_col = next((c for c in acx_cands if c in df.columns), None)
    empi_col = 'empi' if 'empi' in df.columns else None

    if not diag_col or not acx_col:
        print(f"   Could not find expected bridge columns. Available columns: {list(df.columns[:20])}")
        return None

    # Normalize and reduce
    df = df[[diag_col, acx_col] + ([empi_col] if empi_col else [])].copy()
    df = df.rename(columns={diag_col: 'diag_id', acx_col: 'member_id'})
    df['diag_id'] = df['diag_id'].astype(str).str.strip()
    df['member_id'] = df['member_id'].astype(str).str.strip()
    if empi_col:
        df[empi_col] = df[empi_col].astype(str).str.strip()

    df = df[df['diag_id'].notna() & df['member_id'].notna()]
    df = df.drop_duplicates(subset=['diag_id', 'member_id'])

    try:
        df.to_csv(out_path, index=False)
        print(f"   Wrote bridge to: {out_path} (rows: {len(df)})")
    except Exception:
        print("   Could not write bridge file to disk, continuing with in-memory bridge")

    print(f"   Example mappings (first 5):\n{df.head().to_string(index=False)}")
    return df


def create_bridge_via_demographics(acxiom_df, demographics_path='demographics.csv', out_path='id_bridge.csv'):
    """Create a bridge using `demographics.csv` linking `sys_mbr_sk` <-> `empi`,
    then linking `empi` -> `member_id` from the provided `acxiom_df`.

    Returns bridge DataFrame with columns: diag_sys_mbr_sk, empi, member_id
    """
    try:
        print(f"\nLoading demographics: {demographics_path}...")
        demo = pd.read_csv(demographics_path, low_memory=False, dtype=str)
    except FileNotFoundError:
        print(f"   demographics file not found: {demographics_path}")
        return None
    except Exception as e:
        print(f"   Failed to read demographics file: {e}")
        return None

    # Find candidate columns
    demo_sys_cols = [c for c in demo.columns if any(t in c.lower() for t in ['sys_mbr_sk', 'clm_sys_mbr_sk', 'sys_mbr_id', 'clm_mbr_sk'])]
    demo_empi_cols = [c for c in demo.columns if any(t in c.lower() for t in ['empi', 'lumeris_empi', 'cerner_empi'])]
    acx_empi_cols = [c for c in acxiom_df.columns if any(t in c.lower() for t in ['empi', 'lumeris_empi', 'cerner_empi'])]
    acx_member_cols = [c for c in acxiom_df.columns if 'member_id' in c.lower()]

    # If Acxiom doesn't have a `member_id` column but does have an EMPI (lumeris_empi),
    # we'll treat EMPI as the member identifier (map empi->empi and use as member_id)
    use_empi_as_member = False
    if not acx_member_cols and acx_empi_cols:
        use_empi_as_member = True
        acx_member_cols = [acx_empi_cols[0]]

    if not demo_sys_cols or not demo_empi_cols:
        print(f"   Could not find sys_mbr_sk/empi in demographics. Found: sys_cols={demo_sys_cols}, empi_cols={demo_empi_cols}")
        return None
    if not acx_empi_cols:
        print(f"   Could not find any EMPI-like column in Acxiom. Found: acx_empi={acx_empi_cols}")
        return None

    demo_sys_col = demo_sys_cols[0]
    demo_empi_col = demo_empi_cols[0]
    acx_empi_col = acx_empi_cols[0]
    acx_member_col = acx_member_cols[0]

    # Normalize
    demo_subset = demo[[demo_sys_col, demo_empi_col]].copy()
    demo_subset = demo_subset.rename(columns={demo_sys_col: 'sys_mbr_sk', demo_empi_col: 'empi'})
    demo_subset['sys_mbr_sk'] = demo_subset['sys_mbr_sk'].astype(str).str.strip()
    demo_subset['empi'] = demo_subset['empi'].astype(str).str.strip()
    demo_subset = demo_subset.dropna(subset=['sys_mbr_sk', 'empi'])
    demo_subset = demo_subset.drop_duplicates()

    # Acxiom EMPI -> member_id
    acx_subset = acxiom_df[[acx_empi_col, acx_member_col]].copy()
    acx_subset = acx_subset.rename(columns={acx_empi_col: 'empi', acx_member_col: 'member_id'})
    acx_subset['empi'] = acx_subset['empi'].astype(str).str.strip()
    acx_subset['member_id'] = acx_subset['member_id'].astype(str).str.strip()
    acx_subset = acx_subset.dropna(subset=['empi', 'member_id'])
    acx_subset = acx_subset.drop_duplicates()

    # If we're using EMPI as the member identifier (no explicit member_id column),
    # normalize by copying empi -> member_id so downstream logic can treat 'member_id'
    # as the canonical Acxiom key.
    if use_empi_as_member:
        acx_subset['member_id'] = acx_subset['empi']

    # Join via empi
    bridge = demo_subset.merge(acx_subset, on='empi', how='inner')
    bridge = bridge.rename(columns={'sys_mbr_sk': 'diag_sys_mbr_sk'})

    if bridge.empty:
        print("   Demographics -> Acxiom join produced zero rows. Check formats (leading zeros, types).")
        return None

    # Write to disk
    try:
        bridge.to_csv(out_path, index=False)
        print(f"   Wrote demographics-based bridge to {out_path} (rows: {len(bridge)})")
    except Exception:
        print("   Could not write bridge file to disk, continuing with in-memory bridge")

    print(f"   Example bridge rows:\n{bridge.head().to_string(index=False)}")
    return bridge


# =============================================================================
# Step 1: Load and Link Datasets
# =============================================================================

def load_and_link_datasets(acxiom_path, diagnosis_path):
    """
    Load both datasets and find patients present in both.
    
    Returns:
        acxiom_df, diagnosis_df, common_patient_ids
    """
    print("=" * 70)
    print("Step 1: Loading and Linking Datasets")
    print("=" * 70)
    
    # Load Acxiom (SDoH features)
    print(f"\nLoading Acxiom data from {acxiom_path}...")
    acxiom = pd.read_csv(acxiom_path, low_memory=False, dtype=str)
    print(f"  Shape: {acxiom.shape}")
    print(f"  Columns (first 10): {list(acxiom.columns[:10])}")
    
    # Load Diagnosis
    print(f"\nLoading Diagnosis data from {diagnosis_path}...")
    diagnosis = pd.read_csv(diagnosis_path, low_memory=False, dtype=str)
    print(f"  Shape: {diagnosis.shape}")
    print(f"  Columns (first 10): {list(diagnosis.columns[:10])}")
    
    # Find best candidate ID columns by matching likely ID-like column names.
    # Prefer real member ID columns and explicitly avoid zipcode-like columns.
    token_candidates = ['id', 'member', 'empi', 'sys', 'sk', 'clm', 'patient']
    zip_tokens = ['zip', 'zipcode', 'memberzipcode', 'postal']

    def find_id_candidates(df):
        cols = [c for c in df.columns if any(t in c.lower() for t in token_candidates)]
        # exclude obvious zipcode-like columns by name
        cols = [c for c in cols if not any(z in c.lower() for z in zip_tokens)]
        return cols

    acx_candidates = find_id_candidates(acxiom)
    diag_candidates = find_id_candidates(diagnosis)

    # If nothing obvious, fall back to columns that contain 'id' specifically,
    # but still avoid zipcode-like names
    if len(acx_candidates) == 0:
        acx_candidates = [c for c in acxiom.columns if 'id' in c.lower() and not any(z in c.lower() for z in zip_tokens)]
    if len(diag_candidates) == 0:
        diag_candidates = [c for c in diagnosis.columns if 'id' in c.lower() and not any(z in c.lower() for z in zip_tokens)]

    # Score candidate pairs by overlap and pick the best pair
    best_pair = (None, None)
    best_overlap = -1

    # Convert values to strings and strip for comparison
    def values_set(df, col):
        try:
            return set(df[col].astype(str).str.strip().dropna().unique())
        except Exception:
            return set()

    for a_col in acx_candidates:
        a_vals = values_set(acxiom, a_col)
        if len(a_vals) == 0:
            continue
        for d_col in diag_candidates:
            d_vals = values_set(diagnosis, d_col)
            overlap = len(a_vals.intersection(d_vals))
            if overlap > best_overlap:
                best_overlap = overlap
                best_pair = (a_col, d_col)

    # If best_pair looks like a zipcode (values are 5-digit numeric strings), deprioritize
    def looks_like_zip(df, col):
        try:
            sample = df[col].dropna().astype(str).head(200)
            if sample.empty:
                return False
            matches = sample.str.match(r"^\d{5}$").sum()
            return matches / len(sample) > 0.8
        except Exception:
            return False

    if best_pair[0] is not None and looks_like_zip(acxiom, best_pair[0]):
        # try to choose a preferred ID column if present
        # prefer Lumeris EMPI explicitly over member_id when available
        preferred = ['lumeris_empi', 'member_id', 'empi', 'sys_mbr_sk', 'clm_sys_mbr_sk', 'patient_id']
        pref_a = next((c for c in preferred if c in acxiom.columns), None)
        pref_d = next((c for c in preferred if c in diagnosis.columns), None)
        if pref_a and pref_d:
            # Check overlap for preferred pair
            ov = len(values_set(acxiom, pref_a).intersection(values_set(diagnosis, pref_d)))
            if ov >= best_overlap:
                best_pair = (pref_a, pref_d)
                best_overlap = ov

    # If no overlap found using candidates, fall back to a default list
    if best_pair[0] is None or best_pair[1] is None:
        possible_ids = ['sys_mbr_sk', 'clm_sys_mbr_sk', 'lumeris_empi', 'empi', 'member_id', 'patient_id']
        acx_id_col = None
        for col in possible_ids:
            if col in acxiom.columns:
                acx_id_col = col
                break
        diag_id_col = None
        for col in possible_ids:
            if col in diagnosis.columns:
                diag_id_col = col
                break
        if not acx_id_col or not diag_id_col:
            print(f"\n❌ Could not confidently identify ID columns.")
            print(f"Acxiom candidates: {acx_candidates}")
            print(f"Diagnosis candidates: {diag_candidates}")
            print(f"Available Acxiom columns: {list(acxiom.columns[:20])}")
            print(f"Available Diagnosis columns: {list(diagnosis.columns[:20])}")
            raise ValueError("Cannot find patient ID columns")
        else:
            acx_id_col = acx_id_col
            diag_id_col = diag_id_col
    else:
        acx_id_col, diag_id_col = best_pair

    print(f"\n✅ ID columns chosen:")
    print(f"   Acxiom: {acx_id_col}")
    print(f"   Diagnosis: {diag_id_col}")

    if best_overlap >= 0:
        print(f"   Overlap between these columns: {best_overlap}")
    
    # Normalize IDs
    acxiom[acx_id_col] = acxiom[acx_id_col].astype(str).str.strip()
    diagnosis[diag_id_col] = diagnosis[diag_id_col].astype(str).str.strip()
    
    # Find common patients
    acx_ids = set(acxiom[acx_id_col].unique())
    diag_ids = set(diagnosis[diag_id_col].unique())
    common_ids = acx_ids.intersection(diag_ids)
    
    print(f"\n📊 Patient overlap:")
    print(f"   Patients in Acxiom: {len(acx_ids):,}")
    print(f"   Patients in Diagnosis: {len(diag_ids):,}")
    print(f"   Patients in BOTH: {len(common_ids):,} ({len(common_ids)/len(acx_ids)*100:.1f}% of Acxiom)")
    
    if len(common_ids) < 50:
        print(f"\n⚠️ WARNING: Only {len(common_ids)} common patients!")
        print("   This may be too small for reliable analysis.")
    
    # If no overlap (or very small), try to use a bridge file that links
    # diagnosis-side IDs to Acxiom `member_id`. The bridge file is typically
    # `diagnosis_with_acxiom.csv` and we write/read `id_bridge.csv`.
    if len(common_ids) == 0:
        print("\nNo direct patient ID overlap found — attempting to build/use ID bridge...")
        bridge = None
        try:
            # Prefer an existing id_bridge.csv if present
            bridge = pd.read_csv('id_bridge.csv', dtype=str)
            print(f"   Loaded existing id_bridge.csv (rows: {len(bridge)})")
        except Exception:
            bridge = create_bridge_from_diagnosis_with_acxiom('diagnosis_with_acxiom.csv', out_path='id_bridge.csv')

        # If diagnosis_with_acxiom didn't produce a bridge, try demographics + acxiom mapping
        if (bridge is None or bridge.empty):
            try:
                bridge = create_bridge_via_demographics(acxiom, demographics_path='demographics.csv', out_path='id_bridge.csv')
            except Exception as e:
                print(f"   demographics-based bridge failed: {e}")

        if bridge is not None and not bridge.empty:
            # Build mapping diag_id -> member_id
            bridge_map = dict(zip(bridge['diag_id'].astype(str).str.strip(), bridge['member_id'].astype(str).str.strip()))

            # Map diagnosis IDs to member_id
            diagnosis['mapped_member_id'] = diagnosis[diag_id_col].astype(str).str.strip().map(bridge_map)
            mapped_ids = set(diagnosis['mapped_member_id'].dropna().unique())

            # Prefer Acxiom member_id column if present
            if 'member_id' in acxiom.columns:
                acx_id_col = 'member_id'

            acx_ids = set(acxiom[acx_id_col].astype(str).str.strip().unique())
            common_ids = acx_ids.intersection(mapped_ids)

            print(f"   Bridge-enabled overlap: {len(common_ids)} patients")
            if len(common_ids) == 0:
                print("   Bridge found but no matching member_ids in Acxiom — bridge may use different formats.")
        else:
            print("   No usable bridge available.")
    # Filter to common patients
    acxiom_filtered = acxiom[acxiom[acx_id_col].astype(str).str.strip().isin(common_ids)].copy()
    # If we created a mapped_member_id via bridge use it for filtering
    if 'mapped_member_id' in diagnosis.columns:
        diagnosis_filtered = diagnosis[diagnosis['mapped_member_id'].astype(str).str.strip().isin(common_ids)].copy()
    else:
        diagnosis_filtered = diagnosis[diagnosis[diag_id_col].astype(str).str.strip().isin(common_ids)].copy()
    
    # Rename ID columns to standard name
    acxiom_filtered = acxiom_filtered.rename(columns={acx_id_col: 'patient_id'})
    # Rename diagnosis id to patient_id — prefer mapped_member_id if available
    if 'mapped_member_id' in diagnosis_filtered.columns:
        diagnosis_filtered = diagnosis_filtered.rename(columns={'mapped_member_id': 'patient_id'})
    else:
        diagnosis_filtered = diagnosis_filtered.rename(columns={diag_id_col: 'patient_id'})
    
    return acxiom_filtered, diagnosis_filtered, list(common_ids)


# =============================================================================
# Step 2: Extract Diagnosis Information
# =============================================================================

def extract_diagnosis_codes(diagnosis_df):
    """
    Extract diagnosis codes and create patient-diagnosis matrix.
    
    Returns:
        patient_diagnoses: dict {patient_id: set(diagnosis_codes)}
        diagnosis_prevalence: Series with diagnosis counts
    """
    print("\n" + "=" * 70)
    print("Step 2: Extracting Diagnosis Codes")
    print("=" * 70)
    
    # Find diagnosis code column using substring matching (more flexible)
    diag_code_tokens = ['diagnosis', 'diag', 'dx', 'icd', 'icd9', 'icd10', 'code']
    diag_col = None

    for col in diagnosis_df.columns:
        col_l = col.lower()
        if any(tok in col_l for tok in diag_code_tokens):
            diag_col = col
            break

    # If not found yet, try exact matches as a final fallback
    if not diag_col:
        diag_code_candidates = ['diagnosis_code', 'dx_code', 'diag_code', 'icd_code',
                                'icd10', 'icd9', 'dx', 'diagnosis', 'code']
        cols_lower = {c.lower(): c for c in diagnosis_df.columns}
        for cand in diag_code_candidates:
            if cand.lower() in cols_lower:
                diag_col = cols_lower[cand.lower()]
                break

    if not diag_col:
        print(f"\n❌ No diagnosis code column found. Tried tokens: {diag_code_tokens}")
        print(f"Available columns: {list(diagnosis_df.columns[:20])}")
        raise ValueError("Cannot find diagnosis code column")

    print(f"\n✅ Diagnosis code column: {diag_col}")
    
    # Build patient → diagnoses mapping
    patient_diagnoses = defaultdict(set)
    for _, row in diagnosis_df.iterrows():
        patient_id = row['patient_id']
        diag_code = str(row[diag_col]).strip()
        if diag_code and diag_code != 'nan':
            patient_diagnoses[patient_id].add(diag_code)
    
    # Calculate prevalence
    all_diagnoses = []
    for diagnoses in patient_diagnoses.values():
        all_diagnoses.extend(diagnoses)
    
    diagnosis_prevalence = pd.Series(all_diagnoses).value_counts()
    
    print(f"\n📊 Diagnosis statistics:")
    print(f"   Patients with diagnoses: {len(patient_diagnoses):,}")
    print(f"   Unique diagnosis codes: {len(diagnosis_prevalence):,}")
    print(f"   Total diagnosis records: {len(all_diagnoses):,}")
    print(f"   Avg diagnoses per patient: {len(all_diagnoses)/len(patient_diagnoses):.1f}")
    
    print(f"\n🔝 Top 20 diagnoses:")
    for dx, count in diagnosis_prevalence.head(20).items():
        pct = count / len(patient_diagnoses) * 100
        print(f"   {dx}: {count} patients ({pct:.1f}%)")
    
    return dict(patient_diagnoses), diagnosis_prevalence


# =============================================================================
# Step 3: Validate Target Diagnoses
# =============================================================================

def validate_target_diagnoses(diagnosis_prevalence, target_diagnoses, total_patients):
    """
    Validate which target diagnoses are available in the dataset.
    
    Returns:
        List of valid diagnosis codes with their prevalence info
    """
    print("\n" + "=" * 70)
    print("Step 3: Validating Target Diagnosis Codes")
    print("=" * 70)
    
    valid_diagnoses = []
    
    print(f"\nChecking {len(target_diagnoses)} target diagnoses...")
    
    for dx_code in target_diagnoses:
        if dx_code in diagnosis_prevalence.index:
            count = diagnosis_prevalence[dx_code]
            prevalence = count / total_patients
            valid_diagnoses.append({
                'code': dx_code,
                'count': count,
                'prevalence': prevalence
            })
            print(f"  ✓ {dx_code}: {count} patients ({prevalence*100:.1f}%)")
        else:
            print(f"  ✗ {dx_code}: Not found in dataset")
    
    print(f"\n✅ Found {len(valid_diagnoses)}/{len(target_diagnoses)} target diagnoses in dataset")
    
    if len(valid_diagnoses) == 0:
        print("\n⚠️ WARNING: No target diagnoses found!")
        print("Available diagnosis codes (top 20):")
        for dx, count in diagnosis_prevalence.head(20).items():
            print(f"   {dx}: {count} patients")
    
    return valid_diagnoses


# =============================================================================
# Step 4: Create Labels for Single Diagnosis
# =============================================================================

def create_labels_for_diagnosis(patient_ids, patient_diagnoses, diagnosis_code):
    """
    Create binary labels: 1 if patient has the specific diagnosis, 0 otherwise.
    
    Returns:
        labels: Series with patient_id as index
    """
    labels = {}
    for patient_id in patient_ids:
        patient_dx = patient_diagnoses.get(patient_id, set())
        has_diagnosis = diagnosis_code in patient_dx
        labels[patient_id] = 1 if has_diagnosis else 0
    
    return pd.Series(labels)


# =============================================================================
# Step 5: Prepare Features (SDoH only, NO LEAKAGE)
# =============================================================================

def prepare_features(acxiom_df):
    """
    Extract SDoH features (NO diagnosis features to avoid leakage).
    
    Returns:
        X_raw: DataFrame with raw features (with NaNs)
        feature_names: List of feature names
    """
    print("\n" + "=" * 70)
    print("Step 4: Preparing SDoH Features (NO LEAKAGE)")
    print("=" * 70)
    
    # Identify SDoH columns (pattern: 2-4 letters + numbers, or ibe*)
    # Examples: ab123, xyz45, demo1234, ibe789
    pattern = re.compile(r'^[a-z]{2,4}\d+|^ibe\d+', re.IGNORECASE)
    sdoh_cols = [col for col in acxiom_df.columns if pattern.match(col)]
    
    print(f"\n🔍 Detected {len(sdoh_cols)} potential SDoH columns")
    print(f"   Pattern: 2-4 letters followed by numbers (e.g., ab123, xyz45, demo1234)")
    print(f"   Examples found: {sdoh_cols[:15]}")

    # Exclude any columns that look like diagnosis/claim fields to avoid leakage
    exclusion_tokens = ['diag', 'icd', 'dx', 'diagnosis', 'claim', 'clm', 'disease',
                        'clinical', 'code', 'procedure', 'proc', 'admit', 'encounter']
    excluded = [c for c in sdoh_cols if any(tok in c.lower() for tok in exclusion_tokens)]
    if excluded:
        print(f"\n⚠️ Excluding {len(excluded)} potential leakage columns by name")
        print(f"   Examples excluded: {excluded[:10]}")
    sdoh_cols = [c for c in sdoh_cols if c not in excluded]

    print(f"\n✅ Found {len(sdoh_cols)} valid SDoH columns (after exclusion)")
    print(f"   Final examples: {sdoh_cols[:15]}")
    
    if len(sdoh_cols) == 0:
        print("\n⚠️ No SDoH columns found, using all non-ID columns")
        sdoh_cols = [col for col in acxiom_df.columns if col != 'patient_id']
    
    # Extract features - KEEP RAW with NaNs
    X_raw = acxiom_df[sdoh_cols].copy()
    
    # Convert to numeric
    for col in X_raw.columns:
        X_raw[col] = pd.to_numeric(X_raw[col], errors='coerce')
    
    # Remove completely empty columns only
    all_nan_cols = X_raw.columns[X_raw.isna().all()]
    if len(all_nan_cols) > 0:
        print(f"\n⚠️ Removing {len(all_nan_cols)} completely empty columns")
        X_raw = X_raw.drop(columns=all_nan_cols)
        sdoh_cols = [c for c in sdoh_cols if c not in all_nan_cols]
    
    # Report missingness
    missing_pct = X_raw.isna().mean()
    print(f"\n📊 Missingness statistics:")
    print(f"   Average: {missing_pct.mean():.2%}")
    print(f"   Features with >50% missing: {(missing_pct > 0.5).sum()}")
    
    print(f"\n✅ Prepared RAW feature matrix: {X_raw.shape}")
    print("   (NaNs preserved for leakage-free CV)")
    
    return X_raw, sdoh_cols




# =============================================================================
# Step 6: Create Leakage-Free Model Pipelines with Hyperparameter Variations
# =============================================================================

def create_model_pipelines():
    """
    Create pipelines with 2-3 hyperparameter configurations per model.
    This provides alternatives in case one configuration fails or predicts one class.
    """
    models = {}
    
    print("🔒 Creating leakage-free pipelines with hyperparameter variations...")
    print("   All preprocessing happens INSIDE cross-validation folds")
    print("   Multiple configs per model for robustness")
    
    # Random Forest - 3 configurations
    rf_configs = [
        ('Conservative', {'max_depth': 4, 'min_samples_leaf': 15, 'max_features': 'sqrt', 'n_estimators': 100}),
        ('Moderate', {'max_depth': 6, 'min_samples_leaf': 10, 'max_features': 'sqrt', 'n_estimators': 100}),
        ('Relaxed', {'max_depth': 8, 'min_samples_leaf': 5, 'max_features': 'sqrt', 'n_estimators': 100}),
    ]
    for name, config in rf_configs:
        params = {
            'random_state': RANDOM_STATE, 'n_jobs': -1, 'class_weight': 'balanced',
            **config
        }
        models[f'RF-{name}'] = Pipeline([
            ('dropper', ColumnDropper(MISSING_THRESHOLD, VARIANCE_THRESHOLD)),
            ('imputer', ConditionalImputer(knn_neighbors=5, feature_threshold=500)),
            ('scaler', StandardScaler()),
            ('model', RandomForestClassifier(**params))
        ])
    
    # Gradient Boosting - 3 configurations
    gb_configs = [
        ('Conservative', {'max_depth': 3, 'learning_rate': 0.01, 'min_samples_leaf': 20, 'subsample': 0.7, 'n_estimators': 100}),
        ('Moderate', {'max_depth': 4, 'learning_rate': 0.02, 'min_samples_leaf': 10, 'subsample': 0.8, 'n_estimators': 100}),
        ('Relaxed', {'max_depth': 5, 'learning_rate': 0.03, 'min_samples_leaf': 5, 'subsample': 0.8, 'n_estimators': 100}),
    ]
    for name, config in gb_configs:
        params = {
            'random_state': RANDOM_STATE, 'validation_fraction': 0.1, 
            'n_iter_no_change': 10, 'tol': 1e-4,
            **config
        }
        models[f'GB-{name}'] = Pipeline([
            ('dropper', ColumnDropper(MISSING_THRESHOLD, VARIANCE_THRESHOLD)),
            ('imputer', ConditionalImputer(knn_neighbors=5, feature_threshold=500)),
            ('scaler', StandardScaler()),
            ('model', GradientBoostingClassifier(**params))
        ])
    
    # LightGBM - 3 configurations
    if LIGHTGBM_AVAILABLE:
        lgbm_configs = [
            ('Conservative', {'max_depth': 3, 'learning_rate': 0.01, 'num_leaves': 15, 'min_child_samples': 30, 
                            'subsample': 0.7, 'n_estimators': 100}),
            ('Moderate', {'max_depth': 4, 'learning_rate': 0.02, 'num_leaves': 20, 'min_child_samples': 20, 
                         'subsample': 0.8, 'n_estimators': 100}),
            ('Relaxed', {'max_depth': 5, 'learning_rate': 0.03, 'num_leaves': 25, 'min_child_samples': 15, 
                        'subsample': 0.8, 'n_estimators': 100}),
        ]
        for name, config in lgbm_configs:
            params = {
                'colsample_bytree': 0.7, 'reg_alpha': 0.1, 'reg_lambda': 1.0,
                'random_state': RANDOM_STATE, 'n_jobs': -1, 'verbosity': -1, 
                'class_weight': 'balanced',
                **config
            }
            models[f'LGBM-{name}'] = Pipeline([
                ('dropper', ColumnDropper(MISSING_THRESHOLD, VARIANCE_THRESHOLD)),
                ('imputer', ConditionalImputer(knn_neighbors=5, feature_threshold=500)),
                ('scaler', StandardScaler()),
                ('model', lgb.LGBMClassifier(**params))
            ])
    
    # XGBoost - 3 configurations
    if XGBOOST_AVAILABLE:
        xgb_configs = [
            ('Conservative', {'max_depth': 3, 'learning_rate': 0.01, 'min_child_weight': 15, 'subsample': 0.7, 
                            'gamma': 0.3, 'n_estimators': 100}),
            ('Moderate', {'max_depth': 4, 'learning_rate': 0.02, 'min_child_weight': 10, 'subsample': 0.8, 
                         'gamma': 0.2, 'n_estimators': 100}),
            ('Relaxed', {'max_depth': 5, 'learning_rate': 0.03, 'min_child_weight': 5, 'subsample': 0.8, 
                        'gamma': 0.1, 'n_estimators': 100}),
        ]
        for name, config in xgb_configs:
            params = {
                'colsample_bytree': 0.7, 'reg_alpha': 0.1, 'reg_lambda': 1.5,
                'random_state': RANDOM_STATE, 'n_jobs': -1, 'eval_metric': 'logloss',
                **config
            }
            models[f'XGB-{name}'] = Pipeline([
                ('dropper', ColumnDropper(MISSING_THRESHOLD, VARIANCE_THRESHOLD)),
                ('imputer', ConditionalImputer(knn_neighbors=5, feature_threshold=500)),
                ('scaler', StandardScaler()),
                ('model', xgb.XGBClassifier(**params))
            ])
    
    # CatBoost - 2 configurations (slower to train)
    if CATBOOST_AVAILABLE:
        catboost_configs = [
            ('Conservative', {'depth': 4, 'learning_rate': 0.01, 'min_data_in_leaf': 20, 'l2_leaf_reg': 5, 
                            'subsample': 0.7, 'iterations': 100}),
            ('Moderate', {'depth': 5, 'learning_rate': 0.02, 'min_data_in_leaf': 10, 'l2_leaf_reg': 3, 
                         'subsample': 0.8, 'iterations': 100}),
        ]
        for name, config in catboost_configs:
            params = {
                'random_state': RANDOM_STATE, 'verbose': False, 'thread_count': -1,
                'auto_class_weights': 'Balanced', 'max_leaves': 16,
                **config
            }
            models[f'CatBoost-{name}'] = Pipeline([
                ('dropper', ColumnDropper(MISSING_THRESHOLD, VARIANCE_THRESHOLD)),
                ('imputer', ConditionalImputer(knn_neighbors=5, feature_threshold=500)),
                ('scaler', StandardScaler()),
                ('model', CatBoostClassifier(**params))
            ])
    
    print(f"✅ Created {len(models)} model pipeline variations")
    model_counts = {}
    for model_name in models.keys():
        base_name = model_name.split('-')[0]
        model_counts[base_name] = model_counts.get(base_name, 0) + 1
    
    count_str = ', '.join([f'{k}: {v}' for k, v in model_counts.items()])
    print(f"   Configs per model: {count_str}")
    
    return models


# =============================================================================
# Step 7: Evaluate Single Diagnosis with Quality Filters
# =============================================================================

def detect_single_class_predictions(pipeline, X_raw, y, cv_folds=3):
    """
    Check if a model predicts only one class across CV folds (class collapse).
    
    Returns:
        True if model predicts multiple classes, False if single-class only
    """
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    
    unique_predictions_per_fold = []
    
    for train_idx, test_idx in skf.split(X_raw, y):
        X_train, X_test = X_raw.iloc[train_idx], X_raw.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Fit and predict
        try:
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            
            # Count unique predictions
            unique_predictions_per_fold.append(len(np.unique(y_pred)))
        except Exception as e:
            # If prediction fails, consider it a failure
            return False
    
    # Check if any fold predicted only one class
    single_class_folds = [n for n in unique_predictions_per_fold if n == 1]
    
    if len(single_class_folds) > 0:
        return False  # Model collapsed to single class
    return True  # Model predicts multiple classes


def evaluate_diagnosis(X_raw, y, diagnosis_code, models):
    """
    Run leakage-free CV for one diagnosis code with quality filters.
    
    Filters out models that:
    - Predict only one class (class collapse)
    - Overfit excessively (gap > 0.20)
    - Perform too poorly (F1 < 0.48)
    
    Returns:
        DataFrame with results for each valid model
    """
    skf = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1',
        'roc_auc': 'roc_auc'
    }
    
    results = []
    
    print(f"\n🔍 Evaluating {len(models)} model configurations...")
    
    for model_name, pipeline in models.items():
        try:
            # Step 1: Check for single-class predictions (quick test)
            print(f"   Testing {model_name}...", end=" ", flush=True)
            predicts_multiple_classes = detect_single_class_predictions(pipeline, X_raw, y, cv_folds=3)
            
            if not predicts_multiple_classes:
                print("❌ REJECTED: Predicts only one class (class collapse)")
                continue
            
            # Step 2: Run full cross-validation
            cv_results = cross_validate(
                pipeline, X_raw, y,
                cv=skf,
                scoring=scoring,
                n_jobs=1,
                return_train_score=True,
                error_score='raise'
            )
            
            # Calculate metrics
            f1_score_val = cv_results['test_f1'].mean()
            overfit_gap = cv_results['train_accuracy'].mean() - cv_results['test_accuracy'].mean()
            
            # Step 3: Quality filters
            rejection_reasons = []
            
            if f1_score_val < 0.48:
                rejection_reasons.append(f"Low F1={f1_score_val:.3f}")
            
            if overfit_gap > 0.20:
                rejection_reasons.append(f"Excessive overfit gap={overfit_gap:.3f}")
            
            if rejection_reasons:
                print(f"❌ REJECTED: {', '.join(rejection_reasons)}")
                continue
            
            # Step 4: Model passed all checks
            result = {
                'Diagnosis': diagnosis_code,
                'Model': model_name,
                'Accuracy': cv_results['test_accuracy'].mean(),
                'Accuracy_Std': cv_results['test_accuracy'].std(),
                'Precision': cv_results['test_precision'].mean(),
                'Recall': cv_results['test_recall'].mean(),
                'F1': f1_score_val,
                'ROC_AUC': cv_results['test_roc_auc'].mean(),
                'Train_Acc': cv_results['train_accuracy'].mean(),
                'Overfit_Gap': overfit_gap
            }
            results.append(result)
            
            # Quality indicator
            quality = "🟢" if overfit_gap < 0.05 else "🟡" if overfit_gap < 0.10 else "🟠"
            print(f"{quality} ACCEPTED: F1={f1_score_val:.3f}, Gap={overfit_gap:.3f}")
            
        except Exception as e:
            print(f"  ❌ {model_name} failed: {str(e)}")
    
    results_df = pd.DataFrame(results)
    
    if results_df.empty:
        print("\n⚠️ WARNING: All models were rejected or failed!")
        print("   Consider relaxing hyperparameters or checking data quality")
    else:
        print(f"\n✅ {len(results_df)}/{len(models)} models passed quality checks")
        
        # Add composite score that balances F1 and generalization
        # Formula: F1 * (1 - overfitting_penalty)
        # Penalty increases sharply with overfit gap
        results_df['Composite_Score'] = results_df.apply(
            lambda row: row['F1'] * (1 - min(row['Overfit_Gap'] * 2.0, 0.5)),
            axis=1
        )
        print("\n📊 Composite scoring applied (balances F1 vs generalization)")
    
    return results_df


# =============================================================================
# Step 8: Extract Feature Importances for Best Combination
# =============================================================================

def extract_feature_importances(X_raw, y, feature_names, best_model_pipeline):
    """
    Train best model on full data and extract feature importances.
    
    Returns:
        DataFrame with feature importances sorted by importance
    """
    print("\n" + "=" * 70)
    print("Extracting Feature Importances from Best Model")
    print("=" * 70)
    
    # Fit the full pipeline on all data
    best_model_pipeline.fit(X_raw, y)
    
    # Get the trained model from the pipeline
    trained_model = best_model_pipeline.named_steps['model']
    
    # Get feature names after column dropping
    dropper = best_model_pipeline.named_steps['dropper']
    kept_features = dropper.cols_to_keep_
    
    # Extract feature importances if available
    if hasattr(trained_model, 'feature_importances_'):
        importances = trained_model.feature_importances_
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': kept_features,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        print(f"\n✅ Extracted {len(importance_df)} feature importances")
        print(f"\nTop 20 Most Important Features:")
        print(importance_df.head(20).to_string(index=False))
        
        return importance_df
    else:
        print(f"\n⚠️ Model does not provide feature importances")
        return None


def plot_feature_importances(importance_df, top_n=20, filename='feature_importances.png'):
    """
    Plot top N feature importances.
    """
    if importance_df is None:
        return
    
    print(f"\n📊 Creating feature importance plot...")
    
    top_features = importance_df.head(top_n)
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(top_features)), top_features['importance'], color='steelblue')
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title(f'Top {top_n} Feature Importances for Diagnosis Prediction', 
              fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {filename}")
    plt.close()


def plot_correlation_heatmap(X_raw, y, importance_df, top_n=20, filename='correlation_heatmap.png'):
    """
    Plot correlation heatmap for top features.
    
    NOTE: 'diagnosis_label' is added ONLY for visualization purposes in this function.
    It is NOT part of the feature set used for model training. The label is added
    temporarily to this separate dataframe (df_corr) to compute and visualize
    correlations between features and the target.
    """
    if importance_df is None:
        return
    
    print(f"\n📊 Creating correlation heatmap...")
    print("   ℹ️  Note: 'diagnosis_label' used ONLY for correlation visualization (NOT in training)")
    
    top_features = importance_df.head(top_n)['feature'].tolist()
    
    # Create SEPARATE DataFrame for correlation visualization (NOT used in training!)
    df_corr = X_raw[top_features].copy()
    
    # Impute NaNs for correlation calculation (use median, fast and safe)
    print("   Imputing NaNs for correlation calculation...")
    for col in df_corr.columns:
        if df_corr[col].isna().any():
            df_corr[col].fillna(df_corr[col].median(), inplace=True)
    
    # Add label ONLY to this temporary visualization dataframe
    df_corr['diagnosis_label'] = y
    
    # Calculate correlation matrix
    print("   Computing correlations...")
    corr_matrix = df_corr.corr()
    
    # Plot heatmap
    plt.figure(figsize=(14, 12))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                vmin=-1, vmax=1)
    plt.title(f'Correlation Heatmap: Top {top_n} Features vs Diagnosis\n(diagnosis_label shown for visualization only)', 
              fontsize=14, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {filename}")
    plt.close()
    
    # Print correlations with target
    print(f"\n📊 Correlations with Diagnosis Label (visualization only):")
    target_corr = corr_matrix['diagnosis_label'].drop('diagnosis_label').sort_values(ascending=False)
    print(target_corr.to_string())
    
    # Clean up - remove the temporary visualization dataframe
    del df_corr


def plot_model_comparison(final_results, filename='model_comparison.png'):
    """
    Plot comparison of model performance across all combinations.
    """
    print(f"\n📊 Creating model comparison plot...")
    
    # Average performance per model across all combinations
    model_avg = final_results.groupby('Model').agg({
        'F1': ['mean', 'std'],
        'ROC_AUC': ['mean', 'std'],
        'Overfit_Gap': ['mean', 'std']
    }).reset_index()
    
    model_avg.columns = ['Model', 'F1_mean', 'F1_std', 'ROC_AUC_mean', 
                         'ROC_AUC_std', 'Gap_mean', 'Gap_std']
    model_avg = model_avg.sort_values('F1_mean', ascending=False)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: F1-Score comparison
    ax1 = axes[0]
    models = model_avg['Model']
    f1_means = model_avg['F1_mean']
    f1_stds = model_avg['F1_std']
    
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(models)))
    bars = ax1.barh(range(len(models)), f1_means, xerr=f1_stds, 
                     color=colors, capsize=5)
    ax1.set_yticks(range(len(models)))
    ax1.set_yticklabels(models)
    ax1.set_xlabel('F1-Score', fontsize=12)
    ax1.set_title('Model Comparison: F1-Score (Mean ± Std)', fontsize=13, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    ax1.invert_yaxis()
    
    for i, (mean, std) in enumerate(zip(f1_means, f1_stds)):
        ax1.text(mean + std + 0.01, i, f'{mean:.3f}', 
                va='center', fontsize=10, fontweight='bold')
    
    # Plot 2: Overfitting analysis
    ax2 = axes[1]
    gaps = model_avg['Gap_mean']
    colors2 = ['red' if gap > 0.1 else 'green' for gap in gaps]
    
    bars = ax2.barh(range(len(models)), gaps, color=colors2, alpha=0.7)
    ax2.set_yticks(range(len(models)))
    ax2.set_yticklabels(models)
    ax2.set_xlabel('Overfit Gap (Train - Test)', fontsize=12)
    ax2.set_title('Overfitting Analysis', fontsize=13, fontweight='bold')
    ax2.axvline(x=0.1, color='orange', linestyle='--', linewidth=2, label='Warning threshold')
    ax2.grid(axis='x', alpha=0.3)
    ax2.legend()
    ax2.invert_yaxis()
    
    for i, gap in enumerate(gaps):
        ax2.text(gap + 0.005, i, f'{gap:.3f}', 
                va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {filename}")
    plt.close()


# =============================================================================
# Step 9: Refined Analysis with Top Features
# =============================================================================

def refined_analysis_top_features(X_raw, y, importance_df, best_model_name, 
                                  top_n_features=20, diagnosis_code=""):
    """
    Rerun analysis using ONLY the top N most important features.
    
    Returns:
        Refined results, pipeline, predictions, and feature list
    """
    print("\n" + "=" * 70)
    print(f"REFINED ANALYSIS: Top {top_n_features} Features ({diagnosis_code})")
    print("=" * 70)
    
    # Select top features
    top_features = importance_df.head(top_n_features)['feature'].tolist()
    X_refined = X_raw[top_features].copy()
    
    print(f"\n✅ Selected top {len(top_features)} features:")
    for i, feat in enumerate(top_features[:10], 1):
        imp = importance_df[importance_df['feature'] == feat]['importance'].values[0]
        print(f"   {i}. {feat}: {imp:.4f}")
    if len(top_features) > 10:
        print(f"   ... and {len(top_features) - 10} more")
    
    # Create model pipeline
    models = create_model_pipelines()
    refined_pipeline = models[best_model_name]
    
    # Run CV
    print(f"\n🔬 Running cross-validation with {best_model_name}...")
    skf = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1',
        'roc_auc': 'roc_auc'
    }
    
    cv_results = cross_validate(
        refined_pipeline, X_refined, y,
        cv=skf,
        scoring=scoring,
        n_jobs=1,
        return_train_score=True
    )
    
    # Print results
    print(f"\n📊 Refined Model Performance:")
    print(f"   Accuracy: {cv_results['test_accuracy'].mean():.3f} ± {cv_results['test_accuracy'].std():.3f}")
    print(f"   Precision: {cv_results['test_precision'].mean():.3f}")
    print(f"   Recall: {cv_results['test_recall'].mean():.3f}")
    print(f"   F1-Score: {cv_results['test_f1'].mean():.3f}")
    print(f"   ROC-AUC: {cv_results['test_roc_auc'].mean():.3f}")
    print(f"   Train Acc: {cv_results['train_accuracy'].mean():.3f}")
    print(f"   Overfit Gap: {cv_results['train_accuracy'].mean() - cv_results['test_accuracy'].mean():.3f}")
    
    # Train final model on all data
    print(f"\n🎯 Training final refined model...")
    refined_pipeline.fit(X_refined, y)
    
    # Get predictions
    y_pred = refined_pipeline.predict(X_refined)
    y_pred_proba = refined_pipeline.predict_proba(X_refined)[:, 1]
    
    # Save results
    refined_results = {
        'diagnosis': diagnosis_code,
        'n_features': len(top_features),
        'model': best_model_name,
        'accuracy': cv_results['test_accuracy'].mean(),
        'precision': cv_results['test_precision'].mean(),
        'recall': cv_results['test_recall'].mean(),
        'f1': cv_results['test_f1'].mean(),
        'roc_auc': cv_results['test_roc_auc'].mean(),
        'overfit_gap': cv_results['train_accuracy'].mean() - cv_results['test_accuracy'].mean()
    }
    
    return refined_results, refined_pipeline, X_refined, y_pred, y_pred_proba, top_features


def create_model_comparison_for_diagnosis(results_df, diagnosis_code, viz_folder):
    """
    Create visualization comparing all models for a single diagnosis.
    """
    print(f"\n📊 Creating model comparison for {diagnosis_code}...")
    
    # Sort by Composite Score (smart selection)
    results_sorted = results_df.sort_values('Composite_Score', ascending=False)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    models = results_sorted['Model'].values
    n_models = len(models)
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, n_models))
    
    # 1. Composite Score (SMART SELECTION)
    ax1 = axes[0, 0]
    bars1 = ax1.barh(range(n_models), results_sorted['Composite_Score'], color=colors)
    ax1.set_yticks(range(n_models))
    ax1.set_yticklabels(models, fontsize=9)
    ax1.set_xlabel('Composite Score', fontsize=11, fontweight='bold')
    ax1.set_title(f'🏆 Composite Score (Smart Selection)\n({diagnosis_code})', fontsize=12, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    ax1.invert_yaxis()
    for i, (bar, val) in enumerate(zip(bars1, results_sorted['Composite_Score'])):
        ax1.text(val + 0.01, i, f'{val:.3f}', va='center', fontsize=9, fontweight='bold')
    
    # 2. F1-Score comparison
    ax2 = axes[0, 1]
    bars2 = ax2.barh(range(n_models), results_sorted['F1'], color=colors)
    ax2.set_yticks(range(n_models))
    ax2.set_yticklabels(models, fontsize=9)
    ax2.set_xlabel('F1-Score', fontsize=11)
    ax2.set_title(f'F1-Score by Model\n({diagnosis_code})', fontsize=12, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    ax2.invert_yaxis()
    for i, (bar, val) in enumerate(zip(bars2, results_sorted['F1'])):
        ax2.text(val + 0.01, i, f'{val:.3f}', va='center', fontsize=9, fontweight='bold')
    
    # 3. ROC-AUC comparison
    ax3 = axes[0, 2]
    bars3 = ax3.barh(range(n_models), results_sorted['ROC_AUC'], color=colors)
    ax3.set_yticks(range(n_models))
    ax3.set_yticklabels(models, fontsize=9)
    ax3.set_xlabel('ROC-AUC', fontsize=11)
    ax3.set_title(f'ROC-AUC by Model\n({diagnosis_code})', fontsize=12, fontweight='bold')
    ax3.grid(axis='x', alpha=0.3)
    ax3.invert_yaxis()
    for i, (bar, val) in enumerate(zip(bars3, results_sorted['ROC_AUC'])):
        ax3.text(val + 0.01, i, f'{val:.3f}', va='center', fontsize=9, fontweight='bold')
    
    # 4. Accuracy with error bars
    ax4 = axes[1, 0]
    bars4 = ax4.barh(range(n_models), results_sorted['Accuracy'], 
                     xerr=results_sorted['Accuracy_Std'], color=colors, capsize=5)
    ax4.set_yticks(range(n_models))
    ax4.set_yticklabels(models, fontsize=9)
    ax4.set_xlabel('Accuracy ± Std', fontsize=11)
    ax4.set_title(f'Accuracy by Model\n({diagnosis_code})', fontsize=12, fontweight='bold')
    ax4.grid(axis='x', alpha=0.3)
    ax4.invert_yaxis()
    
    # 5. Overfitting analysis
    ax5 = axes[1, 1]
    gap_colors = ['red' if gap > 0.1 else 'orange' if gap > 0.05 else 'green' 
                  for gap in results_sorted['Overfit_Gap']]
    bars5 = ax5.barh(range(n_models), results_sorted['Overfit_Gap'], color=gap_colors, alpha=0.7)
    ax5.set_yticks(range(n_models))
    ax5.set_yticklabels(models, fontsize=9)
    ax5.set_xlabel('Overfit Gap (Train - Test)', fontsize=11)
    ax5.set_title(f'Generalization by Model\n({diagnosis_code})', fontsize=12, fontweight='bold')
    ax5.axvline(x=0.05, color='orange', linestyle='--', alpha=0.5, linewidth=2, label='Warning')
    ax5.axvline(x=0.1, color='red', linestyle='--', alpha=0.5, linewidth=2, label='High')
    ax5.grid(axis='x', alpha=0.3)
    ax5.legend(loc='best', fontsize=9)
    ax5.invert_yaxis()
    
    # 6. F1 vs Overfit Gap scatter
    ax6 = axes[1, 2]
    scatter_colors = ['green' if gap < 0.05 else 'orange' if gap < 0.1 else 'red' 
                     for gap in results_sorted['Overfit_Gap']]
    ax6.scatter(results_sorted['Overfit_Gap'], results_sorted['F1'], 
               c=scatter_colors, s=100, alpha=0.7, edgecolors='black', linewidths=1.5)
    
    # Annotate best model
    best_idx = 0
    ax6.annotate('BEST', 
                xy=(results_sorted.iloc[best_idx]['Overfit_Gap'], results_sorted.iloc[best_idx]['F1']),
                xytext=(10, 10), textcoords='offset points',
                fontsize=10, fontweight='bold', color='darkgreen',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='darkgreen', lw=2))
    
    ax6.set_xlabel('Overfit Gap', fontsize=11, fontweight='bold')
    ax6.set_ylabel('F1-Score', fontsize=11, fontweight='bold')
    ax6.set_title(f'F1 vs Generalization Trade-off\n({diagnosis_code})', fontsize=12, fontweight='bold')
    ax6.axvline(x=0.05, color='orange', linestyle='--', alpha=0.3, linewidth=2)
    ax6.axvline(x=0.1, color='red', linestyle='--', alpha=0.3, linewidth=2)
    ax6.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{viz_folder}/0_all_models_comparison.png', dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved: {viz_folder}/0_all_models_comparison.png")
    plt.close()


def create_best_model_summary(results_df, diagnosis_code, viz_folder):
    """
    Create dedicated visualization for the best performing model.
    """
    print(f"\n📊 Creating best model summary for {diagnosis_code}...")
    
    # Get best model
    best = results_df.sort_values('F1', ascending=False).iloc[0]
    
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Title
    fig.suptitle(f'Best Model Performance: {best["Model"]}\nDiagnosis: {diagnosis_code}', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # 1. Performance metrics (large, top-left)
    ax1 = fig.add_subplot(gs[0:2, 0])
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC_AUC']
    values = [best['Accuracy'], best['Precision'], best['Recall'], best['F1'], best['ROC_AUC']]
    colors_metrics = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']
    
    bars = ax1.barh(metrics, values, color=colors_metrics, alpha=0.8)
    ax1.set_xlim([0, 1])
    ax1.set_xlabel('Score', fontsize=11, fontweight='bold')
    ax1.set_title('Performance Metrics', fontsize=12, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars, values):
        ax1.text(val + 0.02, bar.get_y() + bar.get_height()/2, 
                f'{val:.3f}', va='center', fontsize=11, fontweight='bold')
    
    # 2. Train vs Test Accuracy (top-middle)
    ax2 = fig.add_subplot(gs[0, 1])
    train_test = ['Train', 'Test']
    accuracies = [best['Train_Acc'], best['Accuracy']]
    colors_tt = ['#3498db', '#e67e22']
    bars2 = ax2.bar(train_test, accuracies, color=colors_tt, alpha=0.8)
    ax2.set_ylim([0, 1])
    ax2.set_ylabel('Accuracy', fontsize=10, fontweight='bold')
    ax2.set_title('Train vs Test', fontsize=11, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars2, accuracies):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 0.02, 
                f'{val:.3f}', ha='center', fontsize=10, fontweight='bold')
    
    # 3. Overfitting gauge (top-right)
    ax3 = fig.add_subplot(gs[0, 2])
    gap = best['Overfit_Gap']
    gap_color = 'green' if gap < 0.05 else 'orange' if gap < 0.1 else 'red'
    gap_label = 'Excellent' if gap < 0.05 else 'Good' if gap < 0.1 else 'High'
    
    ax3.barh([0], [gap], color=gap_color, alpha=0.7, height=0.5)
    ax3.set_xlim([0, max(0.15, gap * 1.2)])
    ax3.set_ylim([-0.5, 0.5])
    ax3.set_xlabel('Overfit Gap', fontsize=10, fontweight='bold')
    ax3.set_title('Generalization', fontsize=11, fontweight='bold')
    ax3.set_yticks([])
    ax3.axvline(x=0.05, color='orange', linestyle='--', alpha=0.5, linewidth=2)
    ax3.axvline(x=0.1, color='red', linestyle='--', alpha=0.5, linewidth=2)
    ax3.text(gap + 0.005, 0, f'{gap:.3f}\n({gap_label})', 
            va='center', fontsize=10, fontweight='bold')
    ax3.grid(axis='x', alpha=0.3)
    
    # 4. Model info box (middle)
    ax4 = fig.add_subplot(gs[1, 1:])
    ax4.axis('off')
    
    info_text = f"""
    Model: {best['Model']}
    
    Key Metrics:
    • F1-Score: {best['F1']:.3f}
    • ROC-AUC: {best['ROC_AUC']:.3f}
    • Precision: {best['Precision']:.3f}
    • Recall: {best['Recall']:.3f}
    
    Generalization:
    • Train Accuracy: {best['Train_Acc']:.3f}
    • Test Accuracy: {best['Accuracy']:.3f} ± {best['Accuracy_Std']:.3f}
    • Overfit Gap: {best['Overfit_Gap']:.3f}
    
    Status: {gap_label} generalization
    """
    
    ax4.text(0.1, 0.5, info_text, fontsize=11, family='monospace',
            verticalalignment='center', bbox=dict(boxstyle='round', 
            facecolor='wheat', alpha=0.3))
    
    # 5. Metrics comparison radar/bar (bottom)
    ax5 = fig.add_subplot(gs[2, :])
    
    all_metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC_AUC']
    all_values = [best['Accuracy'], best['Precision'], best['Recall'], 
                  best['F1'], best['ROC_AUC']]
    
    x_pos = np.arange(len(all_metrics))
    bars5 = ax5.bar(x_pos, all_values, color=colors_metrics, alpha=0.8, width=0.6)
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(all_metrics, fontsize=11, fontweight='bold')
    ax5.set_ylim([0, 1])
    ax5.set_ylabel('Score', fontsize=11, fontweight='bold')
    ax5.set_title('Complete Performance Profile', fontsize=12, fontweight='bold')
    ax5.grid(axis='y', alpha=0.3)
    ax5.axhline(y=0.7, color='orange', linestyle='--', alpha=0.3, linewidth=1.5)
    ax5.axhline(y=0.8, color='green', linestyle='--', alpha=0.3, linewidth=1.5)
    
    for bar, val in zip(bars5, all_values):
        ax5.text(bar.get_x() + bar.get_width()/2, val + 0.02, 
                f'{val:.3f}', ha='center', fontsize=10, fontweight='bold')
    
    plt.savefig(f'{viz_folder}/0_best_model_summary.png', dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved: {viz_folder}/0_best_model_summary.png")
    plt.close()


def create_advanced_visualizations(X_refined, y, y_pred, y_pred_proba, 
                                   top_features, refined_pipeline, viz_folder):
    """
    Create comprehensive visualizations for refined model.
    """
    print("\n" + "=" * 70)
    print("Creating Advanced Visualizations")
    print("=" * 70)
    
    # Ensure viz folder exists
    Path(viz_folder).mkdir(parents=True, exist_ok=True)
    
    # 1. Refined Feature Importances
    print("\n📊 1. Refined feature importances...")
    trained_model = refined_pipeline.named_steps['model']
    
    if hasattr(trained_model, 'feature_importances_'):
        dropper = refined_pipeline.named_steps['dropper']
        kept_features = dropper.cols_to_keep_
        
        refined_imp_df = pd.DataFrame({
            'feature': kept_features,
            'importance': trained_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(refined_imp_df)), refined_imp_df['importance'], color='coral')
        plt.yticks(range(len(refined_imp_df)), refined_imp_df['feature'])
        plt.xlabel('Importance', fontsize=12)
        plt.title('Refined Model: Feature Importances', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(f'{viz_folder}/1_refined_feature_importance.png', dpi=300, bbox_inches='tight')
        print(f"   ✅ Saved: {viz_folder}/1_refined_feature_importance.png")
        plt.close()
    
    # 2. Prediction Analysis
    print("\n📊 2. Prediction distribution and confusion matrix...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Predicted probabilities by true class
    ax1 = axes[0]
    for label in [0, 1]:
        probs = y_pred_proba[y == label]
        ax1.hist(probs, bins=30, alpha=0.6, label=f'True Class {label}', density=True)
    ax1.set_xlabel('Predicted Probability', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_title('Prediction Distribution by True Class', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Confusion matrix
    ax2 = axes[1]
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2, cbar=False)
    ax2.set_xlabel('Predicted', fontsize=12)
    ax2.set_ylabel('True', fontsize=12)
    ax2.set_title('Confusion Matrix', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{viz_folder}/2_prediction_analysis.png', dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved: {viz_folder}/2_prediction_analysis.png")
    plt.close()
    
    # 3. Feature Distributions by Class
    print("\n📊 3. Feature distributions by diagnosis class...")
    n_features_to_plot = min(9, len(kept_features))
    n_cols = 3
    n_rows = (n_features_to_plot + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    for idx, feat in enumerate(kept_features[:n_features_to_plot]):
        ax = axes[idx]
        
        # Get feature values (impute for visualization)
        feat_vals = X_refined[feat].fillna(X_refined[feat].median())
        
        for label in [0, 1]:
            vals = feat_vals[y == label]
            ax.hist(vals, bins=20, alpha=0.6, label=f'Diagnosis {label}', density=True)
        
        ax.set_xlabel(feat, fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.set_title(f'{feat}', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    
    # Hide empty subplots
    for idx in range(n_features_to_plot, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{viz_folder}/3_feature_distributions.png', dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved: {viz_folder}/3_feature_distributions.png")
    plt.close()
    
    # 4. Calibration Analysis
    print("\n📊 4. Model calibration and confidence analysis...")
    
    # Bin predictions by confidence
    bins = np.linspace(0, 1, 11)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_accuracies = []
    bin_counts = []
    
    for i in range(len(bins) - 1):
        mask = (y_pred_proba >= bins[i]) & (y_pred_proba < bins[i+1])
        if mask.sum() > 0:
            bin_acc = (y_pred[mask] == y[mask]).mean()
            bin_accuracies.append(bin_acc)
            bin_counts.append(mask.sum())
        else:
            bin_accuracies.append(np.nan)
            bin_counts.append(0)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Calibration curve
    ax1 = axes[0]
    ax1.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=2)
    ax1.plot(bin_centers, bin_accuracies, 'o-', color='coral', 
             label='Model', linewidth=2, markersize=8)
    ax1.set_xlabel('Mean Predicted Probability', fontsize=12)
    ax1.set_ylabel('Fraction of Positives', fontsize=12)
    ax1.set_title('Calibration Curve', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    
    # Sample distribution
    ax2 = axes[1]
    ax2.bar(bin_centers, bin_counts, width=0.08, color='steelblue', alpha=0.7)
    ax2.set_xlabel('Predicted Probability', fontsize=12)
    ax2.set_ylabel('Number of Samples', fontsize=12)
    ax2.set_title('Prediction Confidence Distribution', fontsize=13, fontweight='bold')
    ax2.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{viz_folder}/4_calibration_analysis.png', dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved: {viz_folder}/4_calibration_analysis.png")
    plt.close()
    
    # 5. Performance Metrics Summary
    print("\n📊 5. Performance metrics summary...")
    
    from sklearn.metrics import precision_recall_curve, roc_curve, auc
    
    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    # ROC curve
    fpr, tpr, _ = roc_curve(y, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Precision-Recall
    ax1 = axes[0]
    ax1.plot(recall, precision, color='darkorange', linewidth=2, 
             label=f'PR curve (AUC = {pr_auc:.3f})')
    ax1.set_xlabel('Recall', fontsize=12)
    ax1.set_ylabel('Precision', fontsize=12)
    ax1.set_title('Precision-Recall Curve', fontsize=13, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(alpha=0.3)
    
    # ROC
    ax2 = axes[1]
    ax2.plot(fpr, tpr, color='darkorange', linewidth=2, 
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax2.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random')
    ax2.set_xlabel('False Positive Rate', fontsize=12)
    ax2.set_ylabel('True Positive Rate', fontsize=12)
    ax2.set_title('ROC Curve', fontsize=13, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{viz_folder}/5_performance_curves.png', dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved: {viz_folder}/5_performance_curves.png")
    plt.close()
    
    print("\n✅ All advanced visualizations complete!")


# =============================================================================
# Step 9: Advanced Feature Analysis
# =============================================================================

def analyze_feature_stability(X_raw, y, top_features, best_pipeline, n_folds=5):
    """
    Analyze how stable feature importances are across CV folds.
    ONLY runs on pre-filtered top features (not all 3000+).
    Returns DataFrame with stability metrics for each feature.
    """
    print("\n" + "=" * 70)
    print(f"Feature Stability Analysis (Top {len(top_features)} features)")
    print("=" * 70)
    
    X_top = X_raw[top_features].copy()
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    
    # Store importances from each fold
    fold_importances = defaultdict(list)
    
    print(f"   Running {n_folds}-fold CV to assess stability...")
    
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_top, y), 1):
        X_train = X_top.iloc[train_idx]
        y_train = y.iloc[train_idx]
        
        # Fit pipeline
        best_pipeline.fit(X_train, y_train)
        
        # Get trained model
        trained_model = best_pipeline.named_steps['model']
        
        # Get feature names after column dropping
        dropper = best_pipeline.named_steps['dropper']
        kept_features = dropper.cols_to_keep_
        
        # Extract importances
        if hasattr(trained_model, 'feature_importances_'):
            importances = trained_model.feature_importances_
            
            for feat, imp in zip(kept_features, importances):
                fold_importances[feat].append(imp)
    
    # Calculate stability metrics
    stability_results = []
    for feature in top_features:
        if feature in fold_importances and len(fold_importances[feature]) > 0:
            imps = fold_importances[feature]
            mean_imp = np.mean(imps)
            std_imp = np.std(imps)
            
            stability_results.append({
                'feature': feature,
                'mean_importance': mean_imp,
                'std_importance': std_imp,
                'cv': std_imp / (mean_imp + 1e-10),  # Coefficient of variation
                'min_importance': np.min(imps),
                'max_importance': np.max(imps),
                'range': np.max(imps) - np.min(imps),
                'stability_score': 1.0 / (1.0 + std_imp)  # Higher = more stable
            })
    
    stability_df = pd.DataFrame(stability_results).sort_values('stability_score', ascending=False)
    
    print(f"✅ Analyzed {len(stability_df)} features across {n_folds} folds")
    print(f"\n   Most Stable (Top 5):")
    for idx, row in stability_df.head(5).iterrows():
        print(f"      {row['feature']}: stability={row['stability_score']:.3f}, CV={row['cv']:.3f}")
    
    return stability_df


def analyze_univariate_power(X_raw, y, top_features):
    """
    Calculate univariate discriminative power (ROC-AUC) for top features.
    Much faster than full dataset analysis - only runs on pre-filtered features.
    """
    print("\n" + "=" * 70)
    print(f"Univariate Discriminative Power (Top {len(top_features)} features)")
    print("=" * 70)
    
    from sklearn.metrics import roc_auc_score
    from sklearn.impute import SimpleImputer
    
    univariate_results = []
    
    print(f"   Calculating individual feature ROC-AUC scores...")
    
    for feature in top_features:
        try:
            # Get feature values
            feature_values = X_raw[feature].values.reshape(-1, 1)
            
            # Impute missing values
            imputer = SimpleImputer(strategy='median')
            feature_imputed = imputer.fit_transform(feature_values).ravel()
            
            # Calculate ROC-AUC
            try:
                auc = roc_auc_score(y, feature_imputed)
                # If AUC < 0.5, the feature has inverse relationship
                auc = max(auc, 1 - auc)
            except:
                auc = 0.5
            
            # Calculate effect size (Cohen's d)
            class_vals = y.unique()
            class_0_mean = feature_imputed[y == class_vals[0]].mean()
            class_1_mean = feature_imputed[y == class_vals[1]].mean()
            pooled_std = feature_imputed.std()
            cohens_d = abs(class_1_mean - class_0_mean) / (pooled_std + 1e-10)
            
            univariate_results.append({
                'feature': feature,
                'univariate_auc': auc,
                'cohens_d': cohens_d,
                'class_0_mean': class_0_mean,
                'class_1_mean': class_1_mean,
                'difference': abs(class_1_mean - class_0_mean)
            })
        except Exception as e:
            # Silent fail for problematic features
            pass
    
    univariate_df = pd.DataFrame(univariate_results).sort_values('univariate_auc', ascending=False)
    
    print(f"✅ Analyzed {len(univariate_df)} features")
    print(f"\n   Top 5 by Univariate AUC:")
    for idx, row in univariate_df.head(5).iterrows():
        print(f"      {row['feature']}: AUC={row['univariate_auc']:.3f}, Cohen's d={row['cohens_d']:.3f}")
    
    return univariate_df


def analyze_global_feature_statistics(X_raw, feature_names):
    """
    EFFICIENT global analysis across ALL features (even 3000+).
    Calculates summary statistics without expensive computations.
    """
    print("\n" + "=" * 70)
    print(f"Global Feature Statistics (All {len(feature_names)} features)")
    print("=" * 70)
    
    global_stats = []
    
    print(f"   Computing missingness, variance, and basic stats...")
    
    for feature in feature_names:
        if feature not in X_raw.columns:
            continue
            
        values = X_raw[feature]
        
        # Fast statistics
        missing_pct = values.isna().sum() / len(values)
        
        # Only compute variance for non-missing values
        non_missing = values.dropna()
        if len(non_missing) > 0:
            variance = non_missing.var()
            mean_val = non_missing.mean()
            std_val = non_missing.std()
            n_unique = non_missing.nunique()
        else:
            variance = 0
            mean_val = 0
            std_val = 0
            n_unique = 0
        
        global_stats.append({
            'feature': feature,
            'missing_pct': missing_pct,
            'variance': variance,
            'mean': mean_val,
            'std': std_val,
            'n_unique': n_unique,
            'data_quality': 1.0 - missing_pct  # Higher = better
        })
    
    global_df = pd.DataFrame(global_stats)
    
    print(f"✅ Analyzed {len(global_df)} features")
    print(f"\n   Summary:")
    print(f"      Avg missing: {global_df['missing_pct'].mean()*100:.1f}%")
    print(f"      Zero variance features: {(global_df['variance'] == 0).sum()}")
    print(f"      High quality (>80% complete): {(global_df['missing_pct'] < 0.2).sum()}")
    
    return global_df


def calculate_feature_quality_score(stability_df, univariate_df, importance_df, global_df=None):
    """
    Calculate composite feature quality score combining:
    - Model importance (from trained model)
    - Stability across folds
    - Univariate discriminative power
    - Data quality (if global stats provided)
    
    Focus on TOP features only (not all 3000).
    """
    print("\n" + "=" * 70)
    print("Calculating Composite Feature Quality Scores")
    print("=" * 70)
    
    # Start with model importance (top features only)
    quality_df = importance_df.copy()
    
    # Add stability metrics
    stability_dict = dict(zip(stability_df['feature'], stability_df['stability_score']))
    quality_df['stability_score'] = quality_df['feature'].map(stability_dict).fillna(0)
    
    # Add univariate AUC
    univariate_dict = dict(zip(univariate_df['feature'], univariate_df['univariate_auc']))
    quality_df['univariate_auc'] = quality_df['feature'].map(univariate_dict).fillna(0.5)
    
    # Add data quality if available
    if global_df is not None:
        quality_dict = dict(zip(global_df['feature'], global_df['data_quality']))
        quality_df['data_quality'] = quality_df['feature'].map(quality_dict).fillna(0.5)
    else:
        quality_df['data_quality'] = 1.0  # Assume good quality
    
    # Normalize all metrics to 0-1 range
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    
    quality_df['importance_norm'] = scaler.fit_transform(quality_df[['importance']])
    quality_df['stability_norm'] = scaler.fit_transform(quality_df[['stability_score']])
    quality_df['univariate_norm'] = scaler.fit_transform(quality_df[['univariate_auc']])
    quality_df['data_quality_norm'] = scaler.fit_transform(quality_df[['data_quality']])
    
    # Calculate composite quality score (weighted average)
    quality_df['quality_score'] = (
        0.35 * quality_df['importance_norm'] +      # 35% model importance
        0.25 * quality_df['stability_norm'] +       # 25% stability
        0.25 * quality_df['univariate_norm'] +      # 25% univariate power
        0.15 * quality_df['data_quality_norm']      # 15% data quality
    )
    
    quality_df = quality_df.sort_values('quality_score', ascending=False)
    
    print(f"✅ Calculated quality scores for {len(quality_df)} features")
    print(f"\n   Top 5 Features by Quality Score:")
    for idx, row in quality_df.head(5).iterrows():
        print(f"      #{idx+1}: {row['feature']}")
        print(f"         Quality: {row['quality_score']:.3f}, Importance: {row['importance']:.3f}, "
              f"Stability: {row['stability_score']:.3f}, AUC: {row['univariate_auc']:.3f}")
    
    return quality_df


def create_advanced_feature_analysis_report(X_raw, y, importance_df, best_pipeline, 
                                            diagnosis_code, viz_folder, top_n=20):
    """
    Create comprehensive feature analysis with focus on efficiency.
    
    Strategy:
    1. Global stats on ALL features (fast)
    2. Detailed analysis ONLY on top N features (slow but focused)
    3. Distribution plots ONLY for top 10-12 features
    """
    print("\n" + "=" * 70)
    print(f"ADVANCED FEATURE ANALYSIS FOR {diagnosis_code}")
    print("=" * 70)
    print(f"   Strategy: Global stats (all features) + Deep dive (top {top_n})")
    
    # 1. FAST: Global statistics on all features
    all_feature_names = X_raw.columns.tolist()
    global_df = analyze_global_feature_statistics(X_raw, all_feature_names)
    
    # 2. FOCUSED: Detailed analysis on top N features only
    top_features = importance_df.head(top_n)['feature'].tolist()
    
    print(f"\n{'='*70}")
    print(f"DEEP DIVE ANALYSIS: Top {len(top_features)} Features")
    print(f"{'='*70}")
    
    stability_df = analyze_feature_stability(X_raw, y, top_features, best_pipeline)
    univariate_df = analyze_univariate_power(X_raw, y, top_features)
    
    # 3. Calculate quality scores
    quality_df = calculate_feature_quality_score(stability_df, univariate_df, importance_df, global_df)
    
    # Save comprehensive CSVs
    csv_path = f'{viz_folder}/feature_analysis_complete.csv'
    quality_df.to_csv(csv_path, index=False)
    print(f"\n✅ Saved detailed analysis: {csv_path}")
    
    global_csv_path = f'{viz_folder}/global_feature_stats.csv'
    global_df.to_csv(global_csv_path, index=False)
    print(f"✅ Saved global stats: {global_csv_path}")
    
    # Create visualizations (focused on top features)
    create_feature_quality_dashboard(quality_df, stability_df, univariate_df, 
                                     global_df, diagnosis_code, viz_folder, top_n=min(20, len(quality_df)))
    
    # Distribution grid only for top 10 features
    create_feature_distribution_grid(X_raw, y, quality_df.head(10)['feature'].tolist(), 
                                    diagnosis_code, viz_folder)
    
    create_feature_executive_summary(quality_df, diagnosis_code, viz_folder)
    
    return quality_df


def create_feature_quality_dashboard(quality_df, stability_df, univariate_df, 
                                     global_df, diagnosis_code, viz_folder, top_n=20):
    """
    Create comprehensive dashboard showing all feature quality metrics.
    """
    print(f"\n📊 Creating feature quality dashboard...")
    
    top_features_df = quality_df.head(top_n)
    
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)
    
    # Overall title
    fig.suptitle(f'Comprehensive Feature Quality Analysis: {diagnosis_code}\nTop {top_n} Features', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # 1. Feature Quality Score (composite metric)
    ax1 = fig.add_subplot(gs[0, :])
    colors_quality = plt.cm.RdYlGn(top_features_df['quality_score'] / top_features_df['quality_score'].max())
    bars1 = ax1.barh(range(len(top_features_df)), top_features_df['quality_score'], color=colors_quality)
    ax1.set_yticks(range(len(top_features_df)))
    ax1.set_yticklabels(top_features_df['feature'], fontsize=9)
    ax1.set_xlabel('Composite Quality Score (0-1)', fontsize=11, fontweight='bold')
    ax1.set_title('🏆 Overall Feature Quality Ranking\n(40% Importance + 30% Stability + 30% Univariate Power)', 
                  fontsize=13, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    ax1.invert_yaxis()
    for i, (bar, val) in enumerate(zip(bars1, top_features_df['quality_score'])):
        ax1.text(val + 0.01, i, f'{val:.3f}', va='center', fontsize=8, fontweight='bold')
    
    # 2. Model Importance
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.barh(range(len(top_features_df)), top_features_df['importance'], color='steelblue', alpha=0.7)
    ax2.set_yticks(range(len(top_features_df)))
    ax2.set_yticklabels(top_features_df['feature'], fontsize=8)
    ax2.set_xlabel('Importance', fontsize=10, fontweight='bold')
    ax2.set_title('Model Feature Importance\n(From trained model)', fontsize=11, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    ax2.invert_yaxis()
    
    # 3. Stability Score
    ax3 = fig.add_subplot(gs[1, 1])
    colors_stability = ['green' if s > 0.8 else 'orange' if s > 0.6 else 'red' 
                       for s in top_features_df['stability_score']]
    ax3.barh(range(len(top_features_df)), top_features_df['stability_score'], 
            color=colors_stability, alpha=0.7)
    ax3.set_yticks(range(len(top_features_df)))
    ax3.set_yticklabels(top_features_df['feature'], fontsize=8)
    ax3.set_xlabel('Stability Score', fontsize=10, fontweight='bold')
    ax3.set_title('Cross-Fold Stability\n(Consistency across CV folds)', fontsize=11, fontweight='bold')
    ax3.grid(axis='x', alpha=0.3)
    ax3.invert_yaxis()
    
    # 4. Univariate AUC
    ax4 = fig.add_subplot(gs[1, 2])
    colors_auc = plt.cm.RdYlGn((top_features_df['univariate_auc'] - 0.5) * 2)  # Scale 0.5-1.0 to 0-1
    ax4.barh(range(len(top_features_df)), top_features_df['univariate_auc'], color=colors_auc, alpha=0.7)
    ax4.set_yticks(range(len(top_features_df)))
    ax4.set_yticklabels(top_features_df['feature'], fontsize=8)
    ax4.set_xlabel('Univariate ROC-AUC', fontsize=10, fontweight='bold')
    ax4.set_title('Individual Discriminative Power\n(Feature alone)', fontsize=11, fontweight='bold')
    ax4.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax4.grid(axis='x', alpha=0.3)
    ax4.invert_yaxis()
    
    # 5. Scatter: Quality vs Importance
    ax5 = fig.add_subplot(gs[2, 0])
    scatter = ax5.scatter(top_features_df['importance'], top_features_df['quality_score'],
                         c=top_features_df['stability_score'], cmap='RdYlGn', 
                         s=100, alpha=0.7, edgecolors='black', linewidth=1)
    ax5.set_xlabel('Model Importance', fontsize=10, fontweight='bold')
    ax5.set_ylabel('Quality Score', fontsize=10, fontweight='bold')
    ax5.set_title('Quality vs Importance\n(colored by stability)', fontsize=11, fontweight='bold')
    ax5.grid(alpha=0.3)
    plt.colorbar(scatter, ax=ax5, label='Stability')
    
    # 6. Scatter: Stability vs Univariate
    ax6 = fig.add_subplot(gs[2, 1])
    scatter2 = ax6.scatter(top_features_df['univariate_auc'], top_features_df['stability_score'],
                          c=top_features_df['importance'], cmap='viridis',
                          s=100, alpha=0.7, edgecolors='black', linewidth=1)
    ax6.set_xlabel('Univariate AUC', fontsize=10, fontweight='bold')
    ax6.set_ylabel('Stability Score', fontsize=10, fontweight='bold')
    ax6.set_title('Stability vs Discriminative Power\n(colored by importance)', fontsize=11, fontweight='bold')
    ax6.grid(alpha=0.3)
    plt.colorbar(scatter2, ax=ax6, label='Importance')
    
    # 7. Radar/Spider chart for top 5 features
    ax7 = fig.add_subplot(gs[2, 2], projection='polar')
    top5 = top_features_df.head(5)
    
    categories = ['Importance', 'Stability', 'Univariate\nAUC']
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    
    for idx, row in top5.iterrows():
        values = [
            row['importance_norm'],
            row['stability_norm'],
            row['univariate_norm']
        ]
        values += values[:1]
        ax7.plot(angles, values, 'o-', linewidth=2, label=row['feature'][:15])
        ax7.fill(angles, values, alpha=0.15)
    
    ax7.set_xticks(angles[:-1])
    ax7.set_xticklabels(categories, fontsize=9)
    ax7.set_ylim(0, 1)
    ax7.set_title('Top 5 Features Profile\n(Normalized metrics)', fontsize=11, fontweight='bold', pad=20)
    ax7.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=8)
    ax7.grid(True)
    
    # 8. Data Quality (missingness) - GLOBAL metric
    ax8 = fig.add_subplot(gs[3, :2])
    
    # Merge with global data quality stats
    quality_merged = top_features_df.copy()
    if global_df is not None:
        quality_dict = dict(zip(global_df['feature'], global_df['data_quality']))
        quality_merged['data_quality_pct'] = quality_merged['feature'].map(quality_dict).fillna(1.0) * 100
    else:
        quality_merged['data_quality_pct'] = 100.0
    
    quality_merged = quality_merged.head(15)  # Top 15 for readability
    
    colors_quality = ['green' if q > 90 else 'orange' if q > 75 else 'red' 
                     for q in quality_merged['data_quality_pct']]
    ax8.barh(range(len(quality_merged)), quality_merged['data_quality_pct'], color=colors_quality, alpha=0.7)
    ax8.set_yticks(range(len(quality_merged)))
    ax8.set_yticklabels(quality_merged['feature'], fontsize=8)
    ax8.set_xlabel('Data Completeness (%)', fontsize=10, fontweight='bold')
    ax8.set_title('Feature Data Quality\n(% non-missing values)', 
                 fontsize=11, fontweight='bold')
    ax8.axvline(x=75, color='orange', linestyle='--', alpha=0.7, linewidth=2, label='75%')
    ax8.axvline(x=90, color='green', linestyle='--', alpha=0.7, linewidth=2, label='90%')
    ax8.set_xlim([0, 105])
    ax8.grid(axis='x', alpha=0.3)
    ax8.legend(fontsize=8)
    ax8.invert_yaxis()
    
    # 9. Feature ranking comparison
    ax9 = fig.add_subplot(gs[3, 2])
    top10 = top_features_df.head(10)
    
    # Create ranking comparison
    x = np.arange(len(top10))
    width = 0.25
    
    # Rank by each metric (lower rank = better)
    importance_ranks = top10['importance'].rank(ascending=False, method='min')
    stability_ranks = top10['stability_score'].rank(ascending=False, method='min')
    univariate_ranks = top10['univariate_auc'].rank(ascending=False, method='min')
    
    ax9.barh(x - width, importance_ranks, width, label='Importance Rank', alpha=0.8, color='steelblue')
    ax9.barh(x, stability_ranks, width, label='Stability Rank', alpha=0.8, color='orange')
    ax9.barh(x + width, univariate_ranks, width, label='Univariate Rank', alpha=0.8, color='green')
    
    ax9.set_yticks(x)
    ax9.set_yticklabels(top10['feature'], fontsize=8)
    ax9.set_xlabel('Rank (lower = better)', fontsize=10, fontweight='bold')
    ax9.set_title('Ranking Comparison\n(Top 10 features)', fontsize=11, fontweight='bold')
    ax9.legend(fontsize=8)
    ax9.invert_yaxis()
    ax9.invert_xaxis()
    ax9.grid(axis='x', alpha=0.3)
    
    plt.savefig(f'{viz_folder}/6_feature_quality_dashboard.png', dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved: {viz_folder}/6_feature_quality_dashboard.png")
    plt.close()


def create_feature_distribution_grid(X_raw, y, top_features, diagnosis_code, viz_folder):
    """
    Create grid of distribution plots for top features showing class separation.
    """
    print(f"\n📊 Creating feature distribution grid...")
    
    from sklearn.impute import SimpleImputer
    
    n_features = min(12, len(top_features))
    features_to_plot = top_features[:n_features]
    
    n_cols = 4
    n_rows = int(np.ceil(n_features / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4*n_rows))
    axes = axes.flatten() if n_features > 1 else [axes]
    
    fig.suptitle(f'Feature Distributions by Class: {diagnosis_code}\nTop {n_features} Features', 
                 fontsize=15, fontweight='bold', y=0.998)
    
    class_labels = y.unique()
    colors = ['steelblue', 'coral']
    
    for idx, feature in enumerate(features_to_plot):
        ax = axes[idx]
        
        # Impute missing values
        feature_values = X_raw[feature].values.reshape(-1, 1)
        imputer = SimpleImputer(strategy='median')
        feature_imputed = imputer.fit_transform(feature_values).ravel()
        
        # Split by class
        for class_idx, class_label in enumerate(class_labels):
            class_data = feature_imputed[y == class_label]
            ax.hist(class_data, bins=30, alpha=0.6, label=f'Class {class_label}', 
                   color=colors[class_idx], edgecolor='black', linewidth=0.5)
        
        ax.set_xlabel('Value', fontsize=9)
        ax.set_ylabel('Frequency', fontsize=9)
        ax.set_title(f'{feature}', fontsize=10, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_features, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{viz_folder}/7_feature_distributions.png', dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved: {viz_folder}/7_feature_distributions.png")
    plt.close()


def create_feature_executive_summary(quality_df, diagnosis_code, viz_folder):
    """
    Create single-page executive summary of best features for presentation.
    """
    print(f"\n📊 Creating executive summary...")
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    fig.suptitle(f'Feature Analysis Executive Summary: {diagnosis_code}\nBest Features for Prediction', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    top10 = quality_df.head(10)
    
    # 1. Top 10 Features (left, spans 2 rows)
    ax1 = fig.add_subplot(gs[:2, 0])
    colors = plt.cm.RdYlGn(top10['quality_score'] / top10['quality_score'].max())
    bars = ax1.barh(range(len(top10)), top10['quality_score'], color=colors)
    ax1.set_yticks(range(len(top10)))
    ax1.set_yticklabels([f"#{i+1}: {feat}" for i, feat in enumerate(top10['feature'])], fontsize=11)
    ax1.set_xlabel('Quality Score', fontsize=12, fontweight='bold')
    ax1.set_title('🏆 Top 10 Features\n(Ranked by composite quality)', fontsize=14, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    ax1.invert_yaxis()
    
    for i, (bar, val) in enumerate(zip(bars, top10['quality_score'])):
        ax1.text(val + 0.01, i, f'{val:.3f}', va='center', fontsize=10, fontweight='bold')
    
    # 2. Metrics comparison table (top right)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis('off')
    
    table_data = []
    for _, row in top10.head(5).iterrows():
        table_data.append([
            row['feature'][:20],
            f"{row['importance']:.3f}",
            f"{row['stability_score']:.3f}",
            f"{row['univariate_auc']:.3f}"
        ])
    
    table = ax2.table(cellText=table_data,
                     colLabels=['Feature', 'Importance', 'Stability', 'Univ. AUC'],
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header
    for i in range(4):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(table_data) + 1):
        for j in range(4):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    ax2.set_title('Top 5 Features - Detailed Metrics', fontsize=13, fontweight='bold', pad=20)
    
    # 3. Quality score components breakdown (middle right)
    ax3 = fig.add_subplot(gs[1, 1])
    
    metrics = ['Importance', 'Stability', 'Univ. AUC']
    weights = [0.4, 0.3, 0.3]
    colors_pie = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    wedges, texts, autotexts = ax3.pie(weights, labels=metrics, colors=colors_pie,
                                        autopct='%1.0f%%', startangle=90,
                                        textprops={'fontsize': 11, 'fontweight': 'bold'})
    ax3.set_title('Quality Score Components\n(Weighting scheme)', fontsize=13, fontweight='bold')
    
    # 4. Key insights (bottom, full width)
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')
    
    # Calculate insights
    n_highly_stable = (quality_df['stability_score'] > 0.8).sum()
    avg_univariate_auc = quality_df['univariate_auc'].mean()
    avg_data_quality = quality_df['data_quality'].mean() * 100
    best_feature = quality_df.iloc[0]['feature']
    best_score = quality_df.iloc[0]['quality_score']
    
    insights_text = f"""
    📊 KEY INSIGHTS FOR {diagnosis_code}:
    
    ✅ Best Feature: {best_feature} (Quality Score: {best_score:.3f})
    
    🎯 Feature Statistics:
       • Features analyzed (deep dive): {len(quality_df)}
       • Highly stable features (score > 0.8): {n_highly_stable}
       • Average univariate AUC: {avg_univariate_auc:.3f}
       • Average data completeness: {avg_data_quality:.1f}%
    
    💡 Quality Score Formula:
       = 35% × Model Importance + 25% × Stability + 25% × Univariate Power + 15% × Data Quality
    
    🏆 Top 3 Recommendations:
       1. {quality_df.iloc[0]['feature']} - {quality_df.iloc[0]['quality_score']:.3f}
       2. {quality_df.iloc[1]['feature']} - {quality_df.iloc[1]['quality_score']:.3f}
       3. {quality_df.iloc[2]['feature']} - {quality_df.iloc[2]['quality_score']:.3f}
    
    📈 These features combine:
       • High predictive power in multivariate model
       • Consistent importance across CV folds
       • Strong individual discriminative ability
       • High data completeness (minimal missingness)
    """
    
    ax4.text(0.05, 0.5, insights_text, fontsize=11, family='monospace',
            verticalalignment='center',
            bbox=dict(boxstyle='round,pad=1', facecolor='lightyellow', alpha=0.8, edgecolor='gray', linewidth=2))
    
    plt.savefig(f'{viz_folder}/8_executive_summary.png', dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved: {viz_folder}/8_executive_summary.png")
    plt.close()


# =============================================================================
# Step 10: Comparative Analysis Across All Diagnoses
# =============================================================================

def aggregate_feature_importances(all_feature_importances):
    """
    Aggregate feature importances across all diagnosis runs.
    
    Args:
        all_feature_importances: List of dicts with 'diagnosis' and 'importance_df'
    
    Returns:
        feature_summary_df: Complete aggregated statistics (for CSV)
        top20_features: Top 20 features for visualization
    """
    print("\n" + "=" * 70)
    print("AGGREGATING FEATURE IMPORTANCES ACROSS ALL DIAGNOSES")
    print("=" * 70)
    
    if not all_feature_importances:
        print("⚠️ No feature importances to aggregate")
        return None, None
    
    # Collect all feature importances
    all_features = defaultdict(list)
    
    for item in all_feature_importances:
        dx = item['diagnosis']
        importance_df = item['importance_df']
        
        if importance_df is not None and not importance_df.empty:
            for _, row in importance_df.iterrows():
                all_features[row['feature']].append({
                    'diagnosis': dx,
                    'importance': row['importance']
                })
    
    print(f"\n✅ Collected importances for {len(all_features)} unique features")
    print(f"   Across {len(all_feature_importances)} diagnoses")
    
    # Aggregate statistics
    feature_stats = []
    
    for feature, data in all_features.items():
        importances = [d['importance'] for d in data]
        diagnoses_appeared = [d['diagnosis'] for d in data]
        
        feature_stats.append({
            'feature': feature,
            'frequency': len(importances),  # How many diagnoses this feature appeared in
            'avg_importance': np.mean(importances),
            'std_importance': np.std(importances),
            'min_importance': np.min(importances),
            'max_importance': np.max(importances),
            'median_importance': np.median(importances),
            'total_weight': np.sum(importances),  # Cumulative importance
            'diagnoses': ', '.join(diagnoses_appeared[:5]) + ('...' if len(diagnoses_appeared) > 5 else '')
        })
    
    feature_summary_df = pd.DataFrame(feature_stats)
    
    # Sort by composite metric: frequency * avg_importance
    feature_summary_df['composite_score'] = (
        feature_summary_df['frequency'] * feature_summary_df['avg_importance']
    )
    feature_summary_df = feature_summary_df.sort_values('composite_score', ascending=False)
    
    print(f"\n📊 Feature Statistics:")
    print(f"   Most frequent: {feature_summary_df.iloc[0]['feature']} ({feature_summary_df.iloc[0]['frequency']} diagnoses)")
    print(f"   Highest avg importance: {feature_summary_df.nlargest(1, 'avg_importance').iloc[0]['feature']}")
    
    # Get top 20 for visualization
    top20_features = feature_summary_df.head(20).copy()
    
    return feature_summary_df, top20_features


def visualize_feature_importance_overlap(top20_features, all_feature_importances, filename):
    """
    Create comprehensive visualization of top 20 features across all runs.
    """
    print("\n📊 Creating feature importance overlap visualization...")
    
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # 1. Top 20 by Composite Score (Frequency × Avg Importance)
    ax1 = fig.add_subplot(gs[0, :])
    colors_comp = plt.cm.RdYlGn(top20_features['composite_score'] / top20_features['composite_score'].max())
    bars1 = ax1.barh(range(len(top20_features)), top20_features['composite_score'], color=colors_comp)
    ax1.set_yticks(range(len(top20_features)))
    ax1.set_yticklabels(top20_features['feature'], fontsize=10)
    ax1.set_xlabel('Composite Score (Frequency × Avg Importance)', fontsize=12, fontweight='bold')
    ax1.set_title('Top 20 Most Important Features Across All Diagnoses\n(Ranked by Frequency × Average Importance)', 
                  fontsize=14, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    ax1.invert_yaxis()
    
    # Add value labels
    for i, (bar, val, freq) in enumerate(zip(bars1, top20_features['composite_score'], top20_features['frequency'])):
        ax1.text(val + val*0.02, i, f'{val:.3f} (n={freq})', 
                va='center', fontsize=9, fontweight='bold')
    
    # 2. Frequency Distribution
    ax2 = fig.add_subplot(gs[1, 0])
    colors_freq = plt.cm.Blues(top20_features['frequency'] / top20_features['frequency'].max())
    ax2.barh(range(len(top20_features)), top20_features['frequency'], color=colors_freq)
    ax2.set_yticks(range(len(top20_features)))
    ax2.set_yticklabels(top20_features['feature'], fontsize=9)
    ax2.set_xlabel('Number of Diagnoses', fontsize=11, fontweight='bold')
    ax2.set_title('Feature Frequency\n(How many diagnoses each feature appeared in)', 
                  fontsize=12, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    ax2.invert_yaxis()
    
    # 3. Average Importance
    ax3 = fig.add_subplot(gs[1, 1])
    colors_imp = plt.cm.Oranges(top20_features['avg_importance'] / top20_features['avg_importance'].max())
    ax3.barh(range(len(top20_features)), top20_features['avg_importance'], 
            xerr=top20_features['std_importance'], color=colors_imp, capsize=3)
    ax3.set_yticks(range(len(top20_features)))
    ax3.set_yticklabels(top20_features['feature'], fontsize=9)
    ax3.set_xlabel('Average Importance ± Std', fontsize=11, fontweight='bold')
    ax3.set_title('Average Feature Importance\n(Mean across all diagnoses ± Std)', 
                  fontsize=12, fontweight='bold')
    ax3.grid(axis='x', alpha=0.3)
    ax3.invert_yaxis()
    
    # 4. Consistency Analysis (std/mean ratio)
    ax4 = fig.add_subplot(gs[2, 0])
    consistency = top20_features['std_importance'] / (top20_features['avg_importance'] + 1e-10)
    colors_cons = ['green' if c < 0.5 else 'orange' if c < 1.0 else 'red' for c in consistency]
    ax4.barh(range(len(top20_features)), consistency, color=colors_cons, alpha=0.7)
    ax4.set_yticks(range(len(top20_features)))
    ax4.set_yticklabels(top20_features['feature'], fontsize=9)
    ax4.set_xlabel('Coefficient of Variation (Std/Mean)', fontsize=11, fontweight='bold')
    ax4.set_title('Feature Consistency\n(Lower = more consistent across diagnoses)', 
                  fontsize=12, fontweight='bold')
    ax4.axvline(x=0.5, color='orange', linestyle='--', alpha=0.5, label='Moderate (0.5)')
    ax4.axvline(x=1.0, color='red', linestyle='--', alpha=0.5, label='High (1.0)')
    ax4.grid(axis='x', alpha=0.3)
    ax4.legend(fontsize=9)
    ax4.invert_yaxis()
    
    # 5. Heatmap: Top 10 features across diagnoses
    ax5 = fig.add_subplot(gs[2, 1])
    
    # Create matrix: diagnoses × top 10 features
    top10_features = top20_features.head(10)['feature'].tolist()
    diagnoses = list(set([item['diagnosis'] for item in all_feature_importances]))[:15]  # Limit to 15 diagnoses for readability
    
    heatmap_data = []
    for dx in diagnoses:
        row = []
        for feat in top10_features:
            # Find importance for this feature in this diagnosis
            importance = 0
            for item in all_feature_importances:
                if item['diagnosis'] == dx and item['importance_df'] is not None:
                    match = item['importance_df'][item['importance_df']['feature'] == feat]
                    if not match.empty:
                        importance = match.iloc[0]['importance']
                        break
            row.append(importance)
        heatmap_data.append(row)
    
    heatmap_df = pd.DataFrame(heatmap_data, index=diagnoses, columns=top10_features)
    
    sns.heatmap(heatmap_df, annot=True, fmt='.3f', cmap='YlOrRd', 
                ax=ax5, cbar_kws={'label': 'Importance'}, linewidths=0.5)
    ax5.set_xlabel('Feature', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Diagnosis', fontsize=11, fontweight='bold')
    ax5.set_title('Top 10 Features Across Diagnoses\n(Feature importance heatmap)', 
                  fontsize=12, fontweight='bold')
    plt.setp(ax5.get_xticklabels(), rotation=45, ha='right', fontsize=9)
    plt.setp(ax5.get_yticklabels(), rotation=0, fontsize=9)
    
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved: {filename}")
    plt.close()


def create_comparative_analysis(all_diagnosis_results, all_refined_results, all_feature_importances):
    """
    Create comparative visualizations across all diagnoses.
    """
    print("\n" + "=" * 70)
    print("COMPARATIVE ANALYSIS ACROSS ALL DIAGNOSES")
    print("=" * 70)
    
    # 0. Feature Importance Aggregation and Visualization
    feature_summary_df, top20_features = aggregate_feature_importances(all_feature_importances)
    
    if feature_summary_df is not None:
        # Save complete feature statistics to CSV
        csv_path = f'{COMPARATIVE_FOLDER}/feature_importance_summary.csv'
        feature_summary_df.to_csv(csv_path, index=False)
        print(f"\n✅ Saved complete feature statistics: {csv_path}")
        print(f"   Total features: {len(feature_summary_df)}")
        print(f"   Top 20 visualized in plot")
        
        # Create visualization
        visualize_feature_importance_overlap(
            top20_features, 
            all_feature_importances,
            f'{COMPARATIVE_FOLDER}/0_feature_importance_analysis.png'
        )
    
    # 1. Performance comparison across diagnoses
    print("\n📊 1. Creating performance comparison...")
    
    # Get best model per diagnosis using COMPOSITE SCORE (balances F1 and generalization)
    best_per_dx = all_diagnosis_results.loc[
        all_diagnosis_results.groupby('Diagnosis')['Composite_Score'].idxmax()
    ]
    
    best_per_dx = best_per_dx.sort_values('F1', ascending=False)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # F1 scores
    ax1 = axes[0, 0]
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(best_per_dx)))
    ax1.barh(range(len(best_per_dx)), best_per_dx['F1'], color=colors)
    ax1.set_yticks(range(len(best_per_dx)))
    ax1.set_yticklabels(best_per_dx['Diagnosis'])
    ax1.set_xlabel('F1-Score', fontsize=12)
    ax1.set_title('F1-Score by Diagnosis Code', fontsize=13, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    ax1.invert_yaxis()
    
    # ROC-AUC scores
    ax2 = axes[0, 1]
    ax2.barh(range(len(best_per_dx)), best_per_dx['ROC_AUC'], color=colors)
    ax2.set_yticks(range(len(best_per_dx)))
    ax2.set_yticklabels(best_per_dx['Diagnosis'])
    ax2.set_xlabel('ROC-AUC', fontsize=12)
    ax2.set_title('ROC-AUC by Diagnosis Code', fontsize=13, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    ax2.invert_yaxis()
    
    # Overfit gaps
    ax3 = axes[1, 0]
    gap_colors = ['red' if gap > 0.1 else 'orange' if gap > 0.05 else 'green' 
                  for gap in best_per_dx['Overfit_Gap']]
    ax3.barh(range(len(best_per_dx)), best_per_dx['Overfit_Gap'], color=gap_colors, alpha=0.7)
    ax3.set_yticks(range(len(best_per_dx)))
    ax3.set_yticklabels(best_per_dx['Diagnosis'])
    ax3.set_xlabel('Overfit Gap (Train - Test)', fontsize=12)
    ax3.set_title('Overfitting by Diagnosis', fontsize=13, fontweight='bold')
    ax3.axvline(x=0.05, color='orange', linestyle='--', alpha=0.5, label='Warning (0.05)')
    ax3.axvline(x=0.1, color='red', linestyle='--', alpha=0.5, label='High (0.1)')
    ax3.grid(axis='x', alpha=0.3)
    ax3.legend()
    ax3.invert_yaxis()
    
    # Model selection frequency
    ax4 = axes[1, 1]
    model_counts = best_per_dx['Model'].value_counts()
    ax4.bar(range(len(model_counts)), model_counts.values, color='steelblue', alpha=0.7)
    ax4.set_xticks(range(len(model_counts)))
    ax4.set_xticklabels(model_counts.index, rotation=45, ha='right')
    ax4.set_ylabel('Number of Diagnoses', fontsize=12)
    ax4.set_title('Best Model Selection Frequency', fontsize=13, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{COMPARATIVE_FOLDER}/1_performance_comparison.png', dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved: {COMPARATIVE_FOLDER}/1_performance_comparison.png")
    plt.close()
    
    # 2. Refined model comparison
    if len(all_refined_results) > 0:
        print("\n📊 2. Creating refined model comparison...")
        
        refined_df = pd.DataFrame(all_refined_results)
        refined_df = refined_df.sort_values('f1', ascending=False)
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Performance comparison
        ax1 = axes[0]
        x = np.arange(len(refined_df))
        width = 0.35
        ax1.bar(x - width/2, refined_df['f1'], width, label='F1-Score', alpha=0.8, color='coral')
        ax1.bar(x + width/2, refined_df['roc_auc'], width, label='ROC-AUC', alpha=0.8, color='steelblue')
        ax1.set_xticks(x)
        ax1.set_xticklabels(refined_df['diagnosis'], rotation=45, ha='right')
        ax1.set_ylabel('Score', fontsize=12)
        ax1.set_title('Refined Model Performance (Top 20 Features)', fontsize=13, fontweight='bold')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # Feature count vs performance
        ax2 = axes[1]
        scatter = ax2.scatter(refined_df['overfit_gap'], refined_df['f1'], 
                             s=100, c=refined_df['f1'], cmap='viridis', alpha=0.6)
        for idx, row in refined_df.iterrows():
            ax2.annotate(row['diagnosis'], (row['overfit_gap'], row['f1']),
                        fontsize=8, alpha=0.7)
        ax2.set_xlabel('Overfit Gap', fontsize=12)
        ax2.set_ylabel('F1-Score', fontsize=12)
        ax2.set_title('Performance vs Generalization', fontsize=13, fontweight='bold')
        ax2.grid(alpha=0.3)
        plt.colorbar(scatter, ax=ax2, label='F1-Score')
        
        plt.tight_layout()
        plt.savefig(f'{COMPARATIVE_FOLDER}/2_refined_comparison.png', dpi=300, bbox_inches='tight')
        print(f"   ✅ Saved: {COMPARATIVE_FOLDER}/2_refined_comparison.png")
        plt.close()
    
    # 3. Summary statistics table
    print("\n📊 3. Creating summary table...")
    
    summary = best_per_dx[['Diagnosis', 'Model', 'F1', 'ROC_AUC', 'Accuracy', 'Overfit_Gap']].copy()
    summary = summary.round(3)
    summary.to_csv(f'{COMPARATIVE_FOLDER}/summary_statistics.csv', index=False)
    print(f"   ✅ Saved: {COMPARATIVE_FOLDER}/summary_statistics.csv")
    
    # 4. Print summary
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    print(f"\nBest performing diagnoses (by F1-score):")
    for idx, row in best_per_dx.head(5).iterrows():
        print(f"  {row['Diagnosis']}: F1={row['F1']:.3f}, AUC={row['ROC_AUC']:.3f}, Model={row['Model']}")
    
    print(f"\nMost challenging diagnoses (by F1-score):")
    for idx, row in best_per_dx.tail(5).iterrows():
        print(f"  {row['Diagnosis']}: F1={row['F1']:.3f}, AUC={row['ROC_AUC']:.3f}, Model={row['Model']}")
    
    print(f"\nAverage metrics across all diagnoses:")
    print(f"  F1-Score: {best_per_dx['F1'].mean():.3f} ± {best_per_dx['F1'].std():.3f}")
    print(f"  ROC-AUC: {best_per_dx['ROC_AUC'].mean():.3f} ± {best_per_dx['ROC_AUC'].std():.3f}")
    print(f"  Overfit Gap: {best_per_dx['Overfit_Gap'].mean():.3f} ± {best_per_dx['Overfit_Gap'].std():.3f}")
    
    print("\n✅ Comparative analysis complete!")


# =============================================================================
# Main Execution
# =============================================================================

def create_final_feature_report(all_feature_importances):
    """
    Create comprehensive cross-diagnosis feature aggregation and visualizations.
    
    Aggregates features at 3 levels: Top 20, Top 50, Top 100
    For each level, saves:
    - Detailed CSV with frequency, diagnosis codes, importance scores, sum
    - Comprehensive visualizations comparing across levels
    
    All outputs saved to 'final_report/' folder
    """
    print("\n" + "=" * 70)
    print("CREATING FINAL CROSS-DIAGNOSIS FEATURE REPORT")
    print("=" * 70)
    
    if not all_feature_importances:
        print("⚠️ No feature importances to analyze")
        return
    
    # Create output folder
    FINAL_REPORT_FOLDER = f"{BASE_VIZ_FOLDER}/final_report"
    Path(FINAL_REPORT_FOLDER).mkdir(parents=True, exist_ok=True)
    
    print(f"\n📁 Output folder: {FINAL_REPORT_FOLDER}/")
    print(f"   Analyzing features from {len(all_feature_importances)} diagnoses")
    
    # Step 1: Collect ALL feature importances with full details
    print("\n" + "=" * 70)
    print("Step 1: Collecting Feature Importances from All Diagnoses")
    print("=" * 70)
    
    feature_details = defaultdict(lambda: {
        'diagnoses': [],
        'importances': [],
        'ranks': []  # Track what rank this feature had in each diagnosis
    })
    
    for item in all_feature_importances:
        dx = item['diagnosis']
        importance_df = item['importance_df']
        
        if importance_df is not None and not importance_df.empty:
            for rank, (_, row) in enumerate(importance_df.iterrows(), 1):
                feature = row['feature']
                importance = row['importance']
                
                feature_details[feature]['diagnoses'].append(dx)
                feature_details[feature]['importances'].append(importance)
                feature_details[feature]['ranks'].append(rank)
    
    print(f"✅ Collected {len(feature_details)} unique features across all diagnoses")
    
    # Step 2: Create aggregated datasets for Top 20, 50, 100
    print("\n" + "=" * 70)
    print("Step 2: Creating Aggregations for Top 20, Top 50, Top 100")
    print("=" * 70)
    
    def create_aggregation(top_n):
        """Create aggregation for features appearing in top N of any diagnosis"""
        print(f"\n   Processing Top {top_n}...")
        
        aggregation = []
        
        for feature, details in feature_details.items():
            # Check if this feature appeared in top N of any diagnosis
            if any(rank <= top_n for rank in details['ranks']):
                # Count how many times it appeared in top N
                appearances_in_topN = sum(1 for rank in details['ranks'] if rank <= top_n)
                
                # Get importance scores only from diagnoses where it was in top N
                topN_diagnoses = []
                topN_importances = []
                all_diagnoses = []
                all_importances = []
                
                for dx, imp, rank in zip(details['diagnoses'], details['importances'], details['ranks']):
                    all_diagnoses.append(dx)
                    all_importances.append(imp)
                    
                    if rank <= top_n:
                        topN_diagnoses.append(dx)
                        topN_importances.append(imp)
                
                aggregation.append({
                    'feature': feature,
                    'frequency_in_topN': appearances_in_topN,
                    'total_appearances': len(details['diagnoses']),
                    'diagnosis_codes_topN': '; '.join(topN_diagnoses),
                    'diagnosis_codes_all': '; '.join(all_diagnoses),
                    'importance_scores_topN': '; '.join([f"{imp:.6f}" for imp in topN_importances]),
                    'importance_scores_all': '; '.join([f"{imp:.6f}" for imp in all_importances]),
                    'sum_importance_topN': sum(topN_importances),
                    'sum_importance_all': sum(all_importances),
                    'avg_importance_topN': np.mean(topN_importances),
                    'avg_importance_all': np.mean(all_importances),
                    'std_importance_topN': np.std(topN_importances),
                    'std_importance_all': np.std(all_importances),
                    'min_rank': min(details['ranks']),
                    'max_rank': max(details['ranks']),
                    'avg_rank': np.mean(details['ranks'])
                })
        
        # Create DataFrame and sort by frequency in top N, then by sum of importances
        df = pd.DataFrame(aggregation)
        df = df.sort_values(['frequency_in_topN', 'sum_importance_topN'], ascending=[False, False])
        
        print(f"      ✅ {len(df)} features appeared in Top {top_n} of at least one diagnosis")
        print(f"         Most frequent: {df.iloc[0]['feature']} ({df.iloc[0]['frequency_in_topN']} diagnoses)")
        
        return df
    
    # Create three aggregation levels
    top20_df = create_aggregation(20)
    top50_df = create_aggregation(50)
    top100_df = create_aggregation(100)
    
    # Step 3: Save CSVs
    print("\n" + "=" * 70)
    print("Step 3: Saving Detailed CSVs")
    print("=" * 70)
    
    csv_files = {
        'top20_features_aggregated.csv': top20_df,
        'top50_features_aggregated.csv': top50_df,
        'top100_features_aggregated.csv': top100_df
    }
    
    for filename, df in csv_files.items():
        filepath = f"{FINAL_REPORT_FOLDER}/{filename}"
        df.to_csv(filepath, index=False)
        print(f"✅ Saved: {filepath}")
        print(f"   Rows: {len(df)}, Columns: {len(df.columns)}")
    
    # Step 4: Create comprehensive visualizations
    print("\n" + "=" * 70)
    print("Step 4: Creating Comprehensive Visualizations")
    print("=" * 70)
    
    create_final_report_visualizations(top20_df, top50_df, top100_df, FINAL_REPORT_FOLDER)
    
    print("\n" + "=" * 70)
    print("✅ FINAL FEATURE REPORT COMPLETE")
    print("=" * 70)
    print(f"\n📁 All outputs saved to: {FINAL_REPORT_FOLDER}/")
    print(f"\n📄 CSV Files:")
    print(f"   - top20_features_aggregated.csv ({len(top20_df)} features)")
    print(f"   - top50_features_aggregated.csv ({len(top50_df)} features)")
    print(f"   - top100_features_aggregated.csv ({len(top100_df)} features)")
    print(f"\n📊 Visualizations:")
    print(f"   - 1_frequency_comparison.png")
    print(f"   - 2_importance_comparison.png")
    print(f"   - 3_top_features_detailed.png")
    print(f"   - 4_rank_analysis.png")
    print(f"   - 5_comprehensive_dashboard.png")


def create_final_report_visualizations(top20_df, top50_df, top100_df, output_folder):
    """
    Create comprehensive visualizations comparing Top 20, 50, 100 aggregations.
    """
    print("\n📊 Generating visualizations...")
    
    # === VISUALIZATION 1: Frequency Comparison ===
    print("   Creating 1_frequency_comparison.png...")
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('Feature Frequency Analysis: Top 20 vs Top 50 vs Top 100', 
                 fontsize=16, fontweight='bold')
    
    # 1.1: Top 15 features by frequency in Top 20
    ax1 = axes[0, 0]
    top15_freq = top20_df.nlargest(15, 'frequency_in_topN')
    colors = plt.cm.Reds(top15_freq['frequency_in_topN'] / top15_freq['frequency_in_topN'].max())
    ax1.barh(range(len(top15_freq)), top15_freq['frequency_in_topN'], color=colors)
    ax1.set_yticks(range(len(top15_freq)))
    ax1.set_yticklabels(top15_freq['feature'], fontsize=9)
    ax1.set_xlabel('Frequency (# diagnoses)', fontweight='bold')
    ax1.set_title('Top 15 Features by Frequency in Top 20', fontweight='bold')
    ax1.invert_yaxis()
    ax1.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, val in enumerate(top15_freq['frequency_in_topN']):
        ax1.text(val + 0.1, i, str(int(val)), va='center', fontweight='bold')
    
    # 1.2: Comparison across levels (top 10 features)
    ax2 = axes[0, 1]
    top10_features = top20_df.head(10)['feature'].tolist()
    
    freq_comparison = []
    for feat in top10_features:
        freq_20 = top20_df[top20_df['feature'] == feat]['frequency_in_topN'].values[0] if feat in top20_df['feature'].values else 0
        freq_50 = top50_df[top50_df['feature'] == feat]['frequency_in_topN'].values[0] if feat in top50_df['feature'].values else 0
        freq_100 = top100_df[top100_df['feature'] == feat]['frequency_in_topN'].values[0] if feat in top100_df['feature'].values else 0
        freq_comparison.append({'feature': feat, 'Top 20': freq_20, 'Top 50': freq_50, 'Top 100': freq_100})
    
    freq_comp_df = pd.DataFrame(freq_comparison)
    x = np.arange(len(freq_comp_df))
    width = 0.25
    
    ax2.barh(x - width, freq_comp_df['Top 20'], width, label='Top 20', color='#d62728', alpha=0.8)
    ax2.barh(x, freq_comp_df['Top 50'], width, label='Top 50', color='#ff7f0e', alpha=0.8)
    ax2.barh(x + width, freq_comp_df['Top 100'], width, label='Top 100', color='#2ca02c', alpha=0.8)
    
    ax2.set_yticks(x)
    ax2.set_yticklabels(freq_comp_df['feature'], fontsize=9)
    ax2.set_xlabel('Frequency', fontweight='bold')
    ax2.set_title('Top 10 Features Across Cutoff Levels', fontweight='bold')
    ax2.legend()
    ax2.invert_yaxis()
    ax2.grid(axis='x', alpha=0.3)
    
    # 1.3: Feature count distribution
    ax3 = axes[1, 0]
    level_stats = {
        'Top 20': len(top20_df),
        'Top 50': len(top50_df),
        'Top 100': len(top100_df)
    }
    colors_bar = ['#d62728', '#ff7f0e', '#2ca02c']
    bars = ax3.bar(level_stats.keys(), level_stats.values(), color=colors_bar, alpha=0.7, edgecolor='black', linewidth=2)
    ax3.set_ylabel('Number of Unique Features', fontweight='bold')
    ax3.set_title('Total Features Appearing in Each Level', fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, level_stats.values()):
        ax3.text(bar.get_x() + bar.get_width()/2, val + val*0.02, str(val), 
                ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    # 1.4: Frequency distribution histogram
    ax4 = axes[1, 1]
    ax4.hist(top20_df['frequency_in_topN'], bins=15, alpha=0.7, label='Top 20', color='#d62728', edgecolor='black')
    ax4.hist(top50_df['frequency_in_topN'], bins=15, alpha=0.5, label='Top 50', color='#ff7f0e', edgecolor='black')
    ax4.hist(top100_df['frequency_in_topN'], bins=15, alpha=0.3, label='Top 100', color='#2ca02c', edgecolor='black')
    ax4.set_xlabel('Frequency (# diagnoses)', fontweight='bold')
    ax4.set_ylabel('Number of Features', fontweight='bold')
    ax4.set_title('Distribution of Feature Frequencies', fontweight='bold')
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_folder}/1_frequency_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("      ✅ Saved 1_frequency_comparison.png")
    
    # === VISUALIZATION 2: Importance Comparison ===
    print("   Creating 2_importance_comparison.png...")
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('Feature Importance Analysis: Sum and Average', 
                 fontsize=16, fontweight='bold')
    
    # 2.1: Top 15 by sum of importances (Top 20 level)
    ax1 = axes[0, 0]
    top15_sum = top20_df.nlargest(15, 'sum_importance_topN')
    colors = plt.cm.Oranges(top15_sum['sum_importance_topN'] / top15_sum['sum_importance_topN'].max())
    ax1.barh(range(len(top15_sum)), top15_sum['sum_importance_topN'], color=colors)
    ax1.set_yticks(range(len(top15_sum)))
    ax1.set_yticklabels(top15_sum['feature'], fontsize=9)
    ax1.set_xlabel('Sum of Importances', fontweight='bold')
    ax1.set_title('Top 15 Features by Sum of Importances (Top 20)', fontweight='bold')
    ax1.invert_yaxis()
    ax1.grid(axis='x', alpha=0.3)
    
    # 2.2: Top 15 by average importance (Top 20 level)
    ax2 = axes[0, 1]
    top15_avg = top20_df.nlargest(15, 'avg_importance_topN')
    ax2.barh(range(len(top15_avg)), top15_avg['avg_importance_topN'], 
            xerr=top15_avg['std_importance_topN'], color=plt.cm.Greens(0.7), capsize=3)
    ax2.set_yticks(range(len(top15_avg)))
    ax2.set_yticklabels(top15_avg['feature'], fontsize=9)
    ax2.set_xlabel('Average Importance ± Std', fontweight='bold')
    ax2.set_title('Top 15 Features by Average Importance (Top 20)', fontweight='bold')
    ax2.invert_yaxis()
    ax2.grid(axis='x', alpha=0.3)
    
    # 2.3: Sum comparison across levels
    ax3 = axes[1, 0]
    top10_features = top20_df.head(10)['feature'].tolist()
    
    sum_comparison = []
    for feat in top10_features:
        sum_20 = top20_df[top20_df['feature'] == feat]['sum_importance_topN'].values[0] if feat in top20_df['feature'].values else 0
        sum_50 = top50_df[top50_df['feature'] == feat]['sum_importance_topN'].values[0] if feat in top50_df['feature'].values else 0
        sum_100 = top100_df[top100_df['feature'] == feat]['sum_importance_topN'].values[0] if feat in top100_df['feature'].values else 0
        sum_comparison.append({'feature': feat, 'Top 20': sum_20, 'Top 50': sum_50, 'Top 100': sum_100})
    
    sum_comp_df = pd.DataFrame(sum_comparison)
    x = np.arange(len(sum_comp_df))
    width = 0.25
    
    ax3.barh(x - width, sum_comp_df['Top 20'], width, label='Top 20', color='#d62728', alpha=0.8)
    ax3.barh(x, sum_comp_df['Top 50'], width, label='Top 50', color='#ff7f0e', alpha=0.8)
    ax3.barh(x + width, sum_comp_df['Top 100'], width, label='Top 100', color='#2ca02c', alpha=0.8)
    
    ax3.set_yticks(x)
    ax3.set_yticklabels(sum_comp_df['feature'], fontsize=9)
    ax3.set_xlabel('Sum of Importances', fontweight='bold')
    ax3.set_title('Sum of Importances: Top 10 Features Across Levels', fontweight='bold')
    ax3.legend()
    ax3.invert_yaxis()
    ax3.grid(axis='x', alpha=0.3)
    
    # 2.4: Scatter: Frequency vs Sum of Importances
    ax4 = axes[1, 1]
    ax4.scatter(top20_df['frequency_in_topN'], top20_df['sum_importance_topN'], 
               alpha=0.6, s=100, c='#d62728', label='Top 20', edgecolors='black')
    ax4.scatter(top50_df['frequency_in_topN'], top50_df['sum_importance_topN'], 
               alpha=0.4, s=80, c='#ff7f0e', label='Top 50', edgecolors='black')
    ax4.scatter(top100_df['frequency_in_topN'], top100_df['sum_importance_topN'], 
               alpha=0.2, s=60, c='#2ca02c', label='Top 100', edgecolors='black')
    
    ax4.set_xlabel('Frequency (# diagnoses)', fontweight='bold')
    ax4.set_ylabel('Sum of Importances', fontweight='bold')
    ax4.set_title('Frequency vs Total Importance', fontweight='bold')
    ax4.legend()
    ax4.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_folder}/2_importance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("      ✅ Saved 2_importance_comparison.png")
    
    # === VISUALIZATION 3: Top Features Detailed View ===
    print("   Creating 3_top_features_detailed.png...")
    
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)
    
    fig.suptitle('Detailed Analysis of Top 20 Most Important Features', 
                 fontsize=16, fontweight='bold')
    
    top20_display = top20_df.head(20)
    
    # 3.1: Combined score (frequency × avg importance)
    ax1 = fig.add_subplot(gs[0, :])
    combined_score = top20_display['frequency_in_topN'] * top20_display['avg_importance_topN']
    colors = plt.cm.RdYlGn(combined_score / combined_score.max())
    bars = ax1.barh(range(len(top20_display)), combined_score, color=colors)
    ax1.set_yticks(range(len(top20_display)))
    ax1.set_yticklabels(top20_display['feature'], fontsize=10)
    ax1.set_xlabel('Combined Score (Frequency × Avg Importance)', fontsize=12, fontweight='bold')
    ax1.set_title('Top 20 Features: Overall Impact Score', fontsize=14, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    ax1.invert_yaxis()
    
    for i, (bar, val, freq) in enumerate(zip(bars, combined_score, top20_display['frequency_in_topN'])):
        ax1.text(val + val*0.02, i, f'{val:.3f} (n={int(freq)})', 
                va='center', fontsize=9, fontweight='bold')
    
    # 3.2: Frequency
    ax2 = fig.add_subplot(gs[1, 0])
    colors_freq = plt.cm.Blues(top20_display['frequency_in_topN'] / top20_display['frequency_in_topN'].max())
    ax2.barh(range(len(top20_display)), top20_display['frequency_in_topN'], color=colors_freq)
    ax2.set_yticks(range(len(top20_display)))
    ax2.set_yticklabels(top20_display['feature'], fontsize=9)
    ax2.set_xlabel('Frequency', fontweight='bold')
    ax2.set_title('How Many Diagnoses Each Feature Appeared In', fontweight='bold')
    ax2.invert_yaxis()
    ax2.grid(axis='x', alpha=0.3)
    
    # 3.3: Sum of importances
    ax3 = fig.add_subplot(gs[1, 1])
    colors_sum = plt.cm.Oranges(top20_display['sum_importance_topN'] / top20_display['sum_importance_topN'].max())
    ax3.barh(range(len(top20_display)), top20_display['sum_importance_topN'], color=colors_sum)
    ax3.set_yticks(range(len(top20_display)))
    ax3.set_yticklabels(top20_display['feature'], fontsize=9)
    ax3.set_xlabel('Sum of Importances', fontweight='bold')
    ax3.set_title('Cumulative Importance Across All Diagnoses', fontweight='bold')
    ax3.invert_yaxis()
    ax3.grid(axis='x', alpha=0.3)
    
    # 3.4: Average importance with std
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.barh(range(len(top20_display)), top20_display['avg_importance_topN'],
            xerr=top20_display['std_importance_topN'], color=plt.cm.Greens(0.7), capsize=3)
    ax4.set_yticks(range(len(top20_display)))
    ax4.set_yticklabels(top20_display['feature'], fontsize=9)
    ax4.set_xlabel('Average Importance ± Std', fontweight='bold')
    ax4.set_title('Average Importance Across Diagnoses', fontweight='bold')
    ax4.invert_yaxis()
    ax4.grid(axis='x', alpha=0.3)
    
    # 3.5: Rank distribution
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.barh(range(len(top20_display)), top20_display['min_rank'], 
            color='lightblue', alpha=0.7, label='Min Rank')
    ax5.barh(range(len(top20_display)), top20_display['avg_rank'], 
            color='orange', alpha=0.7, label='Avg Rank')
    ax5.set_yticks(range(len(top20_display)))
    ax5.set_yticklabels(top20_display['feature'], fontsize=9)
    ax5.set_xlabel('Rank (lower = better)', fontweight='bold')
    ax5.set_title('Feature Ranking Statistics', fontweight='bold')
    ax5.invert_yaxis()
    ax5.invert_xaxis()  # Lower rank is better
    ax5.legend()
    ax5.grid(axis='x', alpha=0.3)
    
    plt.savefig(f'{output_folder}/3_top_features_detailed.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("      ✅ Saved 3_top_features_detailed.png")
    
    # === VISUALIZATION 4: Rank Analysis ===
    print("   Creating 4_rank_analysis.png...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Feature Ranking Analysis Across Diagnoses', 
                 fontsize=16, fontweight='bold')
    
    # 4.1: Best rank achieved
    ax1 = axes[0, 0]
    top15_rank = top20_df.nsmallest(15, 'min_rank')
    colors = plt.cm.RdYlGn_r(top15_rank['min_rank'] / 20)  # Reverse: lower rank = greener
    ax1.barh(range(len(top15_rank)), top15_rank['min_rank'], color=colors)
    ax1.set_yticks(range(len(top15_rank)))
    ax1.set_yticklabels(top15_rank['feature'], fontsize=9)
    ax1.set_xlabel('Best Rank Achieved', fontweight='bold')
    ax1.set_title('Top 15 Features by Best Ranking', fontweight='bold')
    ax1.invert_yaxis()
    ax1.invert_xaxis()  # Lower is better
    ax1.grid(axis='x', alpha=0.3)
    
    # 4.2: Average rank
    ax2 = axes[0, 1]
    top15_avg_rank = top20_df.nsmallest(15, 'avg_rank')
    ax2.barh(range(len(top15_avg_rank)), top15_avg_rank['avg_rank'], color=plt.cm.Purples(0.7))
    ax2.set_yticks(range(len(top15_avg_rank)))
    ax2.set_yticklabels(top15_avg_rank['feature'], fontsize=9)
    ax2.set_xlabel('Average Rank', fontweight='bold')
    ax2.set_title('Top 15 Features by Average Ranking', fontweight='bold')
    ax2.invert_yaxis()
    ax2.invert_xaxis()
    ax2.grid(axis='x', alpha=0.3)
    
    # 4.3: Rank variability (max - min)
    ax3 = axes[1, 0]
    top20_df_copy = top20_df.copy()
    top20_df_copy['rank_range'] = top20_df_copy['max_rank'] - top20_df_copy['min_rank']
    top15_variability = top20_df_copy.nsmallest(15, 'rank_range')
    colors_var = ['green' if r < 10 else 'orange' if r < 20 else 'red' 
                  for r in top15_variability['rank_range']]
    ax3.barh(range(len(top15_variability)), top15_variability['rank_range'], color=colors_var, alpha=0.7)
    ax3.set_yticks(range(len(top15_variability)))
    ax3.set_yticklabels(top15_variability['feature'], fontsize=9)
    ax3.set_xlabel('Rank Variability (Max - Min)', fontweight='bold')
    ax3.set_title('Most Consistent Features (low variability)', fontweight='bold')
    ax3.invert_yaxis()
    ax3.grid(axis='x', alpha=0.3)
    
    # 4.4: Distribution of minimum ranks
    ax4 = axes[1, 1]
    ax4.hist(top20_df['min_rank'], bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    ax4.axvline(top20_df['min_rank'].median(), color='red', linestyle='--', 
               linewidth=2, label=f"Median: {top20_df['min_rank'].median():.1f}")
    ax4.set_xlabel('Minimum Rank Achieved', fontweight='bold')
    ax4.set_ylabel('Number of Features', fontweight='bold')
    ax4.set_title('Distribution of Best Rankings', fontweight='bold')
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_folder}/4_rank_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("      ✅ Saved 4_rank_analysis.png")
    
    # === VISUALIZATION 5: Comprehensive Dashboard ===
    print("   Creating 5_comprehensive_dashboard.png...")
    
    fig = plt.figure(figsize=(24, 16))
    gs = fig.add_gridspec(4, 3, hspace=0.4, wspace=0.3)
    
    fig.suptitle('Comprehensive Cross-Diagnosis Feature Analysis Dashboard', 
                 fontsize=18, fontweight='bold')
    
    # Get top 12 for detailed display
    top12 = top20_df.head(12)
    
    # 5.1: Frequency
    ax1 = fig.add_subplot(gs[0, 0])
    colors1 = plt.cm.Reds(top12['frequency_in_topN'] / top12['frequency_in_topN'].max())
    ax1.barh(range(len(top12)), top12['frequency_in_topN'], color=colors1)
    ax1.set_yticks(range(len(top12)))
    ax1.set_yticklabels(top12['feature'], fontsize=9)
    ax1.set_xlabel('Frequency', fontweight='bold')
    ax1.set_title('A. Feature Frequency', fontweight='bold', fontsize=12)
    ax1.invert_yaxis()
    ax1.grid(axis='x', alpha=0.3)
    
    # 5.2: Sum of importances
    ax2 = fig.add_subplot(gs[0, 1])
    colors2 = plt.cm.Oranges(top12['sum_importance_topN'] / top12['sum_importance_topN'].max())
    ax2.barh(range(len(top12)), top12['sum_importance_topN'], color=colors2)
    ax2.set_yticks(range(len(top12)))
    ax2.set_yticklabels(top12['feature'], fontsize=9)
    ax2.set_xlabel('Sum of Importances', fontweight='bold')
    ax2.set_title('B. Cumulative Importance', fontweight='bold', fontsize=12)
    ax2.invert_yaxis()
    ax2.grid(axis='x', alpha=0.3)
    
    # 5.3: Average importance
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.barh(range(len(top12)), top12['avg_importance_topN'],
            xerr=top12['std_importance_topN'], color=plt.cm.Greens(0.7), capsize=2)
    ax3.set_yticks(range(len(top12)))
    ax3.set_yticklabels(top12['feature'], fontsize=9)
    ax3.set_xlabel('Avg Importance ± Std', fontweight='bold')
    ax3.set_title('C. Average Importance', fontweight='bold', fontsize=12)
    ax3.invert_yaxis()
    ax3.grid(axis='x', alpha=0.3)
    
    # 5.4: Combined score
    ax4 = fig.add_subplot(gs[1, :])
    combined = top12['frequency_in_topN'] * top12['avg_importance_topN']
    colors4 = plt.cm.RdYlGn(combined / combined.max())
    bars4 = ax4.barh(range(len(top12)), combined, color=colors4, edgecolor='black', linewidth=1.5)
    ax4.set_yticks(range(len(top12)))
    ax4.set_yticklabels(top12['feature'], fontsize=11, fontweight='bold')
    ax4.set_xlabel('Combined Impact Score (Frequency × Avg Importance)', fontsize=12, fontweight='bold')
    ax4.set_title('D. Overall Feature Impact (Top 12 Features)', fontweight='bold', fontsize=14)
    ax4.invert_yaxis()
    ax4.grid(axis='x', alpha=0.3)
    
    for i, (bar, val, freq) in enumerate(zip(bars4, combined, top12['frequency_in_topN'])):
        ax4.text(val + val*0.02, i, f'{val:.3f} (n={int(freq)})', 
                va='center', fontsize=10, fontweight='bold')
    
    # 5.5: Scatter matrix (frequency vs importance)
    ax5 = fig.add_subplot(gs[2, 0])
    scatter = ax5.scatter(top20_df['frequency_in_topN'], top20_df['avg_importance_topN'],
                         s=top20_df['sum_importance_topN']*1000, 
                         c=top20_df['frequency_in_topN'], cmap='viridis',
                         alpha=0.6, edgecolors='black', linewidth=1.5)
    ax5.set_xlabel('Frequency', fontweight='bold')
    ax5.set_ylabel('Avg Importance', fontweight='bold')
    ax5.set_title('E. Feature Landscape\n(size = sum importance)', fontweight='bold', fontsize=12)
    ax5.grid(alpha=0.3)
    plt.colorbar(scatter, ax=ax5, label='Frequency')
    
    # Annotate top 5
    for idx, row in top20_df.head(5).iterrows():
        ax5.annotate(row['feature'], 
                    (row['frequency_in_topN'], row['avg_importance_topN']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    # 5.6: Rank distribution
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.barh(range(len(top12)), top12['min_rank'], color='lightblue', alpha=0.7, label='Best Rank')
    ax6.barh(range(len(top12)), top12['avg_rank'], color='orange', alpha=0.7, label='Avg Rank')
    ax6.set_yticks(range(len(top12)))
    ax6.set_yticklabels(top12['feature'], fontsize=9)
    ax6.set_xlabel('Rank (lower = better)', fontweight='bold')
    ax6.set_title('F. Ranking Statistics', fontweight='bold', fontsize=12)
    ax6.invert_yaxis()
    ax6.invert_xaxis()
    ax6.legend(fontsize=9)
    ax6.grid(axis='x', alpha=0.3)
    
    # 5.7: Level comparison statistics
    ax7 = fig.add_subplot(gs[2, 2])
    level_data = {
        'Total Features': [len(top20_df), len(top50_df), len(top100_df)],
        'Avg Frequency': [top20_df['frequency_in_topN'].mean(), 
                         top50_df['frequency_in_topN'].mean(),
                         top100_df['frequency_in_topN'].mean()],
        'Avg Sum Imp': [top20_df['sum_importance_topN'].mean(),
                       top50_df['sum_importance_topN'].mean(),
                       top100_df['sum_importance_topN'].mean()]
    }
    x_pos = np.arange(3)
    width = 0.25
    
    ax7.bar(x_pos - width, level_data['Total Features'], width, label='Total Features', color='#d62728')
    ax7_twin1 = ax7.twinx()
    ax7_twin1.bar(x_pos, level_data['Avg Frequency'], width, label='Avg Frequency', color='#ff7f0e')
    ax7_twin2 = ax7.twinx()
    ax7_twin2.spines['right'].set_position(('outward', 60))
    ax7_twin2.bar(x_pos + width, level_data['Avg Sum Imp'], width, label='Avg Sum Imp', color='#2ca02c')
    
    ax7.set_xticks(x_pos)
    ax7.set_xticklabels(['Top 20', 'Top 50', 'Top 100'])
    ax7.set_ylabel('Total Features', fontweight='bold', color='#d62728')
    ax7_twin1.set_ylabel('Avg Frequency', fontweight='bold', color='#ff7f0e')
    ax7_twin2.set_ylabel('Avg Sum Imp', fontweight='bold', color='#2ca02c')
    ax7.set_title('G. Level Comparison', fontweight='bold', fontsize=12)
    ax7.grid(axis='y', alpha=0.3)
    
    # 5.8: Key statistics panel
    ax8 = fig.add_subplot(gs[3, :])
    ax8.axis('off')
    
    stats_text = f"""
    📊 KEY STATISTICS - CROSS-DIAGNOSIS FEATURE ANALYSIS
    
    🔝 TOP 20 LEVEL:
       • Unique features: {len(top20_df)}
       • Most frequent feature: {top20_df.iloc[0]['feature']} (appeared in {int(top20_df.iloc[0]['frequency_in_topN'])} diagnoses)
       • Highest sum of importance: {top20_df.nlargest(1, 'sum_importance_topN').iloc[0]['feature']} (sum = {top20_df.nlargest(1, 'sum_importance_topN').iloc[0]['sum_importance_topN']:.4f})
       • Average frequency per feature: {top20_df['frequency_in_topN'].mean():.2f} diagnoses
       • Average importance per feature: {top20_df['avg_importance_topN'].mean():.4f}
    
    🔝 TOP 50 LEVEL:
       • Unique features: {len(top50_df)}
       • Average frequency per feature: {top50_df['frequency_in_topN'].mean():.2f} diagnoses
       • Average importance per feature: {top50_df['avg_importance_topN'].mean():.4f}
    
    🔝 TOP 100 LEVEL:
       • Unique features: {len(top100_df)}
       • Average frequency per feature: {top100_df['frequency_in_topN'].mean():.2f} diagnoses
       • Average importance per feature: {top100_df['avg_importance_topN'].mean():.4f}
    
    💡 INSIGHTS:
       • {len([f for f in top20_df['feature'] if f in top50_df['feature'].values and f in top100_df['feature'].values])} features appear in all three levels
       • Most consistent feature (lowest rank variability): {top20_df_copy.nsmallest(1, 'rank_range').iloc[0]['feature']}
       • Best overall performer (highest combined score): {top20_df.iloc[0]['feature']}
    """
    
    ax8.text(0.05, 0.95, stats_text, transform=ax8.transAxes, 
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.savefig(f'{output_folder}/5_comprehensive_dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("      ✅ Saved 5_comprehensive_dashboard.png")
    
    print("\n✅ All visualizations created successfully!")


def main():
    print("\n" + "=" * 70)
    print("INDIVIDUAL DIAGNOSIS PREDICTION FROM SDoH FEATURES")
    print("Leakage-Free Analysis with Comparative Study")
    print("=" * 70)
    
    # Step 1: Load and link datasets
    acxiom_df, diagnosis_df, common_ids = load_and_link_datasets(ACXIOM_PATH, DIAGNOSIS_PATH)
    
    # Step 2: Extract diagnoses
    patient_diagnoses, diagnosis_prevalence = extract_diagnosis_codes(diagnosis_df)
    
    # Step 3: Validate target diagnoses
    total_patients = len(patient_diagnoses)
    valid_diagnoses = validate_target_diagnoses(diagnosis_prevalence, TARGET_DIAGNOSES, total_patients)
    
    if len(valid_diagnoses) == 0:
        print("\n❌ No valid diagnoses found!")
        return
    
    # Step 4: Prepare features
    X_raw, feature_names = prepare_features(acxiom_df)
    
    # Align X_raw with patient_diagnoses
    X_raw['patient_id'] = acxiom_df['patient_id']
    X_raw = X_raw.set_index('patient_id')
    
    # Step 5: Create models with hyperparameter variations
    print("\n" + "=" * 70)
    print("Step 5: Creating Leakage-Free Model Pipelines with Variations")
    print("=" * 70)
    models = create_model_pipelines()
    
    # Step 6: Evaluate each diagnosis
    print("\n" + "=" * 70)
    print(f"Step 6: Evaluating {len(valid_diagnoses)} Individual Diagnoses")
    print("=" * 70)
    
    all_results = []
    all_refined_results = []
    all_feature_importances = []  # Track feature importances across all runs
    
    for i, dx_info in enumerate(valid_diagnoses, 1):
        dx_code = dx_info['code']
        dx_safe = dx_code.replace('.', '_')  # Safe folder name
        viz_folder = f"{BASE_VIZ_FOLDER}/visualisations_{dx_safe}"
        
        print(f"\n{'='*70}")
        print(f"🔬 Analyzing Diagnosis {i}/{len(valid_diagnoses)}: {dx_code}")
        print(f"{'='*70}")
        print(f"Patients: {dx_info['count']} ({dx_info['prevalence']*100:.1f}%)")
        
        # Create labels
        y = create_labels_for_diagnosis(X_raw.index, patient_diagnoses, dx_code)
        y = y.reindex(X_raw.index, fill_value=0)
        
        print(f"\nLabel distribution:")
        print(f"  Positive (has {dx_code}): {(y==1).sum()} ({(y==1).mean()*100:.1f}%)")
        print(f"  Negative (no {dx_code}): {(y==0).sum()} ({(y==0).mean()*100:.1f}%)")
        
        # Skip if too imbalanced
        prevalence = (y==1).mean()
        if prevalence < 0.05 or prevalence > 0.95:
            print(f"\n⚠️ Skipping {dx_code}: Too imbalanced ({prevalence*100:.1f}%)")
            continue
        
        # Evaluate
        results_df = evaluate_diagnosis(X_raw, y, dx_code, models)
        all_results.append(results_df)
        
        # Print summary
        if not results_df.empty:
            # Sort by COMPOSITE SCORE (balances F1 and generalization)
            best = results_df.sort_values('Composite_Score', ascending=False).iloc[0]
            print(f"\n✅ Best model (by composite score): {best['Model']}")
            print(f"   F1: {best['F1']:.3f}, ROC-AUC: {best['ROC_AUC']:.3f}, Gap: {best['Overfit_Gap']:.3f}")
            print(f"   Composite Score: {best['Composite_Score']:.3f} (F1 × generalization penalty)")
            
            # Show comparison with pure F1 selection
            best_f1 = results_df.sort_values('F1', ascending=False).iloc[0]
            if best_f1['Model'] != best['Model']:
                print(f"\n   ℹ️  Note: Pure F1 would select {best_f1['Model']} (F1={best_f1['F1']:.3f}, Gap={best_f1['Overfit_Gap']:.3f})")
                print(f"   But composite score prefers {best['Model']} for better generalization")
            
            # Create visualizations comparing all models and highlighting best
            Path(viz_folder).mkdir(parents=True, exist_ok=True)
            create_model_comparison_for_diagnosis(results_df, dx_code, viz_folder)
            create_best_model_summary(results_df, dx_code, viz_folder)
            
            # Step 7: Feature importance and refined analysis for this diagnosis
            best_model_name = best['Model']
            best_pipeline = models[best_model_name]
            
            # Extract feature importances
            importance_df = extract_feature_importances(X_raw, y, X_raw.columns.tolist(), best_pipeline)
            
            # Store for cross-diagnosis analysis
            all_feature_importances.append({
                'diagnosis': dx_code,
                'importance_df': importance_df
            })
            
            if importance_df is not None:
                # Save visualizations for this diagnosis
                Path(viz_folder).mkdir(parents=True, exist_ok=True)
                
                plot_feature_importances(importance_df, top_n=30, 
                                        filename=f'{viz_folder}/initial_feature_importances.png')
                plot_correlation_heatmap(X_raw, y, importance_df, top_n=20,
                                        filename=f'{viz_folder}/initial_correlation_heatmap.png')
                
                # Refined analysis with top 20 features
                refined_results, refined_pipeline, X_refined, y_pred, y_pred_proba, top_features = \
                    refined_analysis_top_features(
                        X_raw, y, importance_df, best_model_name, 
                        top_n_features=20, diagnosis_code=dx_code
                    )
                
                # Advanced feature analysis (NEW!)
                quality_df = create_advanced_feature_analysis_report(
                    X_raw, y, importance_df, best_pipeline, 
                    dx_code, viz_folder, top_n=20
                )
                
                # Create advanced visualizations
                create_advanced_visualizations(
                    X_refined, y, y_pred, y_pred_proba, 
                    top_features, refined_pipeline, viz_folder
                )
                
                # Save refined results
                refined_df = pd.DataFrame([refined_results])
                refined_df.to_csv(f'{viz_folder}/refined_model_results.csv', index=False)
                all_refined_results.append(refined_results)
                
                print(f"\n✅ Saved all visualizations to: {viz_folder}/")
    
    # Combine all results
    if len(all_results) > 0:
        final_results = pd.concat(all_results, ignore_index=True)
        final_results.to_csv(OUTPUT_PATH, index=False)
        print(f"\n✅ Saved all results to: {OUTPUT_PATH}")
        
        # Step 8: Comparative Analysis
        print("\n" + "=" * 70)
        print("Step 8: COMPARATIVE ANALYSIS")
        print("=" * 70)
        
        create_comparative_analysis(final_results, all_refined_results, all_feature_importances)
        
        # Step 9: Cross-Diagnosis Feature Aggregation & Final Report
        print("\n" + "=" * 70)
        print("Step 9: CROSS-DIAGNOSIS FEATURE AGGREGATION")
        print("=" * 70)
        
        create_final_feature_report(all_feature_importances)
        
        # Final Summary
        print("\n" + "=" * 70)
        print("✅ COMPLETE ANALYSIS FINISHED")
        print("=" * 70)
        print(f"\n📁 Output Structure:")
        print(f"\n   Main Results:")
        print(f"   - {OUTPUT_PATH}")
        print(f"\n   Individual Diagnosis Folders: {BASE_VIZ_FOLDER}/")
        print(f"   - visualisations_I10/")
        print(f"   - visualisations_E78_5/")
        print(f"   - ... (one folder per diagnosis)")
        print(f"\n   Comparative Analysis: {COMPARATIVE_FOLDER}/")
        print(f"   - 0_feature_importance_analysis.png (NEW! Top 20 features)")
        print(f"   - feature_importance_summary.csv (NEW! All features with stats)")
        print(f"   - 1_performance_comparison.png")
        print(f"   - 2_refined_comparison.png")
        print(f"   - summary_statistics.csv")
        print(f"\n   Each diagnosis folder contains:")
        print(f"   - 0_all_models_comparison.png")
        print(f"   - 0_best_model_summary.png")
        print(f"   - initial_feature_importances.png")
        print(f"   - initial_correlation_heatmap.png")
        print(f"   - 1_refined_feature_importance.png")
        print(f"   - 2_prediction_analysis.png")
        print(f"   - 3_feature_distributions.png")
        print(f"   - 4_calibration_analysis.png")
        print(f"   - 5_performance_curves.png")
        print(f"   - 6_feature_quality_dashboard.png (NEW! Advanced metrics)")
        print(f"   - 7_feature_distributions.png (NEW! Class separation)")
        print(f"   - 8_executive_summary.png (NEW! Presentation-ready)")
        print(f"   - feature_analysis_complete.csv (NEW! All quality metrics)")
        print(f"   - refined_model_results.csv")
        
        print("\n" + "=" * 70)
        print(f"✨ Analyzed {len(valid_diagnoses)} diagnoses")
        print(f"✨ Created {len(valid_diagnoses)} individual visualization folders")
        print(f"✨ Generated comparative analysis across all diagnoses")
        print("=" * 70)
    else:
        print("\n❌ No diagnoses were successfully analyzed!")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()

