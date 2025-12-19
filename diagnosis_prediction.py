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
    
    # Identify SDoH columns (pattern: 2 letters + numbers, or ibe*)
    pattern = re.compile(r'^[a-z]{2}\d+|^ibe\d+', re.IGNORECASE)
    sdoh_cols = [col for col in acxiom_df.columns if pattern.match(col)]

    # Exclude any columns that look like diagnosis/claim fields to avoid leakage
    exclusion_tokens = ['diag', 'icd', 'dx', 'diagnosis', 'claim', 'clm', 'disease',
                        'clinical', 'code', 'procedure', 'proc', 'admit', 'encounter']
    excluded = [c for c in sdoh_cols if any(tok in c.lower() for tok in exclusion_tokens)]
    if excluded:
        print(f"\n⚠️ Excluding {len(excluded)} potential leakage columns by name")
        print(f"   Examples excluded: {excluded[:10]}")
    sdoh_cols = [c for c in sdoh_cols if c not in excluded]

    print(f"\n✅ Found {len(sdoh_cols)} SDoH columns (after exclusion)")
    print(f"   Examples: {sdoh_cols[:10]}")
    
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
# Step 6: Create Leakage-Free Model Pipelines
# =============================================================================

def create_model_pipelines():
    """Create pipelines with all preprocessing inside."""
    models = {}
    
    print("🔒 Creating leakage-free pipelines...")
    print("   All preprocessing happens INSIDE cross-validation folds")
    
    models['Random Forest'] = Pipeline([
        ('dropper', ColumnDropper(MISSING_THRESHOLD, VARIANCE_THRESHOLD)),
        ('imputer', ConditionalImputer(knn_neighbors=5, feature_threshold=500)),
        ('scaler', StandardScaler()),
        ('model', RandomForestClassifier(**RF_PARAMS))
    ])
    
    models['Gradient Boosting'] = Pipeline([
        ('dropper', ColumnDropper(MISSING_THRESHOLD, VARIANCE_THRESHOLD)),
        ('imputer', ConditionalImputer(knn_neighbors=5, feature_threshold=500)),
        ('scaler', StandardScaler()),
        ('model', GradientBoostingClassifier(**GB_PARAMS))
    ])
    
    if LIGHTGBM_AVAILABLE:
        lgbm_params = {
            'n_estimators': 50, 'max_depth': 4, 'learning_rate': 0.01,
            'num_leaves': 15, 'min_child_samples': 30, 'subsample': 0.7,
            'colsample_bytree': 0.7, 'reg_alpha': 0.1, 'reg_lambda': 1.0,
            'random_state': RANDOM_STATE, 'n_jobs': -1, 'verbosity': -1, 
            'class_weight': 'balanced'
        }
        models['LightGBM'] = Pipeline([
            ('dropper', ColumnDropper(MISSING_THRESHOLD, VARIANCE_THRESHOLD)),
            ('imputer', ConditionalImputer(knn_neighbors=5, feature_threshold=500)),
            ('scaler', StandardScaler()),
            ('model', lgb.LGBMClassifier(**lgbm_params))
        ])
    
    if XGBOOST_AVAILABLE:
        xgb_params = {
            'n_estimators': 50, 'max_depth': 3, 'learning_rate': 0.01,
            'min_child_weight': 10, 'subsample': 0.7, 'colsample_bytree': 0.7,
            'gamma': 0.2, 'reg_alpha': 0.1, 'reg_lambda': 1.5,
            'random_state': RANDOM_STATE, 'n_jobs': -1, 'eval_metric': 'logloss'
        }
        models['XGBoost'] = Pipeline([
            ('dropper', ColumnDropper(MISSING_THRESHOLD, VARIANCE_THRESHOLD)),
            ('imputer', ConditionalImputer(knn_neighbors=5, feature_threshold=500)),
            ('scaler', StandardScaler()),
            ('model', xgb.XGBClassifier(**xgb_params))
        ])
    
    if CATBOOST_AVAILABLE:
        catboost_params = {
            'iterations': 50, 'depth': 4, 'learning_rate': 0.01,
            'l2_leaf_reg': 5, 'random_state': RANDOM_STATE,
            'verbose': False, 'thread_count': -1,
            'auto_class_weights': 'Balanced',
            'min_data_in_leaf': 10, 'max_leaves': 16, 'subsample': 0.7
        }
        models['CatBoost'] = Pipeline([
            ('dropper', ColumnDropper(MISSING_THRESHOLD, VARIANCE_THRESHOLD)),
            ('imputer', ConditionalImputer(knn_neighbors=5, feature_threshold=500)),
            ('scaler', StandardScaler()),
            ('model', CatBoostClassifier(**catboost_params))
        ])
    
    print(f"✅ Created {len(models)} model pipelines")
    
    return models


# =============================================================================
# Step 7: Evaluate Single Diagnosis
# =============================================================================

def evaluate_diagnosis(X_raw, y, diagnosis_code, models):
    """
    Run leakage-free CV for one diagnosis code.
    
    Returns:
        DataFrame with results for each model
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
    
    for model_name, pipeline in models.items():
        try:
            cv_results = cross_validate(
                pipeline, X_raw, y,
                cv=skf,
                scoring=scoring,
                n_jobs=1,
                return_train_score=True,
                error_score='raise'
            )
            
            result = {
                'Diagnosis': diagnosis_code,
                'Model': model_name,
                'Accuracy': cv_results['test_accuracy'].mean(),
                'Accuracy_Std': cv_results['test_accuracy'].std(),
                'Precision': cv_results['test_precision'].mean(),
                'Recall': cv_results['test_recall'].mean(),
                'F1': cv_results['test_f1'].mean(),
                'ROC_AUC': cv_results['test_roc_auc'].mean(),
                'Train_Acc': cv_results['train_accuracy'].mean(),
                'Overfit_Gap': cv_results['train_accuracy'].mean() - cv_results['test_accuracy'].mean()
            }
            results.append(result)
            
        except Exception as e:
            print(f"  ❌ {model_name} failed: {str(e)}")
    
    return pd.DataFrame(results)


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
    """
    if importance_df is None:
        return
    
    print(f"\n📊 Creating correlation heatmap...")
    
    top_features = importance_df.head(top_n)['feature'].tolist()
    
    # Create DataFrame with top features and target
    df_corr = X_raw[top_features].copy()
    
    # Impute NaNs for correlation calculation (use median, fast and safe)
    print("   Imputing NaNs for correlation calculation...")
    for col in df_corr.columns:
        if df_corr[col].isna().any():
            df_corr[col].fillna(df_corr[col].median(), inplace=True)
    
    df_corr['diagnosis_label'] = y
    
    # Calculate correlation matrix
    print("   Computing correlations...")
    corr_matrix = df_corr.corr()
    
    # Plot heatmap
    plt.figure(figsize=(14, 12))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                vmin=-1, vmax=1)
    plt.title(f'Correlation Heatmap: Top {top_n} Features vs Diagnosis', 
              fontsize=14, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {filename}")
    plt.close()
    
    # Print correlations with target
    print(f"\n📊 Correlations with Diagnosis Label:")
    target_corr = corr_matrix['diagnosis_label'].drop('diagnosis_label').sort_values(ascending=False)
    print(target_corr.to_string())


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
    
    # Sort by F1 score
    results_sorted = results_df.sort_values('F1', ascending=False)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    models = results_sorted['Model'].values
    n_models = len(models)
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, n_models))
    
    # 1. F1-Score comparison
    ax1 = axes[0, 0]
    bars1 = ax1.barh(range(n_models), results_sorted['F1'], color=colors)
    ax1.set_yticks(range(n_models))
    ax1.set_yticklabels(models)
    ax1.set_xlabel('F1-Score', fontsize=11)
    ax1.set_title(f'F1-Score by Model\n({diagnosis_code})', fontsize=12, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    ax1.invert_yaxis()
    # Add values on bars
    for i, (bar, val) in enumerate(zip(bars1, results_sorted['F1'])):
        ax1.text(val + 0.01, i, f'{val:.3f}', va='center', fontsize=9, fontweight='bold')
    
    # 2. ROC-AUC comparison
    ax2 = axes[0, 1]
    bars2 = ax2.barh(range(n_models), results_sorted['ROC_AUC'], color=colors)
    ax2.set_yticks(range(n_models))
    ax2.set_yticklabels(models)
    ax2.set_xlabel('ROC-AUC', fontsize=11)
    ax2.set_title(f'ROC-AUC by Model\n({diagnosis_code})', fontsize=12, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    ax2.invert_yaxis()
    for i, (bar, val) in enumerate(zip(bars2, results_sorted['ROC_AUC'])):
        ax2.text(val + 0.01, i, f'{val:.3f}', va='center', fontsize=9, fontweight='bold')
    
    # 3. Accuracy with error bars
    ax3 = axes[1, 0]
    bars3 = ax3.barh(range(n_models), results_sorted['Accuracy'], 
                     xerr=results_sorted['Accuracy_Std'], color=colors, capsize=5)
    ax3.set_yticks(range(n_models))
    ax3.set_yticklabels(models)
    ax3.set_xlabel('Accuracy ± Std', fontsize=11)
    ax3.set_title(f'Accuracy by Model\n({diagnosis_code})', fontsize=12, fontweight='bold')
    ax3.grid(axis='x', alpha=0.3)
    ax3.invert_yaxis()
    
    # 4. Overfitting analysis
    ax4 = axes[1, 1]
    gap_colors = ['red' if gap > 0.1 else 'orange' if gap > 0.05 else 'green' 
                  for gap in results_sorted['Overfit_Gap']]
    bars4 = ax4.barh(range(n_models), results_sorted['Overfit_Gap'], color=gap_colors, alpha=0.7)
    ax4.set_yticks(range(n_models))
    ax4.set_yticklabels(models)
    ax4.set_xlabel('Overfit Gap (Train - Test)', fontsize=11)
    ax4.set_title(f'Generalization by Model\n({diagnosis_code})', fontsize=12, fontweight='bold')
    ax4.axvline(x=0.05, color='orange', linestyle='--', alpha=0.5, linewidth=2, label='Warning')
    ax4.axvline(x=0.1, color='red', linestyle='--', alpha=0.5, linewidth=2, label='High')
    ax4.grid(axis='x', alpha=0.3)
    ax4.legend(loc='best', fontsize=9)
    ax4.invert_yaxis()
    
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
# Step 10: Comparative Analysis Across All Diagnoses
# =============================================================================

def create_comparative_analysis(all_diagnosis_results, all_refined_results):
    """
    Create comparative visualizations across all diagnoses.
    """
    print("\n" + "=" * 70)
    print("COMPARATIVE ANALYSIS ACROSS ALL DIAGNOSES")
    print("=" * 70)
    
    # 1. Performance comparison across diagnoses
    print("\n📊 1. Creating performance comparison...")
    
    # Get best model per diagnosis
    best_per_dx = all_diagnosis_results.loc[
        all_diagnosis_results.groupby('Diagnosis')['F1'].idxmax()
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
    
    # Step 5: Create models
    print("\n" + "=" * 70)
    print("Step 5: Creating Leakage-Free Model Pipelines")
    print("=" * 70)
    models = create_model_pipelines()
    print(f"\n✅ Created {len(models)} model pipelines")
    
    # Step 6: Evaluate each diagnosis
    print("\n" + "=" * 70)
    print(f"Step 6: Evaluating {len(valid_diagnoses)} Individual Diagnoses")
    print("=" * 70)
    
    all_results = []
    all_refined_results = []
    
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
            best = results_df.sort_values('F1', ascending=False).iloc[0]
            print(f"\n✅ Best model: {best['Model']}")
            print(f"   F1: {best['F1']:.3f}, ROC-AUC: {best['ROC_AUC']:.3f}, Gap: {best['Overfit_Gap']:.3f}")
            
            # Create visualizations comparing all models and highlighting best
            Path(viz_folder).mkdir(parents=True, exist_ok=True)
            create_model_comparison_for_diagnosis(results_df, dx_code, viz_folder)
            create_best_model_summary(results_df, dx_code, viz_folder)
            
            # Step 7: Feature importance and refined analysis for this diagnosis
            best_model_name = best['Model']
            best_pipeline = models[best_model_name]
            
            # Extract feature importances
            importance_df = extract_feature_importances(X_raw, y, X_raw.columns.tolist(), best_pipeline)
            
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
        
        create_comparative_analysis(final_results, all_refined_results)
        
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
        print(f"   - 1_performance_comparison.png")
        print(f"   - 2_refined_comparison.png")
        print(f"   - summary_statistics.csv")
        print(f"\n   Each diagnosis folder contains:")
        print(f"   - 0_all_models_comparison.png (NEW!)")
        print(f"   - 0_best_model_summary.png (NEW!)")
        print(f"   - initial_feature_importances.png")
        print(f"   - initial_correlation_heatmap.png")
        print(f"   - 1_refined_feature_importance.png")
        print(f"   - 2_prediction_analysis.png")
        print(f"   - 3_feature_distributions.png")
        print(f"   - 4_calibration_analysis.png")
        print(f"   - 5_performance_curves.png")
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

