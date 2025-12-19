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

# Model parameters (aggressive regularization)
RANDOM_STATE = 42
N_CV_FOLDS = 5
N_COMBINATIONS = 10  # Number of diagnosis combinations to test

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
# Step 3: Create Balanced Diagnosis Combinations
# =============================================================================

def create_balanced_combinations(patient_diagnoses, diagnosis_prevalence, 
                                n_combinations=10, target_prevalence=0.5, 
                                min_prevalence=0.4, max_prevalence=0.6):
    """
    Create N random combinations of diagnoses that result in balanced labels.
    
    Args:
        patient_diagnoses: dict {patient_id: set(diagnosis_codes)}
        diagnosis_prevalence: Series with diagnosis counts
        n_combinations: Number of combinations to create
        target_prevalence: Target prevalence (0.5 = 50%)
        min_prevalence: Minimum acceptable prevalence
        max_prevalence: Maximum acceptable prevalence
    
    Returns:
        List of diagnosis combinations (each is a set of diagnosis codes)
    """
    print("\n" + "=" * 70)
    print("Step 3: Creating Balanced Diagnosis Combinations")
    print("=" * 70)
    
    total_patients = len(patient_diagnoses)
    target_count = int(total_patients * target_prevalence)
    
    print(f"\nTarget: {target_count} patients ({target_prevalence*100:.0f}%) per combination")
    print(f"Acceptable range: {int(total_patients*min_prevalence)}-{int(total_patients*max_prevalence)} patients")
    
    # Filter to diagnoses with reasonable prevalence (5-40%)
    min_patients = int(total_patients * 0.05)
    max_patients = int(total_patients * 0.40)
    
    candidate_dx = diagnosis_prevalence[
        (diagnosis_prevalence >= min_patients) & 
        (diagnosis_prevalence <= max_patients)
    ]
    
    print(f"\nCandidate diagnoses (5-40% prevalence): {len(candidate_dx)}")
    
    if len(candidate_dx) < 5:
        print("⚠️ Too few candidate diagnoses, loosening criteria...")
        candidate_dx = diagnosis_prevalence.head(50)
    
    combinations = []
    attempts = 0
    max_attempts = n_combinations * 100
    
    np.random.seed(RANDOM_STATE)
    
    while len(combinations) < n_combinations and attempts < max_attempts:
        attempts += 1
        
        # Randomly select 2-6 diagnoses
        n_dx = np.random.randint(2, 7)
        selected_dx = set(np.random.choice(candidate_dx.index, size=min(n_dx, len(candidate_dx)), 
                                          replace=False))
        
        # Count patients with ANY of these diagnoses
        count = sum(1 for pt_dx in patient_diagnoses.values() 
                   if len(pt_dx.intersection(selected_dx)) > 0)
        
        prevalence = count / total_patients
        
        # Check if prevalence is in acceptable range
        if min_prevalence <= prevalence <= max_prevalence:
            # Check this combination isn't too similar to existing ones
            is_unique = True
            for existing_combo in combinations:
                overlap = len(selected_dx.intersection(existing_combo))
                if overlap / len(selected_dx) > 0.7:  # >70% overlap
                    is_unique = False
                    break
            
            if is_unique:
                combinations.append(selected_dx)
                print(f"\n✓ Combination {len(combinations)}: {count} patients ({prevalence*100:.1f}%)")
                print(f"  Diagnoses: {list(selected_dx)}")
    
    if len(combinations) < n_combinations:
        print(f"\n⚠️ Only found {len(combinations)} combinations (wanted {n_combinations})")
    
    return combinations


# =============================================================================
# Step 4: Create Labels for Each Combination
# =============================================================================

def create_labels_for_combination(patient_ids, patient_diagnoses, diagnosis_combination):
    """
    Create binary labels: 1 if patient has ANY diagnosis from combination, 0 otherwise.
    
    Returns:
        labels: Series with patient_id as index
    """
    labels = {}
    for patient_id in patient_ids:
        patient_dx = patient_diagnoses.get(patient_id, set())
        has_diagnosis = len(patient_dx.intersection(diagnosis_combination)) > 0
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
# Step 7: Evaluate Each Combination
# =============================================================================

def evaluate_combination(X_raw, y, combination_name, models):
    """
    Run leakage-free CV for one diagnosis combination.
    
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
                'Combination': combination_name,
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
    plt.show()


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
    df_corr['diagnosis_label'] = y
    
    # Calculate correlation matrix
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
    plt.show()
    
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
    plt.show()


    return pd.DataFrame(results)


# =============================================================================
# Main Execution
# =============================================================================

def main():
    print("\n" + "=" * 70)
    print("DIAGNOSIS PREDICTION FROM SDoH FEATURES")
    print("Balanced, Leakage-Free Analysis")
    print("=" * 70)
    
    # Step 1: Load and link datasets
    acxiom_df, diagnosis_df, common_ids = load_and_link_datasets(ACXIOM_PATH, DIAGNOSIS_PATH)
    
    # Step 2: Extract diagnoses
    patient_diagnoses, diagnosis_prevalence = extract_diagnosis_codes(diagnosis_df)
    
    # Step 3: Create balanced combinations
    combinations = create_balanced_combinations(
        patient_diagnoses, diagnosis_prevalence,
        n_combinations=N_COMBINATIONS
    )
    
    if len(combinations) == 0:
        print("\n❌ No balanced combinations found!")
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
    
    # Step 6: Evaluate each combination
    print("\n" + "=" * 70)
    print(f"Step 6: Evaluating {len(combinations)} Diagnosis Combinations")
    print("=" * 70)
    
    all_results = []
    
    for i, combination in enumerate(combinations, 1):
        combo_name = f"Combo_{i}"
        print(f"\n{'='*70}")
        print(f"🔬 Testing Combination {i}/{len(combinations)}")
        print(f"{'='*70}")
        print(f"Diagnoses: {list(combination)}")
        
        # Create labels
        y = create_labels_for_combination(X_raw.index, patient_diagnoses, combination)
        y = y.reindex(X_raw.index, fill_value=0)
        
        print(f"\nLabel distribution:")
        print(f"  Positive (has diagnosis): {(y==1).sum()} ({(y==1).mean()*100:.1f}%)")
        print(f"  Negative (no diagnosis): {(y==0).sum()} ({(y==0).mean()*100:.1f}%)")
        
        # Evaluate
        results_df = evaluate_combination(X_raw, y, combo_name, models)
        all_results.append(results_df)
        
        # Print summary
        if not results_df.empty:
            best = results_df.sort_values('F1', ascending=False).iloc[0]
            print(f"\n✅ Best model: {best['Model']}")
            print(f"   F1: {best['F1']:.3f}, ROC-AUC: {best['ROC_AUC']:.3f}, Gap: {best['Overfit_Gap']:.3f}")
    
    # Combine all results
    final_results = pd.concat(all_results, ignore_index=True)
    
    # Save results
    final_results.to_csv(OUTPUT_PATH, index=False)
    print(f"\n✅ Saved results to: {OUTPUT_PATH}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Best Results per Combination")
    print("=" * 70)
    
    for combo in final_results['Combination'].unique():
        combo_results = final_results[final_results['Combination'] == combo]
        best = combo_results.sort_values('F1', ascending=False).iloc[0]
        
        print(f"\n{combo}:")
        print(f"  Best Model: {best['Model']}")
        print(f"  Accuracy: {best['Accuracy']:.3f} ± {best['Accuracy_Std']:.3f}")
        print(f"  F1-Score: {best['F1']:.3f}")
        print(f"  ROC-AUC: {best['ROC_AUC']:.3f}")
        print(f"  Overfit Gap: {best['Overfit_Gap']:.3f}")
        
        if best['Overfit_Gap'] < 0.05:
            print(f"  ✅ Excellent generalization")
        elif best['Overfit_Gap'] < 0.10:
            print(f"  ⚡ Good generalization")
        else:
            print(f"  ⚠️ Some overfitting")
    
    # Overall best
    overall_best = final_results.sort_values('F1', ascending=False).iloc[0]
    print(f"\n🏆 Overall Best Performance:")
    print(f"   Combination: {overall_best['Combination']}")
    print(f"   Model: {overall_best['Model']}")
    print(f"   F1-Score: {overall_best['F1']:.3f}")
    print(f"   ROC-AUC: {overall_best['ROC_AUC']:.3f}")
    
    # Step 7: Extract feature importances for best combination
    print("\n" + "=" * 70)
    print("Step 7: Feature Importance Analysis")
    print("=" * 70)
    
    best_combo_idx = int(overall_best['Combination'].split('_')[1]) - 1
    best_combo = combinations[best_combo_idx]
    best_model_name = overall_best['Model']
    
    print(f"\nAnalyzing best combination: {overall_best['Combination']}")
    print(f"Diagnoses: {list(best_combo)}")
    print(f"Best model: {best_model_name}")
    
    # Create labels for best combination
    y_best = create_labels_for_combination(X_raw.index, patient_diagnoses, best_combo)
    y_best = y_best.reindex(X_raw.index, fill_value=0)
    
    # Get the best model pipeline
    best_pipeline = models[best_model_name]
    
    # Extract feature importances
    importance_df = extract_feature_importances(X_raw, y_best, X_raw.columns.tolist(), best_pipeline)
    
    # Visualizations
    if importance_df is not None:
        plot_feature_importances(importance_df, top_n=20)
        plot_correlation_heatmap(X_raw, y_best, importance_df, top_n=20)
    
    # Model comparison plot
    plot_model_comparison(final_results)
    
    print("\n" + "=" * 70)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\n📁 Output Files:")
    print(f"   - {OUTPUT_PATH}")
    print(f"   - feature_importances.png")
    print(f"   - correlation_heatmap.png")
    print(f"   - model_comparison.png")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()

