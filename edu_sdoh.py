"""
edu_sdoh.py

ED Utilization Analysis with SDoH Features
------------------------------------------
Calculates ED visit counts per patient, creates utilization categories,
and analyzes feature importance and correlations with SDoH data.

ED Utilization Classes:
- 0: No ED visits
- 1: One ED visit
- 2: High utilization (2+ ED visits)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, make_scorer
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

NYU_EDU_PATH = "nyu_edu.csv"
ACXIOM_PATH = "full_acxiom.csv"
OUTPUT_PATH = "full_acxiom_with_ed_label.csv"

# Model parameters
RANDOM_STATE = 42
N_CV_FOLDS = 5  # Number of cross-validation folds
TEST_SIZE = 0.2  # Hold-out test set size

# Model-specific parameters (to prevent overfitting)
RF_PARAMS = {
    'n_estimators': 100,
    'max_depth': 6,
    'min_samples_split': 10,
    'min_samples_leaf': 5,
    'max_features': 'sqrt',
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
    'class_weight': 'balanced'
}

GB_PARAMS = {
    'n_estimators': 100,
    'max_depth': 4,
    'learning_rate': 0.05,
    'min_samples_split': 10,
    'min_samples_leaf': 5,
    'max_features': 'sqrt',
    'random_state': RANDOM_STATE,
    'subsample': 0.8
}

LGBM_PARAMS = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.05,
    'num_leaves': 31,
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
    'verbosity': -1,
    'class_weight': 'balanced'
}

XGB_PARAMS = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.05,
    'min_child_weight': 5,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'gamma': 0.1,
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
    'tree_method': 'hist'
}

CATBOOST_PARAMS = {
    'iterations': 100,
    'depth': 6,
    'learning_rate': 0.05,
    'l2_leaf_reg': 3,
    'random_state': RANDOM_STATE,
    'verbose': False,
    'thread_count': -1,
    'auto_class_weights': 'Balanced'
}

# Analysis parameters
TOP_N_FEATURES = 20


# =============================================================================
# Step 1: Calculate ED Visits per Patient
# =============================================================================

def calculate_ed_visits(nyu_edu_path):
    """
    Calculate total ED visits per patient from NYU ED data.
    
    Args:
        nyu_edu_path: Path to nyu_edu.csv
    
    Returns:
        DataFrame with sys_mbr_sk and total_ed_visits
    """
    print("=" * 70)
    print("Step 1: Calculating ED Visits per Patient")
    print("=" * 70)
    
    # Load NYU ED data
    df_edu = pd.read_csv(nyu_edu_path)
    print(f"Loaded {len(df_edu):,} ED records")
    
    # The ed_count column contains the number of ED visits per record
    # We need to sum by patient
    if 'ed_count' not in df_edu.columns:
        print("\nAvailable columns:", list(df_edu.columns[:20]))
        raise ValueError("Column 'ed_count' not found in nyu_edu.csv")
    
    # Aggregate ED visits by patient
    ed_visits = df_edu.groupby('sys_mbr_sk')['ed_count'].sum().reset_index()
    ed_visits.columns = ['sys_mbr_sk', 'total_ed_visits']
    
    # Fill NaN with 0
    ed_visits['total_ed_visits'] = ed_visits['total_ed_visits'].fillna(0).astype(int)
    
    print(f"\nTotal unique patients: {len(ed_visits):,}")
    print(f"ED visit distribution:")
    print(ed_visits['total_ed_visits'].describe())
    
    return ed_visits


# =============================================================================
# Step 2: Create ED Utilization Labels
# =============================================================================

def create_ed_utilization_labels(ed_visits_df):
    """
    Create ED utilization categories:
    - 0: No ED visits
    - 1: One ED visit
    - 2: High utilization (2+ ED visits)
    
    Args:
        ed_visits_df: DataFrame with sys_mbr_sk and total_ed_visits
    
    Returns:
        DataFrame with added ed_utilization_class column
    """
    print("\n" + "=" * 70)
    print("Step 2: Creating ED Utilization Labels")
    print("=" * 70)
    
    df = ed_visits_df.copy()
    
    # Create labels
    df['ed_utilization_class'] = df['total_ed_visits'].apply(
        lambda x: 0 if x == 0 else (1 if x == 1 else 2)
    )
    
    # Print distribution
    print("\nED Utilization Class Distribution:")
    class_counts = df['ed_utilization_class'].value_counts().sort_index()
    for cls, count in class_counts.items():
        cls_name = {0: "No visits", 1: "One visit", 2: "High (2+) visits"}[cls]
        pct = count / len(df) * 100
        print(f"  Class {cls} ({cls_name}): {count:,} patients ({pct:.1f}%)")
    
    return df


# =============================================================================
# Step 3: Merge with Acxiom Data
# =============================================================================

def merge_ed_labels_with_acxiom(ed_labels_df, acxiom_path, output_path=None):
    """
    Merge ED utilization labels with Acxiom SDoH data.
    
    Args:
        ed_labels_df: DataFrame with sys_mbr_sk and ed_utilization_class
        acxiom_path: Path to full_acxiom.csv
        output_path: Optional path to save merged data
    
    Returns:
        Merged DataFrame
    """
    print("\n" + "=" * 70)
    print("Step 3: Merging ED Labels with Acxiom Data")
    print("=" * 70)
    
    # Load Acxiom data
    df_acxiom = pd.read_csv(acxiom_path, low_memory=False)
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
# Step 5: Prepare Data for Machine Learning
# =============================================================================

def prepare_ml_data(df, sdoh_cols, include_diagnosis=True):
    """
    Prepare feature matrix and target for machine learning.
    
    Args:
        df: Merged DataFrame
        sdoh_cols: List of SDoH column names
        include_diagnosis: Whether to include dx_* columns as features
    
    Returns:
        X (features), y (target), feature_names
    """
    print("\n" + "=" * 70)
    print("Step 5: Preparing Data for Machine Learning")
    print("=" * 70)
    
    # Get diagnosis columns if requested
    dx_cols = []
    if include_diagnosis:
        dx_cols = [col for col in df.columns if col.startswith('dx_') and col != 'dx_other_count']
        print(f"Found {len(dx_cols)} diagnosis columns")
    
    # Combine features
    feature_cols = sdoh_cols + dx_cols
    print(f"Total features: {len(feature_cols)} ({len(sdoh_cols)} SDoH + {len(dx_cols)} diagnoses)")
    
    # Extract features and target
    X = df[feature_cols].copy()
    y = df['ed_utilization_class'].copy()
    
    # Remove rows with missing target
    valid_mask = y.notna()
    X = X[valid_mask]
    y = y[valid_mask]
    
    print(f"\nSamples: {len(X):,}")
    print(f"Target distribution:")
    print(y.value_counts().sort_index())
    
    # Convert to numeric and handle missing values
    print("\nConverting features to numeric...")
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    
    # Check missingness
    missing_pct = X.isna().mean()
    print(f"\nFeatures with >50% missing: {(missing_pct > 0.5).sum()}")
    
    # Remove features that are all NaN
    all_nan_cols = X.columns[X.isna().all()]
    if len(all_nan_cols) > 0:
        print(f"Removing {len(all_nan_cols)} completely empty features")
        X = X.drop(columns=all_nan_cols)
        feature_cols = [c for c in feature_cols if c not in all_nan_cols]
    
    # Impute missing values with median
    print("\nImputing missing values with median...")
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    X = pd.DataFrame(X_imputed, columns=feature_cols, index=X.index)
    
    print(f"\nFinal feature matrix: {X.shape}")
    
    return X, y, feature_cols


# =============================================================================
# Step 6: Cross-Validation Model Comparison
# =============================================================================

def create_model_pipelines():
    """
    Create pipelines for different models.
    
    Returns:
        Dictionary of model names and their pipeline objects
    """
    models = {}
    
    # Random Forest
    models['Random Forest'] = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', RandomForestClassifier(**RF_PARAMS))
    ])
    
    # Gradient Boosting
    models['Gradient Boosting'] = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', GradientBoostingClassifier(**GB_PARAMS))
    ])
    
    # LightGBM (if available)
    if LIGHTGBM_AVAILABLE:
        models['LightGBM'] = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('model', lgb.LGBMClassifier(**LGBM_PARAMS))
        ])
    
    # XGBoost (if available)
    if XGBOOST_AVAILABLE:
        models['XGBoost'] = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('model', xgb.XGBClassifier(**XGB_PARAMS))
        ])
    
    # CatBoost (if available) - works well with categorical features
    if CATBOOST_AVAILABLE:
        models['CatBoost'] = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('model', CatBoostClassifier(**CATBOOST_PARAMS))
        ])
    
    return models


def cross_validate_models(X, y, models, n_folds=N_CV_FOLDS):
    """
    Perform stratified k-fold cross-validation for all models.
    
    Args:
        X: Feature matrix
        y: Target vector
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
    print(f"\nClasses present: {unique_classes}")
    print(f"Class distribution:")
    for cls in unique_classes:
        count = (y == cls).sum()
        pct = count / len(y) * 100
        print(f"  Class {cls}: {count:,} samples ({pct:.1f}%)")
    
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
                n_jobs=-1,
                return_train_score=True,
                error_score='raise'
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
            
            if result['Overfit_Gap'] > 0.1:
                print(f"  ⚠️ Warning: Potential overfitting (gap: {result['Overfit_Gap']:.3f})")
            
        except Exception as e:
            print(f"  ❌ Error training {model_name}: {str(e)}")
            continue
    
    results_df = pd.DataFrame(results)
    
    # Sort by F1 score
    results_df = results_df.sort_values('CV_F1_Mean', ascending=False)
    
    print("\n" + "=" * 70)
    print("Cross-Validation Results Summary")
    print("=" * 70)
    print(results_df[['Model', 'CV_Accuracy_Mean', 'CV_F1_Mean', 'CV_ROC_AUC_Mean', 'Overfit_Gap']].to_string(index=False))
    
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
        importances_df = pd.DataFrame({
            'feature': feature_names,
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
            scores = cross_val_score(pipeline, X, y, cv=skf, scoring='f1_macro', n_jobs=-1)
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
    
    # Step 4: Identify SDoH columns
    sdoh_cols = identify_sdoh_columns(df_merged)
    
    # Check if we have enough data
    if len(sdoh_cols) == 0:
        print("\n⚠️ WARNING: No SDoH columns found!")
        print("The analysis will proceed with diagnosis features only.")
    
    # Step 5: Prepare ML data
    X, y, feature_names = prepare_ml_data(df_merged, sdoh_cols, include_diagnosis=True)
    
    if len(X) == 0 or len(feature_names) == 0:
        print("\n❌ ERROR: No valid features or samples for analysis.")
        print("Please check your data files.")
        return
    
    # Step 6: Create model pipelines and perform cross-validation
    models = create_model_pipelines()
    cv_results_df = cross_validate_models(X, y, models, n_folds=N_CV_FOLDS)
    
    # Save CV results
    cv_results_df.to_csv('cv_results.csv', index=False)
    print("\n✅ Saved: cv_results.csv")
    
    # Step 7: Train best model and get feature importances
    importances, best_model, best_model_name = train_best_model_and_get_importances(
        X, y, feature_names, cv_results_df
    )
    
    # Step 8: Model comparison visualization
    plot_model_comparison(cv_results_df)
    
    # Step 9: Plot feature importances
    plot_feature_importances(importances, top_n=TOP_N_FEATURES)
    
    # Step 10: Correlation heatmap
    plot_correlation_heatmap(df_merged, importances, top_n=TOP_N_FEATURES)
    
    # Step 11: CV metrics distribution
    plot_cv_metrics_boxplot(models, X, y, n_folds=N_CV_FOLDS)
    
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
        print(f"   ⚠️ Model shows significant overfitting (overfit gap: {best_overfit:.3f})")
        print("      Consider: more regularization, more data, or simpler model")
    
    print("\n" + "=" * 70)
    

if __name__ == "__main__":
    main()

