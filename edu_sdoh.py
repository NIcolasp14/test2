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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# Configuration
# =============================================================================

NYU_EDU_PATH = "nyu_edu.csv"
ACXIOM_PATH = "full_acxiom.csv"
OUTPUT_PATH = "full_acxiom_with_ed_label.csv"

# Random Forest parameters
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH = 10
RF_RANDOM_STATE = 42
RF_TEST_SIZE = 0.3

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
    
    # Merge on sys_mbr_sk
    df_merged = df_acxiom.merge(
        ed_labels_df[['sys_mbr_sk', 'total_ed_visits', 'ed_utilization_class']], 
        on='sys_mbr_sk', 
        how='left'
    )
    
    # Fill NaN for patients not in ED data (they have 0 visits)
    df_merged['total_ed_visits'] = df_merged['total_ed_visits'].fillna(0).astype(int)
    df_merged['ed_utilization_class'] = df_merged['ed_utilization_class'].fillna(0).astype(int)
    
    print(f"Merged data shape: {df_merged.shape}")
    print(f"\nPatients with ED data: {(df_merged['total_ed_visits'] > 0).sum():,}")
    print(f"Patients without ED data: {(df_merged['total_ed_visits'] == 0).sum():,}")
    
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
# Step 6: Train Random Forest and Get Feature Importances
# =============================================================================

def train_rf_and_get_importances(X, y, feature_names, n_estimators=RF_N_ESTIMATORS, 
                                  max_depth=RF_MAX_DEPTH, random_state=RF_RANDOM_STATE,
                                  test_size=RF_TEST_SIZE):
    """
    Train Random Forest Classifier and extract feature importances.
    
    Args:
        X: Feature matrix
        y: Target vector
        feature_names: List of feature names
        n_estimators: Number of trees
        max_depth: Maximum depth of trees
        random_state: Random seed
        test_size: Test set proportion
    
    Returns:
        DataFrame with feature importances, trained model, test metrics
    """
    print("\n" + "=" * 70)
    print("Step 6: Training Random Forest Classifier")
    print("=" * 70)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Training set: {len(X_train):,} samples")
    print(f"Test set: {len(X_test):,} samples")
    
    # Scale features
    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest
    print(f"\nTraining Random Forest (n_estimators={n_estimators}, max_depth={max_depth})...")
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
        class_weight='balanced'  # Handle class imbalance
    )
    rf.fit(X_train_scaled, y_train)
    
    # Evaluate
    train_score = rf.score(X_train_scaled, y_train)
    test_score = rf.score(X_test_scaled, y_test)
    
    print(f"\n✅ Training complete!")
    print(f"Training accuracy: {train_score:.3f}")
    print(f"Test accuracy: {test_score:.3f}")
    
    # Predictions
    y_pred = rf.predict(X_test_scaled)
    
    print("\n" + "=" * 70)
    print("Classification Report")
    print("=" * 70)
    print(classification_report(y_test, y_pred, 
                                target_names=['No visits', 'One visit', 'High (2+) visits']))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Get feature importances
    importances = pd.DataFrame({
        'feature': feature_names,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n" + "=" * 70)
    print(f"Top {TOP_N_FEATURES} Most Important Features")
    print("=" * 70)
    print(importances.head(TOP_N_FEATURES).to_string(index=False))
    
    return importances, rf, {'train_score': train_score, 'test_score': test_score, 
                            'y_test': y_test, 'y_pred': y_pred}


# =============================================================================
# Step 7: Visualize Feature Importances
# =============================================================================

def plot_feature_importances(importances_df, top_n=TOP_N_FEATURES):
    """
    Plot top N feature importances.
    
    Args:
        importances_df: DataFrame with feature and importance columns
        top_n: Number of top features to plot
    """
    print("\n" + "=" * 70)
    print("Step 7: Visualizing Feature Importances")
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
# Step 8: Correlation Heatmap for Top Features
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
    print("\n" + "=" * 70)
    print("Step 8: Creating Correlation Heatmap")
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
# Step 9: Additional Visualizations
# =============================================================================

def plot_confusion_matrix(y_test, y_pred):
    """
    Plot confusion matrix heatmap.
    
    Args:
        y_test: True labels
        y_pred: Predicted labels
    """
    print("\n" + "=" * 70)
    print("Step 9: Creating Confusion Matrix Visualization")
    print("=" * 70)
    
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
    
    # Step 6: Train RF and get importances
    importances, rf_model, metrics = train_rf_and_get_importances(X, y, feature_names)
    
    # Step 7: Plot feature importances
    plot_feature_importances(importances, top_n=TOP_N_FEATURES)
    
    # Step 8: Correlation heatmap
    plot_correlation_heatmap(df_merged, importances, top_n=TOP_N_FEATURES)
    
    # Step 9: Additional visualizations
    plot_confusion_matrix(metrics['y_test'], metrics['y_pred'])
    plot_ed_utilization_distribution(df_merged)
    
    # Summary
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\n✅ Output files generated:")
    print(f"   - {OUTPUT_PATH}")
    print(f"   - feature_importances.png")
    print(f"   - correlation_heatmap.png")
    print(f"   - confusion_matrix.png")
    print(f"   - ed_utilization_distribution.png")
    
    print(f"\n📊 Key Results:")
    print(f"   - Total patients: {len(df_merged):,}")
    print(f"   - Features analyzed: {len(feature_names)}")
    print(f"   - Test accuracy: {metrics['test_score']:.3f}")
    print(f"   - Top feature: {importances.iloc[0]['feature']} (importance: {importances.iloc[0]['importance']:.4f})")
    
    print("\n" + "=" * 70)
    

if __name__ == "__main__":
    main()

