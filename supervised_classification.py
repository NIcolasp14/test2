"""
supervised_classification.py

Supervised classification tasks: Predict STATE, CITY, ZIP5, ZIP3 from SDoH features.
Uses stratified k-fold cross-validation with multinomial logistic regression.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    top_k_accuracy_score,
)

from utils import build_sdoh_X, collapse_rare_classes, choose_n_splits, pick_col


def run_multiclass_task(X_df, y_series, task_name, min_count=50, top_n=None, 
                        topk=5, desired_splits=5):
    """
    Run a multiclass classification task with cross-validation.
    
    Args:
        X_df: Feature dataframe
        y_series: Target series
        task_name: Name for display
        min_count: Minimum count to keep a class
        top_n: If specified, keep only top N classes
        topk: K for top-k accuracy metric
        desired_splits: Desired number of CV splits
    
    Returns:
        Dictionary with results
    """
    # Collapse rare labels so the task is feasible
    y = collapse_rare_classes(y_series, min_count=min_count, top_n=top_n)

    # Encode labels
    y_cat = pd.Categorical(y)
    y_codes = y_cat.codes
    n_classes = len(y_cat.categories)

    # If everything collapsed into 1 class, can't classify
    if n_classes < 2:
        print(f"\n=== {task_name} ===")
        print("Not enough classes after collapsing. Increase diversity or lower min_count/top_n.")
        return None

    n_splits = choose_n_splits(y_codes, desired=desired_splits)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Model pipeline: KNN impute -> scale -> multinomial logistic regression
    pipe = Pipeline([
        ("impute", KNNImputer(n_neighbors=10, weights="distance")),
        ("scale", StandardScaler(with_mean=True, with_std=True)),
        ("clf", LogisticRegression(
            multi_class="multinomial",
            solver="saga",
            max_iter=5000,
            n_jobs=-1
        ))
    ])

    X_np = X_df.to_numpy(dtype=float)
    accs, baccs, f1s, topks = [], [], [], []
    k_use = min(topk, n_classes)

    for tr, te in skf.split(X_np, y_codes):
        pipe.fit(X_np[tr], y_codes[tr])
        probs = pipe.predict_proba(X_np[te])
        pred = probs.argmax(axis=1)

        accs.append(accuracy_score(y_codes[te], pred))
        baccs.append(balanced_accuracy_score(y_codes[te], pred))
        f1s.append(f1_score(y_codes[te], pred, average="macro"))
        topks.append(top_k_accuracy_score(y_codes[te], probs, k=k_use, 
                                          labels=np.arange(n_classes)))

    print(f"\n=== {task_name} ===")
    print(f"Classes (after collapse): {n_classes:,}")
    print(f"Splits: {n_splits}")
    print(f"Accuracy:          {np.mean(accs):.3f} ± {np.std(accs):.3f}")
    print(f"Balanced accuracy: {np.mean(baccs):.3f} ± {np.std(baccs):.3f}")
    print(f"Macro-F1:          {np.mean(f1s):.3f} ± {np.std(f1s):.3f}")
    print(f"Top-{k_use} acc:         {np.mean(topks):.3f} ± {np.std(topks):.3f}")

    # show label counts
    print("\nTop label counts:")
    print(pd.Series(y).value_counts().head(15))

    return {
        "task": task_name,
        "classes": n_classes,
        "splits": n_splits,
        "acc": accs,
        "bacc": baccs,
        "macro_f1": f1s,
        f"top{k_use}": topks,
    }


def run_geographic_classification(sdoh_all, row_nan_frac_max=0.40, col_nan_frac_max=0.95):
    """
    Run classification tasks to predict geographic variables from SDoH features.
    
    Args:
        sdoh_all: DataFrame with 'flag' column marking start of SDoH features
        row_nan_frac_max: Maximum fraction of NaN per row for QC
        col_nan_frac_max: Maximum fraction of NaN per column for QC
    
    Returns:
        Dictionary with results for each geographic task
    """
    # Build X (SDoH features) + QC
    df_qc, X_qc = build_sdoh_X(sdoh_all, 
                                row_nan_frac_max=row_nan_frac_max,
                                col_nan_frac_max=col_nan_frac_max)

    # Build targets (ZIP/CITY/STATE)
    zip_col = pick_col(df_qc, ["memberzipcode", "memberzip", "zipcode", "zip", "member_zipcode"])
    city_col = pick_col(df_qc, ["city", "membercity", "member_city"])
    state_col = pick_col(df_qc, ["state", "memberstate", "member_state"])

    if zip_col is None or city_col is None or state_col is None:
        print("Available columns (first 100):", list(df_qc.columns[:100]))
        raise ValueError("Couldn't find zip/city/state columns. Update candidates in pick_col().")

    df_qc["zip_str"] = df_qc[zip_col].astype("string").str.extract(r"(\d{5})", expand=False)
    df_qc["zip3"] = df_qc["zip_str"].str[:3]
    df_qc["city_str"] = df_qc[city_col].astype("string").str.strip()
    df_qc["state_str"] = df_qc[state_col].astype("string").str.strip()

    # Run supervised classifications
    results = {}

    # STATE: usually few classes; keep all
    results['state'] = run_multiclass_task(
        X_qc, df_qc["state_str"],
        task_name="Predict STATE from SDoH",
        min_count=1,
        topk=3
    )

    # CITY: many classes; collapse rare cities
    results['city'] = run_multiclass_task(
        X_qc, df_qc["city_str"],
        task_name="Predict CITY from SDoH",
        min_count=100,
        topk=5
    )

    # ZIP5: very high classes; usually collapse hard
    results['zip5'] = run_multiclass_task(
        X_qc, df_qc["zip_str"],
        task_name="Predict ZIP5 from SDoH",
        min_count=200,
        topk=10
    )

    # ZIP3: recommended compromise
    results['zip3'] = run_multiclass_task(
        X_qc, df_qc["zip3"],
        task_name="Predict ZIP3 from SDoH",
        min_count=50,
        topk=5
    )

    return results, df_qc









