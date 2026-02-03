"""
Conservative runner to create balanced diagnosis groups and evaluate with Random Forest (3-fold CV)
Generates: diagnosis_group_results_conservative.csv and diagnosis_group_best_model_scores_conservative.png
"""
import edu_sdoh
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate

OUTPUT_CSV = 'diagnosis_group_results_conservative.csv'
OUTPUT_PNG = 'diagnosis_group_best_model_scores_conservative.png'
DIAG_PATH = 'diagnosis_with_acxiom3.csv'

def main():
    print('Runner: building merged data and merging diagnosis...')

    # Recreate merged acxiom+ED labels
    ed_visits = edu_sdoh.calculate_ed_visits(edu_sdoh.NYU_EDU_PATH)
    ed_labels = edu_sdoh.create_ed_utilization_labels(ed_visits)
    df_merged = edu_sdoh.merge_ed_labels_with_acxiom(ed_labels, edu_sdoh.ACXIOM_PATH, output_path=None)

    sdoh_cols = edu_sdoh.identify_sdoh_columns(df_merged)

    # Prepare ML data to get feature list (keep diagnosis excluded for now)
    X_raw, y, feature_cols, class_mapping = edu_sdoh.prepare_ml_data(
        df_merged, sdoh_cols, include_diagnosis=True, exclude_dx_for_temporal=False
    )

    # Load and merge diagnosis into df_merged (adds dx_ columns)
    df_with_dx = edu_sdoh.load_and_merge_diagnosis(df_merged, DIAG_PATH)

    # Identify dx_ columns
    dx_cols = [c for c in df_with_dx.columns if c.startswith('dx_')]
    print(f'Found {len(dx_cols)} dx_ columns after merging')
    if len(dx_cols) == 0:
        print('No diagnosis columns found after merge — aborting runner.')
        return

    # Create balanced groups (10 groups, target ~50%)
    groups = edu_sdoh.create_balanced_groups(df_with_dx, dx_cols, n_groups=10, target_frac=0.5)
    print(f'Created {len(groups)} groups')

    # Use only Random Forest to keep runtime small
    models = {'Random Forest': edu_sdoh.create_model_pipelines()['Random Forest']}

    results = []
    for i, grp in enumerate(groups):
        label = f'group_{i+1}'
        print(f'\nEvaluating {label} — {len(grp)} dx codes')
        # build binary label: any dx present
        try:
            y_series = df_with_dx[grp].apply(lambda row: (pd.to_numeric(row, errors='coerce').fillna(0) > 0).any(), axis=1).astype(int)
        except Exception:
            y_series = (df_with_dx[grp] > 0).any(axis=1).astype(int)

        pos = int(y_series.sum())
        neg = int((y_series == 0).sum())
        total = len(y_series)
        print(f' pos={pos}, neg={neg}, total={total}')
        if pos < 5 or neg < 5:
            print(' Skipping — insufficient pos/neg')
            continue

        # No undersampling per request — evaluate on full set
        X_eval = df_with_dx[feature_cols].copy()
        # align X and y
        mask = y_series.notna()
        X_eval = X_eval[mask]
        y_eval = y_series[mask]

        # Convert features to numeric where possible (NaNs preserved)
        for col in X_eval.columns:
            X_eval[col] = pd.to_numeric(X_eval[col], errors='coerce')

        # Conservative 3-fold CV (reduce runtime)
        folds = min(3, max(2, int((y_eval==1).sum())))
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=edu_sdoh.RANDOM_STATE)

        scoring = {'f1': 'f1', 'roc_auc': 'roc_auc', 'accuracy': 'accuracy'}

        # Run only Random Forest
        pipeline = models['Random Forest']
        try:
                # Run serially to avoid joblib/process pickling issues on this environment
                cv_res = cross_validate(pipeline, X_eval, y_eval, cv=skf, scoring=scoring, n_jobs=1, return_train_score=False)
        except Exception as e:
            print(' CV failed for group', label, ' — ', e)
            continue

        folds_f1 = list(cv_res['test_f1'])
        mean_f1 = float(np.mean(folds_f1))
        std_f1 = float(np.std(folds_f1))
        mean_roc = float(np.mean(cv_res['test_roc_auc'])) if 'test_roc_auc' in cv_res else np.nan

        print(f" Random Forest: F1={mean_f1:.3f} ± {std_f1:.3f}, ROC-AUC={mean_roc:.3f}")

        results.append({
            'group': label,
            'dx_codes': grp,
            'pos': pos,
            'neg': neg,
            'model': 'Random Forest',
            'mean_f1': mean_f1,
            'std_f1': std_f1,
            'mean_roc': mean_roc,
            'folds': folds_f1
        })

    # Save results
    if results:
        rows = []
        for r in results:
            rows.append({
                'group': r['group'],
                'dx_codes': ';'.join(r['dx_codes']),
                'model': r['model'],
                'mean_f1': r['mean_f1'],
                'std_f1': r['std_f1'],
                'mean_roc': r['mean_roc'],
                'folds': str(r['folds']),
                'pos': r['pos'],
                'neg': r['neg']
            })
        pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False)
        print(f'Wrote results to {OUTPUT_CSV}')

        # Convert to edu_sdoh-results format for plotting
        edu_results = []
        for r in results:
            edu_results.append({
                'group': r['group'],
                'dx_codes': r['dx_codes'],
                'pos': r['pos'],
                'neg': r['neg'],
                'models': {
                    'Random Forest': {
                        'mean_f1': r['mean_f1'],
                        'std_f1': r['std_f1'],
                        'mean_roc': r['mean_roc'],
                        'folds': r['folds']
                    }
                },
                'best_model': 'Random Forest',
                'best_mean_f1': r['mean_f1']
            })

        edu_sdoh.plot_group_distributions(edu_results, out_png=OUTPUT_PNG)
        print(f'Wrote plot to {OUTPUT_PNG}')
    else:
        print('No group results to save.')

if __name__ == '__main__':
    main()
