import pandas as pd
import edu_sdoh as es

fm = pd.read_csv('full_acxiom_with_ed_label.csv', low_memory=False)
print('fm', fm.shape)

sdoh = es.identify_sdoh_columns(fm)
X_raw, y, feature_names, class_map = es.prepare_ml_data(fm, sdoh, include_diagnosis=False, exclude_dx_for_temporal=True)
print('X_raw', X_raw.shape)

df_out = es.load_and_merge_diagnosis(fm, 'diagnosis_with_acxiom3.csv')
print('df_out', df_out.shape)
print('df_out cols sample:', df_out.columns[:30])

# try to find column in df_out that contains empi values
empi_cols = [c for c in df_out.columns if c.lower()=='empi' or c.lower().endswith('empi') or c=='_empi_bridge']
print('empi-like cols in df_out:', empi_cols)

paired_ids = fm.loc[X_raw.index, 'member_id'].astype(str).str.strip()
print('paired_ids sample:', paired_ids.head())

# build paired subset using empi-like column if available
if empi_cols:
    col = empi_cols[0]
    df_paired = df_out[df_out[col].astype(str).str.strip().isin(set(paired_ids))].copy()
    print('df_paired via empi column:', df_paired.shape)
    # check diagnostics positives counts
    dx_cols = [c for c in df_paired.columns if c.startswith('dx_')]
    if dx_cols:
        for c in dx_cols[:10]:
            s = pd.to_numeric(df_paired[c], errors='coerce')
            print(c, 'pos=', s.fillna(0).astype(int).sum())
else:
    print('No empi-like column found in df_out; cannot pair')

print('done')
