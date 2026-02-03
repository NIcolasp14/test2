import pandas as pd

fm = pd.read_csv('full_acxiom_with_ed_label.csv', low_memory=False)
# Identify SDoH cols like in main
sdoh = [c for c in fm.columns if isinstance(c, str) and c[:2].isalpha() and c[2:].isdigit()]

# Prepare ml data subset similar to main (exclude dx)
from edu_sdoh import prepare_ml_data
X_raw, y, feature_names, class_map = prepare_ml_data(fm, sdoh, include_diagnosis=False, exclude_dx_for_temporal=True)

# Load diagnosis and merge
from edu_sdoh import load_and_merge_diagnosis
df_out = load_and_merge_diagnosis(fm[['member_id']].merge(fm, how='left'), 'diagnosis_with_acxiom3.csv') if False else load_and_merge_diagnosis(fm, 'diagnosis_with_acxiom3.csv')

# attempt to restrict to the ML subset using member_id mapping if possible
try:
    paired_ids = fm.loc[X_raw.index, 'member_id'].astype(str).str.strip()
    df_paired = df_out[df_out['member_id'].astype(str).str.strip().isin(set(paired_ids))].copy()
except Exception:
    df_paired = df_out.copy()

print('df_paired shape:', df_paired.shape)
# dx cols
dx_cols = [c for c in df_paired.columns if c.startswith('dx_')]
print('dx count:', len(dx_cols))

N = len(df_paired)
all_union = pd.Series(False, index=df_paired.index)
for c in dx_cols:
    try:
        arr = pd.to_numeric(df_paired[c], errors='coerce').fillna(0) > 0
    except Exception:
        arr = df_paired[c].astype(str).str.strip().isin(['1','True','true','Y','y'])
    all_union = all_union | arr

print('union positives:', int(all_union.sum()), 'of', N)

# print top dx counts
counts = {c: int(pd.to_numeric(df_paired[c], errors='coerce').fillna(0).gt(0).sum()) for c in dx_cols}
for k,v in sorted(counts.items(), key=lambda x:-x[1])[:20]:
    print(k, v)
