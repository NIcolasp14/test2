import pandas as pd
pd.set_option('display.max_columns', None)

def safe_read(path):
    try:
        df = pd.read_csv(path, low_memory=False)
        print(f"Loaded {path}: {df.shape}")
        return df
    except Exception as e:
        print(f"Could not load {path}: {e}")
        return None

fm = safe_read('full_acxiom_with_ed_label.csv')
dx = safe_read('diagnosis_with_acxiom3.csv')

for name, df in [('merged', fm), ('diag', dx)]:
    if df is None:
        continue
    print(f"\nColumns present in {name}:\n", df.columns.tolist()[:50])
    for col in ['sys_mbr_sk','member_id','empi','acxiom_id','clm_sys_mbr_sk','member_sk']:
        print(f" {col} in {name}?", col in df.columns)

if fm is not None and dx is not None:
    for col in ['sys_mbr_sk','member_id','empi','acxiom_id']:
        if col in fm.columns and col in dx.columns:
            mset = set(fm[col].astype(str).str.strip().unique())
            dset = set(dx[col].astype(str).str.strip().unique())
            print(f"Common ids for {col}:", len(mset & dset))

print('\nDone')
