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

dem = safe_read('demographics.csv')
if dem is not None:
    print('\nColumns:', dem.columns.tolist()[:50])
    for col in ['sys_mbr_sk','member_id','empi','member_sk','acxiom_id']:
        print(f"{col} in demographics?", col in dem.columns)
    # show sample mapping if possible
    for a,b in [('sys_mbr_sk','empi'), ('sys_mbr_sk','member_id'), ('sys_mbr_sk','member_sk')]:
        if a in dem.columns and b in dem.columns:
            s = dem[[a,b]].dropna().head(10)
            print(f"\nSample mapping {a} -> {b} (first 10):")
            print(s.to_string(index=False))
print('\nDone')
