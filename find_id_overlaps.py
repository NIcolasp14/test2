import pandas as pd

dg = pd.read_csv('diagnosis.csv', dtype=str, usecols=lambda c: True)
ac = pd.read_csv('full_acxiom.csv', dtype=str, usecols=lambda c: True)

# candidate id-like cols
ac_cols = [c for c in ac.columns if any(tok in c.lower() for tok in ['id','member','empi','sys','sk','clm','patient','lumeris','cerner'])]
# include zipcode to detect accidental matches
ac_cols += [c for c in ac.columns if 'zip' in c.lower()]
ac_cols = list(dict.fromkeys(ac_cols))

dg_cols = [c for c in dg.columns if any(tok in c.lower() for tok in ['id','member','empi','sys','sk','clm','patient','lumeris','cerner'])]

dg_ids = {}
for dc in dg_cols:
    vals = set(dg[dc].astype(str).str.strip().dropna().unique())
    dg_ids[dc] = vals

print('Checking overlaps (counts):')
for ac_col in ac_cols:
    ac_vals = set(ac[ac_col].astype(str).str.strip().dropna().unique())
    for dg_col, dg_vals in dg_ids.items():
        overlap = len(ac_vals.intersection(dg_vals))
        if overlap>0:
            print(f'Overlap {overlap}: Acxiom.{ac_col} <-> Diagnosis.{dg_col}')

print('\nSummary: no overlapping ID columns found above indicates a mapping issue.')
