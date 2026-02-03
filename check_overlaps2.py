import pandas as pd
fm = pd.read_csv('full_acxiom_with_ed_label.csv', low_memory=False)
dem = pd.read_csv('demographics.csv', low_memory=False)
dx = pd.read_csv('diagnosis_with_acxiom3.csv', low_memory=False)

for a,b in [('member_id','client_id'), ('member_id','empi'), ('member_id','sys_mbr_sk')]:
    print('\nOverlap between full_acxiom.%s and demographics.%s:'% (a,b))
    if a in fm.columns and b in dem.columns:
        s1=set(fm[a].astype(str).str.strip().unique())
        s2=set(dem[b].astype(str).str.strip().unique())
        print(len(s1 & s2))
    else:
        print('column(s) missing')

print('\nOverlap between diagnosis.empi and demographics.empi:')
if 'empi' in dx.columns and 'empi' in dem.columns:
    s1=set(dx['empi'].astype(str).str.strip().unique())
    s2=set(dem['empi'].astype(str).str.strip().unique())
    print(len(s1 & s2))
else:
    print('empi missing')

print('\nOverlap between diagnosis.sys_mbr_sk and demographics.sys_mbr_sk:')
if 'sys_mbr_sk' in dx.columns and 'sys_mbr_sk' in dem.columns:
    s1=set(dx['sys_mbr_sk'].astype(str).str.strip().unique())
    s2=set(dem['sys_mbr_sk'].astype(str).str.strip().unique())
    print(len(s1 & s2))
else:
    print('sys_mbr_sk missing')

print('\nDone')
