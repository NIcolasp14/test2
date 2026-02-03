import pandas as pd

sdoh_all = pd.read_csv(r"C:\Users\npigadas\OneDrive - Lumeris Solutions Company, LLC\Desktop\full_sdoh\acxiom_full.csv")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from umap import UMAP
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
 
 
# -----------------------------

# Helpers

# -----------------------------

def pick_col(df, candidates):

    for c in candidates:

        if c in df.columns:

            return c

    return None
 
def plot_umap_categorical(df_plot, color_col, title=None, top_n=15, point_size=6, alpha=0.6):

    d = df_plot.copy()

    s = d[color_col].astype("string").fillna("Missing")

    vc = s.value_counts(dropna=False)

    if vc.shape[0] > top_n:

        top = set(vc.head(top_n).index)

        s = s.where(s.isin(top), other="Other")
 
    cats = pd.Categorical(s)

    d["_cat_codes"] = cats.codes
 
    plt.figure()

    plt.scatter(d["umap_x"], d["umap_y"], c=d["_cat_codes"], s=point_size, alpha=alpha)

    plt.xlabel("UMAP-1"); plt.ylabel("UMAP-2")

    plt.title(title or f"UMAP colored by {color_col}")

    plt.tight_layout()

    plt.show()
 
def plot_umap_continuous(df_plot, color_col, title=None, point_size=6, alpha=0.6):

    plt.figure()

    sc = plt.scatter(df_plot["umap_x"], df_plot["umap_y"], c=df_plot[color_col], s=point_size, alpha=alpha)

    plt.xlabel("UMAP-1"); plt.ylabel("UMAP-2")

    plt.title(title or f"UMAP colored by {color_col}")

    plt.colorbar(sc, label=color_col)

    plt.tight_layout()

    plt.show()
 
 
# -----------------------------

# Config

# -----------------------------

ROW_NAN_FRAC_MAX = 0.40

COL_NAN_FRAC_MAX = 0.95

SENTINELS = {9, 99, 999, 9999}  # comment out if these are meaningful categories
 
UMAP_N_NEIGHBORS = 30

UMAP_MIN_DIST = 0.10

UMAP_METRIC = "cosine"

RANDOM_STATE = 42
 
# Choose imputer: "iterative" (MICE-like) or "knn"

IMPUTER_MODE = "iterative"  # <- change to "knn" if iterative is too slow
 
 
# -----------------------------

# 1) Extract SDoH features = all columns AFTER 'flag' (excluding 'flag')

# -----------------------------

if "flag" not in sdoh_all.columns:

    raise ValueError("Column 'flag' not found in sdoh_all.")
 
flag_idx = sdoh_all.columns.get_loc("flag")

sdoh_cols = list(sdoh_all.columns[flag_idx + 1:])

if len(sdoh_cols) == 0:

    raise ValueError("No columns found after 'flag'.")
 
X = sdoh_all[sdoh_cols].copy()
 
# numeric coercion

for c in X.columns:

    X[c] = pd.to_numeric(X[c], errors="coerce")
 
# sentinel-to-missing (optional)

X = X.mask(X.isin(SENTINELS))
 
# -----------------------------

# 2) QC: drop columns + rows with lots of NaNs

# -----------------------------

col_nan_frac = X.isna().mean()

keep_cols = col_nan_frac[col_nan_frac <= COL_NAN_FRAC_MAX].index

X = X[keep_cols]
 
row_nan_frac = X.isna().mean(axis=1)

keep_rows = row_nan_frac <= ROW_NAN_FRAC_MAX
 
print(f"QC: kept {keep_rows.sum():,} / {len(keep_rows):,} rows "

      f"(dropped {(~keep_rows).sum():,} rows with >{ROW_NAN_FRAC_MAX:.0%} NaNs)")

print(f"QC: kept {X.shape[1]:,} / {len(sdoh_cols):,} SDoH cols "

      f"(dropped {(len(sdoh_cols)-X.shape[1]):,} cols with >{COL_NAN_FRAC_MAX:.0%} NaNs)")
 
X = X.loc[keep_rows].copy()

df_qc = sdoh_all.loc[keep_rows].copy()
 
if X.shape[0] < 10:

    raise ValueError(f"Too few rows left after QC ({X.shape[0]}). Loosen ROW_NAN_FRAC_MAX.")
 
 
# -----------------------------

# 3) Sophisticated imputation + scaling

# -----------------------------

X_np = X.to_numpy(dtype=float)
 
if IMPUTER_MODE == "iterative":

    # MICE-style: model each feature from others; BayesianRidge is robust + fast-ish.

    # n_nearest_features keeps it feasible in very wide data (uses a subset of predictors per feature).

    imputer = IterativeImputer(

        estimator=BayesianRidge(),

        max_iter=10,

        tol=1e-3,

        n_nearest_features=min(50, X.shape[1]),  # key knob for speed on wide data

        imputation_order="random",

        skip_complete=True,

        random_state=RANDOM_STATE

    )

    X_imp = imputer.fit_transform(X_np)
 
elif IMPUTER_MODE == "knn":

    # KNNImputer: often much faster than full iterative on wide matrices.

    imputer = KNNImputer(n_neighbors=10, weights="distance")

    X_imp = imputer.fit_transform(X_np)
 
else:

    raise ValueError("IMPUTER_MODE must be 'iterative' or 'knn'.")
 
# Drop zero-variance columns after imputation (helps scaling + UMAP)

var = X_imp.var(axis=0)

nonzero = var > 0

X_imp = X_imp[:, nonzero]

kept_features = np.array(X.columns)[nonzero]

print(f"Features used after zero-variance drop: {X_imp.shape[1]:,}")
 
scaler = StandardScaler(with_mean=True, with_std=True)

X_scaled = scaler.fit_transform(X_imp)
 
# -----------------------------

# 4) UMAP per row

# -----------------------------

umap = UMAP(

    n_components=2,

    n_neighbors=UMAP_N_NEIGHBORS,

    min_dist=UMAP_MIN_DIST,

    metric=UMAP_METRIC,

    random_state=RANDOM_STATE,

)

emb = umap.fit_transform(X_scaled)
 
df_qc["umap_x"] = emb[:, 0]

df_qc["umap_y"] = emb[:, 1]
 
# -----------------------------

# 5) Geo columns + extra colorings

# -----------------------------

zip_col = pick_col(df_qc, ["memberzipcode", "memberzip", "zip", "zipcode", "member_zipcode"])

city_col = pick_col(df_qc, ["city", "membercity", "member_city"])

state_col = pick_col(df_qc, ["state", "memberstate", "member_state"])
 
if zip_col is not None:

    df_qc["zip_str"] = df_qc[zip_col].astype("string").str.extract(r"(\d{5})", expand=False)

    df_qc["zip3"] = df_qc["zip_str"].str[:3]

if city_col is not None:

    df_qc["city_str"] = df_qc[city_col].astype("string")

if state_col is not None:

    df_qc["state_str"] = df_qc[state_col].astype("string")
 
# additional useful colorings

df_qc["sdoh_missing_frac"] = row_nan_frac.loc[keep_rows].values

df_qc["sdoh_nonzero_count"] = (X.fillna(0).to_numpy() != 0).sum(axis=1)
 
# quick clustering (another “whatever else we can”)

k = 8

km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init="auto")

df_qc["cluster_kmeans8"] = km.fit_predict(X_scaled)
 
# -----------------------------

# 6) Plots

# -----------------------------

plt.figure()

plt.scatter(df_qc["umap_x"], df_qc["umap_y"], s=6, alpha=0.6)

plt.title("UMAP of SDoH features (after sophisticated imputation)")

plt.xlabel("UMAP-1"); plt.ylabel("UMAP-2")

plt.tight_layout()

plt.show()
 
if "zip_str" in df_qc.columns:

    plot_umap_categorical(df_qc, "zip_str", title="UMAP colored by ZIP (top + Other)", top_n=15)

if "zip3" in df_qc.columns:

    plot_umap_categorical(df_qc, "zip3", title="UMAP colored by ZIP3", top_n=25)

if "city_str" in df_qc.columns:

    plot_umap_categorical(df_qc, "city_str", title="UMAP colored by City (top + Other)", top_n=15)

if "state_str" in df_qc.columns:

    plot_umap_categorical(df_qc, "state_str", title="UMAP colored by State", top_n=60)
 
plot_umap_continuous(df_qc, "sdoh_missing_frac", title="UMAP colored by SDoH missingness (row-wise)")

plot_umap_continuous(df_qc, "sdoh_nonzero_count", title="UMAP colored by SDoH non-zero count (proxy burden)")

plot_umap_categorical(df_qc, "cluster_kmeans8", title="UMAP colored by KMeans clusters (k=8)", top_n=20)
 
df_qc.head()

 # pick a cluster id to zoom
cluster_id = 3  # change this
sub = df_qc[df_qc["cluster_kmeans8"] == cluster_id].copy()
 
plt.figure()
plt.scatter(df_qc["umap_x"], df_qc["umap_y"], s=4, alpha=0.15)
plt.scatter(sub["umap_x"], sub["umap_y"], s=10, alpha=0.8)
plt.title(f"Zoom highlight: cluster_kmeans8 = {cluster_id}")
plt.xlabel("UMAP-1"); plt.ylabel("UMAP-2")
plt.tight_layout()
plt.show()
 
# true zoom (set axis limits around that cluster)
pad = 0.5
xmin, xmax = sub["umap_x"].min()-pad, sub["umap_x"].max()+pad
ymin, ymax = sub["umap_y"].min()-pad, sub["umap_y"].max()+pad
 
plt.figure()
plt.scatter(sub["umap_x"], sub["umap_y"], s=10, alpha=0.8)
plt.xlim(xmin, xmax); plt.ylim(ymin, ymax)
plt.title(f"Zoomed view: cluster {cluster_id}")
plt.xlabel("UMAP-1"); plt.ylabel("UMAP-2")
plt.tight_layout()
plt.show()

# rebuild raw SDoH matrix from df_qc
flag_idx = df_qc.columns.get_loc("flag")
sdoh_cols = list(df_qc.columns[flag_idx + 1:])
X_raw = df_qc[sdoh_cols].apply(pd.to_numeric, errors="coerce")
X_raw = X_raw.mask(X_raw.isin({9,99,999,9999}))
 
# compare subset vs rest (mean difference; you can swap mean->median)
sub_idx = sub.index
rest_idx = df_qc.index.difference(sub_idx)
 
sub_mean = X_raw.loc[sub_idx].mean(numeric_only=True)
rest_mean = X_raw.loc[rest_idx].mean(numeric_only=True)
 
diff = (sub_mean - rest_mean).sort_values(ascending=False)
print("Top features enriched in subset:")
display(diff.head(20))
 
print("\nTop features depleted in subset:")
display(diff.tail(20))

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
 
# define proxy label: top 20% burden = 1
y = (df_qc["sdoh_nonzero_count"] >= df_qc["sdoh_nonzero_count"].quantile(0.8)).astype(int).to_numpy()
 
clf = LogisticRegression(max_iter=2000, n_jobs=-1, class_weight="balanced")
 
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
aucs, prs, f1s = [], [], []
 
for tr, te in skf.split(X_scaled, y):
    clf.fit(X_scaled[tr], y[tr])
    p = clf.predict_proba(X_scaled[te])[:, 1]
    yhat = (p >= 0.5).astype(int)
 
    aucs.append(roc_auc_score(y[te], p))
    prs.append(average_precision_score(y[te], p))
    f1s.append(f1_score(y[te], yhat))
 
print("Proxy task (high burden):")
print(f"ROC-AUC: {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")
print(f"PR-AUC : {np.mean(prs):.3f} ± {np.std(prs):.3f}")
print(f"F1     : {np.mean(f1s):.3f} ± {np.std(f1s):.3f}")

# ============================

# Supervised classification ONLY:

# Predict STATE, CITY, ZIP5, ZIP3 from SDoH (cols after 'flag' excluding 'flag')

# Assumes your dataframe is named: sdoh_all

# ============================
 
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
 
 
# -----------------------------

# Helpers

# -----------------------------

def pick_col(df, candidates):

    for c in candidates:

        if c in df.columns:

            return c

    return None
 
def build_sdoh_X(sdoh_all, row_nan_frac_max=0.40, col_nan_frac_max=0.95, sentinels=(9, 99, 999, 9999)):

    if "flag" not in sdoh_all.columns:

        raise ValueError("Column 'flag' not found in sdoh_all.")
 
    flag_idx = sdoh_all.columns.get_loc("flag")

    sdoh_cols = list(sdoh_all.columns[flag_idx + 1:])

    if not sdoh_cols:

        raise ValueError("No columns found after 'flag'.")
 
    X = sdoh_all[sdoh_cols].copy()
 
    # numeric coercion

    for c in X.columns:

        X[c] = pd.to_numeric(X[c], errors="coerce")
 
    # sentinel-to-missing (optional)

    X = X.mask(X.isin(set(sentinels)))
 
    # drop super-missing columns

    col_nan_frac = X.isna().mean()

    keep_cols = col_nan_frac[col_nan_frac <= col_nan_frac_max].index

    X = X[keep_cols]
 
    # drop rows with too much missingness

    row_nan_frac = X.isna().mean(axis=1)

    keep_rows = row_nan_frac <= row_nan_frac_max
 
    df_qc = sdoh_all.loc[keep_rows].copy()

    X_qc = X.loc[keep_rows].copy()
 
    print(f"QC rows kept: {keep_rows.sum():,}/{len(keep_rows):,}  (row_nan_frac_max={row_nan_frac_max})")

    print(f"QC cols kept: {X_qc.shape[1]:,}/{len(sdoh_cols):,}  (col_nan_frac_max={col_nan_frac_max})")
 
    return df_qc, X_qc
 
def collapse_rare_classes(y, min_count=50, top_n=None):

    """

    Keep classes with count >= min_count (or keep only top_n classes),

    map the rest to 'Other'.

    """

    y = y.astype("string").fillna("Missing")

    vc = y.value_counts(dropna=False)
 
    if top_n is not None:

        keep = set(vc.head(top_n).index)

    else:

        keep = set(vc[vc >= min_count].index)
 
    return y.where(y.isin(keep), other="Other")
 
def choose_n_splits(y_codes, desired=5):

    counts = pd.Series(y_codes).value_counts()

    min_count = counts.min()

    return max(2, min(desired, int(min_count)))
 
def run_multiclass_task(X_df, y_series, task_name, min_count=50, top_n=None, topk=5, desired_splits=5):

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

        topks.append(top_k_accuracy_score(y_codes[te], probs, k=k_use, labels=np.arange(n_classes)))
 
    print(f"\n=== {task_name} ===")

    print(f"Classes (after collapse): {n_classes:,}")

    print(f"Splits: {n_splits}")

    print(f"Accuracy:          {np.mean(accs):.3f} ± {np.std(accs):.3f}")

    print(f"Balanced accuracy: {np.mean(baccs):.3f} ± {np.std(baccs):.3f}")

    print(f"Macro-F1:          {np.mean(f1s):.3f} ± {np.std(f1s):.3f}")

    print(f"Top-{k_use} acc:         {np.mean(topks):.3f} ± {np.std(topks):.3f}")
 
    # show label counts

    print("\nTop label counts:")

    display(pd.Series(y).value_counts().head(15))
 
    return {

        "task": task_name,

        "classes": n_classes,

        "splits": n_splits,

        "acc": accs,

        "bacc": baccs,

        "macro_f1": f1s,

        f"top{k_use}": topks,

    }
 
 
# -----------------------------

# 1) Build X (SDoH features) + QC

# -----------------------------

df_qc, X_qc = build_sdoh_X(sdoh_all, row_nan_frac_max=0.40, col_nan_frac_max=0.95)
 
# -----------------------------

# 2) Build targets (ZIP/CITY/STATE)

# -----------------------------

zip_col   = pick_col(df_qc, ["memberzipcode", "memberzip", "zipcode", "zip", "member_zipcode"])

city_col  = pick_col(df_qc, ["city", "membercity", "member_city"])

state_col = pick_col(df_qc, ["state", "memberstate", "member_state"])
 
if zip_col is None or city_col is None or state_col is None:

    print("Available columns (first 100):", list(df_qc.columns[:100]))

    raise ValueError("Couldn't find zip/city/state columns. Update candidates in pick_col().")
 
df_qc["zip_str"]   = df_qc[zip_col].astype("string").str.extract(r"(\d{5})", expand=False)

df_qc["zip3"]      = df_qc["zip_str"].str[:3]

df_qc["city_str"]  = df_qc[city_col].astype("string").str.strip()

df_qc["state_str"] = df_qc[state_col].astype("string").str.strip()
 
# -----------------------------

# 3) Run supervised classifications

# -----------------------------

# STATE: usually few classes; keep all

res_state = run_multiclass_task(

    X_qc, df_qc["state_str"],

    task_name="Predict STATE from SDoH",

    min_count=1,

    topk=3

)
 
# CITY: many classes; collapse rare cities

res_city = run_multiclass_task(

    X_qc, df_qc["city_str"],

    task_name="Predict CITY from SDoH",

    min_count=100,   # tune: 50/100/200

    topk=5

)
 
# ZIP5: very high classes; usually collapse hard

res_zip5 = run_multiclass_task(

    X_qc, df_qc["zip_str"],

    task_name="Predict ZIP5 from SDoH",

    min_count=200,   # tune: 100/200/500

    topk=10

)
 
# ZIP3: recommended compromise

res_zip3 = run_multiclass_task(

    X_qc, df_qc["zip3"],

    task_name="Predict ZIP3 from SDoH",

    min_count=50,    # tune

    topk=5

)

import matplotlib.pyplot as plt
 
# Choose the column and number of top categories

color_col = "state_str"

top_n = 20
 
# Compute top-N categories

vc = df_qc[color_col].value_counts()

top_cats = set(vc.head(top_n).index)
 
# Build a color array: a colormap for top cats, gray for others

cats = df_qc[color_col].where(df_qc[color_col].isin(top_cats), other="Other")

cat_list = list(cats.cat.categories) if hasattr(cats, "cat") else sorted(cats.unique())

cat_to_code = {c:i for i,c in enumerate(cat_list)}
 
# Assign codes, forcing “Other” to the last index

codes = cats.map(lambda x: cat_to_code[x])

n_colors = len(cat_list)

cmap = plt.get_cmap("tab20", n_colors)  # up to 20 distinct colors
 
# Plot

plt.figure(figsize=(6,5))

# Plot “Other” first in gray

mask_other = cats == "Other"

plt.scatter(df_qc.loc[mask_other, "umap_x"],

            df_qc.loc[mask_other, "umap_y"],

            c="lightgray", s=6, alpha=0.6, label="Other")
 
# Then plot each top category

for i, cat in enumerate(cat_list):

    if cat == "Other": 

        continue

    mask = cats == cat

    plt.scatter(df_qc.loc[mask, "umap_x"],

                df_qc.loc[mask, "umap_y"],

                c=[cmap(i)], s=6, alpha=0.8, label=cat)
 
plt.xlabel("UMAP-1")

plt.ylabel("UMAP-2")

plt.title(f"UMAP colored by top {top_n} {color_col} (+ Other in gray)")

plt.legend(bbox_to_anchor=(1.02,1), loc="upper left", markerscale=2, fontsize="small")

plt.tight_layout()

plt.show()

# option A – if X_scaled is still in memory (fastest, seconds)
emb = umap.fit_transform(X_scaled)           # reuse the fitted `umap` or create a new one
df_qc["umap_x"], df_qc["umap_y"] = emb[:,0], emb[:,1]
 
# option B – if you saved the UMAP-augmented frame earlier
# df_qc = pd.read_csv("with_umap.csv")       # adjust path / filename
print("umap_x in df_qc?", 'umap_x' in df_qc.columns)
print("umap_y in df_qc?", 'umap_y' in df_qc.columns)

def plot_top20(df, col, title):

    vc = df[col].value_counts()

    top = set(vc.head(20).index)
 
    cats = df[col].where(df[col].isin(top), other="Other").astype("string").fillna("Missing")

    codes = pd.Categorical(cats).codes

    cmap  = plt.get_cmap("tab20", len(pd.unique(cats)))
 
    plt.figure(figsize=(6,5))

    # plot Other first in gray

    plt.scatter(df.loc[cats=="Other", "umap_x"],

                df.loc[cats=="Other", "umap_y"],

                c="lightgray", s=6, alpha=0.4, label="Other")
 
    for i, cat in enumerate(pd.unique(cats)):

        if cat == "Other": continue

        plt.scatter(df.loc[cats==cat, "umap_x"],

                    df.loc[cats==cat, "umap_y"],

                    c=[cmap(i)], s=6, alpha=0.8, label=cat)
 
    plt.xlabel("UMAP-1"); plt.ylabel("UMAP-2"); plt.title(title)

    plt.legend(bbox_to_anchor=(1.02,1), loc="upper left", markerscale=2, fontsize="small")

    plt.tight_layout(); plt.show()
 
# call it:

plot_top20(df_qc, "state_str", "UMAP – top-20 states (others gray)")

plot_top20(df_qc, "zip_str",   "UMAP – top-20 ZIP5")

plot_top20(df_qc, "zip3",      "UMAP – top-20 ZIP3")

plot_top20(df_qc, "city_str",  "UMAP – top-20 cities")


# ===========================================================

# Extra visualisations on *existing* UMAP coordinates

# – does NOT overwrite df_qc

# – assumes df_qc already has: umap_x, umap_y, state_str, zip_str, zip3, city_str

# ===========================================================
 
import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

import pandas as pd

import plotly.express as px   # only used if you want interactive later
 
# ------------------------------------------------------------------

# Helper: pick top-N categories but NEVER touch the original df_qc

# ------------------------------------------------------------------

def topN_series(series, n=20):

    vc   = series.value_counts()

    top  = set(vc.head(n).index)

    new  = series.where(series.isin(top), other="Other")

    return new.astype("string").fillna("Missing")
 
# ===========================================================

# 1)  Smaller dots + alpha-stack  (one figure, categorical)

# ===========================================================

def umap_small_alpha(df, col, top_n=20, point_size=4):

    cats = topN_series(df[col], top_n)

    uniq = pd.unique(cats)

    cmap = plt.get_cmap("tab20", len(uniq))
 
    plt.figure(figsize=(6,5))

    # plot gray background (“Other”) first

    mask_other = cats == "Other"

    plt.scatter(df.loc[mask_other,"umap_x"],

                df.loc[mask_other,"umap_y"],

                c="lightgray", s=point_size, alpha=0.15, label="Other")
 
    # overlay each top category

    for i, cat in enumerate(uniq):

        if cat == "Other": continue

        plt.scatter(df.loc[cats==cat,"umap_x"],

                    df.loc[cats==cat,"umap_y"],

                    c=[cmap(i)], s=point_size, alpha=0.85, label=cat)
 
    plt.title(f"UMAP – top {top_n} {col} (others gray)")

    plt.xlabel("UMAP-1"); plt.ylabel("UMAP-2")

    plt.legend(bbox_to_anchor=(1.02,1), loc="upper left", markerscale=2, fontsize="small")

    plt.tight_layout(); plt.show()
 
# example:

umap_small_alpha(df_qc, "state_str", top_n=20)
 
# ===========================================================

# 2)  FacetGrid (small-multiples) for the top 6 categories

#     – avoids hue clutter entirely

# ===========================================================

def umap_facet(df, col, n_facets=6, point_size=6):

    top = df[col].value_counts().head(n_facets).index

    sub = df[df[col].isin(top)].copy()
 
    g = sns.FacetGrid(sub, col=col, col_wrap=3, height=3,

                      sharex=False, sharey=False)

    g.map_dataframe(sns.scatterplot, "umap_x", "umap_y",

                    s=point_size, alpha=.8, color="steelblue")

    for ax in g.axes.flatten():

        ax.set_xlabel(""); ax.set_ylabel("")

    g.fig.subplots_adjust(top=0.9)

    g.fig.suptitle(f"UMAP – each panel is one of top {n_facets} {col}")

    plt.show()
 
# example:

umap_facet(df_qc, "state_str", n_facets=6)
 
# ===========================================================

# 3)  Hexbin / density for ONE category of interest

# ===========================================================

def umap_hexbin_single(df, col, value, gridsize=50):

    mask = df[col] == value

    if mask.sum() == 0:

        print(f"No rows where {col} == {value!r}")

        return

    plt.figure(figsize=(6,5))

    hb = plt.hexbin(df.loc[mask,"umap_x"],

                    df.loc[mask,"umap_y"],

                    gridsize=gridsize, cmap="viridis", mincnt=3)

    plt.colorbar(hb, label="count")

    plt.title(f"{value} density in UMAP space")

    plt.xlabel("UMAP-1"); plt.ylabel("UMAP-2")

    plt.tight_layout(); plt.show()
 
# example: show Missouri (“MO”) concentration

umap_hexbin_single(df_qc, "state_str", "MO", gridsize=60)


# ===========================================================

#  UMAP visualisations – NO recompute, NO mutation of df_qc

# ===========================================================

import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd
 
# ---------- sanity: make sure the basic columns are there ----------

required = {"umap_x", "umap_y", "state_str", "city_str", "zip_str", "zip3"}

missing  = required.difference(df_qc.columns)

if missing:

    raise ValueError(f"df_qc is missing columns: {missing}")
 
# add a convenient alias so the code can use 'zip5'

if "zip5" not in df_qc.columns:

    df_qc["zip5"] = df_qc["zip_str"]
 
# ------------------------------------------------------------------

# Helpers

# ------------------------------------------------------------------

def _topN(series: pd.Series, n=20):

    """Return series with top-n categories retained, all others → 'Other'."""

    vc  = series.value_counts()

    top = set(vc.head(n).index)

    return (

        series.where(series.isin(top), other="Other")

        .astype("string")

        .fillna("Missing")

    )
 
# 1) scatter with small dots + transparency

def umap_small_alpha(df, label_col, top_n=20, point_size=4):

    cats = _topN(df[label_col], top_n)

    uniq = pd.unique(cats)

    cmap = plt.get_cmap("tab20", len(uniq))
 
    plt.figure(figsize=(6,5))

    # gray background

    m_other = cats == "Other"

    plt.scatter(df.loc[m_other,"umap_x"],

                df.loc[m_other,"umap_y"],

                c="lightgray", s=point_size, alpha=0.15, label="Other")
 
    # overlay each top category

    for i, cat in enumerate(uniq):

        if cat == "Other": continue

        m = cats == cat

        plt.scatter(df.loc[m,"umap_x"],

                    df.loc[m,"umap_y"],

                    c=[cmap(i)], s=point_size, alpha=0.85, label=cat)
 
    plt.title(f"UMAP – top {top_n} {label_col} (others gray)")

    plt.xlabel("UMAP-1"); plt.ylabel("UMAP-2")

    plt.legend(bbox_to_anchor=(1.02,1), loc="upper left",

               markerscale=2, fontsize="small")

    plt.tight_layout(); plt.show()
 
# 2) facet grid (one mini-plot per category)

def umap_facet(df, label_col, n_facets=6, point_size=6):

    top = df[label_col].value_counts().head(n_facets).index

    sub = df[df[label_col].isin(top)].copy()
 
    g = sns.FacetGrid(sub, col=label_col, col_wrap=3, height=3,

                      sharex=False, sharey=False)

    g.map_dataframe(sns.scatterplot, "umap_x", "umap_y",

                    s=point_size, alpha=.8, color="steelblue")

    for ax in g.axes.flatten():

        ax.set_xlabel(""); ax.set_ylabel("")

    g.fig.subplots_adjust(top=0.9)

    g.fig.suptitle(f"UMAP – each panel = one of top {n_facets} {label_col}")

    plt.show()
 
# 3) hex-density for ONE category

def umap_hexbin_single(df, label_col, label_value, gridsize=60):

    m = df[label_col] == label_value

    if m.sum() == 0:

        print(f"No rows where {label_col} == {label_value!r}")

        return

    plt.figure(figsize=(6,5))

    hb = plt.hexbin(df.loc[m,"umap_x"], df.loc[m,"umap_y"],

                    gridsize=gridsize, cmap="viridis", mincnt=3)

    plt.colorbar(hb, label="count")

    plt.title(f"{label_value} density in UMAP space")

    plt.xlabel("UMAP-1"); plt.ylabel("UMAP-2")

    plt.tight_layout(); plt.show()
 
# ===========================================================

#  Generate plots

# ===========================================================
 
# --- 1. small-dots + alpha (top-20 vs gray) -----------------

umap_small_alpha(df_qc, "state_str", top_n=20)

umap_small_alpha(df_qc, "zip5",      top_n=20)

umap_small_alpha(df_qc, "zip3",      top_n=20)

umap_small_alpha(df_qc, "city_str",  top_n=20)
 
# --- 2. facet mini-plots (top 6 categories) -----------------

umap_facet(df_qc, "state_str", n_facets=6)

umap_facet(df_qc, "zip3",      n_facets=6)

umap_facet(df_qc, "city_str",  n_facets=6)
 
# --- 3. hex-density for specific examples -------------------

umap_hexbin_single(df_qc, "state_str", "MO")     # Missouri

umap_hexbin_single(df_qc, "zip3",      "631")    # ZIP3 = 631

umap_hexbin_single(df_qc, "city_str",  "Saint Louis")


"""
merge_acxiom_diagnosis.py

Robustly merge `diagnosis.csv` with `acxiom_full.csv`, using `demographics.csv`
as a bridge when needed. The script attempts to automatically detect matching
ID columns and will perform a memory-friendly chunked merge if `acxiom_full.csv`
is large.

Usage (defaults point to the Files V2 used in the notebook):
python merge_acxiom_diagnosis.py \
    --acxiom acxiom_full.csv \
    --diagnosis "C:\\Users\\npigadas\\OneDrive - Lumeris Solutions Company, LLC\\Desktop\\Files V2\\diagnosis.csv" \
    --demographics "C:\\Users\\npigadas\\OneDrive - Lumeris Solutions Company, LLC\\Desktop\\Files V2\\demographics.csv" \
    --output diagnosis_with_acxiom.csv

The script saves `--output` with acxiom columns appended (first-match behavior
for duplicated keys in acxiom).
"""
import os
import sys
from pathlib import Path

import pandas as pd


CANDIDATE_KEYS = [
    "empi",
    "lumeris_empi",
    "empi_id",
    "member_id",
    "sys_mbr_sk",
    "clm_sys_mbr_sk",
    "memberid",
    "acxiom_id",
    "acx_id",
    "subscriber_id",
]


def find_merge_keys(diag_cols, acx_cols):
    # 1) exact match
    for c in acx_cols:
        if c in diag_cols:
            return c, c
    # 2) candidate-based: diag contains candidate and acx has case-insensitive match
    acx_lower = {c.lower(): c for c in acx_cols}
    for cand in CANDIDATE_KEYS:
        if cand in diag_cols and cand.lower() in acx_lower:
            return cand, acx_lower[cand.lower()]
    # 3) case-insensitive exact between the two
    diag_lower = {c.lower(): c for c in diag_cols}
    for c_lower, c_orig in diag_lower.items():
        if c_lower in acx_lower:
            return c_orig, acx_lower[c_lower]
    return None, None


def chunked_bridge_merge(diagnosis, acx_path, diag_key, acx_key, chunksize=200_000):
    # diagnosis: dataframe (left), diag_key column present
    # We'll perform an outer-leaning approach: for each chunk of acxiom,
    # map values for matching keys into the diagnosis dataframe. If acx has
    # multiple rows per key, we keep the first occurrence encountered.
    # Collect distinct diag values (strings) to look up in acxiom chunks
    diag_vals = diagnosis[diag_key].dropna().astype(str).unique()

    # Read acxiom header once to know available columns (cheap operation)
    acx_cols = pd.read_csv(acx_path, nrows=0, dtype=str).columns.tolist()
    if acx_key not in acx_cols:
        raise RuntimeError(f"Expected acx key {acx_key} not in acxiom columns")

    # Add any missing acxiom columns to `diagnosis` in one concat operation
    add_cols = [c for c in acx_cols if c != acx_key and c not in diagnosis.columns]
    if add_cols:
        new_cols = pd.DataFrame({c: pd.NA for c in add_cols}, index=diagnosis.index)
        diagnosis = pd.concat([diagnosis, new_cols], axis=1)

    # Iterate file-by-chunk, mapping first-seen acxiom values into diagnosis
    for chunk in pd.read_csv(acx_path, chunksize=chunksize, dtype=str, low_memory=False):
        # keep only rows for keys present in diagnosis
        chunk_subset = chunk[chunk[acx_key].isin(diag_vals)]
        if chunk_subset.empty:
            continue

        # drop duplicate keys keeping first occurrence
        chunk_subset = chunk_subset.drop_duplicates(subset=[acx_key])

        # build mapping dicts for columns and apply with vectorized fillna
        for col in chunk_subset.columns:
            if col == acx_key:
                continue
            mapping = dict(zip(chunk_subset[acx_key].astype(str), chunk_subset[col]))
            mask = diagnosis[diag_key].notna()
            mapped_series = diagnosis.loc[mask, diag_key].astype(str).map(mapping)
            diagnosis.loc[mask, col] = diagnosis.loc[mask, col].fillna(mapped_series)

    # Defragment frame after many assignments
    diagnosis = diagnosis.copy()
    return diagnosis


def simple_merge(acx_path, diagnosis, left_key, right_key):
    acx_full = pd.read_csv(acx_path, dtype=str, low_memory=False)
    merged = diagnosis.merge(acx_full, left_on=left_key, right_on=right_key, how="left")
    return merged


# Notebook-style top-level execution (hardcoded paths)
# Paths (adjust if needed)
ACX_PATH = Path(r"acxiom_full.csv")
DIAG_PATH = Path(r"C:\Users\npigadas\OneDrive - Lumeris Solutions Company, LLC\Desktop\Files V2\diagnosis.csv")
DEM_PATH = Path(r"C:\Users\npigadas\OneDrive - Lumeris Solutions Company, LLC\Desktop\Files V2\demographics.csv")
OUT_PATH = Path(r"diagnosis_with_acxiom.csv")
CHUNKSIZE = 200_000

print("Reading diagnosis (as strings for robust joins)...")
if not DIAG_PATH.exists():
    raise FileNotFoundError(f"Diagnosis file not found: {DIAG_PATH}")
diagnosis = pd.read_csv(DIAG_PATH, dtype=str, low_memory=False)

print("Reading demographics (for bridging)...")
if not DEM_PATH.exists():
    print(f"Warning: demographics file not found at {DEM_PATH}; bridge options will be limited")
    demographics = None
else:
    demographics = pd.read_csv(DEM_PATH, dtype=str, low_memory=False)

# normalize: if clm_sys_mbr_sk exists, rename to sys_mbr_sk for consistency
if 'clm_sys_mbr_sk' in diagnosis.columns and 'sys_mbr_sk' not in diagnosis.columns:
    diagnosis = diagnosis.rename(columns={'clm_sys_mbr_sk': 'sys_mbr_sk'})

try:
    acx_cols = pd.read_csv(ACX_PATH, nrows=0).columns.tolist()
except Exception as e:
    print("Failed to read acxiom columns preview:", e)
    acx_cols = []

print("Acxiom columns (preview):", acx_cols[:20])

diag_cols = set(diagnosis.columns)
diag_key, acx_key = find_merge_keys(diag_cols, acx_cols)

merged = None

if diag_key and acx_key:
    print(f"Found merge keys: diagnosis.{diag_key} <-> acxiom.{acx_key}")
    acx_size = os.path.getsize(ACX_PATH)
    MB = 1024 * 1024
    if acx_size > 200 * MB:
        print("Large acxiom file detected — performing chunked, memory-friendly merge")
        merged = chunked_bridge_merge(diagnosis.copy(), ACX_PATH, diag_key, acx_key, chunksize=CHUNKSIZE)
    else:
        print("Acxiom file small enough — reading fully for simple merge")
        merged = simple_merge(ACX_PATH, diagnosis, diag_key, acx_key)
else:
    print("No direct key found between diagnosis and acxiom. Trying demographics bridge...")
    if demographics is None:
        raise RuntimeError("No demographics provided — cannot bridge automatically.")

    if 'sys_mbr_sk' not in demographics.columns or 'empi' not in demographics.columns:
        raise RuntimeError("Demographics missing 'sys_mbr_sk' or 'empi' columns — cannot bridge.")

    dem_map = demographics[['sys_mbr_sk', 'empi']].drop_duplicates()
    diag_with_empi = diagnosis.merge(dem_map, left_on='sys_mbr_sk', right_on='sys_mbr_sk', how='left')

    empi_col = None
    for c in acx_cols:
        if c.lower() == 'empi' or 'empi' in c.lower() or c.lower() == 'lumeris_empi':
            empi_col = c
            break

    if empi_col:
        print(f"Found acxiom empi-like column: {empi_col} — merging via demographics bridge")
        acx_size = os.path.getsize(ACX_PATH)
        if acx_size > 200 * 1024 * 1024:
            merged = chunked_bridge_merge(diag_with_empi.copy(), ACX_PATH, 'empi', empi_col, chunksize=CHUNKSIZE)
        else:
            merged = simple_merge(ACX_PATH, diag_with_empi, 'empi', empi_col)
    else:
        bridge_col = None
        for c in acx_cols:
            if demographics is not None and c in demographics.columns:
                bridge_col = c
                break
        if bridge_col:
            print(f"Using demographics bridge via column: {bridge_col}")
            if os.path.getsize(ACX_PATH) > 200 * 1024 * 1024:
                augmented = diagnosis.merge(demographics[['sys_mbr_sk', bridge_col]].drop_duplicates(), on='sys_mbr_sk', how='left')
                merged = chunked_bridge_merge(augmented.copy(), ACX_PATH, bridge_col, bridge_col, chunksize=CHUNKSIZE)
            else:
                acx_full = pd.read_csv(ACX_PATH, dtype=str, low_memory=False)
                merged = diagnosis.merge(demographics[['sys_mbr_sk', bridge_col]].drop_duplicates(), on='sys_mbr_sk', how='left').merge(acx_full, left_on=bridge_col, right_on=bridge_col, how='left')
        else:
            print("No suitable bridge column found automatically.")
            print("Sample of acxiom columns (first 5):", acx_cols[:5])
            raise RuntimeError('No merge key found automatically. Inspect acxiom columns and choose a key to merge on.')

# final save
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
merged.to_csv(OUT_PATH, index=False)
print(f"✅ Saved merged dataset to {OUT_PATH} — shape: {merged.shape}")


if __name__ == '__main__':
    # Hardcoded settings (as you'd run from a notebook cell)
    class Args:
        pass

    args = Args()
    args.acxiom = r"acxiom_full.csv"
    args.diagnosis = r"C:\Users\npigadas\OneDrive - Lumeris Solutions Company, LLC\Desktop\Files V2\diagnosis.csv"
    args.demographics = r"C:\Users\npigadas\OneDrive - Lumeris Solutions Company, LLC\Desktop\Files V2\demographics.csv"
    args.output = r"diagnosis_with_acxiom.csv"
    args.chunksize = 200_000

    print('Running merge with hardcoded paths:')
    print(' acxiom ->', args.acxiom)
    print(' diagnosis ->', args.diagnosis)
    print(' demographics ->', args.demographics)
    print(' output ->', args.output)

    main(args)

diagnoses = pd.read_csv(r"C:\Users\npigadas\OneDrive - Lumeris Solutions Company, LLC\Desktop\full_sdoh\diagnosis.csv")


diagnoses

"""
merge_acxiom_diagnosis.py

Robustly merge `diagnosis.csv` with `acxiom_full.csv`, using `demographics.csv`
as a bridge when needed. The script attempts to automatically detect matching
ID columns and will perform a memory-friendly chunked merge if `acxiom_full.csv`
is large.

Usage (defaults point to the Files V2 used in the notebook):
python merge_acxiom_diagnosis.py \
    --acxiom acxiom_full.csv \
    --diagnosis "C:\\Users\\npigadas\\OneDrive - Lumeris Solutions Company, LLC\\Desktop\\Files V2\\diagnosis.csv" \
    --demographics "C:\\Users\\npigadas\\OneDrive - Lumeris Solutions Company, LLC\\Desktop\\Files V2\\demographics.csv" \
    --output diagnosis_with_acxiom.csv

The script saves `--output` with acxiom columns appended (first-match behavior
for duplicated keys in acxiom).
"""
import os
import sys
from pathlib import Path

import pandas as pd


CANDIDATE_KEYS = [
    "empi",
    "lumeris_empi",
    "empi_id",
    "member_id",
    "sys_mbr_sk",
    "clm_sys_mbr_sk",
    "memberid",
    "acxiom_id",
    "acx_id",
    "subscriber_id",
]

# Columns to look for when finding a diagnosis code column
DIAG_CODE_CANDIDATES = [
    "diagnosis_code",
    "dx_code",
    "diag_code",
    "icd_code",
    "icd10",
    "icd9",
    "dx",
    "diagnosis",
    "code",
]

# Patient identifier candidates (in order of preference)
PATIENT_ID_CANDIDATES = [
    "empi",
    "lumeris_empi",
    "empi_id",
    "member_id",
    "sys_mbr_sk",
    "clm_sys_mbr_sk",
    "memberid",
    "subscriber_id",
]


def find_patient_id_col(cols):
    for c in PATIENT_ID_CANDIDATES:
        if c in cols:
            return c
    # fallback: return first column named like an id
    for c in cols:
        if c.lower().endswith("id") or c.lower().endswith("_id"):
            return c
    return None


def find_diag_code_col(cols):
    # Exact matches first
    for c in DIAG_CODE_CANDIDATES:
        if c in cols:
            return c

    # Case-insensitive exact
    lower = {c.lower(): c for c in cols}
    for cand in DIAG_CODE_CANDIDATES:
        if cand.lower() in lower:
            return lower[cand.lower()]

    # Substring match: allow candidates like 'icd9' to match 'icd9_diagnosis_cd'
    cand_tokens = [t.lower() for t in DIAG_CODE_CANDIDATES]
    for col in cols:
        cl = col.lower()
        for tok in cand_tokens:
            if tok and tok in cl:
                return col

    return None


def aggregate_diagnoses_to_patient(diagnosis_df, patient_col, diag_col, top_n=100, min_count=None):
    """Aggregate diagnosis rows to patient-level and one-hot encode top diagnosis codes.

    Returns a DataFrame indexed by patient id with binary columns for each selected
    diagnosis code (1 = patient has that code at least once).
    """
    if patient_col not in diagnosis_df.columns:
        raise ValueError(f"Patient id column {patient_col!r} not found in diagnosis dataframe")
    if diag_col not in diagnosis_df.columns:
        raise ValueError(f"Diagnosis code column {diag_col!r} not found in diagnosis dataframe")

    # drop rows missing patient id or diagnosis code
    d = diagnosis_df[[patient_col, diag_col]].dropna(subset=[patient_col])

    # Use string form for codes
    d[diag_col] = d[diag_col].astype(str)

    # choose codes to keep
    vc = d[diag_col].value_counts()
    if min_count is not None:
        keep = set(vc[vc >= min_count].index)
    else:
        keep = set(vc.head(top_n).index)

    # mark presence via dummies grouped by patient
    d_keep = d[d[diag_col].isin(keep)].copy()
    if d_keep.empty:
        # If nothing in keep, return an empty frame with patient ids
        patients = pd.DataFrame({patient_col: d[patient_col].unique()})
        patients = patients.set_index(patient_col)
        return patients

    dummies = pd.get_dummies(d_keep[diag_col], prefix="dx")
    d_idx = d_keep[[patient_col]].reset_index(drop=True)
    d_onehot = pd.concat([d_idx, dummies], axis=1)
    patient_onehot = d_onehot.groupby(patient_col).max()

    # optionally add a column counting how many rare/other diagnosis codes the patient has
    other_mask = ~d[diag_col].isin(keep)
    if other_mask.any():
        other_counts = d.loc[other_mask].groupby(patient_col).size()
        patient_onehot["dx_other_count"] = other_counts
        patient_onehot["dx_other_count"] = patient_onehot["dx_other_count"].fillna(0).astype(int)

    # ensure index is a column for downstream merges
    patient_onehot = patient_onehot.reset_index()
    return patient_onehot



def find_merge_keys(diag_cols, acx_cols):
    # 1) exact match
    for c in acx_cols:
        if c in diag_cols:
            return c, c
    # 2) candidate-based: diag contains candidate and acx has case-insensitive match
    acx_lower = {c.lower(): c for c in acx_cols}
    for cand in CANDIDATE_KEYS:
        if cand in diag_cols and cand.lower() in acx_lower:
            return cand, acx_lower[cand.lower()]
    # 3) case-insensitive exact between the two
    diag_lower = {c.lower(): c for c in diag_cols}
    for c_lower, c_orig in diag_lower.items():
        if c_lower in acx_lower:
            return c_orig, acx_lower[c_lower]
    return None, None


def chunked_bridge_merge(diagnosis, acx_path, diag_key, acx_key, chunksize=200_000):
    # diagnosis: dataframe (left), diag_key column present
    # We'll perform an outer-leaning approach: for each chunk of acxiom,
    # map values for matching keys into the diagnosis dataframe. If acx has
    # multiple rows per key, we keep the first occurrence encountered.
    # Collect distinct diag values (strings) to look up in acxiom chunks
    # normalize diag values (strip whitespace) and use a set for fast lookup
    diag_vals = set(v.strip() for v in diagnosis[diag_key].dropna().astype(str).unique())
    print(f"Chunked merge: {len(diag_vals):,} unique diag keys to match (preview)")

    # Read acxiom header once to know available columns (cheap operation)
    acx_cols = pd.read_csv(acx_path, nrows=0, dtype=str).columns.tolist()
    if acx_key not in acx_cols:
        raise RuntimeError(f"Expected acx key {acx_key} not in acxiom columns")

    # Add any missing acxiom columns to `diagnosis` in one concat operation
    add_cols = [c for c in acx_cols if c != acx_key and c not in diagnosis.columns]
    if add_cols:
        new_cols = pd.DataFrame({c: pd.NA for c in add_cols}, index=diagnosis.index)
        diagnosis = pd.concat([diagnosis, new_cols], axis=1)

    # Iterate file-by-chunk, mapping first-seen acxiom values into diagnosis
    matched_keys = set()
    for i, chunk in enumerate(pd.read_csv(acx_path, chunksize=chunksize, dtype=str, low_memory=False)):
        # keep only rows for keys present in diagnosis
        # normalize acx_key values and filter by diag_vals (strip strings)
        chunk[acx_key] = chunk[acx_key].astype(str).str.strip()
        chunk_subset = chunk[chunk[acx_key].isin(diag_vals)]
        if chunk_subset.empty:
            continue

        # drop duplicate keys keeping first occurrence
        chunk_subset = chunk_subset.drop_duplicates(subset=[acx_key])

        # build mapping dicts for columns and apply with vectorized fillna
        matched_in_chunk = set(chunk_subset[acx_key].astype(str).unique())
        matched_keys.update(matched_in_chunk)
        for col in chunk_subset.columns:
            if col == acx_key:
                continue
            mapping = dict(zip(chunk_subset[acx_key].astype(str), chunk_subset[col]))
            mask = diagnosis[diag_key].notna()
            mapped_series = diagnosis.loc[mask, diag_key].astype(str).str.strip().map(mapping)
            diagnosis.loc[mask, col] = diagnosis.loc[mask, col].fillna(mapped_series)

        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1} chunks — matched so far: {len(matched_keys):,} unique keys")

    # report match statistics
    print(f"Chunked merge finished — matched {len(matched_keys):,} unique keys ({len(matched_keys)/max(1,len(diag_vals)):.2%} of diag keys)")
    # Defragment frame after many assignments
    diagnosis = diagnosis.copy()
    return diagnosis


def simple_merge(acx_path, diagnosis, left_key, right_key):
    acx_full = pd.read_csv(acx_path, dtype=str, low_memory=False)
    merged = diagnosis.merge(acx_full, left_on=left_key, right_on=right_key, how="left")
    return merged


# Notebook-style top-level execution (hardcoded paths)
# Paths (adjust if needed)
ACX_PATH = Path(r"acxiom_full.csv")
DIAG_PATH = Path(r"C:\Users\npigadas\OneDrive - Lumeris Solutions Company, LLC\Desktop\Files V2\diagnosis.csv")
DEM_PATH = Path(r"C:\Users\npigadas\OneDrive - Lumeris Solutions Company, LLC\Desktop\Files V2\demographics.csv")
OUT_PATH = Path(r"diagnosis_with_acxiom.csv")
CHUNKSIZE = 200_000

print("Reading diagnosis (as strings for robust joins)...")
if not DIAG_PATH.exists():
    raise FileNotFoundError(f"Diagnosis file not found: {DIAG_PATH}")
diagnosis = pd.read_csv(DIAG_PATH, dtype=str, low_memory=False)

print("Reading demographics (for bridging)...")
if not DEM_PATH.exists():
    print(f"Warning: demographics file not found at {DEM_PATH}; bridge options will be limited")
    demographics = None
else:
    demographics = pd.read_csv(DEM_PATH, dtype=str, low_memory=False)

# normalize: if clm_sys_mbr_sk exists, rename to sys_mbr_sk for consistency
if 'clm_sys_mbr_sk' in diagnosis.columns and 'sys_mbr_sk' not in diagnosis.columns:
    diagnosis = diagnosis.rename(columns={'clm_sys_mbr_sk': 'sys_mbr_sk'})

# -----------------------------
# Aggregate diagnosis rows to patient-level (one row per patient)
# -----------------------------
patient_id_col = find_patient_id_col(diagnosis.columns)
if patient_id_col is None:
    raise RuntimeError("Could not detect a patient identifier column in diagnosis dataframe. Add one of: " + ",".join(PATIENT_ID_CANDIDATES))

diag_code_col = find_diag_code_col(diagnosis.columns)
if diag_code_col is None:
    raise RuntimeError("Could not detect a diagnosis code column in diagnosis dataframe. Add one of: " + ",".join(DIAG_CODE_CANDIDATES))

print(f"Aggregating diagnoses to patient-level using patient id '{patient_id_col}' and diagnosis code '{diag_code_col}'")
patient_df = aggregate_diagnoses_to_patient(diagnosis, patient_id_col, diag_code_col, top_n=200, min_count=None)

print(f"Patient-level diagnoses: {patient_df.shape[0]:,} patients, {patient_df.shape[1]-1:,} diagnosis features")


try:
    acx_cols = pd.read_csv(ACX_PATH, nrows=0).columns.tolist()
except Exception as e:
    print("Failed to read acxiom columns preview:", e)
    acx_cols = []

print("Acxiom columns (preview):", acx_cols[:20])

# Use patient-level columns when choosing merge keys
diag_cols = set(patient_df.columns)
diag_key, acx_key = find_merge_keys(diag_cols, acx_cols)

merged = None

if diag_key and acx_key:
    print(f"Found merge keys: diagnosis.{diag_key} <-> acxiom.{acx_key}")
    acx_size = os.path.getsize(ACX_PATH)
    MB = 1024 * 1024
    if acx_size > 200 * MB:
        print("Large acxiom file detected — performing chunked, memory-friendly merge")
        merged = chunked_bridge_merge(patient_df.copy(), ACX_PATH, diag_key, acx_key, chunksize=CHUNKSIZE)
    else:
        print("Acxiom file small enough — reading fully for simple merge")
        merged = simple_merge(ACX_PATH, patient_df, diag_key, acx_key)
else:
    print("No direct key found between diagnosis and acxiom. Trying demographics bridge...")
    if demographics is None:
        raise RuntimeError("No demographics provided — cannot bridge automatically.")

    if 'sys_mbr_sk' not in demographics.columns or 'empi' not in demographics.columns:
        raise RuntimeError("Demographics missing 'sys_mbr_sk' or 'empi' columns — cannot bridge.")

    dem_map = demographics[['sys_mbr_sk', 'empi']].drop_duplicates()
    diag_with_empi = patient_df.merge(dem_map, left_on='sys_mbr_sk', right_on='sys_mbr_sk', how='left')

    empi_col = None
    for c in acx_cols:
        if c.lower() == 'empi' or 'empi' in c.lower() or c.lower() == 'lumeris_empi':
            empi_col = c
            break

    if empi_col:
        print(f"Found acxiom empi-like column: {empi_col} — merging via demographics bridge")
        acx_size = os.path.getsize(ACX_PATH)
        if acx_size > 200 * 1024 * 1024:
            merged = chunked_bridge_merge(diag_with_empi.copy(), ACX_PATH, 'empi', empi_col, chunksize=CHUNKSIZE)
        else:
            merged = simple_merge(ACX_PATH, diag_with_empi, 'empi', empi_col)
    else:
        bridge_col = None
        for c in acx_cols:
            if demographics is not None and c in demographics.columns:
                bridge_col = c
                break
        if bridge_col:
            print(f"Using demographics bridge via column: {bridge_col}")
            if os.path.getsize(ACX_PATH) > 200 * 1024 * 1024:
                augmented = patient_df.merge(demographics[['sys_mbr_sk', bridge_col]].drop_duplicates(), on='sys_mbr_sk', how='left')
                merged = chunked_bridge_merge(augmented.copy(), ACX_PATH, bridge_col, bridge_col, chunksize=CHUNKSIZE)
            else:
                acx_full = pd.read_csv(ACX_PATH, dtype=str, low_memory=False)
                merged = patient_df.merge(demographics[['sys_mbr_sk', bridge_col]].drop_duplicates(), on='sys_mbr_sk', how='left').merge(acx_full, left_on=bridge_col, right_on=bridge_col, how='left')
        else:
            print("No suitable bridge column found automatically.")
            print("Sample of acxiom columns (first 5):", acx_cols[:5])
            raise RuntimeError('No merge key found automatically. Inspect acxiom columns and choose a key to merge on.')

# final save
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
merged.to_csv(OUT_PATH, index=False)
print(f"✅ Saved merged dataset to {OUT_PATH} — shape: {merged.shape}")


if __name__ == '__main__':
    # Hardcoded settings (as you'd run from a notebook cell)
    class Args:
        pass

    args = Args()
    args.acxiom = r"acxiom_full.csv"
    args.diagnosis = r"C:\Users\npigadas\OneDrive - Lumeris Solutions Company, LLC\Desktop\Files V2\diagnosis.csv"
    args.demographics = r"C:\Users\npigadas\OneDrive - Lumeris Solutions Company, LLC\Desktop\Files V2\demographics.csv"
    args.output = r"diagnosis_with_acxiom.csv"
    args.chunksize = 200_000

    print('Running merge with hardcoded paths:')
    print(' acxiom ->', args.acxiom)
    print(' diagnosis ->', args.diagnosis)
    print(' demographics ->', args.demographics)
    print(' output ->', args.output)

    main(args)


"""
merge_acxiom_diagnosis.py

Robustly merge `diagnosis.csv` with `acxiom_full.csv`, using `demographics.csv`
as a bridge when needed. The script attempts to automatically detect matching
ID columns and will perform a memory-friendly chunked merge if `acxiom_full.csv`
is large.

Usage (defaults point to the Files V2 used in the notebook):
python merge_acxiom_diagnosis.py \
    --acxiom acxiom_full.csv \
    --diagnosis "C:\\Users\\npigadas\\OneDrive - Lumeris Solutions Company, LLC\\Desktop\\Files V2\\diagnosis.csv" \
    --demographics "C:\\Users\\npigadas\\OneDrive - Lumeris Solutions Company, LLC\\Desktop\\Files V2\\demographics.csv" \
    --output diagnosis_with_acxiom.csv

The script saves `--output` with acxiom columns appended (first-match behavior
for duplicated keys in acxiom).
"""
import os
import sys
from pathlib import Path

import pandas as pd


CANDIDATE_KEYS = [
    "empi",
    "lumeris_empi",
    "empi_id",
    "member_id",
    "sys_mbr_sk",
    "clm_sys_mbr_sk",
    "memberid",
    "acxiom_id",
    "acx_id",
    "subscriber_id",
]

# Columns to look for when finding a diagnosis code column
DIAG_CODE_CANDIDATES = [
    "diagnosis_code",
    "dx_code",
    "diag_code",
    "icd_code",
    "icd10",
    "icd9",
    "dx",
    "diagnosis",
    "code",
]

# Patient identifier candidates (in order of preference)
PATIENT_ID_CANDIDATES = [
    "empi",
    "lumeris_empi",
    "empi_id",
    "member_id",
    "sys_mbr_sk",
    "clm_sys_mbr_sk",
    "memberid",
    "subscriber_id",
]


def find_patient_id_col(cols):
    for c in PATIENT_ID_CANDIDATES:
        if c in cols:
            return c
    # fallback: return first column named like an id
    for c in cols:
        if c.lower().endswith("id") or c.lower().endswith("_id"):
            return c
    return None


def find_diag_code_col(cols):
    # Exact matches first
    for c in DIAG_CODE_CANDIDATES:
        if c in cols:
            return c

    # Case-insensitive exact
    lower = {c.lower(): c for c in cols}
    for cand in DIAG_CODE_CANDIDATES:
        if cand.lower() in lower:
            return lower[cand.lower()]

    # Substring match: allow candidates like 'icd9' to match 'icd9_diagnosis_cd'
    cand_tokens = [t.lower() for t in DIAG_CODE_CANDIDATES]
    for col in cols:
        cl = col.lower()
        for tok in cand_tokens:
            if tok and tok in cl:
                return col

    return None


def aggregate_diagnoses_to_patient(diagnosis_df, patient_col, diag_col, top_n=100, min_count=None):
    """Aggregate diagnosis rows to patient-level and one-hot encode top diagnosis codes.

    Returns a DataFrame indexed by patient id with binary columns for each selected
    diagnosis code (1 = patient has that code at least once).
    """
    if patient_col not in diagnosis_df.columns:
        raise ValueError(f"Patient id column {patient_col!r} not found in diagnosis dataframe")
    if diag_col not in diagnosis_df.columns:
        raise ValueError(f"Diagnosis code column {diag_col!r} not found in diagnosis dataframe")

    # drop rows missing patient id or diagnosis code
    d = diagnosis_df[[patient_col, diag_col]].dropna(subset=[patient_col])

    # Use string form for codes
    d[diag_col] = d[diag_col].astype(str)

    # choose codes to keep
    vc = d[diag_col].value_counts()
    if min_count is not None:
        keep = set(vc[vc >= min_count].index)
    else:
        keep = set(vc.head(top_n).index)

    # mark presence via dummies grouped by patient
    d_keep = d[d[diag_col].isin(keep)].copy()
    if d_keep.empty:
        # If nothing in keep, return an empty frame with patient ids
        patients = pd.DataFrame({patient_col: d[patient_col].unique()})
        patients = patients.set_index(patient_col)
        return patients

    dummies = pd.get_dummies(d_keep[diag_col], prefix="dx")
    d_idx = d_keep[[patient_col]].reset_index(drop=True)
    d_onehot = pd.concat([d_idx, dummies], axis=1)
    patient_onehot = d_onehot.groupby(patient_col).max()

    # optionally add a column counting how many rare/other diagnosis codes the patient has
    other_mask = ~d[diag_col].isin(keep)
    if other_mask.any():
        other_counts = d.loc[other_mask].groupby(patient_col).size()
        patient_onehot["dx_other_count"] = other_counts
        patient_onehot["dx_other_count"] = patient_onehot["dx_other_count"].fillna(0).astype(int)

    # ensure index is a column for downstream merges
    patient_onehot = patient_onehot.reset_index()
    return patient_onehot



def find_merge_keys(diag_cols, acx_cols):
    # 1) exact match
    for c in acx_cols:
        if c in diag_cols:
            return c, c
    # 2) candidate-based: diag contains candidate and acx has case-insensitive match
    acx_lower = {c.lower(): c for c in acx_cols}
    for cand in CANDIDATE_KEYS:
        if cand in diag_cols and cand.lower() in acx_lower:
            return cand, acx_lower[cand.lower()]
    # 3) case-insensitive exact between the two
    diag_lower = {c.lower(): c for c in diag_cols}
    for c_lower, c_orig in diag_lower.items():
        if c_lower in acx_lower:
            return c_orig, acx_lower[c_lower]
    return None, None


def chunked_bridge_merge(diagnosis, acx_path, diag_key, acx_key, chunksize=200_000):
    # diagnosis: dataframe (left), diag_key column present
    # We'll perform an outer-leaning approach: for each chunk of acxiom,
    # map values for matching keys into the diagnosis dataframe. If acx has
    # multiple rows per key, we keep the first occurrence encountered.
    # Collect distinct diag values (strings) to look up in acxiom chunks
    # normalize diag values (strip whitespace) and use a set for fast lookup
    diag_vals = set(v.strip() for v in diagnosis[diag_key].dropna().astype(str).unique())
    print(f"Chunked merge: {len(diag_vals):,} unique diag keys to match (preview)")

    # Read acxiom header once to know available columns (cheap operation)
    acx_cols = pd.read_csv(acx_path, nrows=0, dtype=str).columns.tolist()
    if acx_key not in acx_cols:
        raise RuntimeError(f"Expected acx key {acx_key} not in acxiom columns")

    # Add any missing acxiom columns to `diagnosis` in one concat operation
    add_cols = [c for c in acx_cols if c != acx_key and c not in diagnosis.columns]
    if add_cols:
        new_cols = pd.DataFrame({c: pd.NA for c in add_cols}, index=diagnosis.index)
        diagnosis = pd.concat([diagnosis, new_cols], axis=1)

    # Iterate file-by-chunk, mapping first-seen acxiom values into diagnosis
    matched_keys = set()
    for i, chunk in enumerate(pd.read_csv(acx_path, chunksize=chunksize, dtype=str, low_memory=False)):
        # keep only rows for keys present in diagnosis
        # normalize acx_key values and filter by diag_vals (strip strings)
        chunk[acx_key] = chunk[acx_key].astype(str).str.strip()
        chunk_subset = chunk[chunk[acx_key].isin(diag_vals)]
        if chunk_subset.empty:
            continue

        # drop duplicate keys keeping first occurrence
        chunk_subset = chunk_subset.drop_duplicates(subset=[acx_key])

        # build mapping dicts for columns and apply with vectorized fillna
        matched_in_chunk = set(chunk_subset[acx_key].astype(str).unique())
        matched_keys.update(matched_in_chunk)
        for col in chunk_subset.columns:
            if col == acx_key:
                continue
            mapping = dict(zip(chunk_subset[acx_key].astype(str), chunk_subset[col]))
            mask = diagnosis[diag_key].notna()
            mapped_series = diagnosis.loc[mask, diag_key].astype(str).str.strip().map(mapping)
            diagnosis.loc[mask, col] = diagnosis.loc[mask, col].fillna(mapped_series)

        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1} chunks — matched so far: {len(matched_keys):,} unique keys")

    # report match statistics
    print(f"Chunked merge finished — matched {len(matched_keys):,} unique keys ({len(matched_keys)/max(1,len(diag_vals)):.2%} of diag keys)")
    # Defragment frame after many assignments
    diagnosis = diagnosis.copy()
    return diagnosis


def simple_merge(acx_path, diagnosis, left_key, right_key):
    acx_full = pd.read_csv(acx_path, dtype=str, low_memory=False)
    merged = diagnosis.merge(acx_full, left_on=left_key, right_on=right_key, how="left")
    return merged


# Notebook-style top-level execution (hardcoded paths)
# Paths (adjust if needed)
ACX_PATH = Path(r"acxiom_full.csv")
DIAG_PATH = Path(r"C:\Users\npigadas\OneDrive - Lumeris Solutions Company, LLC\Desktop\Files V2\diagnosis.csv")
DEM_PATH = Path(r"C:\Users\npigadas\OneDrive - Lumeris Solutions Company, LLC\Desktop\Files V2\demographics.csv")
OUT_PATH = Path(r"diagnosis_with_acxiom.csv")
CHUNKSIZE = 200_000

print("Reading diagnosis (as strings for robust joins)...")
if not DIAG_PATH.exists():
    raise FileNotFoundError(f"Diagnosis file not found: {DIAG_PATH}")
diagnosis = pd.read_csv(DIAG_PATH, dtype=str, low_memory=False)

print("Reading demographics (for bridging)...")
if not DEM_PATH.exists():
    print(f"Warning: demographics file not found at {DEM_PATH}; bridge options will be limited")
    demographics = None
else:
    demographics = pd.read_csv(DEM_PATH, dtype=str, low_memory=False)

# normalize: if clm_sys_mbr_sk exists, rename to sys_mbr_sk for consistency
if 'clm_sys_mbr_sk' in diagnosis.columns and 'sys_mbr_sk' not in diagnosis.columns:
    diagnosis = diagnosis.rename(columns={'clm_sys_mbr_sk': 'sys_mbr_sk'})

# -----------------------------
# Aggregate diagnosis rows to patient-level (one row per patient)
# -----------------------------
patient_id_col = find_patient_id_col(diagnosis.columns)
if patient_id_col is None:
    raise RuntimeError("Could not detect a patient identifier column in diagnosis dataframe. Add one of: " + ",".join(PATIENT_ID_CANDIDATES))

diag_code_col = find_diag_code_col(diagnosis.columns)
if diag_code_col is None:
    raise RuntimeError("Could not detect a diagnosis code column in diagnosis dataframe. Add one of: " + ",".join(DIAG_CODE_CANDIDATES))

print(f"Aggregating diagnoses to patient-level using patient id '{patient_id_col}' and diagnosis code '{diag_code_col}'")
patient_df = aggregate_diagnoses_to_patient(diagnosis, patient_id_col, diag_code_col, top_n=200, min_count=None)

print(f"Patient-level diagnoses: {patient_df.shape[0]:,} patients, {patient_df.shape[1]-1:,} diagnosis features")


try:
    acx_cols = pd.read_csv(ACX_PATH, nrows=0).columns.tolist()
except Exception as e:
    print("Failed to read acxiom columns preview:", e)
    acx_cols = []

print("Acxiom columns (preview):", acx_cols[:20])

# Use patient-level columns when choosing merge keys
diag_cols = set(patient_df.columns)
diag_key, acx_key = find_merge_keys(diag_cols, acx_cols)

merged = None

if diag_key and acx_key:
    print(f"Found merge keys: diagnosis.{diag_key} <-> acxiom.{acx_key}")
    acx_size = os.path.getsize(ACX_PATH)
    MB = 1024 * 1024
    if acx_size > 200 * MB:
        print("Large acxiom file detected — performing chunked, memory-friendly merge")
        merged = chunked_bridge_merge(patient_df.copy(), ACX_PATH, diag_key, acx_key, chunksize=CHUNKSIZE)
    else:
        print("Acxiom file small enough — reading fully for simple merge")
        merged = simple_merge(ACX_PATH, patient_df, diag_key, acx_key)
else:
    print("No direct key found between diagnosis and acxiom. Trying demographics bridge...")
    if demographics is None:
        raise RuntimeError("No demographics provided — cannot bridge automatically.")

    if 'sys_mbr_sk' not in demographics.columns or 'empi' not in demographics.columns:
        raise RuntimeError("Demographics missing 'sys_mbr_sk' or 'empi' columns — cannot bridge.")

    dem_map = demographics[['sys_mbr_sk', 'empi']].drop_duplicates()
    diag_with_empi = patient_df.merge(dem_map, left_on='sys_mbr_sk', right_on='sys_mbr_sk', how='left')

    empi_col = None
    for c in acx_cols:
        if c.lower() == 'empi' or 'empi' in c.lower() or c.lower() == 'lumeris_empi':
            empi_col = c
            break

    if empi_col:
        print(f"Found acxiom empi-like column: {empi_col} — merging via demographics bridge")
        acx_size = os.path.getsize(ACX_PATH)
        if acx_size > 200 * 1024 * 1024:
            merged = chunked_bridge_merge(diag_with_empi.copy(), ACX_PATH, 'empi', empi_col, chunksize=CHUNKSIZE)
        else:
            merged = simple_merge(ACX_PATH, diag_with_empi, 'empi', empi_col)
    else:
        bridge_col = None
        for c in acx_cols:
            if demographics is not None and c in demographics.columns:
                bridge_col = c
                break
        if bridge_col:
            print(f"Using demographics bridge via column: {bridge_col}")
            if os.path.getsize(ACX_PATH) > 200 * 1024 * 1024:
                augmented = patient_df.merge(demographics[['sys_mbr_sk', bridge_col]].drop_duplicates(), on='sys_mbr_sk', how='left')
                merged = chunked_bridge_merge(augmented.copy(), ACX_PATH, bridge_col, bridge_col, chunksize=CHUNKSIZE)
            else:
                acx_full = pd.read_csv(ACX_PATH, dtype=str, low_memory=False)
                merged = patient_df.merge(demographics[['sys_mbr_sk', bridge_col]].drop_duplicates(), on='sys_mbr_sk', how='left').merge(acx_full, left_on=bridge_col, right_on=bridge_col, how='left')
        else:
            print("No suitable bridge column found automatically.")
            print("Sample of acxiom columns (first 5):", acx_cols[:5])
            raise RuntimeError('No merge key found automatically. Inspect acxiom columns and choose a key to merge on.')

# final save: coerce reasonable dtypes to avoid mixed-type columns on read
if merged is None:
    raise RuntimeError("No merged dataframe produced — aborting save")

# 1) coerce diagnosis one-hot columns (prefix 'dx_') to integer 0/1
dx_cols = [c for c in merged.columns if c.startswith("dx_")]
if dx_cols:
    merged[dx_cols] = merged[dx_cols].fillna(0).astype("int8")

# 2) coerce known Acxiom columns to pandas 'string' dtype for consistency
acx_cols_safe = [c for c in acx_cols if c in merged.columns]
for c in acx_cols_safe:
    try:
        merged[c] = merged[c].astype("string")
    except Exception:
        # best-effort: fallback to generic string conversion
        merged[c] = merged[c].astype(str).replace("nan", "")

# Defragment before write
merged = merged.copy()

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
merged.to_csv(OUT_PATH, index=False)
print(f"✅ Saved merged dataset to {OUT_PATH} — shape: {merged.shape}")


if __name__ == '__main__':
    # Hardcoded settings (as you'd run from a notebook cell)
    class Args:
        pass

    args = Args()
    args.acxiom = r"acxiom_full.csv"
    args.diagnosis = r"C:\Users\npigadas\OneDrive - Lumeris Solutions Company, LLC\Desktop\Files V2\diagnosis.csv"
    args.demographics = r"C:\Users\npigadas\OneDrive - Lumeris Solutions Company, LLC\Desktop\Files V2\demographics.csv"
    args.output = r"diagnosis_with_acxiom.csv"
    args.chunksize = 200_000

    print('Running merge with hardcoded paths:')
    print(' acxiom ->', args.acxiom)
    print(' diagnosis ->', args.diagnosis)
    print(' demographics ->', args.demographics)
    print(' output ->', args.output)

    main(args)


import pandas as pd
 
# read (suppress the dtype warning if you don’t care)
sdoh_diagnoses_patient = pd.read_csv(
    r"C:\Users\npigadas\OneDrive - Lumeris Solutions Company, LLC\Desktop\full_sdoh\diagnosis_with_acxiom.csv",
    low_memory=False   # stops the warning but keeps dtype=object
)
 
# ---------- choose a threshold ----------
max_nan_frac = 0.40      # drop rows with >40 % NaN
# -----------------------------------------
 
row_nan_frac = sdoh_diagnoses_patient.isna().mean(axis=1)
keep_mask    = row_nan_frac <= max_nan_frac
 
clean_df = sdoh_diagnoses_patient.loc[keep_mask].copy()
 
print(f"Kept {keep_mask.sum():,} / {len(keep_mask):,} rows "
      f"(dropped {(~keep_mask).sum():,} rows with > {max_nan_frac:.0%} NaNs)")

# ============================================================

#  UMAP of SDoH (ap-columns) coloured by 10 diagnosis codes

#  ----------------------------------------------------------

#  * single UMAP fit (KNN-imputed, scaled)

#  * picks the 10 most-prevalent dx_* columns automatically

#  * outputs 10 scatter plots (one per diagnosis)

# ============================================================
 
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.impute import KNNImputer

from sklearn.preprocessing import StandardScaler

from umap import UMAP
 
# -----------------------------

# 1)  LOAD  CSV   (low_memory=False suppresses dtype warnings)

# -----------------------------

PATH = r"C:\Users\npigadas\OneDrive - Lumeris Solutions Company, LLC\Desktop\full_sdoh\diagnosis_with_acxiom.csv"

df = pd.read_csv(PATH, low_memory=False)
 
# -----------------------------

# 2)  IDENTIFY COLUMN GROUPS

# -----------------------------

flag_idx   = df.columns.get_loc("flag")                 # marks start of SDoH section

sdoh_cols  = list(df.columns[flag_idx + 1:])            # ap006775, ap006771, ...

dx_cols    = [c for c in df.columns if c.startswith("dx_") and c != "dx_other_count"]
 
# -----------------------------

# 3)  BUILD   SDoH  FEATURE MATRIX  (numeric → sentinels→NaN)

# -----------------------------

X = df[sdoh_cols].apply(pd.to_numeric, errors="coerce")

SENTINELS = {9, 99, 999, 9999}

X = X.mask(X.isin(SENTINELS))
 
# optional QC: drop very-empty rows/cols

X = X.loc[:, X.isna().mean() <= 0.95]           # keep cols <=95 % NaN

row_mask = X.isna().mean(axis=1) <= 0.40        # keep rows <=40 % NaN

X = X.loc[row_mask]

df_qc = df.loc[row_mask].reset_index(drop=True)
 
# -----------------------------

# 4)  IMPUTE  (KNN)  +  SCALE

# -----------------------------

X_imp = KNNImputer(n_neighbors=10, weights="distance").fit_transform(X)

X_scaled = StandardScaler().fit_transform(X_imp)
 
# -----------------------------

# 5)  UMAP  (2-D)

# -----------------------------

umap = UMAP(n_components=2, n_neighbors=30, min_dist=0.1,

            metric="cosine", random_state=42)

emb = umap.fit_transform(X_scaled)

df_qc["umap_x"], df_qc["umap_y"] = emb[:, 0], emb[:, 1]
 
# -----------------------------

# 6)  PICK  10  MOST-COMMON  DIAGNOSIS CODES

# -----------------------------

dx_prevalence = df_qc[dx_cols].sum().sort_values(ascending=False)

top10_dx = list(dx_prevalence.head(10).index)

print("Top-10 diagnosis columns:", top10_dx)
 
# -----------------------------

# 7)  PLOT  — one UMAP per Dx code

# -----------------------------

plt.style.use("default")

for dx in top10_dx:

    values = pd.to_numeric(df_qc[dx], errors="coerce").fillna(0)
 
    plt.figure(figsize=(6, 5))

    # grey background = patients without that diagnosis

    no_mask = values == 0

    plt.scatter(df_qc.loc[no_mask, "umap_x"],

                df_qc.loc[no_mask, "umap_y"],

                c="lightgray", s=6, alpha=0.2, label="Dx = 0")
 
    # coloured foreground = patients with the diagnosis

    yes_mask = values != 0

    plt.scatter(df_qc.loc[yes_mask, "umap_x"],

                df_qc.loc[yes_mask, "umap_y"],

                c="crimson", s=6, alpha=0.8, label="Dx = 1")
 
    plt.xlabel("UMAP-1"); plt.ylabel("UMAP-2")

    plt.title(f"UMAP of SDoH – coloured by {dx}")

    plt.legend(loc="upper right")

    plt.tight_layout()

    plt.show()

 # =======================================================================

#  SDoH-UMAP OVERLAYS  (10 diagnosis flags + geography)

#  ---------------------------------------------------------------------

#  • assumes df_qc already contains:

#        umap_x, umap_y            – the 2-D UMAP coordinates

#        dx_* columns              – binary diagnosis flags

#        state_str, city_str, zip_str (ZIP-5), zip3   (any subset is okay)

#  • NO UMAP recomputation

#  • creates 10 Dx overlays  +  geo overlays (only for columns present)

#  • leaves df_qc untouched (except adds alias df_qc["zip5"] if missing)

# =======================================================================
 
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import matplotlib.colors as mcolors
 
# ──────────────────────────────────────────────────────────────────────

# Helpers

# ──────────────────────────────────────────────────────────────────────

def _topN(series: pd.Series, n=20):

    """Keep top-n categories; others → 'Other'."""

    vc  = series.value_counts()

    top = set(vc.head(n).index)

    return series.where(series.isin(top), other="Other").astype("string")
 
def plot_dx_overlay(df: pd.DataFrame, dx_col: str,

                    yes_colour="#003366",    # navy

                    no_colour ="#8c8c8c",    # mid-gray

                    pt_size=6, alpha_yes=.85, alpha_no=.35):

    """Two-tone overlay: navy = Dx present, gray = Dx absent."""

    yes = df[dx_col].astype(float).fillna(0) != 0
 
    fig, ax = plt.subplots(figsize=(6,5))

    ax.scatter(df.loc[~yes,"umap_x"], df.loc[~yes,"umap_y"],

               c=no_colour, s=pt_size, alpha=alpha_no, label="Dx = 0")

    ax.scatter(df.loc[yes, "umap_x"], df.loc[yes, "umap_y"],

               c=yes_colour, s=pt_size, alpha=alpha_yes, label="Dx = 1")
 
    ax.set_title(f"UMAP of SDoH  –  {dx_col}")

    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")

    ax.legend(frameon=False); plt.tight_layout(); plt.show()
 
def plot_topN_categories(df: pd.DataFrame, col: str, n=20,

                         pt_size=6, alpha_bg=.12, alpha_fg=.85):

    """Colour top-n categories with Tableau palette; others gray."""

    cats = _topN(df[col], n).fillna("Missing")

    uniq = pd.unique(cats)
 
    palette = list(mcolors.TABLEAU_COLORS.values())

    while len(palette) < len(uniq):

        palette.extend(palette)           # repeat if >20 needed
 
    fig, ax = plt.subplots(figsize=(6,5))

    bg = cats == "Other"

    ax.scatter(df.loc[bg,"umap_x"], df.loc[bg,"umap_y"],

               c="#d0d0d0", s=pt_size, alpha=alpha_bg, label="Other")
 
    for i, cat in enumerate(uniq):

        if cat == "Other": continue

        m = cats == cat

        ax.scatter(df.loc[m,"umap_x"], df.loc[m,"umap_y"],

                   c=[palette[i]], s=pt_size, alpha=alpha_fg, label=str(cat))
 
    ax.set_title(f"UMAP – top {n} {col}  (others gray)")

    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")

    ax.legend(bbox_to_anchor=(1.02,1), loc="upper left",

              fontsize="small", markerscale=2)

    plt.tight_layout(); plt.show()
 
# ──────────────────────────────────────────────────────────────────────

# Pre-flight: make sure a ZIP-5 column exists

# ──────────────────────────────────────────────────────────────────────

if "zip5" not in df_qc.columns and "zip_str" in df_qc.columns:

    df_qc["zip5"] = df_qc["zip_str"]
 
# ──────────────────────────────────────────────────────────────────────

# 1)  Ten most-common diagnosis flags

# ──────────────────────────────────────────────────────────────────────

dx_cols = [c for c in df_qc.columns if c.startswith("dx_") and c != "dx_other_count"]

top10   = df_qc[dx_cols].sum().sort_values(ascending=False).head(10).index

print("Top-10 Dx columns:", list(top10))
 
for dx in top10:

    plot_dx_overlay(df_qc, dx)
 
# ──────────────────────────────────────────────────────────────────────

# 2)  Geography overlays  (run only if column exists)

# ──────────────────────────────────────────────────────────────────────

for col in ["state_str", "zip5", "zip3", "city_str"]:

    if col in df_qc.columns:

        plot_topN_categories(df_qc, col, n=20)

    else:

        print(f"⧗  Skipping {col}: column not found.")

 