from diagnosis_prediction import extract_feature_importances
from diagnosis_prediction import ColumnDropper, MissingnessAwareImputer
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Create tiny synthetic dataset
X = pd.DataFrame({
    'f1': [1.0, 2.0, np.nan, 4.0, 5.0],
    'f2': [0, 1, 0, 1, np.nan],
    'f3': [10, 20, 30, 40, 50]
})
y = np.array([0,1,0,1,0])

# Build pipeline matching structure used in diagnosis_prediction
pipe = Pipeline([
    ('dropper', ColumnDropper(missing_threshold=0.95, variance_threshold=1e-5)),
    ('imputer', MissingnessAwareImputer()),
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier(random_state=42, n_estimators=10))
])

# Call extract_feature_importances (it will fit the pipeline internally)
imp_df = extract_feature_importances(X, y, X.columns.tolist(), pipe)
print('\nResulting importances:\n', imp_df)
