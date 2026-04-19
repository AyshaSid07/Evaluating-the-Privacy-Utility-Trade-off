import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.ensemble import  RandomForestRegressor
import os
import sys

# to import util from the parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import util

# taken from src/datasets.py
def get_num_cat_cols(df: pd.DataFrame, y_col: str = None):
    """num_cols, cat_cols, cat_is_num_cols, cat_miss_lable_cols"""
    num_cols = []
    cat_cols = []
    cat_is_num_cols = []
    cat_miss_lable_cols = []
    for col in df.columns:
        if col == y_col:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            num_cols.append(col)
        elif isinstance(df[col].dtype, pd.CategoricalDtype):
            cat_cols.append(col)
            if df[col].astype(str).str.isdigit().all():
                cat_is_num_cols.append(col)
        else:
            cat_miss_lable_cols.append(col)
            cat_cols.append(col)

    return num_cols, cat_cols, cat_is_num_cols, cat_miss_lable_cols

# 1. loading the autism dataset steps taken from data/autism.py
df1 = pd.read_csv("student-mat.csv", encoding="ascii", delimiter=";")
df2 = pd.read_csv("student-por.csv", encoding="ascii", delimiter=";")

df1["subject"] = "Mathematics"
df2["subject"] = "Portuguese"
df = pd.concat([df1, df2]).reset_index(drop=True)

cat_cols = [
    "school",
    "sex",
    "address",
    "famsize",
    "Pstatus",
    "Mjob",
    "Fjob",
    "reason",
    "guardian",
    "schoolsup",
    "famsup",
    "paid",
    "activities",
    "nursery",
    "higher",
    "internet",
    "romantic",
    "subject",
]
for col in cat_cols:
    if not isinstance(df[col].dtype, pd.CategoricalDtype):
        df[col] = df[col].astype("category")

y_col = "G3"

leakage_cols = ['G1', 'G2']
df.drop(columns=[c for c in leakage_cols if c in df.columns], inplace=True)

useless_cols_dict = util.find_useless_colum(df)
df = util.drop_useless(df, useless_cols_dict)


# remove categorical cols with too few samples_per_cat
X = df.drop(columns=[y_col])
y = df[y_col]

# start of the preprocessing steps taken from `src/preprocess.py: split_impute_encode_scale`

num_cols, cat_cols, cat_is_num_cols, cat_miss_lable_cols = get_num_cat_cols(df, y_col)

# TRAIN/TEST SPLIT (From `src/preprocess.py: split_impute_encode_scale`)
X_train_df, X_test_df, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# preprocessing pipelines (From `src/preprocess.py: split_impute_encode_scale`)
numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

# From cat_steps in `src/preprocess.py: split_impute_encode_scale`:
cat_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),
    ("scaler", StandardScaler())
])

trans = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, num_cols),
            ("cat", cat_pipeline, cat_cols),
        ],
        remainder="drop",
    )

# Fit the transformer strictly on the training data, then transform both
X_train = trans.fit_transform(X_train_df)
X_test = trans.transform(X_test_df)

# 3. start model training (From `src/models.py: train_model`)
# They used regression for this dataset.
model = RandomForestRegressor(random_state=12345)
model.fit(X_train, y_train)

# 4. evaluate the model (From `src/models.py: evaluate`)
# for regression, they used MSE, RMSE, MAE, and R2 as metrics.
y_pred = model.predict(X_test)
mse = metrics.mean_squared_error(y_test, y_pred)
rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
mae = metrics.mean_absolute_error(y_test, y_pred)
r2 = metrics.r2_score(y_test, y_pred)

print(f"MSE:  {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE:  {mae:.4f}")
print(f"R2:   {r2:.4f}")