import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
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
df = pd.read_csv("autism_screening.csv")

cat_cols = [col for col in df.columns if col != "age"]
for col in cat_cols:
    if not isinstance(df[col].dtype, pd.CategoricalDtype):
        df[col] = df[col].astype("category")

y_col = "Class/ASD"
df[y_col] = df[y_col].map({"NO": 0, "YES": 1, "no": 0, "yes": 1})

useless_cols_dict = util.find_useless_colum(df)
df = util.drop_useless(df, useless_cols_dict)

# if we we do not drop these columns, the model will achieve 100% accuracy by just looking at these 11 columns, which is not what we want since we want to evaluate the impact of the encoding on the utility of the model.
columns_to_drop = [
    'result',
    'A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score',
    'A6_Score', 'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score'
]

df.drop(columns=columns_to_drop, inplace=True)

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
# They used classification for this dataset.
model = RandomForestClassifier(random_state=12345)
model.fit(X_train, y_train)

# 4. evaluate the model (From `src/models.py: evaluate`)
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")