from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np

df = pd.read_csv('risk_factors_cervical_cancer.csv')
df = df.replace('?', np.nan)
df = df.astype(float)
imputer = SimpleImputer(strategy='median')
df_cleaned = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

df_cleaned.to_csv('risk_factors_cervical_cancer_cleaned.csv', index=False)