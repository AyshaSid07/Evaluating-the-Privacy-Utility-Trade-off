from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd
import leaf
     
hr_data = pd.read_csv('../framingham_heart_study.csv')
# hr_data = pd.read_csv('heart_study_k=2.csv')
if 'index' in hr_data.columns:
    hr_data = hr_data.drop(columns=['index'])

hr_data.rename(columns={'TenYearCHD':"TARGET"}, inplace=True)
print(len(hr_data),'rows loaded.')
print('features =', hr_data.columns.tolist())
hr_X = hr_data.drop('TARGET', axis=1)
hr_Y = hr_data['TARGET'].astype('bool')==False
hr_X = hr_X.astype(str)

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
hr_X_encoded_matrix = encoder.fit_transform(hr_X)

# Convert the encoded matrix back into a Pandas DataFrame so leaf.py has column names
encoded_columns = encoder.get_feature_names_out(hr_X.columns)
hr_X_encoded = pd.DataFrame(hr_X_encoded_matrix, columns=encoded_columns)

hr_class_names=['OK', 'RISK']

hr_ds = (hr_X, hr_Y, hr_class_names, "heartrisk")

# model = sklearn.ensemble.RandomForestClassifier(n_estimators=5, max_depth=5, random_state=12345)
model = LogisticRegression(random_state=99, solver='liblinear')
# model = sklearn.ensemble.GradientBoostingClassifier(n_estimators=5, random_state=34)

# leaf.py will print out the all the measurements
hr_cls = leaf.train_model(hr_X_encoded, hr_Y, model, verbose=False)


