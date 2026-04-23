from sklearn.linear_model import LogisticRegression
import pandas as pd

import leaf
     
hr_data = pd.read_csv('framingham_heart_study.csv')
hr_data.rename(columns={'TenYearCHD':"TARGET"}, inplace=True)
print(len(hr_data),'rows loaded.')
print('features =', hr_data.columns.tolist())
hr_X = hr_data.drop('TARGET', axis=1)
hr_Y = hr_data['TARGET'].astype('bool')==False
hr_X.fillna(hr_X.mean(), inplace=True)
hr_class_names=['OK', 'RISK']


# model = sklearn.ensemble.RandomForestClassifier(n_estimators=5, max_depth=5, random_state=12345)
model = LogisticRegression(random_state=99, solver='liblinear')
# model = sklearn.ensemble.GradientBoostingClassifier(n_estimators=5, random_state=34)

# leaf.py will print out the accuracy
hr_cls = leaf.train_model(hr_X, hr_Y, model, verbose=False)


