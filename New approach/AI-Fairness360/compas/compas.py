import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from aif360.sklearn.datasets import fetch_compas

cols = ['sex', 'race', 'age_cat', 'priors_count', 'c_charge_degree']
X, y = fetch_compas(usecols=cols, binary_race=True)

# Quantize priors count between 0, 1-3, and >3
X['priors_count'] = pd.cut(X['priors_count'], [-1, 0, 3, 100],
                           labels=['0', '1 to 3', 'More than 3'])

compas_df = X.copy()
compas_df['is_recid'] = y
compas_df.to_csv("COMPAS_RAW.csv", index=False)

(X_test, X_train,
y_test, y_train) = train_test_split(X, y, test_size=3694, shuffle=True, random_state=0)

regression = LogisticRegression(solver='lbfgs', C=1.0, penalty='l2', random_state=0)
regression.fit(X_train.apply(lambda s: s.cat.codes), y_train)

pred_clf = regression.predict(X_test.apply(lambda s: s.cat.codes))

print(f"Logistic Regression Accuracy: {accuracy_score(y_test, pred_clf) * 100:.2f}%")