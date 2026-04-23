import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# df = pd.read_csv('../risk_factors_cervical_cancer_cleaned.csv')
df = pd.read_csv('cervical_cancer_k=5.csv')
# drop these two columns since they have a lot of missing values and are not relevant for the prediction
# df = df.drop(['STDs: Time since first diagnosis','STDs: Time since last diagnosis'], axis = 1)

# drop these as these are the other target values
# x = df.drop(columns=['Biopsy', 'Hinselmann', 'Schiller', 'Citology']) 
x = df.drop(columns=["Biopsy"])
y = df['Biopsy'].astype(int)

x = x.astype(str)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=123)

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), x.columns)
    ])

classifier = RandomForestClassifier(random_state=123, class_weight='balanced')

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', classifier)
])

pipeline.fit(x_train, y_train)
predictions = pipeline.predict(x_test)
probabilities = pipeline.predict_proba(x_test)[:, 1]
custom_threshold = 0.15
predictions = (probabilities >= custom_threshold).astype(int)

print(f"Accuracy:  {accuracy_score(y_test, predictions):.4f}")
print(f"F1 score:  {f1_score(y_test, predictions):.4f}")
print(f"Precision: {precision_score(y_test, predictions):.4f}")
print(f"Recall:    {recall_score(y_test, predictions):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, probabilities):.4f}")
