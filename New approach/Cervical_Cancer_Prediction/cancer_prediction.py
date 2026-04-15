import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, classification_report
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

df = pd.read_csv('risk_factors_cervical_cancer_cleaned.csv')

# drop these two columns since they have a lot of missing values and are not relevant for the prediction
df = df.drop(['STDs: Time since first diagnosis','STDs: Time since last diagnosis'], axis = 1)

x = df.drop(columns=['Biopsy', 'Hinselmann', 'Schiller', 'Citology']) 
y = df['Biopsy']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)

rs = SMOTE()
x_train , y_train = rs.fit_resample(x_train, y_train)
x_test , y_test = rs.fit_resample(x_test, y_test)

classifier = RandomForestClassifier()
classifier = classifier.fit(x_train, y_train)

predictions = classifier.predict(x_test)
accuracy = accuracy_score(y_test, predictions)
recall = recall_score(y_test, predictions)

print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Recall:   {recall * 100:.2f}%")
