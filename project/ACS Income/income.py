import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from folktables import ACSDataSource, ACSIncome

# Load the ACS data for California and prepare the features, labels, and group information
data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
acs_data = data_source.get_data(states=["CA"], download=True)
features, label, group = ACSIncome.df_to_numpy(acs_data)

# Save the raw data to a CSV file
al_features, al_labels, _ = ACSIncome.df_to_pandas(acs_data)
combined_df = pd.concat([al_features, al_labels], axis=1)
combined_df.to_csv('folktables_income_RAW.csv', index=False)

# Split the data into training and testing sets and use a logistic regression model to predict income status
X_train, X_test, y_train, y_test, group_train, group_test = train_test_split(
    features, label, group, test_size=0.2, random_state=0)

model = make_pipeline(StandardScaler(), LogisticRegression())
model.fit(X_train, y_train)

yhat = model.predict(X_test)
accuracy = accuracy_score(y_test, yhat)

print(f"Accuracy: {accuracy * 100:.2f}%")