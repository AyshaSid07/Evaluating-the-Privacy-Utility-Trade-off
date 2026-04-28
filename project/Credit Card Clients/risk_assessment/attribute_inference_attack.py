import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from art.estimators.classification import SklearnClassifier
from art.attacks.inference.attribute_inference import AttributeInferenceBlackBox

df_anon = pd.read_csv("../datasets/credit_card_k=2.csv")
# df_anon = pd.read_csv("../datasets/credit-card-clients.csv")

if 'index' in df_anon.columns:
    df_anon = df_anon.drop(columns=['index'])

encoders = {}
for col in df_anon.columns:
    df_anon[col] = df_anon[col].astype(str)
    le = LabelEncoder()
    df_anon[col] = le.fit_transform(df_anon[col])
    encoders[col] = le

target_col = 'default payment'
X = df_anon.drop(columns=[target_col]).values
y = df_anon[target_col].values

model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X, y)

art_classifier = SklearnClassifier(model=model)

attack_feature_index = 3 # index 3 in this case is MARRIAGE

attack = AttributeInferenceBlackBox(art_classifier, attack_feature=attack_feature_index)

attack.fit(X)

model_predictions = model.predict(X).reshape(-1, 1)

X_hacker = np.delete(X, attack_feature_index, axis=1)

inferred_values = attack.infer(X_hacker, pred=model_predictions)

true_values = X[:, attack_feature_index]

accuracy = np.mean(inferred_values == true_values) * 100
print(f"\nAttackens Success Rate: {accuracy:.2f}%")