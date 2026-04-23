import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split
import optuna
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


student = pd.read_csv('../data.csv', sep=';')
# student = pd.read_csv('sml_k=2.csv')
if 'index' in student.columns:
    student = student.drop(columns=['index'])

student['Target'] = student['Target'].map({
    'Dropout': 1,
    'Enrolled': 2,
    'Graduate': 0
})

X = student.drop(columns=['Target'])
Y = student['Target']

X = X.astype(str)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), X.columns)
    ])

rf = RandomForestClassifier(random_state=42)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', rf)
])

pipeline.fit(X_train, Y_train)

rf_pred = pipeline.predict(X_test)

print("Before optimization:")
print(f"Accuracy:  {accuracy_score(Y_test, rf_pred):.4f}")
print(f"F1 score:  {f1_score(Y_test, rf_pred, average='weighted'):.4f}")
print(f"Precision: {precision_score(Y_test, rf_pred, average='weighted'):.4f}")
print(f"Recall:    {recall_score(Y_test, rf_pred, average='weighted'):.4f}")

# print(classification_report(Y_test, rf_pred, target_names=names.values()))

# def rf_objective(trial):
    
#     md = trial.suggest_int('max_depth', 2, 64)
#     mi = trial.suggest_int('min_samples_leaf', 1, 32)
#     crit = trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"])
    
#     clf =  RandomForestClassifier(max_depth=md, min_samples_leaf=mi,criterion=crit, random_state=42, )
#     scores = cross_val_score(clf, X_train, Y_train, cv=10, scoring='f1_weighted')
    
#     return scores.mean()

# # Make the optimization deterministic by setting a fixed seed for the sampler
# # We need determinism to ensure baseline results are consistent    
# sampler = optuna.samplers.TPESampler(seed=42)
# rf_study = optuna.create_study(direction='maximize', sampler=sampler)
# rf_study.optimize(rf_objective, n_trials=20) 

# rf_opt = RandomForestClassifier(**rf_study.best_params, random_state=42)
# rf_opt = rf_opt.fit(X_train,Y_train)
# rf_pred_opt = rf_opt.predict(X_test)

# print("After optimization:")
# print(f"{accuracy_score(Y_test, rf_pred_opt)*100:.2f}%")
# # print(classification_report(Y_test, rf_pred_opt, target_names=names.values()))