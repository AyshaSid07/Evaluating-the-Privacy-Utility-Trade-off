import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split
import optuna
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier


student = pd.read_csv('data.csv', sep=';')
student['Target'] = student['Target'].map({
    'Dropout':1,
    'Enrolled':2,
    'Graduate':0
})
names = {1:'Dropout',2:'Enrolled',0:'Graduate'}

X = student.iloc[:,0:-1]
Y = student.iloc[:,-1]

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2, random_state=42)

rf = RandomForestClassifier(random_state=42)
rf = rf.fit(X_train,Y_train)
rf_pred = rf.predict(X_test)
print("Before optimization:")
print(f"{accuracy_score(Y_test, rf_pred)*100:.2f}%")
# print(classification_report(Y_test, rf_pred, target_names=names.values()))

def rf_objective(trial):
    
    md = trial.suggest_int('max_depth', 2, 64)
    mi = trial.suggest_int('min_samples_leaf', 1, 32)
    crit = trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"])
    
    clf =  RandomForestClassifier(max_depth=md, min_samples_leaf=mi,criterion=crit, random_state=42, )
    scores = cross_val_score(clf, X_train, Y_train, cv=10, scoring='f1_weighted')
    
    return scores.mean()

# Make the optimization deterministic by setting a fixed seed for the sampler
# We need determinism to ensure baseline results are consistent    
sampler = optuna.samplers.TPESampler(seed=42)
rf_study = optuna.create_study(direction='maximize', sampler=sampler)
rf_study.optimize(rf_objective, n_trials=20) 

rf_opt = RandomForestClassifier(**rf_study.best_params, random_state=42)
rf_opt = rf_opt.fit(X_train,Y_train)
rf_pred_opt = rf_opt.predict(X_test)

print("After optimization:")
print(f"{accuracy_score(Y_test, rf_pred_opt)*100:.2f}%")
# print(classification_report(Y_test, rf_pred_opt, target_names=names.values()))