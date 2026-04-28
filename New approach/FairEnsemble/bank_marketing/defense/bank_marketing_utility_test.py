import pandas as pd 

data=pd.read_csv('../data/bank-additional-full.csv', sep = ';')
# data=pd.read_csv('bank_marketing_k=2.csv')
if 'index' in data.columns:
    student = data.drop(columns=['index'])
data.drop_duplicates(keep='first',inplace=True)

from sklearn.preprocessing import LabelEncoder
LE=LabelEncoder()
cat_var=['age', 'job', 'marital', 'education', 'default', 'housing', 'loan','contact', 'month', 'day_of_week','poutcome','y']
# cat_var=['age', 'campaign', 'pdays', 'previous','job', 'marital', 'education', 'loan', 'contact', 'month', 'day_of_week', 'poutcome', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed', 'y']
for i in cat_var:
    data[i]=LE.fit_transform(data[i])
    
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

X = data.drop(columns=['y'])
y = data['y']

sc=StandardScaler()
sc.fit_transform(X)




X_train,X_test,y_train,y_test=train_test_split(X, y,test_size=0.25,random_state=2)

# Dataset is huge, so we will only run the Random forest classifier for the sake of time. The other models are commented out but can be run if needed.

# # ## Logistic Regression

# lr=LogisticRegression(penalty = 'l1',solver = 'liblinear')
# lr.fit(X_train,y_train.values.ravel())
# pred_lr=lr.predict(X_test)
# score_lr= accuracy_score(y_test,pred_lr)
# print("Accuracy Score for Logistic Regression is: ", score_lr)
# print("F1 Score for Logistic Regression is: ", f1_score(y_test,pred_lr))


# # ## KNN

# knn=KNeighborsClassifier()
# knn.fit(X_train,y_train.values.ravel())
# pred_knn=knn.predict(X_test)
# print("Accuracy Score for KNN is: ", accuracy_score(y_test,pred_knn))
# print("F1 Score for KNN is: ", f1_score(y_test,pred_knn))


# # ## Decision Tree Classifier

# dt=DecisionTreeClassifier()
# dt.fit(X_train,y_train.values.ravel())
# pred_dt=dt.predict(X_test)
# #accuracy score:
# print("Accuracy Score for Decision Tree is: ", accuracy_score(y_test,pred_dt))
# print("F1 Score for Decision Tree is: ", f1_score(y_test,pred_dt))

# # ## Random Forest Classifier

rf=RandomForestClassifier()
rf.fit(X_train,y_train.values.ravel())
pred_rf=rf.predict(X_test)

print("Accuracy Score for Random Forest is: ", accuracy_score(y_test,pred_rf))
print("Precision Score for Random Forest is: ", precision_score(y_test,pred_rf))
print("Recall Score for Random Forest is: ", recall_score(y_test,pred_rf))
print("F1 Score for Random Forest is: ", f1_score(y_test,pred_rf))


# # ## XGBoost Classifier


# xgb_clf= xgb.XGBClassifier()
# xgb_clf.fit(X_train,y_train.values.ravel())
# pred_xgb=xgb_clf.predict(X_test)
# print("Accuracy Score for XGBoost is: ", accuracy_score(y_test,pred_xgb))
# print("F1 Score for XGBoost is: ", f1_score(y_test,pred_xgb))






