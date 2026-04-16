import pandas as pd # data processing

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


credit=pd.read_csv("data/german_credit_data.csv")

# **Transformations and feature engineering:**

# The Risk is what we would like to predict: either a 0 for the loan presenting no risk and will be repaid on time, or a 1 indicating that the loan presents a risk and the client will have some payment difficulties.
# To this end, we have two unique categories that’s why we use the map function for Label encoding.

credit['Risk'] = credit['Risk'].map({'bad':1, 'good':0})

# When the time comes to build the machine learning model, we have to fill in these missing values (known as imputation) identified during the data checks phase.
# In our case, this is a small dataset which obligeges to keep all our rows that’s why we have introduced a new category value called “Others” for both Saving account and Checking account columns.

credit['Saving accounts'] = credit['Saving accounts'].fillna('Others')
credit['Checking account'] = credit['Checking account'].fillna('Others')

credit_clean=credit.copy()

cat_features = ['Sex','Housing', 'Saving accounts', 'Checking account','Purpose']
num_features=['Age', 'Job', 'Credit amount', 'Duration','Risk']
for variable in cat_features:
    dummies = pd.get_dummies(credit_clean[cat_features])
    df1= pd.concat([credit_clean[num_features], dummies],axis=1)

Risk= df1['Risk']          
df2=df1.drop(['Risk'],axis=1)

X_train,X_test,Y_train,Y_test = train_test_split(df2,Risk,test_size=0.20,random_state = 30)

# * **Model building process:**

random_forest = RandomForestClassifier(random_state = 100)

#Standardization
sc=StandardScaler()
X_train_std=sc.fit_transform(X_train)
X_test_std=sc.transform(X_test)


random_forest = RandomForestClassifier(random_state = 100, n_estimators=100, max_features='sqrt', max_depth=10, min_samples_split=5, min_samples_leaf=1)
random_forest.fit(X_train_std, Y_train)


Y_test_pred = random_forest.predict(X_test_std)

print('The accuracy of the Random Forest Classifier is',accuracy_score(Y_test, Y_test_pred))