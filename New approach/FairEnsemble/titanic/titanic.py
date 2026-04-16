import pandas as pd

# preprocessing the data
data=pd.read_csv('data/train.csv')

## Assigning the NaN Values with the Ceil values of the mean ages

data['Initial'] = data.Name.str.extract(r' ([A-Za-z]+)\.', expand=False)
data['Initial'] = data['Initial'].replace(
    ['Rev', 'Dr', 'Col', 'Major', 'Capt', 'Jonkheer', 'Don', 'Sir', 'Countess', 'Lady', 'Mme', 'Ms', 'Mlle'], 
    'Other'
)


data.loc[(data.Age.isnull())&(data.Initial=='Mr'),'Age']=33
data.loc[(data.Age.isnull())&(data.Initial=='Mrs'),'Age']=36
data.loc[(data.Age.isnull())&(data.Initial=='Master'),'Age']=5
data.loc[(data.Age.isnull())&(data.Initial=='Miss'),'Age']=22
data.loc[(data.Age.isnull())&(data.Initial=='Other'),'Age']=46

# **Filling the missing values in the Embarked column with the most common value which is S.**
data['Embarked'] = data['Embarked'].fillna('S')

# **Family_Size=0 means that the passeneger is alone.** Clearly, if you are alone or family_size=0,then chances for survival is very low. For family size > 4,the chances decrease too. 
# This also looks to be an important feature for the model. Lets examine this further.
data['Family_Size']=0
data['Family_Size']=data['Parch']+data['SibSp']#family size
data['Alone']=0
data.loc[data.Family_Size==0,'Alone']=1#Alone

# ## Converting String Values into Numeric
# 
# Since we cannot pass strings to a machine learning model, we need to convert features like Sex, Embarked, etc into numeric values.


data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
data['Embarked'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
data['Initial'] = data['Initial'].map({'Mr': 0, 'Mrs': 1, 'Miss': 2, 'Master': 3, 'Other': 4})


# ### Dropping UnNeeded Features
# 
# **Name**--> We don't need name feature as it cannot be converted into any categorical value.
# 
# **Ticket**--> It is any random string that cannot be categorised.
# 
# **Fare**--> We have the Fare_cat feature, so unneeded
# 
# **PassengerId**--> Cannot be categorised.


data.drop(['Name','Ticket','Cabin','PassengerId'], axis=1, inplace=True)


# # Part3: Predictive Modeling
# 
# We have gained some insights from the EDA part. But with that, we cannot accurately predict or tell whether a passenger will survive or die. So now we will predict the whether the Passenger will survive or not using some great Classification Algorithms.Following are the algorithms we will use to make the model:
# 
# 1)Logistic Regression
# 
# 2)Support Vector Machines(Linear and radial)
# 
# 3)Random Forest
# 
# 4)K-Nearest Neighbours
# 
# 5)Naive Bayes
# 
# 6)Decision Tree
# 
# 7)Logistic Regression



#importing all the required ML packages
from sklearn.linear_model import LogisticRegression #logistic regression
from sklearn import svm #support vector Machine
from sklearn.ensemble import RandomForestClassifier #Random Forest
from sklearn.neighbors import KNeighborsClassifier #KNN
from sklearn.naive_bayes import GaussianNB #Naive bayes
from sklearn.tree import DecisionTreeClassifier #Decision Tree
from sklearn.model_selection import train_test_split #training and testing data split
from sklearn import metrics #accuracy measure


train,test=train_test_split(data,test_size=0.3,random_state=0,stratify=data['Survived'])
train_X=train[train.columns[1:]]
train_Y=train[train.columns[:1]]
test_X=test[test.columns[1:]]
test_Y=test[test.columns[:1]]
X=data[data.columns[1:]]
Y=data['Survived']


# ### Radial Support Vector Machines(rbf-SVM)



model=svm.SVC(kernel='rbf',C=1,gamma=0.1)
model.fit(train_X,train_Y.values.ravel())
prediction1=model.predict(test_X)
print('Accuracy for rbf SVM is ',metrics.accuracy_score(prediction1,test_Y))


# ### Linear Support Vector Machine(linear-SVM)


model=svm.SVC(kernel='linear',C=0.1,gamma=0.1)
model.fit(train_X,train_Y.values.ravel())
prediction2=model.predict(test_X)
print('Accuracy for linear SVM is',metrics.accuracy_score(prediction2,test_Y))


# ### Logistic Regression


model = LogisticRegression(max_iter=1000)
model.fit(train_X,train_Y.values.ravel())
prediction3=model.predict(test_X)
print('The accuracy of the Logistic Regression is',metrics.accuracy_score(prediction3,test_Y))


# ### Decision Tree



model=DecisionTreeClassifier()
model.fit(train_X,train_Y.values.ravel())
prediction4=model.predict(test_X)
print('The accuracy of the Decision Tree is',metrics.accuracy_score(prediction4,test_Y))


# ### K-Nearest Neighbours(KNN)



model=KNeighborsClassifier() 
model.fit(train_X,train_Y.values.ravel())
prediction5=model.predict(test_X)
print('The accuracy of the K-Nearest Neighbours is',metrics.accuracy_score(prediction5,test_Y))



# ### Gaussian Naive Bayes



model=GaussianNB()
model.fit(train_X,train_Y.values.ravel())
prediction6=model.predict(test_X)
print('The accuracy of the Gaussian Naive Bayes is',metrics.accuracy_score(prediction6,test_Y))


# ### Random Forests


model=RandomForestClassifier(n_estimators=100)
model.fit(train_X,train_Y.values.ravel())
prediction7=model.predict(test_X)
print('The accuracy of the Random Forest is',metrics.accuracy_score(prediction7,test_Y))

