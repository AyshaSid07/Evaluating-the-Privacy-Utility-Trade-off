import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
df = pd.read_csv('credit-card-clients.csv')   

#preprocessing 
class_label = 'default payment'

df['SEX'] = ["Female" if v == 2 else "Male" for v in df['SEX']]

le = preprocessing.LabelEncoder()
for i in df.columns:
    if df[i].dtypes == 'object':
        df[i] = le.fit_transform(df[i])


length = len(df.columns)
X = df.iloc[:, :length-1]
y = df[class_label]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) 

algorithm = 'DT' # Decision Tree Classifier
if algorithm == 'DT':
    model = tree.DecisionTreeClassifier(random_state=0)  
# The original paper tests the following algorithms: Decision Tree, Naive Bayes, Multi-layer Perceptron and K-Nearest Neighbors. 
# We have use the Decision Tree Classifier as a baseline in this case, but the other algorithms can be run if needed.

# elif algorithm == 'NB': 
#     model = GaussianNB()
# elif algorithm == 'MLP':
#     model = MLPClassifier(random_state=1, max_iter=300)
# elif algorithm == 'kNN':
#     model = KNeighborsClassifier(n_neighbors=5)

model.fit(X_train, y_train)

y_predicts = model.predict(X_test)

print(f"{algorithm} Accuracy: {accuracy_score(y_test, y_predicts) * 100:.2f}%\n")
