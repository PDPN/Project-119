from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd

#Column Name
col_names = ['passengerid', 'pclass', 'sex', 'sibsp', 'parch', 'survived', 'age']

df = pd.read_csv("titanic.csv", names=col_names).iloc[1:]

print(df.head())

features = ['passengerid', 'parch', 'survived', 'age','pclass','sex']
X = df[features]
y = df.label
#splitting data in training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

#Initialising the Decision Tree Model
clf = DecisionTreeClassifier()

#Fitting the data into the model
clf = clf.fit(X_train, y_train)

#Calculating the accuracy of the model
y_pred = clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))