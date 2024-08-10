# iris_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import joblib

# Load and preprocess the data
iris = pd.read_csv("iris.csv")
iris = iris.drop('Unnamed: 0', axis=1)

# Split the dataset
train, test = train_test_split(iris, test_size=0.3)
train_X = train[['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width']]
train_y = train['Species']
test_X = test[['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width']]
test_y = test['Species']

# Train models
model1 = LogisticRegression()
model1.fit(train_X, train_y)
model2 = SVC()
model2.fit(train_X, train_y)
model3 = KNeighborsClassifier(n_neighbors=5)
model3.fit(train_X, train_y)
model4 = GaussianNB()
model4.fit(train_X, train_y)

# Save models
joblib.dump(model1, 'logistic_regression_model.pkl')
joblib.dump(model2, 'svc_model.pkl')
joblib.dump(model3, 'knn_model.pkl')
joblib.dump(model4, 'naive_bayes_model.pkl')
