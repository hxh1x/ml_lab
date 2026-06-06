#importing necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection importtrain_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
#importing iris dataset from sklearn and spliting input and output
from sklearn.datasetsimportload_iris
iris = load_iris()
X = iris.data
y = iris.target
#Train-testsplit
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)
#Implementing Knn Classifier model
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train)
y_pred = knn_model.predict(X_test)
#Checking performance matrices
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)
print("Classification Report:")
print(classification_report(y_test, y_pred))
