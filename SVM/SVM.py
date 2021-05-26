import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

df1 = pd.read_csv('pima-indians-diabetes.csv', header=None).sample(frac=1)
df1.columns = ['col1', 'col2', 'col3', 'col4', 'col5', 'col6', 'col7', 'col8', 'label']
y1 = df1['label']
X1 = df1.drop('label', axis=1)
X1


kfold = KFold(n_splits=2)


svmModel1 = SVC()

numFolds = 1
for idx_train, idx_test in kfold.split(X1):
    X_train, X_test = X1.iloc[idx_train], X1.iloc[idx_test]
    y_train, y_test = y1[idx_train], y1[idx_test]
    
    svmModel1.fit(X_train, y_train)
    y_pred = svmModel1.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    print("Fold: %d -- Accuracy Score: %4f"%(numFolds, score))
    numFolds += 1


df2 = pd.read_csv('winequality-red.csv').sample(frac=1)
y2 = df2['quality']
X2 = df2.drop('quality', axis=1)
X2

svmModel1 = SVC()

numFolds = 1
for idx_train, idx_test in kfold.split(X2):
    X_train, X_test = X2.iloc[idx_train], X2.iloc[idx_test]
    y_train, y_test = y2[idx_train], y2[idx_test]
    
    svmModel1.fit(X_train, y_train)
    y_pred = svmModel1.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    print("Fold: %d -- Accuracy Score: %4f"%(numFolds, score))
    numFolds += 1


df3 = pd.read_csv('new_test.csv').sample(frac=1)
y3 = df3['poutcome']
X3 = df3.drop('poutcome', axis=1)


svmModel1 = SVC()

numFolds = 1
for idx_train, idx_test in kfold.split(X3):
    X_train, X_test = X3.iloc[idx_train], X3.iloc[idx_test]
    y_train, y_test = y3[idx_train], y3[idx_test]
    
    svmModel1.fit(X_train, y_train)
    y_pred = svmModel1.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    print("Fold: %d -- Accuracy Score: %4f"%(numFolds, score))
    numFolds += 1


