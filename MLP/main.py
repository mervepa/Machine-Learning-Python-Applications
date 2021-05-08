import pandas as pd
import numpy  as np
import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold

########################################################################################
#                                   CLASSIFICATION                                     #
########################################################################################

X = pd.read_csv('fashion-mnist_train.csv')[:1000]  # Read the Fashion MNIST dataset
y = X['label']                                     # Save label into y
X = X.drop('label', axis=1)                        # Drop label from X

X = X.astype('float32')                            # Convert colors into Greyscale
X /= 255

y = y.astype('category')                           # Convert labels into categories

k = 2
kfold = KFold(n_splits=k, shuffle=True)

foldNo = 1
for train_idx, test_idx in kfold.split(X, y):
    X_train, y_train, X_test, y_test = X.iloc[train_idx], y[train_idx], X.iloc[test_idx], y[test_idx]
    mlpcModel = MLPClassifier(random_state=1, max_iter=300)
    mlpcModel.fit(X_train, y_train)
    prediction = mlpcModel.predict(X_test)
    score = mlpcModel.score(X_test, y_test)
    print(f"Fold: {foldNo}, \tClassification Prediction Accuracy: {score}")
    foldNo += 1

########################################################################################
#                                   REGRESSION                                         #
########################################################################################

Xreg = pd.read_csv('winequality-red.csv')
yreg = Xreg['quality']
Xreg = Xreg.drop('quality', axis=1)

k = 2
kfold = KFold(n_splits=k, shuffle=True)

foldNo = 1
for train_idx, test_idx in kfold.split(Xreg, yreg):
    X_train, y_train, X_test, y_test = Xreg.iloc[train_idx], yreg[train_idx], Xreg.iloc[test_idx], yreg[test_idx]
    mlprModel = MLPRegressor(random_state=1, max_iter=500)
    mlprModel.fit(X_train, y_train)
    prediction = mlprModel.predict(X_test)
    score = mlprModel.score(X_test, y_test)
    print(f"Fold: {foldNo}, \tRegression Prediction Accuracy: {score}")
    foldNo += 1