import pandas     as pd                           # DataFrame Library
import tensorflow as tf                           # Tensorflow, library to develop and train ML models
import matplotlib.pyplot as plt                   # Plotting Library
from Models.myanfis import ANFIS                  # ANFIS model from: https://github.com/gregorLen/AnfisTensorflow2.0
from Models.myanfis import fis_parameters         # Model Configuration class
from sklearn.utils           import shuffle       # For shuffling the dataset
from sklearn.decomposition   import PCA           # For dimensionality reduction
from sklearn.model_selection import KFold         # k-fold Cross
from sklearn.preprocessing   import MinMaxScaler  # For converting negative ranges into [0,1] range

# Read the dataset, shuffle it, get the first 1000 sample of it
df = shuffle(pd.read_csv("winequality-red.csv"))[0:1000]
df.head()

# Separate Label and Features within the dataset, then reduce the dimensionality of feature set
Y = df['quality']
pca = PCA(n_components=2)
X_pca = pca.fit_transform(df.drop('quality', axis=1))
minMaxScaler = MinMaxScaler()
X = minMaxScaler.fit_transform(X_pca)
print(X)

# Configuration object for ANFIS model
param = fis_parameters(
    n_input = 2,
    n_memb = 2,
    batch_size = 5,
    memb_func = 'gaussian',
    optimizer = 'sgd',
    loss = tf.keras.losses.MeanAbsoluteError(),
    n_epochs = 8
)

# Create model for 2-k Cross Separation
kfold = KFold(n_splits=2)
histories = []

# Train and fit ANFIS models for each train and test sets within each fold
for train_index,test_index in kfold.split(X):
    X_train,X_test = X[train_index],X[test_index]
    Y_train,Y_test = Y.iloc[train_index],Y.iloc[test_index]

    fis = ANFIS(
                n_input=param.n_input,
                n_memb=param.n_memb,
                batch_size=param.batch_size,
                memb_func=param.memb_func,
                name= 'firstAnfis'
                )

    fis.model.compile(
                        optimizer=param.optimizer,
                        loss=param.loss,
                        metrics=['mae']
                     )

    history = fis.fit(
                        X_train,Y_train,
                        epochs=param.n_epochs,
                        batch_size=param.batch_size,
                        validation_data=(X_test,Y_test)
                    )

    histories.append(history)


# Plot the result
fis.plotmfs()
pd.DataFrame(histories[0].history).plot()
pd.DataFrame(histories[1].history).plot()
plt.show()
