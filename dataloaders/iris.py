from sklearn import datasets
import numpy as np

iris = datasets.load_iris()

def load_data(train=0.8, shuffle=True, seed=0):
    # train/test split
    n = len(iris.data)
    n_train = int(n * train)
    n_test = n - n_train
    if shuffle:
        indices = np.random.RandomState(seed).permutation(n)
    else:
        indices = np.arange(n)
    X_train = iris.data[indices[:n_train]]
    y_train = iris.target[indices[:n_train]]
    X_test = iris.data[indices[n_train:]]
    y_test = iris.target[indices[n_train:]]
    return X_train, y_train, X_test, y_test
