# data_processing.py
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

def load_and_preprocess_mnist():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32') / 255.0
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32') / 255.0
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    return (X_train, y_train), (X_test, y_test)

def split_data(X_train, y_train, num_users):
    split_indices = np.array_split(np.arange(len(X_train)), num_users)
    X_train_users = [X_train[indices] for indices in split_indices]
    y_train_users = [y_train[indices] for indices in split_indices]
    return X_train_users, y_train_users
