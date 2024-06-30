# training.py
import time
from tensorflow.keras.models import load_model

def train_model(model, X_train, y_train, X_test, y_test, epochs=1):
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs)
    return history.history['loss']

def evaluate_model(model, X_test, y_test):
    _, accuracy = model.evaluate(X_test, y_test, verbose=0)
    return accuracy
