# training.py
import time
from tensorflow.keras.models import load_model

def train_model(model, X_train, y_train, X_test, y_test, epochs=10):
    
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs)
    # return the loss and accuracy of the model

    return history.history['loss'], history.history['accuracy']

def evaluate_model(model, X_test, y_test):
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    return loss,accuracy
