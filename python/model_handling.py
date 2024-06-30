# model_handling.py
import base64
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, AveragePooling2D, Flatten, Dropout

def create_lenet5_model_with_regularization():
    model = Sequential([
        Conv2D(6, kernel_size=(5, 5), strides=(1, 1), activation='relu', input_shape=(28, 28, 1)),
        AveragePooling2D(pool_size=(2, 2), strides=(2, 2)),
        Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation='relu'),
        AveragePooling2D(pool_size=(2, 2), strides=(2, 2)),
        Flatten(),
        Dense(120, activation='relu', kernel_regularizer='l2'),
        Dropout(0.5),
        Dense(84, activation='relu', kernel_regularizer='l2'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def serialize_model(model):
    model.save('temp_model.h5')
    with open('temp_model.h5', 'rb') as file:
        return base64.b64encode(file.read()).decode('utf-8')

def deserialize_model(model_base64):
    with open('temp_model.h5', 'wb') as file:
        file.write(base64.b64decode(model_base64))
    return load_model('temp_model.h5')
