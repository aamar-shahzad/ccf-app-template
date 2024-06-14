import requests
import json
import os
import numpy as np
import base64
import time
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

tf.config.threading.set_inter_op_parallelism_threads(4)  # Adjust as needed
tf.config.threading.set_intra_op_parallelism_threads(4)  # Adjust as needed

RSA_SIZE = 2048
DEFAULT_CURVE = "secp384r1"
FAST_CURVE = "secp256r1"
SUPPORTED_CURVES = [DEFAULT_CURVE, FAST_CURVE]
DIGEST_SHA384 = "sha384"
DIGEST_SHA256 = "sha256"
server = "https://127.0.0.1:8000"
num_users = 4
url = server + "/app/api/metrics"
workspace_path = "workspace"

def get_workspace_path(file_name):
    return os.path.join(os.getcwd(), workspace_path, file_name)

service_cert_path = get_workspace_path("sandbox_common/service_cert.pem")
user0_cert_path = get_workspace_path("sandbox_common/user0_cert.pem")
user0_privk_path = get_workspace_path("sandbox_common/user0_privk.pem")
user1_cert_path = get_workspace_path("sandbox_common/user1_cert.pem")
user1_privk_path = get_workspace_path("sandbox_common/user1_privk.pem")
member0_cert_path = get_workspace_path("sandbox_common/member0_cert.pem")
member0_privk_path = get_workspace_path("sandbox_common/member0_privk.pem")

def checkServerHealth():
    try:
        response = requests.get(f'{server}/app/status', verify=service_cert_path)
        if response.status_code == 200:
            print("Server is healthy.")
        else:
            print(f"Server is not healthy. Status code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print("Error making request:", e)

checkServerHealth()

try:
    response = requests.get(url, verify=service_cert_path)

    print("Status Code:", response.status_code)
    print("\nResponse Headers:")
    for header, value in response.headers.items():
        print(f"{header}: {value}")

    print("\nResponse Body:")
    try:
        response_json = response.json()
        print(json.dumps(response_json, indent=4))
    except ValueError:
        print(response.text)

except requests.exceptions.RequestException as e:
    print("Error making request:", e)

def compute_gradients(model, X, y):
    with tf.GradientTape() as tape:
        predictions = model(X, training=True)
        loss = tf.keras.losses.categorical_crossentropy(y, predictions)
    gradients = tape.gradient(loss, model.trainable_weights)
    return gradients

def aggregate_weight(client_weights_list):
    if client_weights_list:
        total_weights = sum(client_weights_list, [])
        aggregated_weights = [weight / len(client_weights_list) for weight in total_weights]
        return aggregated_weights
    else:
        return []

def create_model():
    model = Sequential([
        Conv2D(64, kernel_size=3, activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(32, kernel_size=3, activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def aggregate_weights(model_id, round_no, user_cert, user_key):
    response = requests.put(
        url=f"{server}/app/model/aggregate_weights_local?model_id={model_id}&round_no={round_no}",
        verify=service_cert_path,
        cert=(user_cert, user_key)
    )
    if response.status_code == 200:
        print("Aggregation successful for model:", model_id)
    else:
        raise Exception(f"Failed to aggregate weights. Status code: {response.status_code}")

def train_model(model, X_train, y_train, X_test, y_test, user_id, round_no, epochs=1):
    batch_size = 16  # Example reduced batch size
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size)

    return history.history['loss']

def serialize_model(model):
    model.save('temp_model.keras')
    with open('temp_model.keras', 'rb') as file:
        return base64.b64encode(file.read()).decode('utf-8')

def upload_initial_model(model_base64, user_cert, user_key):
    payload = {
        "global_model": {
            "model_name": "CNNModel",
            "model_data": model_base64
        }
    }
    response = requests.post(
        url=f"{server}/app/model/intial_model",
        verify=service_cert_path,
        cert=(user_cert, user_key),
        json=payload
    )

    if response.status_code == 200:
        model_data = response.json()
        model_id = model_data.get("model_id")
        model_name = model_data.get("model_name")
        print(f"Initial global model '{model_name}' (ID: {model_id}) uploaded successfully.")
        return model_id
    else:
        print(f"Failed to upload initial model. Status code: {response.status_code}")
        return None

def flatten_weights(model):
    flat_weights = []
    for layer in model.layers:
        weights = layer.get_weights()
        if weights:
            flat_weights.append(weights[0].flatten())
    return np.concatenate(flat_weights)

def deserialize_weights(serialized_weights, model):
    flat_weights = np.array(serialized_weights)
    unflattened_weights = unflatten_weights(model, flat_weights)
    return unflattened_weights

def unflatten_weights(model, flat_weights):
    unflattened_weights = []
    index = 0
    for layer in model.layers:
        layer_weights = layer.get_weights()
        if layer_weights:
            weights_shape = layer_weights[0].shape
            layer_weights_unflattened = flat_weights[index:index + np.prod(weights_shape)].reshape(weights_shape)
            unflattened_weights.append(layer_weights_unflattened)
            index += np.prod(weights_shape)
    unflattened_weights = [np.array(arr) for arr in unflattened_weights]
    return unflattened_weights

def serialize_gradients(gradients):
    serialized_gradients = [grad.tolist() for grad in gradients]
    return json.dumps(serialized_gradients)

def deserialize_gradients(serialized_gradients):
    gradients_list = json.loads(serialized_gradients)
    return [np.array(grad) for grad in gradients_list]

def upload_gradients(gradients_base64, user_cert, user_key, round_no, model_id=None):
    print(f"Uploading gradients for Round {round_no}...")
    payload = {
        "model_id": model_id,
        "gradients_json": gradients_base64,
        "round_no": round_no
    }
    response = requests.post(
        url=f"{server}/app/model/upload/local_gradients",
        verify=service_cert_path,
        cert=(user_cert, user_key),
        json=payload
    )
    if response.status_code == 200:
        print(f"Gradients uploaded successfully for Round {round_no}.")
    else:
        raise Exception(f"Failed to upload gradients. Status code: {response.status_code}")

def download_global_gradients(user_cert, user_key, model_id):
    try:
        response = requests.get(
            url=f"{server}/app/model/download_global_gradients?model_id={model_id}",
            verify=service_cert_path,
            cert=(user_cert, user_key)
        )
        if response.status_code == 200:
            print("Global gradients downloaded successfully.")
            response_data = response.json()
            global_gradients_value = response_data.get("global_gradients")
            if global_gradients_value:
                gradients = deserialize_gradients(global_gradients_value)
                return gradients
            else:
                print("Global gradients data not found in response.")
        else:
            print(f"Failed to download global gradients. Status code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print("Error making request:", e)
    return None

def apply_gradients(model, gradients):
    model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

def delete_temp_model_file():
    if os.path.exists('temp_model.keras'):
        os.remove('temp_model.keras')

def plot_loss_curve(round_loss_dict):
    rounds = list(round_loss_dict.keys())
    losses_user0 = [round_loss_dict[round][0] for round in rounds]
    losses_user1 = [round_loss_dict[round][1] for round in rounds]

    plt.plot(rounds, losses_user0, label='User 0 Loss')
    plt.plot(rounds, losses_user1, label='User 1 Loss')

    plt.xlabel('Round Number')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curve for Each User')
    plt.grid()
    plt.show()

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

num_users = 2
user_split_size = X_train.shape[0] // num_users
X_train_user0 = X_train[:user_split_size]
y_train_user0 = y_train[:user_split_size]
X_train_user1 = X_train[user_split_size:user_split_size*2]
y_train_user1 = y_train[user_split_size:user_split_size*2]

global_model = create_model()

serialized_model = serialize_model(global_model)
initial_model_id = upload_initial_model(serialized_model, user0_cert_path, user0_privk_path)

num_rounds = 5
round_loss_dict = {}

local_model_user0 = global_model
local_model_user1 = global_model

for round_no in range(1, num_rounds + 1):
    round_loss_dict[round_no] = {}
    for user_id in range(2):
        X_train_user = X_train_user0 if user_id == 0 else X_train_user1
        y_train_user = y_train_user0 if user_id == 0 else y_train_user1
        gradients = compute_gradients(local_model_user0 if user_id == 0 else local_model_user1, X_train_user, y_train_user)
        serialized_gradients = serialize_gradients(gradients)
        print(f"User {user_id} - Round {round_no} - Gradients Length: {len(serialized_gradients)}")
        upload_gradients(serialized_gradients, user0_cert_path if user_id == 0 else user1_cert_path, user0_privk_path if user_id == 0 else user1_privk_path, round_no, model_id=initial_model_id)
    
    global_gradients = download_global_gradients(user0_cert_path, user0_privk_path, model_id=initial_model_id)
    if global_gradients:
        apply_gradients(local_model_user0, global_gradients)
        apply_gradients(local_model_user1, global_gradients)
    else:
        print("No global gradients received for this round.")

print("Federated Learning Process Completed.")
delete_temp_model_file()
plot_loss_curve(round_loss_dict)
