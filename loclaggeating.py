import numpy as np
import requests
import json
import base64
import time
import matplotlib.pyplot as plt
import random
import os
import numpy as np
import requests
import json
import base64
import time
import matplotlib.pyplot as plt
import random
import os
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense, Conv2D, AveragePooling2D, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dropout


server = "https://127.0.0.1:8000"  # Replace with your server URL

def download_global_model_weights(user_cert, user_key, model_id, local_model):
    try:
        print("Downloading global weights for aggregation...")
        start_time = time.time()
        response = requests.get(
            url=f"{server}/app/model/download_gloabl_weights?model_id={model_id}",
            verify="./workspace/sandbox_common/service_cert.pem",
            cert=(user_cert, user_key)
        )
        end_time = time.time()
        latency = end_time - start_time
        print(f"Latency for download_global_model_weights: {latency} seconds")

        if response.status_code == 200:
            print("Global weights downloaded successfully.")
            response_data = json.loads(response.text)
            global_model_value = response_data.get("global_model")
            if global_model_value:
                unflattened_weights = deserialize_weights(global_model_value, local_model)
                print("Global model weights downloaded and unflattened successfully.")
                return unflattened_weights
            else:
                print("Global model data not found in response.")
        else:
            print(f"Failed to download global weights. Status code: {response.status_code}")

        return None
    except Exception as e:
        print(f"An error occurred while downloading global model weights: {e}")
        return None

def aggregate_weight(client_weights_list):
    print("Aggregating weights...")
    if client_weights_list:
        # Initialize with zeros
        aggregated_weights = [np.zeros_like(weights) for weights in client_weights_list[0]]

        # Sum weights from all clients
        for client_weights in client_weights_list:
            for i in range(len(aggregated_weights)):
                aggregated_weights[i] += client_weights[i]

        # Average weights
        aggregated_weights = [weights / len(client_weights_list) for weights in aggregated_weights]
        print("Weights aggregated successfully.")
        return aggregated_weights
        
       
        
       
    else:
        print("No weights to aggregate.")
        aggregated_weights = None
     
def create_complex_model():
    print("Initializing the complex global model...")
    model = Sequential([
        Flatten(input_shape=(28, 28, 1)),
        Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def create_lenet5_model_with_regularization():
    print("Initializing the LeNet-5 model with regularization...")
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
def create_model():
    print("Initializing the global model...")
    model = Sequential([
        Flatten(input_shape=(28, 28, 1)),  # Flatten instead of Conv2D
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def aggregate_weights(model_id, round_no, user_cert, user_key):
    try:
        print("Aggregating weights for model:", model_id)
        start_time = time.time()
        response = requests.put(
            url=f"{server}/app/model/aggregate_weights_local?model_id={model_id}&&round_no={round_no}",
            verify="./workspace/sandbox_common/service_cert.pem",
            cert=(user_cert, user_key)
        )
        end_time = time.time()
        latency = end_time - start_time
        print(f"Latency for aggregate_weights: {latency} seconds")

        if response.status_code == 200:
            print("Aggregation successful for model:", model_id)
        else:
            raise Exception(f"Failed to aggregate weights. Status code: {response.status_code}")
    except Exception as e:
        print(f"An error occurred while aggregating weights: {e}")

def train_model(model, X_train, y_train, X_test, y_test, user_id, round_no, epochs=1):
    try:
        print(f"Training model for User {user_id}, Round {round_no}...")
        start_time = time.time()
        history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs)
        end_time = time.time()
        latency = end_time - start_time
        print(f"Latency for train_model: {latency} seconds")

        return history.history['loss']
    except Exception as e:
        print(f"An error occurred while training the model: {e}")
        return []

def serialize_model(model):
    print("Serializing the model...")
    model.save('temp_model.h5')
    with open('temp_model.h5', 'rb') as file:
        return base64.b64encode(file.read()).decode('utf-8')
def print_time_table(time_records):
    print("\nTime Records Table:")
    print("{:<10} {:<20} {:<15} {:<15} {:<15}".format("Round", "Participants", "Accuracy", "Loss", "Latency (ms)"))
    print("="*80)
    for record in time_records:
        if 'user_id' in record:
            print("{:<10} {:<20} {:<15} {:<15} {:<15}".format(record["round_no"],
                                                                 f"{record['participants']} / {num_users}",
                                                                 f"{record['accuracy']:.4f}",
                                                                 f"{record['loss']:.4f}",
                                                                 f"{record['latency']:.4f}"))
        else:
            print("{:<10} {:<20} {:<15}".format(record["round_no"],
                                                 f"{record['participants']} / {num_users}",
                                                 f"{record['latency']:.4f}"))

def upload_initial_model(model_base64, user_cert, user_key):
    try:
        print("Uploading initial global model...")
        payload = {
            "global_model": {
                "model_name": "FeedforwardModel",
                "model_data": model_base64
            }
        }
        response = requests.post(
            url=f"{server}/app/model/intial_model",
            verify="./workspace/sandbox_common/service_cert.pem",
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
    except Exception as e:
        print(f"An error occurred while uploading initial model: {e}")
        return None

def flatten_weights(model_weights):
    flat_weights = []
    for weights in model_weights:
        flat_weights.append(weights.flatten())
    return np.concatenate(flat_weights)

def serialize_weights(model):
    print("Serializing weights...")
    model_weights = model.get_weights()
    serialized_weights = [weights.tolist() for weights in model_weights]
    return serialized_weights

def deserialize_weights(serialized_weights, model):
    print("Deserializing weights...")
    deserialized_weights = [np.array(weights) for weights in serialized_weights]
    model.set_weights(deserialized_weights)
    return deserialized_weights


def upload_model_weights(model_weights_base64, user_cert, user_key, round_no, model_id=None):
    try:
        print(f"Uploading model weights for Round {round_no}...")
        start_time = time.time()
        payload = {
            "model_id": model_id,
            "weights_json": model_weights_base64,
            "round_no": round_no
        }
        response = requests.post(
            url=f"{server}/app/model/upload/local_model_weights",
            verify="./workspace/sandbox_common/service_cert.pem",
            cert=(user_cert, user_key),
            json=payload
        )
        end_time = time.time()
        latency = end_time - start_time
        print(f"Latency for upload_model_weights: {latency} seconds")

        print(response.text)
        if response:
            print(f"Model weights uploaded successfully for Round {round_no}.")
        else:
            raise Exception(f"Failed to upload model weights. Status code: {response.status_code}")
    except Exception as e:
        print(f"An error occurred while uploading model weights: {e}")

def download_model(user_cert, user_key, user_id, round_no, max_retries=5, model_id=None):
    attempts = 0
    while attempts < max_retries:
        try:
            print(f"Attempting to download model for User {user_id}, Round {round_no}, Attempt {attempts + 1}...")
            start_time = time.time()
            response = requests.get(
                url=f"{server}/app/model/download/global?model_id={model_id}",
                verify="./workspace/sandbox_common/service_cert.pem",
                cert=(user_cert, user_key)
            )
            end_time = time.time()
            latency = end_time - start_time
            print(f"Latency for download_model: {latency} seconds")

            if response.status_code == 200:
                model_data = response.json().get("model_details", {})
                model_base64 = model_data

                if model_base64:
                    with open('temp_model.h5', 'wb') as file:
                        file.write(base64.b64decode(model_base64))
                    return load_model('temp_model.h5')
                else:
                    print("Model data not found in response, retrying...")
            else:
                print(f"Failed to download model. Status code: {response.status_code}, retrying...")

            time.sleep(2)
            attempts += 1
        except Exception as e:
            print(f"An error occurred during download_model: {e}")

    raise Exception("Failed to download model after maximum retries.")


def download_local_model_weights(user_cert, user_key, model_id, round_no):
    try:
        print(f"Downloading local weights for round {round_no}...")
        start_time = time.time()
        response = requests.get(
            url=f"{server}/app/model/download/local_model_weights?model_id={model_id}&round_no={round_no}",
            verify="./workspace/sandbox_common/service_cert.pem",
            cert=(user_cert, user_key)
        )
        end_time = time.time()
        latency = end_time - start_time
        print(f"Latency for download_local_model_weights: {latency} seconds")

        if response.status_code == 200:
            print("Local weights downloaded successfully.")
            local_weights_data = json.loads(response.text)
            # iteravte the local weights data and deserialize the weights 
            deserialize_weights_data = []
            for  weights in local_weights_data:
                deserialized_weights = deserialize_weights(weights, global_model)
                deserialize_weights_data.append(deserialized_weights)
                
            print("Local weights deserialized successfully.")
            return deserialize_weights_data
            
        

           
          
            
           
            
           
           
            
            

           
          
           
        
        else:
            print(f"Failed to download local weights. Status code: {response.status_code}")

        return None
    except Exception as e:
        print(f"An error occurred while downloading local model weights: {e}")
        return None


def delete_temp_model_file():
    temp_model_path = 'temp_model.h5'
    if os.path.exists(temp_model_path):
        os.remove(temp_model_path)
        print(f"Deleted temporary model file: {temp_model_path}")
    else:
        print(f"No temporary model file found at: {temp_model_path}")

def plot_loss_and_accuracy(user_losses, user_accuracies, num_participating_users, num_rounds):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))

    for user_id, losses in user_losses.items():
        rounds = list(range(1, num_rounds + 1))  # Use num_rounds for x-axis
        ax1.plot(rounds, losses[:num_rounds], label=f'User {user_id}')

    ax1.set_title('Model Loss per Training Round')
    ax1.set_ylabel('Loss')
    ax1.set_xlabel('Round')
    ax1.legend()

    for user_id, accuracies in user_accuracies.items():
        rounds = list(range(1, num_rounds + 1))  # Use num_rounds for x-axis
        accuracies_to_plot = accuracies[:num_rounds] + [None] * (num_rounds - len(accuracies))  # Pad with None if needed
        ax2.plot(rounds, accuracies_to_plot, label=f'User {user_id}')

    ax2.set_title('Model Accuracy per Training Round')
    ax2.set_ylabel('Accuracy')
    ax2.set_xlabel('Round')
    ax2.legend()

    rounds = list(range(1, num_rounds + 1))  # Use num_rounds for x-axis
    ax3.plot(rounds, num_participating_users[:num_rounds], marker='o', linestyle='-', color='b')
    ax3.set_title('Number of Participating Users per Round')
    ax3.set_ylabel('Number of Users')
    ax3.set_xlabel('Round')

    plt.tight_layout()
    plt.show()


def evaluate_model(model, X_test, y_test):
    """Evaluate the model on the test set and return accuracy."""
    _, accuracy = model.evaluate(X_test, y_test, verbose=0)
    return accuracy

# Load and preprocess Fashion MNIST dataset
# Load and preprocess MNIST dataset
print("Loading and preprocessing MNIST dataset...")
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32') / 255.0
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32') / 255.0
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
delete_temp_model_file()


# Number of users
num_users = 2

# Split the dataset for multiple users
print(f"Splitting dataset for {num_users} users...")
split_indices = np.array_split(np.arange(len(X_train)), num_users)
X_train_users = [X_train[indices] for indices in split_indices]
y_train_users = [y_train[indices] for indices in split_indices]

# Initialize and train the global model
# Initialize and train the global model (using LeNet-5)
global_model = create_lenet5_model_with_regularization()
train_model(global_model, X_train, y_train, X_test, y_test, user_id=0, round_no=0)

# Rest of the code remains the same

# Serialize and upload initial global model
model_base64 = serialize_model(global_model)
initial_model_id = upload_initial_model(model_base64, "./workspace/sandbox_common/user0_cert.pem", "./workspace/sandbox_common/user0_privk.pem")

if initial_model_id is None:
    raise Exception("Failed to upload the initial global model. Stopping the process.")

# Number of rounds and users
num_rounds = 3
num_participating_users = []

user_losses = {i: [] for i in range(num_users)}
user_accuracies = {i: [] for i in range(num_users)}

time_records = []

for round_no in range(1, num_rounds + 1):
    round_start_time = time.time()
    participating_users = random.randint(1, num_users)
    num_participating_users.append(participating_users)

    for user_id in range(num_users):
        if random.random() < 0.8:
            user_start_time = time.time()
            X_train_user = X_train_users[user_id]
            y_train_user = y_train_users[user_id]
            loss = train_model(global_model, X_train_user, y_train_user, X_test, y_test, user_id, round_no)
            accuracy = evaluate_model(global_model, X_test, y_test)
            user_losses[user_id].extend(loss)
            user_accuracies[user_id].append(accuracy)

            local_serialize_weights = serialize_weights(global_model)
            if local_serialize_weights:
                upload_model_weights(
                    local_serialize_weights,
                    f"./workspace/sandbox_common/user{user_id}_cert.pem",
                    f"./workspace/sandbox_common/user{user_id}_privk.pem",
                    round_no,
                    model_id=initial_model_id
                )

            user_end_time = time.time()
            user_latency = (user_end_time - user_start_time) * 1000  # Convert to milliseconds
            time_records.append({
                "round_no": round_no,
                "user_id": user_id,
                "participants": participating_users,
                "accuracy": accuracy,
                "loss": loss[-1],
                "latency": user_latency
            })

    local_weights = download_local_model_weights(
        "./workspace/sandbox_common/user0_cert.pem", 
        "./workspace/sandbox_common/user0_privk.pem",
        model_id=initial_model_id,
        round_no=round_no
    )

    if local_weights:
        aggregated_weights = aggregate_weight(local_weights)
        global_model.set_weights(aggregated_weights)
    else:
        print("Mismatch in the number of weights. Check aggregation logic.")

    round_end_time = time.time()
    round_latency = (round_end_time - round_start_time) * 1000  # Convert to milliseconds
    time_records.append({
        "round_no": round_no,
        "participants": participating_users,
        "latency": round_latency
    })


