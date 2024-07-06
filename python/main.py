import os
import random
import time
import json
import pandas as pd
import matplotlib.pyplot as plt
from config import cert_paths
from data_processing import load_and_preprocess_mnist, split_data
from model_handling import create_lenet5_model_with_regularization, serialize_model, deserialize_model
from server_communication import download_local_model_weights, serialize_weights, upload_model_weights, upload_initial_model, check_server_health,download_global_model
from training import train_model, evaluate_model
from aggregation import aggregate_weights
from plotting import plot_training_testing_results

# Function to clear the contents of the results directory
def clear_results_directory(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                os.rmdir(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

# Create results directory if it doesn't exist
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)
# clear_results_directory(results_dir)

check_server_health()

(X_train, y_train), (X_test, y_test) = load_and_preprocess_mnist()
num_users = 5
X_train_users, y_train_users = split_data(X_train, y_train, num_users)
global_model = create_lenet5_model_with_regularization()

model_base64 = serialize_model(global_model)
initial_model_id = upload_initial_model(model_base64, cert_paths["user0_cert"], cert_paths["user0_privk"])
if initial_model_id is not None:
    print("Initial model uploaded successfully")
else:
    raise Exception("Initial model upload failed")
num_rounds = 20
epoch = 1
num_participating_users = []
user_losses_training = {i: [] for i in range(num_users)}
user_accuracies_training = {i: [] for i in range(num_users)}
user_losses_testing = {i: [] for i in range(num_users)}
user_accuracies_testing = {i: [] for i in range(num_users)}
time_records = []

local_model_copies = []
for user_id in range(num_users):
    try:
        local_modelCopy = download_global_model(cert_paths[f"user{user_id}_cert"], cert_paths[f"user{user_id}_privk"], user_id=user_id, model_id=initial_model_id)
        if local_modelCopy is not None:
            print(f"Global weights downloaded successfully for user {user_id}")
            local_model_copies.append(local_modelCopy)
        else:
            raise Exception("Global weights download failed")
    except Exception as e:
        print(f"Error creating local model for user {user_id}: {e}")
    finally:
        time.sleep(1)

for round_no in range(1, num_rounds + 1):
    round_start_time = time.time()
    participating_users = random.randint(1, num_users)
    num_participating_users.append(participating_users)
    local_weights = []

    for user_id in range(num_users):
        print(f"User {user_id} is participating in round {round_no}")
        X_train_user = X_train_users[user_id]
        y_train_user = y_train_users[user_id]
        local_model_user = local_model_copies[user_id]
        train_loss, train_accuracy = train_model(local_model_user, X_train_user, y_train_user, X_test, y_test, epochs=epoch)
        test_loss, test_accuracy = evaluate_model(local_model_user, X_test, y_test)
        user_losses_training[user_id].append(train_loss)
        user_accuracies_training[user_id].append(train_accuracy)
        user_losses_testing[user_id].append(test_loss)
        user_accuracies_testing[user_id].append(test_accuracy)

        local_serialize_weights = serialize_weights(local_model_user)
        if local_serialize_weights is not None:
            local_weights.append(local_serialize_weights)
            upload_model_weights(local_serialize_weights, cert_paths[f"user{user_id}_cert"], cert_paths[f"user{user_id}_privk"], round_no, initial_model_id)
            print(f"User {user_id} weights uploaded successfully")

    local_weights_from_server = download_local_model_weights(cert_paths[f"user{user_id}_cert"], cert_paths[f"user{user_id}_privk"], model_id=initial_model_id, round_no=round_no, model=global_model)
    if local_weights_from_server is not None:
        print("Global weights downloaded successfully")

        aggregated_weights = aggregate_weights(local_weights_from_server)
        global_model.set_weights(aggregated_weights)
        for user_id in range(num_users):
            local_model_user = local_model_copies[user_id]
            local_model_user.set_weights(aggregated_weights)
            local_model_copies[user_id] = local_model_user

        round_end_time = time.time()
        time_records.append({
            "round_no": round_no,
            "participants": participating_users,
            "latency": (round_end_time - round_start_time) * 1000
        })

        print(f"Round {round_no} completed in {round_end_time - round_start_time} seconds")

plot_training_testing_results(user_losses_training, user_accuracies_training, user_losses_testing, user_accuracies_testing, time_records, results_dir)