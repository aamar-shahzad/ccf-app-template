import random
import time
import json
import pandas as pd
import matplotlib.pyplot as plt
from config import cert_paths
from data_processing import load_and_preprocess_mnist, split_data
from model_handling import create_lenet5_model_with_regularization, serialize_model, deserialize_model
from server_communication import download_local_model_weights, serialize_weights, upload_model_weights, upload_initial_model, check_server_health
from training import train_model, evaluate_model
from aggregation import aggregate_weights

check_server_health()

(X_train, y_train), (X_test, y_test) = load_and_preprocess_mnist()
num_users = 2
X_train_users, y_train_users = split_data(X_train, y_train, num_users)
global_model = create_lenet5_model_with_regularization()

# Train initial global model
# train_model(global_model, X_train, y_train, X_test, y_test)
model_base64 = serialize_model(global_model)
initial_model_id = upload_initial_model(model_base64, cert_paths["user0_cert"], cert_paths["user0_privk"])
if initial_model_id is not None:
    print("Initial model uploaded successfully")
else:
    raise Exception("Initial model upload failed")

num_rounds = 5
num_participating_users = []
user_losses_training = {i: [] for i in range(num_users)}
user_accuracies_testing = {i: [] for i in range(num_users)}
time_records = []

for round_no in range(1, num_rounds + 1):
    round_start_time = time.time()
    participating_users = random.randint(1, num_users)
    num_participating_users.append(participating_users)
    local_weights = []

    for user_id in range(num_users):
        print(f"User {user_id} is participating in round {round_no}")
        X_train_user = X_train_users[user_id]
        y_train_user = y_train_users[user_id]
        train_loss, train_accuracy = train_model(global_model, X_train_user, y_train_user, X_test, y_test)
        test_loss, test_accuracy = evaluate_model(global_model, X_test, y_test)
        # add both loss and accuracy to the respective lists
        user_losses_training[user_id].append(train_loss)
        user_losses_training[user_id].append(train_accuracy)
        user_accuracies_testing[user_id].append(test_loss)
        user_accuracies_testing[user_id].append(test_accuracy)

        local_serialize_weights = serialize_weights(global_model)
        if local_serialize_weights is not None:
            local_weights.append(local_serialize_weights)
            upload_model_weights(local_serialize_weights, cert_paths[f"user{user_id}_cert"], cert_paths[f"user{user_id}_privk"], round_no, initial_model_id)
            print(f"User {user_id} weights uploaded successfully")

    local_weights_from_server = download_local_model_weights(cert_paths[f"user{user_id}_cert"], cert_paths[f"user{user_id}_privk"], model_id=initial_model_id, round_no=round_no, model=global_model)
    if local_weights_from_server is not None:
        print("Global weights downloaded successfully")

        aggregated_weights = aggregate_weights(local_weights_from_server)
        global_model.set_weights(aggregated_weights)

        round_end_time = time.time()
        time_records.append({
            "round_no": round_no,
            "participants": participating_users,
            "latency": (round_end_time - round_start_time) * 1000
        })

# Function to save combined plots for each user
def save_combined_plots(user_losses_training, user_accuracies_testing):
    plt.figure(figsize=(14, 7))

    # Plot training loss for all users
    plt.subplot(1, 2, 1)
    for user_id in user_losses_training.keys():
        plt.plot(user_losses_training[user_id][::2], label=f'User {user_id} Loss')  # Even indices are losses
    plt.title('Training Loss for All Users')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot training accuracy for all users
    plt.subplot(1, 2, 2)
    for user_id in user_losses_training.keys():
        plt.plot(user_losses_training[user_id][1::2], label=f'User {user_id} Accuracy')  # Odd indices are accuracies
    plt.title('Training Accuracy for All Users')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig('combined_training_plots.png')
    plt.close()

    # Plot testing accuracy for each user
    for user_id in user_accuracies_testing.keys():
        plt.figure()
        plt.plot(user_accuracies_testing[user_id][1::2], label=f'User {user_id} Testing Accuracy')  # Odd indices are accuracies
        plt.title(f'User {user_id} Testing Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.savefig(f'user_{user_id}_testing.png')
        plt.close()

# Call the function with the user data
save_combined_plots(user_losses_training, user_accuracies_testing)
