
    
import matplotlib.pyplot as plt
import os
from datetime import datetime

def plot_training_testing_results(user_losses_training, user_accuracies_training, user_losses_testing, user_accuracies_testing, time_records, results_base_dir="results"):
    """
    Function to plot training and testing results.
    
    Parameters:
    - user_losses_training: Dictionary with training losses for each user
    - user_accuracies_training: Dictionary with training accuracies for each user
    - user_losses_testing: Dictionary with testing losses for each user
    - user_accuracies_testing: Dictionary with testing accuracies for each user
    - time_records: List of dictionaries with latency data for each round
    - results_base_dir: Base directory to save the plots
    """
    
    # Create a unique directory based on the current date and time
    timestamp = datetime.now().strftime('%b-%d-%Y_%I-%M-%S_%p')
    results_dir = os.path.join(results_base_dir, timestamp)
    os.makedirs(results_dir, exist_ok=True)

    # Save all other information in a text file in the directory
    with open(os.path.join(results_dir, 'info.txt'), 'w') as f:
        f.write(f"Results Directory: {results_dir}\n")
        f.write(f"Number of Users: {len(user_losses_training)}\n")
        f.write(f"Number of Rounds: {len(next(iter(user_losses_training.values())))}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Results saved at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Merge the training losses and accuracies for each user
    merged_user_losses_training = {
        user_id: [loss for round_losses in losses for loss in round_losses]
        for user_id, losses in user_losses_training.items()
    }

    merged_user_accuracies_training = {
        user_id: [accuracy * 100 for round_accuracies in accuracies for accuracy in round_accuracies]
        for user_id, accuracies in user_accuracies_training.items()
    }

    # Plot the training loss for each user
    plt.figure(figsize=(10, 5))
    for user_id, losses in merged_user_losses_training.items():
        plt.plot(range(1, len(losses) + 1), losses, label=f"User {user_id} Training Loss")

    plt.xlabel("Round")
    plt.ylabel("Training Loss")
    plt.legend()
    plt.title("Training Loss per Round for Each User")
    plt.grid()
    plt.savefig(os.path.join(results_dir, 'training_loss.png'))
    plt.close()
    print(f"Training loss chart saved to {results_dir}/training_loss.png")

    # Plot the training accuracy for each user
    plt.figure(figsize=(10, 5))
    for user_id, accuracies in merged_user_accuracies_training.items():
        plt.plot(range(1, len(accuracies) + 1), accuracies, label=f"User {user_id} Training Accuracy")

    plt.xlabel("Round")
    plt.ylabel("Training Accuracy (%)")
    plt.legend()
    plt.title("Training Accuracy per Round for Each User")
    plt.grid()
    plt.savefig(os.path.join(results_dir, 'training_accuracy.png'))
    plt.close()
    print(f"Training accuracy chart saved to {results_dir}/training_accuracy.png")

    # Plot the testing loss for each user by round
    plt.figure(figsize=(10, 5))
    for user_id, losses in user_losses_testing.items():
        plt.plot(range(1, len(losses) + 1), losses, label=f"User {user_id} Testing Loss")

    plt.xlabel("Round")
    plt.ylabel("Testing Loss")
    plt.legend()
    plt.title("Testing Loss per Round for Each User")
    plt.grid()
    plt.savefig(os.path.join(results_dir, 'testing_loss.png'))
    plt.close()
    print(f"Testing loss chart saved to {results_dir}/testing_loss.png")

    # Plot the testing accuracy for each user by round
    plt.figure(figsize=(10, 5))
    for user_id, accuracies in user_accuracies_testing.items():
        plt.plot(range(1, len(accuracies) + 1), accuracies, label=f"User {user_id} Testing Accuracy")

    plt.xlabel("Round")
    plt.ylabel("Testing Accuracy (%)")
    plt.legend()
    plt.title("Testing Accuracy per Round for Each User")
    plt.grid()
    plt.savefig(os.path.join(results_dir, 'testing_accuracy.png'))
    plt.close()
    print(f"Testing accuracy chart saved to {results_dir}/testing_accuracy.png")

    # Plot the latency for each round
    rounds = [record['round_no'] for record in time_records]
    latencies = [record['latency'] for record in time_records]
    
    plt.figure(figsize=(10, 5))
    plt.bar(rounds, latencies, color='skyblue')
    
    plt.xlabel("Round")
    plt.ylabel("Latency (ms)")
    plt.title("Latency per Round")
    plt.grid()
    plt.savefig(os.path.join(results_dir, 'latency_per_round.png'))
    plt.close()
    print(f"Latency chart saved to {results_dir}/latency_per_round.png")
