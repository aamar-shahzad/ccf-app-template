import matplotlib.pyplot as plt
import os

def plot_training_testing_results(user_losses_training, user_accuracies_training, user_losses_testing, user_accuracies_testing, results_dir="results"):
    """
    Function to plot training and testing results.
    
    Parameters:
    - user_losses_training: Dictionary with training losses for each user
    - user_accuracies_training: Dictionary with training accuracies for each user
    - user_losses_testing: Dictionary with testing losses for each user
    - user_accuracies_testing: Dictionary with testing accuracies for each user
    - results_dir: Directory to save the plots
    """
    
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

    # Set plot labels and title
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.legend()
    plt.title("Training Loss per Epoch for Each User")
    plt.grid()

    # Save the plot to an image file
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(f"{results_dir}/training_loss.png")
    plt.close()

    print(f"Training loss chart saved to {results_dir}/training_loss.png")

    # Plot the training accuracy for each user
    plt.figure(figsize=(10, 5))
    for user_id, accuracies in merged_user_accuracies_training.items():
        plt.plot(range(1, len(accuracies) + 1), accuracies, label=f"User {user_id} Training Accuracy")

    # Set plot labels and title
    # plt.xlabel("Epoch")
    plt.ylabel("Training Accuracy (%)")
    plt.legend()
    plt.title("Training Accuracy per Epoch for Each User")
    plt.grid()

    # Save the plot to an image file
    plt.savefig(f"{results_dir}/training_accuracy.png")
    plt.close()

    print(f"Training accuracy chart saved to {results_dir}/training_accuracy.png")

    # Plot the testing loss for each user by round
    plt.figure(figsize=(10, 5))
    for user_id, losses in user_losses_testing.items():
        plt.plot(range(1, len(losses) + 1), losses, label=f"User {user_id} Testing Loss")

    # Set plot labels and title
    # plt.xlabel("Round")
    plt.ylabel("Testing Loss")
    plt.legend()
    plt.title("Testing Loss per Round for Each User")
    plt.grid()

    # Save the plot to an image file
    plt.savefig(f"{results_dir}/testing_loss.png")
    plt.close()

    print(f"Testing loss chart saved to {results_dir}/testing_loss.png")

    # Plot the testing accuracy for each user by round
    plt.figure(figsize=(10, 5))
    for user_id, accuracies in user_accuracies_testing.items():
        plt.plot(range(1, len(accuracies) + 1), accuracies, label=f"User {user_id} Testing Accuracy")

    # Set plot labels and title
    # plt.xlabel("Round")
    plt.ylabel("Testing Accuracy (%)")
    plt.legend()
    plt.title("Testing Accuracy per Round for Each User")
    plt.grid()

    # Save the plot to an image file
    plt.savefig(f"{results_dir}/testing_accuracy.png")
    plt.close()

    print(f"Testing accuracy chart saved to {results_dir}/testing_accuracy.png")
