# aggregation.py
import numpy as np

def aggregate_weights(client_weights_list):
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
        