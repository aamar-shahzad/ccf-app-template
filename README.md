# Introduction
The CCF Confidential Consortium Framework is a blockchain that utilizes Trusted Execution Environments (TEEs) for secure and multiparty computation. It features a built-in map for data storage and exposes a RESTful endpoint, allowing it to be used as a secure service. The framework employs a Distributed Ledger Technology (DLT) solution for decentralized federated learning.

## Getting Started
The repository includes a development container that creates an isolated environment for installing and running the application.

### Steps:

1. **Clone the repository:**
    ```bash
    git clone https://github.com/aamar-shahzad/ccf-app-template.git
    ```

2. **Open the project in Visual Studio Code:**
    - It will prompt you to open the code in the development container.
    - Choose "Open in dev-container," and it will start installing the required packages.

3. **Run the development container:**
    ```bash
    make run-virtual
    ```
    - This command starts the network and exposes a default server address with a port number.

4. **Main Working Files:**
    - `test.ipynb`: Contains client-side code that consumes the CCF REST API.
    - `/cpp/app.cpp`: Source file of the CCF REST API. It defines endpoints and handlers.

    **Handler Actions:**
    1. Upload initial models to the CCF network using the REST API. This endpoint returns a `model_id` for use in subsequent steps.
    2. Download initial models for each client, initializing them with the same model after data distribution. Pass the previously generated `model_id` as a parameter.
    3. Train the local model and upload local weights to the CCF network with the specified `model_id` for each local client. Assume users are training nodes, and they can call these endpoints by passing their certificate and private key for authentication.
    4. Each local training node uploads local model weights with `round_no` and `model_id`.
    5. After calling the aggregation function, compute `FedAvg` from submitted local model updates to update the `global_model_weights`. Use these weights in the next round for each client.
    6. Download the global weights and update the local model weights at the end of each round.
    7. Continue this process until the model converges at the desired threshold limit.
