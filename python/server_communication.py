# server_communication.py
import requests
import json
import time
from config import server, cert_paths
import numpy as np
from tensorflow.keras.models import Sequential, load_model
import os
import base64

def download_global_model_weights(user_cert, user_key, model_id):
    try:
        response = requests.get(
            url=f"{server}/model/download_global_weights?model_id={model_id}",
            verify=cert_paths["service_cert"],
            cert=(user_cert, user_key)
        )
        if response.status_code == 200:
            response_data = response.json()
            print("Global model weights downloaded successfully.")
          
            return response_data.get("global_model")
        return None
    except Exception as e:
        print(f"Error downloading global model weights: {e}")
        return None

def upload_model_weights(model_weights_base64, user_cert, user_key, round_no, model_id):
    try:
        payload = {
            "model_id": model_id,
            "weights_json": model_weights_base64,
            "round_no": round_no
        }
        response = requests.post(
            url=f"{server}/model/upload/local_model_weights",
            verify=cert_paths["service_cert"],
            cert=(user_cert, user_key),
            json=payload
        )
        
        if response.status_code == 200:
            print("Model weights uploaded successfully.")
            return True
        else:
            print(f"Failed to upload model weights. Status code: {response.status_code}")
    except Exception as e:
        print(f"Error uploading model weights: {e}")
        return False

def upload_initial_model(model_base64, user_cert, user_key):
    try:
        payload = {
            "global_model": {
                "model_name": "FeedforwardModel",
                "model_data": model_base64
            }
        }
        response = requests.post(
             url=f"{server}/model/intial_model",
            verify=cert_paths["service_cert"],
            cert=(user_cert, user_key),
            json=payload
        )
        print(f"Response status code: {response.status_code}")
        if response.status_code == 200:

            model_data = response.json()
            model_id = model_data.get("model_id")
            model_name = model_data.get("model_name")
            print(f"Initial global model '{model_name}' (ID: {model_id}) uploaded successfully.")
            return model_id
        return None
    except Exception as e:
        print(f"Error uploading initial model: {e}")
        return None
    
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

def download_local_model_weights(user_cert, user_key, model_id, round_no,model):
    try:
        print(f"Downloading local weights for round {round_no}...")
        start_time = time.time()
        response = requests.get(
            url=f"{server}/model/download/local_model_weights?model_id={model_id}&round_no={round_no}",
            verify=cert_paths["service_cert"],
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
                deserialized_weights = deserialize_weights(weights, model= model)
                deserialize_weights_data.append(deserialized_weights)
                
            print("Local weights deserialized successfully.")
            return deserialize_weights_data
        
        else:
            print(f"Failed to download local weights. Status code: {response.status_code}")

        return None
    except Exception as e:
        print(f"An error occurred while downloading local model weights: {e}")
        return None

# check server health
def check_server_health():
    try:
        response = requests.get(
            url=f"{server}/status",
            verify=cert_paths["service_cert"]
        )
        if response.status_code == 200:
            print("Server is healthy.")
            return True
        elif response.status_code == 503:
            print("Server is overloaded.")
            return False
        else:
            print(f"Server returned status code {response.status_code}.")
            return False
    
    except Exception as e:
        print(f"Error checking server health: {e}")
        return False
    


def download_global_model(user_cert, user_key, user_id, model_id=None, save_folder='models'):
    """
    Downloads the global model for the specified user and saves it to a folder with the user ID in the filename.
    
    Parameters:
    - user_cert: Path to the user's certificate
    - user_key: Path to the user's private key
    - user_id: ID of the user
    - model_id: ID of the model to download (default is None)
    - save_folder: Folder to save the downloaded model (default is 'models')
    
    Returns:
    - Loaded Keras model
    """
    try:
        if(model_id is None):
            print("Model ID not provided, retrying...")
            return None
     
        else:
            print("model_id is provided",model_id)
            print(f"Downloading global model for user {user_id} for round ...")
 
 
        response = requests.get(
            url=f"{server}/model/download/global?model_id={model_id}",
            verify=cert_paths["service_cert"],
            cert=(user_cert, user_key)
        )

        if response.status_code == 200:
            print("Global model downloaded successfully.")
            model_data = response.json().get("model_details", {})
            model_base64 = model_data

            if model_base64:
                # Ensure the save folder exists
                os.makedirs(save_folder, exist_ok=True)
                model_path = os.path.join(save_folder, f'user_{user_id}_model.h5')
                
                # Write the model data to a file
                with open(model_path, 'wb') as file:
                    file.write(base64.b64decode(model_base64))
                
                # Load and return the model
                return load_model(model_path)
            else:
                print("Model data not found in response, retrying...")
        if response.status_code == 404:
            print("Model not found, retrying...")
        if response.status_code == 503:
            print("Server is overloaded, retrying...")
        
        if response.status_code == 500:
            print("Internal server error, retrying...")
        
        if response.status_code == 400:
            print("Bad request, retrying...")
        else:
            print(f"Failed to download global model. Status code: {response.status_code}")   
        return None
    except Exception as e:
        print(f"Error downloading global model weights: {e}")
        return None
 

 