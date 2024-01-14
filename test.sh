#!/bin/bash

# Define colors
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Number of times to run the loop
num_runs=5

for ((i=1; i<=$num_runs; i++)); do
  echo "Running iteration $i"

  # Generate random values for parameters
  model_id=$((1 + RANDOM % 1000))
  param1=$(awk -v min=0.1 -v max=1 -v scale=100 'BEGIN{srand(); print int(min*scale+rand()*(max-min)*scale)/scale}')
  param2=$(awk -v min=0.1 -v max=1 -v scale=100 'BEGIN{srand(); print int(min*scale+rand()*(max-min)*scale)/scale}')

  # Send the POST request to create a model with random values
  response=$(curl -X POST https://127.0.0.1:8000/app/model \
    --cacert ./workspace/sandbox_common/service_cert.pem \
    -H "Content-Type: application/json" \
    -d "{\"msg\": {\"model_id\":$model_id,\"modelName\":\"CNN\",\"weights\":{\"param1\":$param1,\"param2\":$param2}}}")

  # Use the plain response as model_id
  model_id="$response"

  # Check if model_id is not empty
  if [ -n "$model_id" ]; then
    # Send the GET request to retrieve the model using the model_id
    result=$(curl -X GET "https://127.0.0.1:8000/app/model?model_id=$model_id" \
      --cacert ./workspace/sandbox_common/service_cert.pem)
    
    # Print test result in green for success and red for error
    if [[ "$result" == *"Successful"* ]]; then
      echo -e "${GREEN}Test successful${NC}"
    else
      echo -e "${RED}Test failed: $result${NC}"
    fi
  else
    echo -e "${RED}Failed to retrieve model_id from the response.${NC}"
  fi

  # Sleep for a short duration between iterations if needed
  sleep 1
done
