#!/bin/bash

# Define colors
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Number of times to run the loop
num_runs=200

# Initialize variables for total time, total successful requests, and total failed requests
total_time=0
total_success=0
total_fail=0

# Initialize variables for calculating requests per second
start_timestamp=$(date +%s)
requests_per_second=0

# Initialize variables for the progress bar
progress_bar=""

# Function to print a progress bar
print_progress() {
  local percent=$1
  local fill=$(printf "%0.s=" $(seq 1 $percent))
  local empty=$(printf "%0.s " $(seq $((100 - percent))))
  progress_bar="\r[$fill$empty] $percent%"
  echo -ne "$progress_bar"
}

# Loop through the specified number of runs
for ((i=1; i<=$num_runs; i++)); do
  start_time=$(date +%s%N)

  # Generate random values for parameters
  model_id=$((1 + RANDOM % 1000))
  param1=$(awk -v min=0.1 -v max=1 -v scale=100 'BEGIN{srand(); print int(min*scale+rand()*(max-min)*scale)/scale}')
  param2=$(awk -v min=0.1 -v max=1 -v scale=100 'BEGIN{srand(); print int(min*scale+rand()*(max-min)*scale)/scale}')

  # Send the POST request to create a model with random values
  response=$(curl -X POST https://127.0.0.1:8000/app/model \
    --cacert ./workspace/sandbox_common/service_cert.pem \
    -H "Content-Type: application/json" \
    -d "{\"msg\": \"{\\\"model_id\\\":$model_id,\\\"modelName\\\":\\\"CNN\\\",\\\"weights\\\":{\\\"param1\\\":$param1,\\\"param2\\\":$param2}}\"}")

  end_time=$(date +%s%N)

  # Calculate the time taken for the request in milliseconds
  time_taken=$(( (end_time - start_time) / 1000000 ))

  # Add the time taken to the total time
  total_time=$((total_time + time_taken))

  # Use the plain response as model_id
  model_id="$response"

  # Check if model_id is not empty
  if [ -n "$model_id" ] && [ "$model_id" != "null" ]; then
    # Send the GET request to retrieve the model using the model_id
    result=$(curl -s -o /dev/null -w "%{http_code}" -X GET "https://127.0.0.1:8000/app/model?model_id=$model_id" \
      --cacert ./workspace/sandbox_common/service_cert.pem)
    
    # Print test result in green for success and red for failure
    if [ "$result" -eq 200 ]; then
      total_success=$((total_success + 1))
    else
      total_fail=$((total_fail + 1))
    fi
  else
    total_fail=$((total_fail + 1))
  fi

  # Calculate requests per second
  end_timestamp=$(date +%s)
  elapsed_time=$((end_timestamp - start_timestamp))
  
  # Avoid division by zero
  if [ "$elapsed_time" -ne 0 ]; then
    requests_per_second=$((i / elapsed_time))
  else
    requests_per_second=0
  fi

  # Update and print progress bar
  progress=$(( (i * 100) / num_runs ))
  print_progress $progress
done

# Calculate average time per request
average_time=$((total_time / num_runs))

# Print summary
echo -e "\n\nTotal Successful Requests: $total_success"
echo "Total Failed Requests: $total_fail"
echo "Average Time per Request: $average_time ms"
echo "Total Time: $total_time ms"
echo "Requests Per Second: $requests_per_second"
