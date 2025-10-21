#!/bin/bash

# Enhanced script to run experiments with multiple models and datasets
# for comparing federated learning with and without feature selection

# Define all datasets to test
DATASETS=("income")

# Define all models to test
MODELS=("MLPC")

# Common parameters
NUM_CLIENTS=3
NUM_ROUNDS=10
RANDOM_STATE=42
PORT_BASE=8090  # We'll increment this for each experiment to avoid conflicts

# Freedom rates to test
FREEDOM_RATES=(0.1 0.3 0.7)

# Feature selection methods to test
FS_METHODS=("lasso")

# Flag to control whether to run baseline (non-FS) experiments
# Set to "false" to skip baseline experiments
RUN_BASELINE=${RUN_BASELINE:-false}

# Create a function to run an individual experiment
run_experiment() {
    local dataset=$1
    local model=$2
    local use_fs=$3
    local freedom_rate=$4
    local fs_method=$5
    local port=$6
    
    # Create experiment name
    local exp_name="${dataset}_${model}"
    if [ "$use_fs" = true ]; then
        exp_name="${exp_name}_${fs_method}_FS_${freedom_rate}"
    else
        exp_name="${exp_name}_NoFS"
    fi
    
    echo "================================================="
    echo "Running experiment: $exp_name"
    echo "Dataset: $dataset"
    echo "Model: $model"
    echo "Feature Selection: $use_fs"
    if [ "$use_fs" = true ]; then
        echo "Feature Selection Method: $fs_method"
        echo "Freedom Rate: $freedom_rate"
    fi
    echo "Port: $port"
    echo "================================================="
    
    # Start the server
    if [ "$use_fs" = true ]; then
        python3 flower-server.py --dataset $dataset --model $model --rounds $NUM_ROUNDS \
            --min-clients $NUM_CLIENTS --port $port --random-state $RANDOM_STATE \
            --feature-selection --freedom-rate $freedom_rate &
    else
        python3 flower-server.py --dataset $dataset --model $model --rounds $NUM_ROUNDS \
            --min-clients $NUM_CLIENTS --port $port --random-state $RANDOM_STATE &
    fi
    
    SERVER_PID=$!
    echo "Server started with PID: $SERVER_PID"
    
    # Wait for server to initialize
    sleep 3
    
    # Start the clients
    CLIENT_PIDS=()
    for ((i=0; i<$NUM_CLIENTS; i++)); do
        if [ "$use_fs" = true ]; then
            python3 flower-client.py --client-id $i --dataset $dataset --server 127.0.0.1:$port \
                --model $model --random-state $RANDOM_STATE --feature-selection --fs-method $fs_method &
        else
            python3 flower-client.py --client-id $i --dataset $dataset --server 127.0.0.1:$port \
                --model $model --random-state $RANDOM_STATE &
        fi
        
        CLIENT_PIDS[$i]=$!
        echo "Client $i started with PID: ${CLIENT_PIDS[$i]}"
    done
    
    # Wait for server to complete
    wait $SERVER_PID
    
    # Cleanup any remaining client processes
    for pid in "${CLIENT_PIDS[@]}"; do
        if ps -p $pid > /dev/null; then
            kill $pid
        fi
    done
    
    echo "Experiment $exp_name completed"
    echo "================================================="
    echo ""
    
    # Allow some time for ports to be released
    sleep 5
}

# Main experiment loop
port=$PORT_BASE

# Loop through all datasets
for dataset in "${DATASETS[@]}"; do
    # Loop through all models
    for model in "${MODELS[@]}"; do
        # Run baseline experiment (no feature selection) if enabled
        if [ "$RUN_BASELINE" = true ]; then
            echo "Running baseline (non-FS) experiment..."
            run_experiment "$dataset" "$model" false 0 "none" $port
            port=$((port + 1))
        else
            echo "Skipping baseline (non-FS) experiment..."
        fi
        
        # Run experiments with feature selection at different freedom rates
        for fs_method in "${FS_METHODS[@]}"; do
            for freedom_rate in "${FREEDOM_RATES[@]}"; do
                run_experiment "$dataset" "$model" true $freedom_rate "$fs_method" $port
                port=$((port + 1))
            done
        done
    done
done

echo "All experiments completed!"
echo "Check the results directory for detailed measurements and comparisons."
