# Federated Feature Selection with PyImpetus in Flower

This implementation adds federated feature selection using PyImpetus to the Flower federated learning framework. The goal is to analyze how federated feature selection improves model performance and reduces model parameter size.

## Files Overview

1. `utils/feature_election/impetus.py` - Implementation of PyImpetus feature selector
2. `utils/feature_selection_utils.py` - Utility functions for feature selection
3. `updated-flower-client.py` - Modified client that supports feature selection
4. `updated-flower-server.py` - Modified server that coordinates feature selection
5. `run_experiment.sh` - Script to run experiments with and without feature selection

## How Feature Selection Works

The feature selection process works in 3 phases:

1. **Local Feature Selection**: Each client performs feature selection on its local data using PyImpetus, which combines mutual information and random forest feature importance.

2. **Federated Feature Selection Aggregation**: The server collects the feature masks and scores from all clients and performs an aggregation based on:
   - Intersection (features selected by ALL clients)
   - Top-K ranked features from the difference set (union - intersection) based on the freedom rate

3. **Global Feature Mask Application**: The server sends the global feature mask back to all clients who apply it to their local data.

## Key Parameters

- `--feature-selection` - Enable feature selection
- `--fs-method` - Feature selection method (currently only 'impetus' is supported)
- `--freedom-rate` - How many features from the difference set to include (0-1):
  - 0 = Only include features in the intersection (most conservative)
  - 1 = Include all features in the union (most inclusive)
  - 0.5 = Include top 50% of features from the difference set (balanced)

## Running the Experiments

1. First, ensure you have the correct directory structure for your dataset:
   ```
   ./datasets/<dataset_name>_federated/<dataset_name>_client_<id>_train.csv
   ./datasets/<dataset_name>_federated/<dataset_name>_client_<id>_test.csv
   ```

2. Install the required dependencies:
   ```bash
   pip install flwr pandas scikit-learn numpy
   ```

3. Run an experiment:
   ```bash
   # Without feature selection
   python updated-flower-server.py --dataset your_dataset --model GNB --rounds 5 --min-clients 3
   
   # In separate terminals for each client
   python updated-flower-client.py --client-id 0 --dataset your_dataset --model GNB
   python updated-flower-client.py --client-id 1 --dataset your_dataset --model GNB
   python updated-flower-client.py --client-id 2 --dataset your_dataset --model GNB
   
   # With feature selection
   python updated-flower-server.py --dataset your_dataset --model GNB --rounds 5 --min-clients 3 --feature-selection --freedom-rate 0.5
   
   # In separate terminals for each client
   python updated-flower-client.py --client-id 0 --dataset your_dataset --model GNB --feature-selection
   python updated-flower-client.py --client-id 1 --dataset your_dataset --model GNB --feature-selection
   python updated-flower-client.py --client-id 2 --dataset your_dataset --model GNB --feature-selection
   ```

4. Alternatively, use the provided experiment script:
   ```bash
   # Update the dataset name in run_experiment.sh first
   chmod +x run_experiment.sh
   ./run_experiment.sh
   ```

## Analyzing Results

After running experiments, results will be stored in:
- `./results/run_YYYYMMDD_HHMMSS/feature_selection_results.json` - Details of feature selection process
- `./results/run_YYYYMMDD_HHMMSS/final_results.json` - Final evaluation metrics

Key metrics to compare:
1. F1 score - Evaluate model performance improvement
2. Model size (bytes) - Evaluate parameter size reduction
3. Number of features - See how many features were selected

## Expected Outcomes

1. **Performance**: Feature selection should improve or maintain F1 scores by eliminating noisy features.
2. **Model Size**: Feature selection should significantly reduce model parameter size, especially for models like GNB where parameters scale linearly with features.
3. **Communication Efficiency**: Smaller models mean less data transmitted between clients and server.

## Advanced Analysis

To track detailed model performance across rounds:
- Performance before feature selection
- Performance after local feature selection
- Performance after global feature mask application
- Performance improvement over training rounds

## Troubleshooting

If you encounter issues:
1. Check that dataset paths are correct
2. Ensure all dependencies are installed
3. Look for error messages in client and server output
4. Verify feature selection parameters are in valid ranges
