import os
import argparse
import json
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from datetime import datetime
import time
from collections import defaultdict
from joblib import dump, load
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold

# ML models
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier

# FS methods
from utils.feature_election.impetus import PyImpetusSelector
from utils.feature_election.lasso_fs import LassoFeatureSelector
from utils.feature_election.elastic_net_fs import ElasticNetFeatureSelector
from utils.feature_election.SequentialAttentionOptimizer import SequentialAttentionOptimizer


class CentralizedExperimentRunner:
    def __init__(self):
        self.batch_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.results_dir = f'.centralized_results/run_{self.batch_timestamp}'
        os.makedirs(self.results_dir, exist_ok=True)

        # Store best results by configuration
        self.best_results = defaultdict(lambda: {'best_f1': -1})

        # Create config log file
        self.config_log_file = os.path.join(self.results_dir, 'experiment_configs.txt')
        with open(self.config_log_file, 'w') as f:
            f.write("Timestamp -> Config -> Dataset\n")
            f.write("-" * 50 + "\n")

    def _load_combined_dataset(self, dataset_name):
        """Load and combine all client datasets into a single dataset"""
        base_path = f"./datasets/{dataset_name}_federated/"

        # Find all client files
        train_files = [f for f in os.listdir(base_path) if
                       f.startswith(f"{dataset_name}_client_") and f.endswith("_train.csv")]
        test_files = [f for f in os.listdir(base_path) if
                      f.startswith(f"{dataset_name}_client_") and f.endswith("_test.csv")]

        # Extract client IDs
        client_ids = sorted(list(set([f.split('_client_')[1].split('_')[0] for f in train_files])))
        print(f"Found {len(client_ids)} clients for dataset {dataset_name}: {client_ids}")

        # Initialize combined dataframes
        all_train_data = []
        all_test_data = []

        # Load each client's data
        for client_id in client_ids:
            train_path = os.path.join(base_path, f"{dataset_name}_client_{client_id}_train.csv")
            test_path = os.path.join(base_path, f"{dataset_name}_client_{client_id}_test.csv")

            # Load and add client identifier column
            train_df = pd.read_csv(train_path)
            train_df['client_id'] = client_id
            all_train_data.append(train_df)

            test_df = pd.read_csv(test_path)
            test_df['client_id'] = client_id
            all_test_data.append(test_df)

        # Combine all data
        combined_train = pd.concat(all_train_data, ignore_index=True).dropna()
        combined_test = pd.concat(all_test_data, ignore_index=True).dropna()

        print(f"Combined dataset shapes - Train: {combined_train.shape}, Test: {combined_test.shape}")

        return combined_train, combined_test

    def _preprocess_data(self, train_df, test_df):
        """Preprocess the data similar to how the client does it"""
        # Prepare target
        target_encoder = LabelEncoder()
        unique_targets = sorted(train_df['target'].unique())
        target_encoder.fit(unique_targets)
        y_train = target_encoder.transform(train_df['target'])

        # Transform test data, mapping unknown values to a new class
        y_test = np.array([
            target_encoder.transform([val])[0] if val in target_encoder.classes_
            else -1 for val in test_df['target']
        ])

        # Get categorical columns (except target and client_id)
        categorical_columns = sorted(train_df.select_dtypes(include=['object', 'category']).columns)
        categorical_columns = [col for col in categorical_columns if col not in ['target', 'client_id']]

        # Process features
        X_train = train_df.drop(['target', 'client_id'], axis=1)
        X_test = test_df.drop(['target', 'client_id'], axis=1)

        # Initialize feature encoders dictionary
        feature_encoders = {}

        # Process each categorical feature
        for col in categorical_columns:
            encoder = LabelEncoder()
            # Fit on training data only
            encoder.fit(X_train[col])
            n_classes_feat = len(encoder.classes_)
            feature_encoders[col] = encoder

            # Transform training data
            X_train[col] = encoder.transform(X_train[col])
            # Transform test data, mapping unknown values to a new class
            X_test[col] = np.array([
                encoder.transform([val])[0] if val in encoder.classes_
                else n_classes_feat for val in X_test[col]
            ])

        return X_train, X_test, y_train, y_test

    def _initialize_selector(self, selector_name):
        """Initialize the feature selector based on the method name"""
        match selector_name:
            case 'lasso':
                return LassoFeatureSelector(n_trials=150, random_state=42)
            case 'elastic_net':
                return ElasticNetFeatureSelector()
            case 'sequential':
                return None  # Handled specially later
            case 'impetus':
                return PyImpetusSelector(task='classification', verbose=False)
            case _:
                raise ValueError(f"Unknown feature selector: {selector_name}")

    def _get_model(self, model_name):
        """Get model instance by name"""
        match model_name:
            case 'GNB':
                return GaussianNB()
            case 'SGDC':
                return SGDClassifier(
                    loss='log_loss',
                    penalty='l2',
                    alpha=0.01,
                    max_iter=1000,
                    tol=1e-3,
                    learning_rate='adaptive',
                    eta0=0.005,
                    power_t=0.25,
                    warm_start=True,
                    average=False,
                    random_state=42
                )
            case 'MLPC':
                return MLPClassifier(random_state=42)
            case _:
                raise ValueError(f"Unknown model type: {model_name}")

    def feature_selection(self, X_train, y_train, method='lasso'):
        """Perform feature selection using the specified method"""
        if method == 'none':
            return X_train, None, None  # No feature selection

        # Keep the original column names for later
        if isinstance(X_train, pd.DataFrame):
            original_columns = X_train.columns.tolist()
        else:
            # If not a DataFrame, convert to one for consistent handling
            X_train = pd.DataFrame(X_train)
            original_columns = X_train.columns.tolist()

        selector = self._initialize_selector(method)

        # Special case for Sequential Attention
        if method == 'sequential':
            from sklearn.ensemble import RandomForestClassifier

            # Create a RF classifier that scales with the number of features
            n_features = X_train.shape[1]
            min_samples_factor = max(2, int(np.log2(n_features)))
            base_model = RandomForestClassifier(
                n_estimators=max(100, int(np.log2(n_features))),
                max_features=max(1, int(np.sqrt(n_features))),
                min_samples_split=min_samples_factor,
                min_samples_leaf=max(1, min_samples_factor // 2),
                random_state=42,
                n_jobs=-1
            )

            optimizer = SequentialAttentionOptimizer(
                X=X_train,
                y=y_train,
                base_model=base_model,
                n_trials=200
            )
            results = optimizer.optimize_and_select()

            # Get binary array and scores
            binary_array, scores_array = optimizer.align_arrays(X_train.shape[1])

            # Select features using the column names
            selected_indices = np.where(binary_array == 1)[0]
            selected_columns = [original_columns[i] for i in selected_indices]
            X_train_selected = X_train[selected_columns]

            return X_train_selected, binary_array, scores_array, selected_columns
        else:
            # Fit the selector
            results = selector.fit(X_train, y_train)

            # Get binary array and scores
            binary_array, scores_array = selector.align_arrays(X_train.shape[1])

            # Select features using the column names
            selected_indices = np.where(binary_array == 1)[0]
            selected_columns = [original_columns[i] for i in selected_indices]
            X_train_selected = X_train[selected_columns]

            return X_train_selected, binary_array, scores_array, selected_columns

    def transform_test_set(self, X_test, selected_columns):
        """Transform test set using the selected column names"""
        if selected_columns is None:
            return X_test  # No feature selection

        # Ensure X_test is a DataFrame with column names
        if not isinstance(X_test, pd.DataFrame):
            X_test = pd.DataFrame(X_test)

        # Select the same columns as in the training set
        X_test_selected = X_test[selected_columns]

        return X_test_selected

    def run_experiment(self, dataset_name, model_name, fs_method='none'):
        """Run a single experiment with the specified dataset, model, and FS method"""
        print(f"\n{'=' * 50}")
        print(f"Running experiment for dataset: {dataset_name}")
        print(f"Model: {model_name}, Feature Selection: {fs_method}")
        print(f"{'=' * 50}\n")

        start_time = time.time()
        experiment_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Log configuration
        with open(self.config_log_file, 'a') as f:
            config = {
                'model': model_name,
                'fs_method': fs_method,
            }
            f.write(f"\n{experiment_timestamp} -> {config} -> {dataset_name}\n")

        # Create experiment directory
        experiment_dir = os.path.join(self.results_dir,
                                      f"{experiment_timestamp}_{dataset_name}_{model_name}_{fs_method}")
        os.makedirs(experiment_dir, exist_ok=True)

        try:
            # Load and preprocess data
            train_df, test_df = self._load_combined_dataset(dataset_name)
            X_train, X_test, y_train, y_test = self._preprocess_data(train_df, test_df)

            # Save original data shapes
            data_info = {
                'original_train_shape': X_train.shape,
                'original_test_shape': X_test.shape,
                'train_samples': len(y_train),
                'test_samples': len(y_test),
                'num_features': X_train.shape[1],
                'class_distribution': {str(c): int((y_train == c).sum()) for c in np.unique(y_train)}
            }

            with open(os.path.join(experiment_dir, 'data_info.json'), 'w') as f:
                json.dump(data_info, f, indent=4)

            # Perform feature selection if needed
            if fs_method != 'none':
                print(f"Performing feature selection using {fs_method}...")
                X_train_selected, binary_array, feature_scores, selected_columns = self.feature_selection(X_train,
                                                                                                          y_train,
                                                                                                          fs_method)
                X_test_selected = self.transform_test_set(X_test, selected_columns)

                # Save feature selection results
                fs_results = {
                    'num_selected_features': int(np.sum(binary_array)),
                    'selected_features_ratio': float(np.sum(binary_array) / len(binary_array)),
                    'binary_array': binary_array.tolist(),
                    'feature_scores': feature_scores.tolist() if feature_scores is not None else None,
                    'selected_columns': selected_columns
                }

                with open(os.path.join(experiment_dir, 'feature_selection_results.json'), 'w') as f:
                    json.dump(fs_results, f, indent=4)

                print(
                    f"Selected {fs_results['num_selected_features']} features ({fs_results['selected_features_ratio']:.2%})")
            else:
                X_train_selected = X_train
                X_test_selected = X_test

            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_selected)
            X_test_scaled = scaler.transform(X_test_selected)

            # Train model
            print(f"Training {model_name} model...")
            model = self._get_model(model_name)
            model.fit(X_train_scaled, y_train)

            # Evaluate model
            y_train_pred = model.predict(X_train_scaled)
            y_test_pred = model.predict(X_test_scaled)

            train_f1 = f1_score(y_train, y_train_pred, average='weighted')
            test_f1 = f1_score(y_test, y_test_pred, average='weighted')

            # Save model and results
            performance = {
                'train_f1': float(train_f1),
                'test_f1': float(test_f1),
                'train_to_test_ratio': float(train_f1 / test_f1) if test_f1 > 0 else float('inf'),
                'training_time': float(time.time() - start_time)
            }

            with open(os.path.join(experiment_dir, 'performance.json'), 'w') as f:
                json.dump(performance, f, indent=4)

            # Save model
            dump(model, os.path.join(experiment_dir, 'model.joblib'))

            # Update best results
            config_key = f"{dataset_name}_{model_name}_{fs_method}"
            if test_f1 > self.best_results[config_key]['best_f1']:
                self.best_results[config_key] = {
                    'best_f1': test_f1,
                    'experiment_dir': experiment_dir,
                    'timestamp': experiment_timestamp
                }

            print(f"\nResults for {dataset_name}, {model_name}, {fs_method}:")
            print(f"Train F1: {train_f1:.4f}")
            print(f"Test F1: {test_f1:.4f}")
            print(f"Runtime: {time.time() - start_time:.2f} seconds")

            return {
                'dataset': dataset_name,
                'model': model_name,
                'fs_method': fs_method,
                'train_f1': train_f1,
                'test_f1': test_f1,
                'runtime': time.time() - start_time,
                'experiment_dir': experiment_dir,
                'timestamp': experiment_timestamp
            }

        except Exception as e:
            print(f"Error during experiment: {str(e)}")

            # Log the error
            with open(os.path.join(experiment_dir, 'error.txt'), 'w') as f:
                f.write(f"Error: {str(e)}\n")

            return {
                'dataset': dataset_name,
                'model': model_name,
                'fs_method': fs_method,
                'error': str(e),
                'timestamp': experiment_timestamp
            }

    def run_experiments(self, datasets, models, fs_methods):
        """Run all specified experiments"""
        all_results = []

        for dataset in datasets:
            for model in models:
                for fs_method in fs_methods:
                    result = self.run_experiment(dataset, model, fs_method)
                    all_results.append(result)
                    self._save_results(all_results)

                    # Clean up memory
                    import gc
                    gc.collect()
                    time.sleep(1)  # Small delay between experiments

        return all_results

    def _save_results(self, results):
        """Save experiment results"""
        # Save all results
        summary_file = os.path.join(self.results_dir, 'experiments_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(results, f, indent=4)

        # Save best results
        best_results_file = os.path.join(self.results_dir, 'best_results.json')
        with open(best_results_file, 'w') as f:
            json.dump(self.best_results, f, indent=4)

        # Generate readable summary
        summary = []
        summary.append("CENTRALIZED LEARNING RESULTS SUMMARY")
        summary.append("=" * 80)
        summary.append("")

        # Group results by dataset
        by_dataset = defaultdict(list)
        for result in results:
            if 'error' not in result:
                by_dataset[result['dataset']].append(result)

        for dataset, dataset_results in by_dataset.items():
            summary.append(f"Dataset: {dataset}")
            summary.append("-" * 40)

            # Sort by test F1 score
            dataset_results.sort(key=lambda x: x['test_f1'], reverse=True)

            summary.append("{:<15} {:<15} {:<10} {:<10} {:<10}".format(
                "Model", "FS Method", "Train F1", "Test F1", "Runtime(s)"
            ))
            summary.append("-" * 65)

            for result in dataset_results:
                summary.append("{:<15} {:<15} {:<10.4f} {:<10.4f} {:<10.2f}".format(
                    result['model'],
                    result['fs_method'],
                    result['train_f1'],
                    result['test_f1'],
                    result['runtime']
                ))

            summary.append("")
            summary.append("=" * 80)
            summary.append("")

        # Add errors if any
        error_results = [r for r in results if 'error' in r]
        if error_results:
            summary.append("ERRORS:")
            summary.append("-" * 40)
            for result in error_results:
                summary.append(f"Dataset: {result['dataset']}, Model: {result['model']}, "
                               f"FS Method: {result['fs_method']}")
                summary.append(f"Error: {result['error']}")
                summary.append("")

        # Save readable summary
        with open(os.path.join(self.results_dir, 'results_summary.txt'), 'w') as f:
            f.write('\n'.join(summary))

        # Also create a comparison CSV for easier analysis
        comparison_data = []
        for result in results:
            if 'error' not in result:
                comparison_data.append({
                    'dataset': result['dataset'],
                    'model': result['model'],
                    'fs_method': result['fs_method'],
                    'train_f1': result['train_f1'],
                    'test_f1': result['test_f1'],
                    'runtime': result['runtime'],
                    'timestamp': result['timestamp']
                })

        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            comparison_df.to_csv(os.path.join(self.results_dir, 'results_comparison.csv'), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run centralized learning experiments for comparison with FL")

    # Dataset configuration
    parser.add_argument('--datasets', nargs='+',
                        default=['heartdisease', 'mushroom', 'breastcancer', 'income', 'TUANDROMD'],
                        help='List of datasets to test')

    # Experiment configuration
    parser.add_argument('--models', nargs='+',
                        default=['GNB', 'SGDC', 'MLPC'],
                        help='List of ML models to test')

    parser.add_argument('--fs-methods', nargs='+',
                        default=['none', 'impetus', 'lasso', 'sequential'],
                        help='List of feature selection methods to test')

    args = parser.parse_args()

    # Validate arguments
    if not args.datasets:
        raise ValueError("No datasets specified")

    if not args.models:
        raise ValueError("No models specified")

    if not args.fs_methods:
        raise ValueError("No feature selection methods specified")

    # Run experiments
    runner = CentralizedExperimentRunner()
    results = runner.run_experiments(
        datasets=args.datasets,
        models=args.models,
        fs_methods=args.fs_methods
    )

    print(f"\nAll experiments completed.")
    print(f"Results saved in: {runner.results_dir}")
    print("Check results_summary.txt for a summary of all results.")