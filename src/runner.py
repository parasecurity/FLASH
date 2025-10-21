import subprocess
import os
import json
import pandas as pd
from datetime import datetime
import time
import argparse
from itertools import product
from collections import defaultdict


class FSExperimentRunner:
    def __init__(self, base_port=8080):
        self.base_port = base_port
        self.batch_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.results_dir = f'./backup/runner/fs_experiments_{self.batch_timestamp}'
        os.makedirs(self.results_dir, exist_ok=True)

        # Store best results by configuration
        self.best_results = defaultdict(lambda: {'best_fr': None, 'best_f1': -1})

        # Create config log file
        self.config_log_file = os.path.join(self.results_dir, 'experiment_configs.txt')
        with open(self.config_log_file, 'w') as f:
            f.write("Timestamp -> Server Config -> Dataset\n")
            f.write("-" * 50 + "\n")

    def _get_next_port(self, offset):
        """Get unique port number for each experiment run"""
        return self.base_port + offset

    def run_single_dataset_experiment(self, dataset, num_clients, configs_for_dataset, enable_tuning=False, combs=1):
        """Run all experiments for a single dataset"""
        print(f"\n{'=' * 50}")
        print(f"Starting experiments for dataset: {dataset} with {num_clients} clients")
        print(f"{'=' * 50}\n")

        results_for_dataset = []

        for config in configs_for_dataset:
            print(f"\nRunning experiment with:")
            print(f"  Model: {config['model']}")
            print(f"  FS Method: {config['fs_method']}")
            print(f"  Freedom Rate: {config['freedom_rate']}")
            print(f"  Weighted: {config['weighted']}")
            if enable_tuning:
                print(f"  Tuning Enabled: Yes")
                print(f"  Combinations per parameter: {combs}\n")
            else:
                print(f"  Tuning Enabled: No\n")

            start_time = time.time()
            port = self._get_next_port(0)

            # Generate new timestamp for this experiment
            experiment_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            os.environ['EXPERIMENT_TIME'] = experiment_timestamp
            os.environ['JOBLIB_MULTIPROCESSING'] = '0'  # Disable multiprocessing in joblib

            # Create experiment name (just for logging)
            experiment_name = f"{dataset}_{config['model']}_fs{config['fs_method']}_fr{str(config['freedom_rate']).replace('.', '_')}_{'w' if config['weighted'] else 'u'}"
            if enable_tuning:
                experiment_name += f"_tuned{combs}"

            # Log configuration for this experiment
            with open(self.config_log_file, 'a') as f:
                server_config = {
                    'port': port,
                    'clients': num_clients,
                    'model': config['model'],
                    'fs-method': config['fs_method'],
                    'freedom': config['freedom_rate'],
                    'weighted': config['weighted'],
                    'tuning': enable_tuning,
                    'combs': combs if enable_tuning else 1
                }
                f.write(f"\n{experiment_timestamp} -> {server_config} -> {dataset}\n")

            # Start server process
            server_cmd = [
                'python3', 'server.py',
                '--port', str(port),
                '--clients', str(num_clients),
                '--model', config['model'],
                '--fe-method', config['fs_method'],
                '--freedom', str(config['freedom_rate']),
            ]
            if config['weighted']:
                server_cmd.append('--fe_weighted')
            if enable_tuning:
                server_cmd.extend(['--tuning', '--combs', str(combs)])

            try:
                # Start server process
                server_process = subprocess.Popen(server_cmd)
                time.sleep(2)  # Give server time to start

                # Start client processes
                client_processes = []
                for _ in range(num_clients):
                    client_cmd = [
                        'python3', 'client.py',
                        '--ip', '127.0.0.1',
                        '--port', str(port),
                        '--dataset', dataset
                    ]
                    client_process = subprocess.Popen(client_cmd)
                    client_processes.append(client_process)

                # Wait for all processes to complete
                server_process.wait()
                for client_process in client_processes:
                    client_process.wait()

            except Exception as e:
                print(f"Error during experiment execution: {str(e)}")
                # Make sure to terminate processes if there's an error
                if 'server_process' in locals():
                    server_process.terminate()
                if 'client_processes' in locals():
                    for p in client_processes:
                        p.terminate()
            finally:
                # Add a garbage collection call to help clean up resources
                import gc
                gc.collect()
            # Collect and analyze results
            avg_f1 = self._collect_results(experiment_name, num_clients, experiment_timestamp)
            end_time = time.time()

            result = {
                'dataset': dataset,
                'num_clients': num_clients,
                'fs_method': config['fs_method'],
                'freedom_rate': config['freedom_rate'],
                'model': config['model'],
                'weighted': config['weighted'],
                'tuning_enabled': enable_tuning,
                'tuning_combs': combs if enable_tuning else 1,
                'experiment_name': experiment_name,
                'log_dir': f'./logs/client*/{{experiment_timestamp}}_{dataset}',
                'timestamp': experiment_timestamp,
                'avg_f1': avg_f1,
                'runtime': end_time - start_time
            }
            results_for_dataset.append(result)

            # Update best results
            config_key = f"{dataset}_{config['model']}_{config['fs_method']}_{'w' if config['weighted'] else 'u'}"
            if enable_tuning:
                config_key += f"_tuned{combs}"

            if avg_f1 > self.best_results[config_key]['best_f1']:
                self.best_results[config_key] = {
                    'best_fr': config['freedom_rate'],
                    'best_f1': avg_f1,
                    'experiment_name': experiment_name,
                    'timestamp': experiment_timestamp,
                    'tuning_enabled': enable_tuning,
                    'tuning_combs': combs if enable_tuning else 1
                }

            # Save intermediate results after each experiment
            self._save_summary(results_for_dataset)

            print(f"\nCompleted experiment: {experiment_name}")
            print(f"Average F1 Score: {avg_f1:.4f}")
            print(f"Runtime: {end_time - start_time:.2f} seconds")

            # Clean up memory between experiments
            import gc
            gc.collect()

            # Sleep between experiments to allow for resource cleanup
            time.sleep(3)

        return results_for_dataset

    def _collect_results(self, experiment_name, num_clients, timestamp):
        """Collect and analyze results from the experiment directory"""
        client_f1_scores = []
        logs_dir = './logs'

        # Get dataset name from experiment name while preserving underscores
        dataset_name = "_".join(
            experiment_name.split("_")[:2] if "heart_disease" in experiment_name else [experiment_name.split("_")[0]])

        # For each client
        for client_id in range(1, num_clients + 1):
            client_dir = os.path.join(logs_dir, f'client{client_id}', f'{timestamp}_{dataset_name}')

            try:
                # Find the highest round number
                round_files = [f for f in os.listdir(client_dir) if
                               f.startswith('round_') and f.endswith('_results.json')]
                if not round_files:
                    print(f"Warning: No round files found for client {client_id} in {client_dir}")
                    continue

                # Get highest round number
                max_round = max([int(f.split('_')[1]) for f in round_files])
                final_round_file = f'round_{max_round}_results.json'

                # Read the final round results
                with open(os.path.join(client_dir, final_round_file), 'r') as f:
                    results = json.load(f)
                    client_f1_scores.append(float(results['test']))

            except FileNotFoundError:
                print(f"Warning: Could not find results for client {client_id} in {client_dir}")
                continue
            except (KeyError, ValueError) as e:
                print(f"Error reading results for client {client_id}: {str(e)}")
                continue

        # Calculate average F1 score across all clients
        if client_f1_scores:
            avg_f1 = sum(client_f1_scores) / len(client_f1_scores)
            print(f"Client F1 scores: {client_f1_scores}")
            print(f"Average F1 score: {avg_f1:.4f}")
        else:
            avg_f1 = 0
            print(f"Warning: No valid results found for experiment: {experiment_name}")

        return avg_f1

    def run_batch_experiments(self, dataset_configs, all_configs, enable_tuning=False, combs=1):
        """Run experiments for each dataset sequentially"""
        all_results = []

        # Group configs by dataset
        configs_by_dataset = defaultdict(list)
        for config in all_configs:
            configs_by_dataset[config['dataset']].append(config)

        # Run experiments for each dataset sequentially
        for dataset, num_clients in dataset_configs.items():
            dataset_results = self.run_single_dataset_experiment(
                dataset,
                num_clients,
                configs_by_dataset[dataset],
                enable_tuning,
                combs
            )
            all_results.extend(dataset_results)

        return all_results

    def _save_summary(self, results):
        """Save summary of all experiments and best results by configuration"""
        # Save all results
        summary_file = os.path.join(self.results_dir, 'experiments_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(results, f, indent=4)

        # Save best results by configuration
        best_results_file = os.path.join(self.results_dir, 'best_results_by_config.json')
        with open(best_results_file, 'w') as f:
            json.dump(self.best_results, f, indent=4)

        # Generate readable summary
        summary = []
        for config, result in self.best_results.items():
            parts = config.split('_')

            # Parse configuration parts
            if 'heart_disease' in config:
                dataset = '_'.join(parts[:2])
                remaining_parts = parts[2:]
            else:
                dataset = parts[0]
                remaining_parts = parts[1:]

            # Find tuning information
            tuning_info = ""
            if any(p.startswith('tuned') for p in remaining_parts):
                tuning_part = next(p for p in remaining_parts if p.startswith('tuned'))
                tuning_info = f", tuning={result['tuning_enabled']}, combs={result['tuning_combs']}"
                remaining_parts.remove(tuning_part)

            # Extract other parts
            model = remaining_parts[-3]
            fs_method = '_'.join(remaining_parts[-2:-1])
            weighted = remaining_parts[-1]

            summary.append(f"Configuration: {dataset}, {model}, {fs_method}, weighted={weighted}{tuning_info}")
            summary.append(f"Best freedom rate: {result['best_fr']}")
            summary.append(f"Best F1 score: {result['best_f1']:.4f}")
            summary.append(f"Timestamp: {result['timestamp']}")
            summary.append("-" * 50)

        # Save readable summary
        with open(os.path.join(self.results_dir, 'best_results_summary.txt'), 'w') as f:
            f.write('\n'.join(summary))

    def run_specific_combinations(self, dataset_configs, specific_configs, enable_tuning=False, combs=1):
        """Run experiments for specific combinations of dataset, feature selector, model, and freedom rate"""
        all_results = []

        # Group configs by dataset
        configs_by_dataset = defaultdict(list)
        for config in specific_configs:
            dataset = config['dataset']
            if dataset in dataset_configs:
                configs_by_dataset[dataset].append(config)
            else:
                print(f"Warning: Dataset {dataset} not found in dataset configurations. Skipping combination.")

        # Run experiments for each dataset
        for dataset, configs in configs_by_dataset.items():
            num_clients = dataset_configs[dataset]
            print(f"\nProcessing combinations for dataset: {dataset} with {num_clients} clients")

            dataset_results = self.run_single_dataset_experiment(
                dataset,
                num_clients,
                configs,
                enable_tuning,
                combs
            )
            all_results.extend(dataset_results)

        return all_results


def create_experiment_configs(
        dataset_configs,
        fs_methods=['impetus', 'lasso', 'sequential', 'none'],
        models=['GNB', 'SGDC', 'MLPC'],
        freedom_rates=[0.1, 0.3, 0.4, 0.5, 0.7, 1.0],
        weighted_options=[True]
):
    """Create all combinations of experiment configurations"""
    configs = []

    for dataset, num_clients in dataset_configs.items():
        # Handle 'none' FS method separately (only once per model)
        for model in models:
            # Add configuration with 'none' FS method (freedom rate doesn't matter)
            if 'none' in fs_methods:
                configs.append({
                    'dataset': dataset,
                    'num_clients': num_clients,
                    'fs_method': 'none',
                    'freedom_rate': 0.0,  # Arbitrary value since it doesn't matter
                    'model': model,
                    'weighted': weighted_options[0]
                })

            # Add configurations for other FS methods
            for params in product(
                    [dataset],
                    [num_clients],
                    [m for m in fs_methods if m != 'none'],  # Exclude 'none'
                    freedom_rates,
                    [model],
                    weighted_options
            ):
                config = {
                    'dataset': params[0],
                    'num_clients': params[1],
                    'fs_method': params[2],
                    'freedom_rate': params[3],
                    'model': params[4],
                    'weighted': params[5]
                }
                configs.append(config)

    return configs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run feature selection experiments")
    parser.add_argument('--base-port', type=int, default=8080, help='Base port number')
    parser.add_argument('--dataset-clients', nargs='+',
                        default=['heartdisease:3', 'mushroom:3', 'breastcancer:3', 'income:3', 'TUANDROMD:3'],
                        help='Dataset and client count pairs in format dataset_name:num_clients')

    # Model tuning arguments
    tuning_group = parser.add_argument_group('Model Tuning Options')
    tuning_group.add_argument('--tuning', action='store_true', default=False,
                              help='Enable model fine-tuning')
    tuning_group.add_argument('--combs', type=int, default=1,
                              help='Number of combinations per target parameter for tuning')

    # Experiment configuration arguments
    config_group = parser.add_argument_group('Experiment Configuration')
    config_group.add_argument('--fs-methods', nargs='+',
                              default=['impetus', 'lasso', 'sequential', 'none'],
                              help='List of feature selection methods to test')
    config_group.add_argument('--models', nargs='+',
                              default=['GNB', 'SGDC', 'MLPC'],
                              help='List of ML models to test')
    config_group.add_argument('--freedom-rates', nargs='+', type=float,
                              default=[0.1, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0],
                              help='List of freedom rates to test')
    config_group.add_argument('--weighted-options', nargs='+', type=bool,
                              default=[True],
                              help='List of weighted options to test')

    # Specific combinations mode
    specific_group = parser.add_argument_group('Specific Combinations Mode')
    specific_group.add_argument('--specific-combinations', action='store_true',
                                help='Run specific combinations instead of all combinations')
    specific_group.add_argument('--combinations-file', type=str,
                                help='JSON file containing specific combinations to run')

    args = parser.parse_args()

    # Parse dataset:client_count pairs
    dataset_configs = {}
    for pair in args.dataset_clients:
        try:
            dataset, num_clients = pair.split(':')
            dataset_configs[dataset] = int(num_clients)
            print(f"Setting {num_clients} clients for dataset {dataset}")
        except ValueError:
            print(f"Warning: Invalid dataset-client pair format: {pair}. Expected format: dataset_name:num_clients")
            continue

    if not dataset_configs:
        raise Exception("No valid dataset configurations found")

    runner = FSExperimentRunner(base_port=args.base_port)

    if args.specific_combinations:
        if not args.combinations_file:
            raise Exception("Must provide --combinations-file when using --specific-combinations")

        # Load specific combinations from JSON file
        with open(args.combinations_file, 'r') as f:
            specific_configs = json.load(f)

        # Validate combinations
        for config in specific_configs:
            if 'dataset' not in config:
                raise Exception("Each combination in the JSON file must include a 'dataset' field")

        # Run specific combinations
        results = runner.run_specific_combinations(
            dataset_configs,
            specific_configs,
            enable_tuning=args.tuning,
            combs=args.combs
        )
    else:
        # Create and run all experiment configurations
        configs = create_experiment_configs(
            dataset_configs=dataset_configs,
            fs_methods=args.fs_methods,
            models=args.models,
            freedom_rates=args.freedom_rates,
            weighted_options=args.weighted_options
        )

        results = runner.run_batch_experiments(
            dataset_configs,
            configs,
            enable_tuning=args.tuning,
            combs=args.combs
        )

    print(f"\nCompleted all experiments")
    print(f"Results saved in: {runner.results_dir}")
    print("Check best_results_summary.txt for the best freedom rates by configuration")