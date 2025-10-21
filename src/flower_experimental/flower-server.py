import argparse
import os
import datetime
import json
from typing import Dict, List, Tuple, Optional, Union, Callable
import numpy as np

import flwr as fl
from flwr.common import Metrics, Parameters, FitRes, EvaluateRes, NDArrays
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg


class FeatureSelectionUtils:
    """Utility class for feature selection aggregation"""

    @staticmethod
    def get_features_intersection(masks: np.ndarray) -> np.ndarray:
        """Get intersection of feature masks (features selected by ALL clients)"""
        return np.all(masks == 1, axis=0)

    @staticmethod
    def get_features_union(masks: np.ndarray) -> np.ndarray:
        """Get union of feature masks (features selected by ANY client)"""
        return np.any(masks == 1, axis=0)

    @staticmethod
    def get_features_difference(masks: np.ndarray) -> np.ndarray:
        """Get features in (union - intersection)"""
        union = FeatureSelectionUtils.get_features_union(masks)
        intersection = FeatureSelectionUtils.get_features_intersection(masks)
        return union & ~intersection

    @staticmethod
    def aggregate_feature_selection(client_masks, client_scores, client_weights, freedom_rate=0.5):
        """
        Aggregate feature selections from multiple clients

        Args:
            client_masks: List of binary arrays indicating selected features for each client
            client_scores: List of score arrays for each feature from each client
            client_weights: List of weights for each client (usually based on sample count)
            freedom_rate: Percentage of non-intersection features to include (0-1)

        Returns:
            global_mask: Binary array of globally selected features
        """
        # Convert lists to numpy arrays
        masks = np.array(client_masks)
        scores = np.array(client_scores)
        weights = np.array(client_weights)

        # Get intersection and union masks
        intersection_mask = FeatureSelectionUtils.get_features_intersection(masks)
        union_mask = FeatureSelectionUtils.get_features_union(masks)

        # Handle edge cases
        if freedom_rate == 0:
            return intersection_mask
        elif freedom_rate == 1:
            return union_mask

        # Get features in (union - intersection)
        difference_mask = FeatureSelectionUtils.get_features_difference(masks)

        # Scale scores and zero out non-selected features for each client
        scaled_scores = []
        for client_idx, (client_mask, client_score) in enumerate(zip(masks, scores)):
            # First scale all selected features for the client
            scaled_row = np.zeros_like(client_score)
            selected_features = client_mask == 1

            if np.any(selected_features):
                selected_scores = client_score[selected_features]
                max_score = np.max(selected_scores)
                min_score = np.min(selected_scores)
                range_score = max_score - min_score

                if range_score > 0:
                    scaled_row[selected_features] = (client_score[selected_features] - min_score) / range_score
                else:
                    scaled_row[selected_features] = 1.0

            # Zero out intersection features (we'll include them all anyway)
            scaled_row[intersection_mask] = 0.0

            # Apply client weight
            weighted_row = weights[client_idx] * scaled_row
            scaled_scores.append(weighted_row)

        # Aggregate scaled scores
        aggregated_scores = np.sum(scaled_scores, axis=0)

        # Calculate number of additional features to select
        n_additional = int(np.ceil(np.sum(difference_mask) * freedom_rate))

        # Select top features from difference set
        diff_indices = np.where(difference_mask)[0]
        diff_scores = aggregated_scores[difference_mask]

        if len(diff_scores) > 0:
            # Sort difference features by score and select top n_additional
            top_indices = diff_indices[np.argsort(diff_scores)[-min(n_additional, len(diff_scores)):]]

            # Create final mask combining intersection and selected difference features
            final_mask = np.zeros_like(union_mask)
            final_mask[np.where(intersection_mask)[0]] = 1  # Include all intersection features
            final_mask[top_indices] = 1  # Include top difference features

            return final_mask
        else:
            return intersection_mask


# Create a custom strategy based on FedAvg that coordinates feature selection
class FeatureSelectionStrategy(FedAvg):
    """Custom strategy that includes a feature selection phase"""

    def __init__(
            self,
            min_clients: int,
            model_type: str,
            num_rounds: int,
            dataset_name: str,
            perform_fs: bool = False,
            freedom_rate: float = 0.5,
            random_state: int = 42,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.min_clients = min_clients
        self.model_type = model_type
        self.dataset_name = dataset_name
        self.current_round = 0
        self.num_rounds = num_rounds
        self.random_state = random_state
        self.initial_parameters = None

        # Feature selection properties
        self.perform_fs = perform_fs
        self.freedom_rate = freedom_rate
        self.fs_completed = False
        self.global_feature_mask = None
        self.client_fs_results = None
        self.fs_metrics = {}

        # Track the timestamp for file naming
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"Starting server with {self.model_type} model")
        print(f"Requiring minimum {self.min_clients} clients")
        print(f"Feature selection: {'Enabled' if perform_fs else 'Disabled'}, Freedom rate: {freedom_rate}")

        # Create results directory
        self.results_dir = f"./results/run_{self.timestamp}"
        os.makedirs(self.results_dir, exist_ok=True)

    def initialize_parameters(self, client_manager) -> Optional[Parameters]:
        """Initialize parameters by aggregating from all available clients."""
        print("Initializing server parameters")

        # If we're doing feature selection, we'll initialize parameters after feature selection is done
        if self.perform_fs:
            print("Parameter initialization deferred until after feature selection")
            return None

        # Get clients
        clients = client_manager.sample(
            num_clients=client_manager.num_available(), min_num_clients=3
        )

        if not clients:
            return None

        # For standard FL without feature selection, we could request parameters from clients
        # but for simplicity we'll just return None to let the framework handle it
        return None

    def aggregate_fit(self, server_round: int, results, failures):
        """Aggregate fit results using weighted averaging"""
        # If this is the feature selection round (Round 1), process feature selection results
        if self.perform_fs and not self.fs_completed and server_round == 1 and results:
            print(f"Aggregating feature selection results from {len(results)} clients")
            # Extract feature selection results from clients
            client_masks = []
            client_scores = []
            client_weights = []
            client_metrics = []

            for client_proxy, fit_res in results:
                # Parse feature mask and scores from client metrics
                if "feature_selection" in fit_res.metrics and fit_res.metrics["feature_selection"]:
                    feature_mask = np.array(json.loads(fit_res.metrics["feature_mask"]))
                    feature_scores = np.array(json.loads(fit_res.metrics["feature_scores"]))

                    client_masks.append(feature_mask)
                    client_scores.append(feature_scores)
                    client_weights.append(fit_res.num_examples / sum(r.num_examples for _, r in results))

                    # Store client-specific metrics
                    client_metrics.append({
                        "client_id": client_proxy.cid,
                        "num_examples": fit_res.num_examples,
                        "score_before_fs": fit_res.metrics["score_before_fs"],
                        "score_after_local_fs": fit_res.metrics["score_after_local_fs"],
                        "n_features_before": fit_res.metrics["n_features_before"],
                        "n_features_after": fit_res.metrics["n_features_after"],
                    })

            # Store client results for reporting
            self.client_fs_results = client_metrics

            if client_masks:
                # Aggregate feature selections
                self.global_feature_mask = FeatureSelectionUtils.aggregate_feature_selection(
                    client_masks,
                    client_scores,
                    client_weights,
                    self.freedom_rate
                )

                print(
                    f"Feature selection completed. Selected {int(np.sum(self.global_feature_mask))} out of {len(self.global_feature_mask)} features")

                # Save feature selection results
                self.save_feature_selection_results(client_masks, client_metrics)

                # Mark feature selection as completed
                self.fs_completed = True

                # Return empty parameters to ensure clients get updated in next round
                return [], {}

        # If this is the round to apply global mask (Round 2), collect metrics
        elif self.perform_fs and self.fs_completed and self.global_feature_mask is not None and server_round == 2 and results:
            print(f"Processing results after applying global feature mask to {len(results)} clients")
            # Collect metrics after applying global mask
            global_mask_metrics = {}
            total_examples = 0

            for client_proxy, fit_res in results:
                if "applied_global_mask" in fit_res.metrics and fit_res.metrics["applied_global_mask"]:
                    cid = client_proxy.cid
                    global_mask_metrics[cid] = {
                        "score_with_global_mask": fit_res.metrics["score_with_global_mask"],
                        "n_features": fit_res.metrics["n_features"],
                        "num_examples": fit_res.num_examples
                    }
                    total_examples += fit_res.num_examples

            # Calculate weighted average score across clients
            if global_mask_metrics:
                avg_score = sum(
                    m["score_with_global_mask"] * m["num_examples"]
                    for m in global_mask_metrics.values()
                ) / total_examples

                self.fs_metrics["global_mask"] = {
                    "client_metrics": global_mask_metrics,
                    "avg_score_with_global_mask": avg_score,
                    "n_selected_features": int(np.sum(self.global_feature_mask))
                }

                print(f"Applied global feature mask. Average F1 score: {avg_score:.4f}")

                # Save updated feature selection results
                self.save_feature_selection_results(None, None)

                # Return empty parameters to force clients to start fresh in next round
                return [], {}

        # For regular rounds (Round 3+), use the parent class aggregation
        return super().aggregate_fit(server_round, results, failures)

    def aggregate_evaluate(self, server_round: int, results, failures) -> Optional[Tuple[float, Dict[str, float]]]:
        """Aggregate evaluation results using unweighted average"""
        if not results:
            return None

        # Extract f1 scores and model sizes from client results
        f1_scores = [res.metrics.get("f1_score", 0.0) for _, res in results]
        model_sizes = [res.metrics.get("model_size_bytes", 0.0) for _, res in results]
        n_features = [res.metrics.get("n_features", 0) for _, res in results]

        # Calculate unweighted averages
        if f1_scores:
            avg_f1 = sum(f1_scores) / len(f1_scores)
            # Convert to loss
            loss = 1.0 - avg_f1
        else:
            # Default if no scores available
            avg_f1 = 0.0
            loss = 1.0

        # Calculate average model size
        avg_model_size = sum(model_sizes) / len(model_sizes) if model_sizes else 0
        avg_n_features = sum(n_features) / len(n_features) if n_features else 0

        # Create metrics dictionary
        metrics = {
            "f1_score": avg_f1,
            "avg_model_size_bytes": avg_model_size,
            "avg_n_features": avg_n_features
        }

        # Log detailed metrics for this round
        print(f"Round {server_round} evaluation completed:")
        print(f"  Unweighted Test F1 score: {avg_f1:.4f}")
        print(f"  Average model size: {avg_model_size:.2f} bytes")
        print(f"  Average feature count: {avg_n_features:.1f}")

        # Log client-specific results
        print(f"  Client results:")
        for client, res in results:
            print(f"    Client {client.cid}: "
                  f"Test F1 score = {res.metrics.get('f1_score', 'N/A')}, "
                  f"Model size = {res.metrics.get('model_size_bytes', 'N/A')} bytes, "
                  f"Features = {res.metrics.get('n_features', 'N/A')}, "
                  f"Samples = {res.num_examples}")

        # Print final results banner
        if server_round == self.num_rounds:
            print("\nAll rounds completed! Final results:")
            print(f"  Final loss: {loss:.4f}")
            print(f"  Final Unweighted Test F1 score: {avg_f1:.4f}")
            print(f"  Final average model size: {avg_model_size:.2f} bytes")
            print(f"  Final average feature count: {avg_n_features:.1f}")

            # Save final results
            self.save_final_results(metrics, results)

            # Print the final result with clear marking as the last line for easy extraction
            print("=" * 50)
            print(f"FINAL_RESULT: dataset={self.dataset_name}, model={self.model_type}, "
                  f"test_f1={avg_f1:.4f}, model_size={avg_model_size:.2f}, "
                  f"feature_selection={'Yes' if self.perform_fs else 'No'}")
            print("=" * 50)

        return loss, metrics

    def configure_fit(self, server_round: int, parameters, client_manager):
        """Configure clients for training with feature selection phases"""
        self.current_round = server_round

        # Get standard client/config pairs from parent class
        client_config_pairs = super().configure_fit(server_round, parameters, client_manager)

        # Check if we need to perform feature selection (Round 1)
        if self.perform_fs and not self.fs_completed and server_round == 1:
            print("Starting feature selection round")
            # Configure clients for feature selection
            for _, fit_ins in client_config_pairs:
                fit_ins.config["feature_selection_round"] = "true"
                fit_ins.config["round"] = str(server_round)
                fit_ins.config["model_type"] = self.model_type
                fit_ins.config["random_state"] = str(self.random_state)

        # Check if we need to apply global feature mask (Round 2)
        elif self.perform_fs and self.fs_completed and self.global_feature_mask is not None and server_round == 2:
            print(
                f"Applying global feature mask to clients (selected {int(np.sum(self.global_feature_mask))} features)")
            # Configure clients to apply global feature mask
            for _, fit_ins in client_config_pairs:
                fit_ins.config["apply_global_mask"] = "true"
                fit_ins.config["global_feature_mask"] = json.dumps(self.global_feature_mask.tolist())
                fit_ins.config["round"] = str(server_round)
                fit_ins.config["model_type"] = self.model_type
                fit_ins.config["random_state"] = str(self.random_state)

        # Regular training rounds (Round 3+)
        else:
            # Add custom configuration to each client config
            for _, fit_ins in client_config_pairs:
                # Add round information and model type
                fit_ins.config["round"] = str(server_round)
                fit_ins.config["total_rounds"] = str(self.num_rounds)
                fit_ins.config["model_type"] = self.model_type
                fit_ins.config["random_state"] = str(self.random_state)

                # If feature selection was performed, ensure all clients know the reduced feature count
                if self.perform_fs and self.global_feature_mask is not None:
                    fit_ins.config["has_feature_selection"] = "true"
                    fit_ins.config["n_selected_features"] = str(int(np.sum(self.global_feature_mask)))

        return client_config_pairs

    def save_feature_selection_results(self, client_masks, client_metrics):
        """Save feature selection results to file"""
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir, exist_ok=True)

        results = {
            "timestamp": self.timestamp,
            "dataset": self.dataset_name,
            "model_type": self.model_type,
            "freedom_rate": self.freedom_rate,
        }

        # Add client-specific results if available
        if self.client_fs_results:
            results["client_results"] = self.client_fs_results

        # Add metrics from applying global mask if available
        if hasattr(self, "fs_metrics") and self.fs_metrics:
            results["global_mask_metrics"] = self.fs_metrics

        # Add global mask information if available
        if self.global_feature_mask is not None:
            results["global_feature_mask"] = {
                "n_total_features": len(self.global_feature_mask),
                "n_selected_features": int(np.sum(self.global_feature_mask)),
                "selected_indices": np.where(self.global_feature_mask)[0].tolist()
            }

        # Save to file
        fs_results_file = os.path.join(self.results_dir, "feature_selection_results.json")
        with open(fs_results_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"Feature selection results saved to {fs_results_file}")

    def save_final_results(self, metrics, client_results):
        """Save final evaluation results to file"""
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir, exist_ok=True)

        results = {
            "timestamp": self.timestamp,
            "dataset": self.dataset_name,
            "model_type": self.model_type,
            "num_rounds": self.num_rounds,
            "perform_fs": self.perform_fs,
            "freedom_rate": self.freedom_rate if self.perform_fs else None,
            "metrics": metrics,
            "client_results": [
                {
                    "client_id": client.cid,
                    "f1_score": res.metrics.get("f1_score", 0.0),
                    "model_size_bytes": res.metrics.get("model_size_bytes", 0.0),
                    "n_features": res.metrics.get("n_features", 0),
                    "num_examples": res.num_examples
                }
                for client, res in client_results
            ]
        }

        # Save to file
        final_results_file = os.path.join(self.results_dir, "final_results.json")
        with open(final_results_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"Final results saved to {final_results_file}")


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Flower server")
    parser.add_argument("--rounds", type=int, default=3, help="Number of rounds")
    parser.add_argument("--min-clients", type=int, default=3, help="Minimum number of clients")
    parser.add_argument("--model", type=str,
                        choices=["GNB", "SGDC", "LogReg", "MLPC"],
                        help="Model type")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--port", type=int, default=8080, help="Server port")
    parser.add_argument("--random-state", type=int, default=42, help="Random state for reproducibility")
    parser.add_argument("--feature-selection", "--fs", dest="feature_selection", action="store_true",
                        help="Enable feature selection")
    parser.add_argument("--freedom-rate", dest="freedom_rate", type=float, default=0.5,
                        help="Freedom rate for feature selection (0-1)")
    args = parser.parse_args()

    # Create output directory for models and results
    os.makedirs("./models", exist_ok=True)
    os.makedirs("./results", exist_ok=True)

    # Configure strategy
    strategy = FeatureSelectionStrategy(
        min_clients=args.min_clients,
        model_type=args.model,
        num_rounds=args.rounds,
        dataset_name=args.dataset,
        perform_fs=args.feature_selection,
        freedom_rate=args.freedom_rate,
        random_state=args.random_state,
        min_fit_clients=args.min_clients,
        min_available_clients=args.min_clients,
        min_evaluate_clients=args.min_clients,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
    )

    # Configure server
    server_config = fl.server.ServerConfig(num_rounds=args.rounds)

    # Print experiment start information
    print("=" * 50)
    print(f"Starting federated learning experiment:")
    print(f"  Dataset: {args.dataset}")
    print(f"  Model: {args.model}")
    print(f"  Rounds: {args.rounds}")
    print(f"  Minimum clients: {args.min_clients}")
    print(f"  Feature selection: {'Enabled' if args.feature_selection else 'Disabled'}")
    if args.feature_selection:
        print(f"  Freedom rate: {args.freedom_rate}")
    print("=" * 50)

    # Start the server using the current Flower API
    fl.server.start_server(
        server_address=f"0.0.0.0:{args.port}",
        config=server_config,
        strategy=strategy
    )


if __name__ == "__main__":
    main()