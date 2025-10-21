import argparse
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
import flwr as fl
from typing import Dict, List, Optional, Tuple, Union
import json

# Import the feature selection implementation
try:
    from utils.feature_election.impetus import PyImpetusSelector
    from utils.feature_election.lasso_fs import LassoFeatureSelector
    from utils.feature_selection_utils import apply_feature_mask
except ImportError:
    # Create minimal implementation if not available
    class PyImpetusSelector:
        def __init__(self, task='classification', verbose=False, **kwargs):
            self.task = task
            self.verbose = verbose
            self.selected_features_mask = None
            self.feature_scores = None

        def fit(self, X, y):
            n_features = X.shape[1]
            # Simple implementation that selects top 50% features based on variance
            if isinstance(X, pd.DataFrame):
                variances = X.var().values
            else:
                variances = np.var(X, axis=0)

            # Select top 50% of features by variance
            threshold = np.median(variances)
            self.selected_features_mask = variances > threshold
            self.feature_scores = variances / np.sum(variances)
            return self

        def transform(self, X):
            if isinstance(X, pd.DataFrame):
                return X.iloc[:, self.selected_features_mask]
            return X[:, self.selected_features_mask]

        def align_arrays(self, n_features):
            binary_array = np.zeros(n_features)
            scores_array = np.zeros(n_features)
            if self.selected_features_mask is not None:
                binary_array[self.selected_features_mask] = 1
                scores_array = self.feature_scores
            return binary_array, scores_array


    class LassoFeatureSelector:
        """Fallback implementation of LassoFeatureSelector if the actual one is not available.
        This tries to match the behavior of the actual implementation as closely as possible."""

        def __init__(self, n_trials=150, random_state=None, timeout=300, max_iter=5000, n_splits=5, **kwargs):
            self.n_trials = n_trials
            self.timeout = timeout
            self.max_iter = max_iter
            self.n_splits = n_splits
            self.random_state = random_state
            self.selected_features_ = None
            self.best_alpha_ = None
            self.final_lasso = None
            self.n_features_total_ = None
            self.scaler_ = None

        def fit(self, X, y):
            """Simplified version of the actual implementation without Optuna."""
            from sklearn.preprocessing import MinMaxScaler
            from sklearn.linear_model import Lasso
            from sklearn.model_selection import StratifiedKFold
            from sklearn.metrics import f1_score
            import numpy as np

            # Convert to numpy array if DataFrame
            X_array = X.values if isinstance(X, pd.DataFrame) else X
            y_array = y.values if isinstance(y, pd.Series) else y

            # Scale features
            self.scaler_ = MinMaxScaler()
            X_scaled = self.scaler_.fit_transform(X_array)
            self.n_features_total_ = X_array.shape[1]

            # Initialize cross-validation
            skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

            # Test a range of alpha values
            best_score = -np.inf
            best_alpha = None

            for alpha in np.logspace(-8, 1, min(20, self.n_trials)):
                scores = []
                feature_counts = []

                for train_idx, valid_idx in skf.split(X_scaled, y_array):
                    X_train, X_valid = X_scaled[train_idx], X_scaled[valid_idx]
                    y_train, y_valid = y_array[train_idx], y_array[valid_idx]

                    # Train LASSO
                    lasso = Lasso(alpha=alpha, random_state=self.random_state, max_iter=self.max_iter)
                    lasso.fit(X_train, y_train)

                    # Get selected features
                    selected_features = np.abs(lasso.coef_) > 0
                    num_selected = np.sum(selected_features)

                    if num_selected > 0:
                        # Use selected features
                        X_valid_selected = X_valid[:, selected_features]
                        X_train_selected = X_train[:, selected_features]

                        # Retrain on selected features
                        lasso_selected = Lasso(alpha=alpha, random_state=self.random_state, max_iter=self.max_iter)
                        lasso_selected.fit(X_train_selected, y_train)
                        y_valid_pred = lasso_selected.predict(X_valid_selected)

                        scores.append(f1_score(y_valid, np.round(y_valid_pred), average='macro'))
                        feature_counts.append(num_selected)
                    else:
                        scores.append(float('-inf'))
                        feature_counts.append(0)

                # Calculate average score and penalty
                mean_score = np.mean(scores)
                mean_features = np.mean(feature_counts)

                # Simple feature ratio penalty
                feature_ratio = mean_features / self.n_features_total_
                penalty = 0
                if feature_ratio < 0.05:
                    penalty = 0.5
                elif feature_ratio > 0.5:
                    penalty = 0.2 * (feature_ratio - 0.5)

                objective_value = mean_score - penalty

                if objective_value > best_score:
                    best_score = objective_value
                    best_alpha = alpha

            # Store best alpha
            self.best_alpha_ = best_alpha

            # Final feature selection with best alpha
            self.final_lasso = Lasso(alpha=self.best_alpha_, random_state=self.random_state, max_iter=self.max_iter)
            self.final_lasso.fit(X_scaled, y_array)

            # Store selected features
            self.selected_features_ = np.where(np.abs(self.final_lasso.coef_) > 0)[0]

            # Ensure at least some features are selected
            if len(self.selected_features_) == 0:
                # Select top 5% features by coefficient magnitude
                top_k = max(1, int(0.05 * self.n_features_total_))
                coef_magnitudes = np.abs(self.final_lasso.coef_)
                self.selected_features_ = np.argsort(coef_magnitudes)[-top_k:]

            return self

        def transform(self, X):
            """Transform X by selecting features."""
            if self.selected_features_ is None:
                raise ValueError("Transformer not fitted. Call 'fit' first.")

            if isinstance(X, pd.DataFrame):
                return X.iloc[:, self.selected_features_]
            else:
                return X[:, self.selected_features_]

        def align_arrays(self, total_num_of_features):
            """Create binary mask and feature importance arrays."""
            if not hasattr(self, 'selected_features_') or self.selected_features_ is None:
                raise ValueError("Transformer not fitted. Call 'fit' first.")

            # Create binary array for selected features
            binary_array = np.zeros(total_num_of_features, dtype=int)
            binary_array[self.selected_features_] = 1

            # Get feature importance from Lasso coefficients
            weights_array = np.zeros(total_num_of_features)
            weights_array[self.selected_features_] = np.abs(self.final_lasso.coef_[self.selected_features_])

            # Normalize weights if possible
            if np.sum(weights_array) > 0:
                weights_array = weights_array / np.sum(weights_array)

            return binary_array, weights_array


    def apply_feature_mask(X, mask):
        """Apply feature mask to data X"""
        # Convert mask to boolean if it's binary 0/1
        if np.all(np.isin(mask, [0, 1])):
            mask = mask.astype(bool)

        # Handle both numpy arrays and pandas DataFrames
        if isinstance(X, np.ndarray):
            return X[:, mask]
        else:
            # Assume pandas DataFrame
            return X.iloc[:, mask]

# Path to dataset files
DATA_PATH = './datasets/'
np.random.seed(42)


# Define a NumPyClient
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, client_id, dataset_name, model_type, perform_fs=False, fs_method='impetus', random_state=42):
        self.client_id = client_id
        self.dataset_name = dataset_name
        self.model_type = model_type
        self.random_state = random_state
        self.scaler = StandardScaler()

        # Feature selection flags and properties
        self.perform_fs = perform_fs
        self.fs_method = fs_method
        self.feature_selector = None
        self.feature_mask = None
        self.feature_scores = None

        # Original data before feature selection
        self.X_train_full = None
        self.X_test_full = None

        # Flag to track if model has been properly initialized
        self.model_initialized = False

        # Track classes seen during training for consistent partial_fit
        self.classes = None

        print(f"Client {client_id} initialized with model type: {model_type} and random_state: {random_state}")
        print(f"Feature selection enabled: {perform_fs}, method: {fs_method}")

        # Load client data
        self.load_data()

        # Create model
        self.create_model()

        # Initialize with a simple fit
        self.simpleFit()

        print(f"Client {client_id} prepared with {len(self.y_train)} training samples")

    def load_data(self):
        """Load client data from CSV files"""
        # Construct file paths
        train_path = f"{DATA_PATH}{self.dataset_name}_federated/{self.dataset_name}_client_{self.client_id}_train.csv"
        test_path = f"{DATA_PATH}{self.dataset_name}_federated/{self.dataset_name}_client_{self.client_id}_test.csv"

        # Load data with a fixed index to ensure consistent ordering
        train_df = pd.read_csv(train_path, header=0)
        test_df = pd.read_csv(test_path, header=0)

        # Sort by index before dropping NA to ensure consistent row ordering
        train_df = train_df.sort_index().dropna()
        test_df = test_df.sort_index().dropna()

        # Prepare target encoder and transform
        self.target_encoder = LabelEncoder()
        # Sort unique values before fitting to ensure consistent encoding
        unique_targets = sorted(train_df['target'].unique())
        self.target_encoder.fit(unique_targets)
        self.y_train = self.target_encoder.transform(train_df['target'])

        # Transform test data, mapping unknown values to a new class
        self.y_test = np.array([
            self.target_encoder.transform([val])[0] if val in self.target_encoder.classes_
            else -1 for val in test_df['target']
        ])

        # Get categorical columns in a deterministic order
        categorical_columns = sorted(train_df.select_dtypes(include=['object', 'category']).columns)
        categorical_columns = [col for col in categorical_columns if col != 'target']

        # Initialize feature encoders dictionary
        self.feature_encoders = {}

        # Process each categorical feature
        X_train = train_df.drop('target', axis=1)
        X_test = test_df.drop('target', axis=1)

        for col in categorical_columns:
            encoder = LabelEncoder()
            # Fit on training data only
            encoder.fit(X_train[col])
            n_classes_feat = len(encoder.classes_)
            self.feature_encoders[col] = encoder

            # Transform training data
            X_train[col] = encoder.transform(X_train[col])
            # Transform test data, mapping unknown values to a new class
            X_test[col] = np.array([
                encoder.transform([val])[0] if val in encoder.classes_
                else n_classes_feat for val in X_test[col]
            ])

        # Save to class variable (both original and working copies)
        self.X_train_full = X_train
        self.X_test_full = X_test
        self.X_train = X_train
        self.X_test = X_test

    def perform_feature_selection(self):
        """Perform feature selection on client data"""
        if not self.perform_fs:
            return None, None

        print(f"Client {self.client_id} performing feature selection with {self.fs_method}")

        # Store original feature count for reporting
        original_feature_count = self.X_train_full.shape[1]

        # Initialize the feature selector
        if self.fs_method == 'impetus':
            self.feature_selector = PyImpetusSelector(task='classification', verbose=True)
        elif self.fs_method == 'lasso':
            # Initialize with parameters matching the actual implementation
            self.feature_selector = LassoFeatureSelector(
                n_trials=150,  # Default number of trials
                timeout=300,  # 5 minutes timeout
                max_iter=5000,  # Maximum iterations for Lasso
                n_splits=5,  # 5-fold cross-validation
                random_state=self.random_state
            )
        else:
            raise ValueError(f"Unsupported feature selection method: {self.fs_method}")

        # Get original model performance before feature selection
        original_scaler = StandardScaler()
        original_X_train_scaled = original_scaler.fit_transform(self.X_train_full)
        original_X_test_scaled = original_scaler.transform(self.X_test_full)

        # Need to create a separate model for original data evaluation to avoid dimension issues
        original_model = GaussianNB() if self.model_type == "GNB" else None
        if original_model:
            original_model.fit(original_X_train_scaled, self.y_train)
            score_before = f1_score(self.y_test, original_model.predict(original_X_test_scaled), average='weighted')
        else:
            # If we can't create a model, just report 0 for the score before
            score_before = 0.0

        # Fit the selector on training data
        self.feature_selector.fit(self.X_train_full, self.y_train)

        # Get binary array and scores for federated aggregation
        binary_array, scores_array = self.feature_selector.align_arrays(self.X_train_full.shape[1])

        # Apply feature selection to data
        self.X_train = self.feature_selector.transform(self.X_train_full)
        self.X_test = self.feature_selector.transform(self.X_test_full)

        print(
            f"Client {self.client_id}: Feature selection reduced features from {original_feature_count} to {self.X_train.shape[1]}")

        # CRITICAL: Recreate the model with the new feature dimensionality
        self.create_model(self.model_type, self.random_state)

        # Reset model initialization flag
        self.model_initialized = False

        # Evaluate new model with reduced features
        reduced_scaler = StandardScaler()
        reduced_X_train_scaled = reduced_scaler.fit_transform(self.X_train)
        reduced_X_test_scaled = reduced_scaler.transform(self.X_test)

        if hasattr(self.model, 'partial_fit') and callable(self.model.partial_fit):
            n_classes = len(self.target_encoder.classes_)
            all_classes = np.arange(n_classes + 1)  # +1 for unknown class
            self.model.partial_fit(reduced_X_train_scaled, self.y_train, classes=all_classes)
        else:
            self.model.fit(reduced_X_train_scaled, self.y_train)

        score_after = f1_score(self.y_test, self.model.predict(reduced_X_test_scaled), average='weighted')

        print(f"Client {self.client_id}: Performance - Before FS: {score_before:.4f}, After FS: {score_after:.4f}")

        # Store feature mask and scores for later reference
        self.feature_mask = binary_array
        self.feature_scores = scores_array

        # Update the scaler for future use
        self.scaler = reduced_scaler

        return binary_array, scores_array

    def apply_global_feature_mask(self, global_mask):
        """Apply a global feature mask received from the server"""
        if global_mask is None:
            return

        # Convert to boolean if necessary
        if np.all(np.isin(global_mask, [0, 1])):
            global_mask = global_mask.astype(bool)

        # Apply to data
        self.X_train = apply_feature_mask(self.X_train_full, global_mask)
        self.X_test = apply_feature_mask(self.X_test_full, global_mask)

        print(f"Client {self.client_id} applied global feature mask. New feature count: {self.X_train.shape[1]}")

        # Critical: Recreate the model with the new feature dimensionality
        # This is necessary because a model initialized with one feature count
        # cannot be used with a different feature count
        self.create_model(self.model_type, self.random_state)

        # Reset model initialization flag since we have a new model
        self.model_initialized = False

    def create_model(self, model_type=None, random_state=42):
        """Create model based on specified type"""
        model_type = model_type or self.model_type
        self.model_type = model_type
        self.random_state = random_state

        print(f"Creating model of type: {model_type} with random_state: {random_state}")

        if model_type == "GNB":
            self.model = GaussianNB()
        elif model_type == "SGDC":
            self.model = SGDClassifier(
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
                random_state=random_state
            )
        elif model_type == "MLPC":
            # MLPClassifier with appropriate settings
            self.model = MLPClassifier(
                warm_start=False
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def initialize_model(self, server_classes=None):
        """Initialize model by fitting it with data to create necessary attributes"""
        print(f"Initializing model for client {self.client_id} with {self.X_train.shape[1]} features")

        # Create a fresh scaler for initialization
        init_scaler = StandardScaler()
        X_train_scaled = init_scaler.fit_transform(self.X_train)

        # Determine classes to use
        classes_to_use = server_classes if server_classes is not None else np.unique(self.y_train)
        self.classes = classes_to_use

        try:
            # Initialize model based on type
            if hasattr(self.model, 'partial_fit') and callable(self.model.partial_fit):
                self.model.partial_fit(X_train_scaled, self.y_train, classes=classes_to_use)
            else:
                self.model.fit(X_train_scaled, self.y_train)

            # Ensure classes_ attribute is set
            if not hasattr(self.model, 'classes_'):
                self.model.classes_ = classes_to_use

            print(f"Model initialized and fit with {len(self.y_train)} samples and {self.X_train.shape[1]} features")
            self.model_initialized = True

            # Save the scaler we used for initialization
            self.scaler = init_scaler

        except Exception as e:
            print(f"Error initializing model: {e}")
            import traceback
            traceback.print_exc()
            print("Will try to continue with default initialization")

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Update model with parameters from server"""
        if not parameters or len(parameters) == 0:
            print("No parameters received from server, skipping parameter update")
            return

        try:
            # Check if model dimensions match parameters
            valid_params = True

            if self.model_type == "GNB":
                # Check if parameters have compatible shape with current model
                if hasattr(self.model, 'var_') and hasattr(self.model, 'theta_'):
                    if (parameters[0].shape != self.model.var_.shape or
                            parameters[1].shape != self.model.theta_.shape):
                        print(f"Parameter shape mismatch: var_ {parameters[0].shape} vs {self.model.var_.shape}, " +
                              f"theta_ {parameters[1].shape} vs {self.model.theta_.shape}")
                        valid_params = False
                else:
                    # Model not initialized, need to fit it first
                    print("GNB model attributes not found, performing initial fit")
                    self.simpleFit()
                    valid_params = False  # Skip parameter update this round

            elif self.model_type == "SGDC":
                # Check if parameters have compatible shape with current model
                if hasattr(self.model, 'coef_') and hasattr(self.model, 'intercept_'):
                    if (parameters[0].shape != self.model.coef_.shape or
                            parameters[1].shape != self.model.intercept_.shape):
                        print(f"Parameter shape mismatch: coef_ {parameters[0].shape} vs {self.model.coef_.shape}, " +
                              f"intercept_ {parameters[1].shape} vs {self.model.intercept_.shape}")
                        valid_params = False
                else:
                    # Model not initialized, need to fit it first
                    print("SGDC model attributes not found, performing initial fit")
                    self.simpleFit()
                    valid_params = False  # Skip parameter update this round

            elif self.model_type == "MLPC":
                # For MLPC we need to check if the model has been initialized with coefs_ and intercepts_
                if not hasattr(self.model, 'coefs_') or not hasattr(self.model, 'intercepts_'):
                    print("MLPC model not initialized, performing initial fit")
                    self.simpleFit()
                    valid_params = False  # Skip parameter update this round
                else:
                    # Check if parameter count matches the model structure
                    n_layers = len(self.model.coefs_)
                    if len(parameters) != n_layers * 2:  # Should have weights + biases
                        print(f"Parameter count mismatch: expected {n_layers * 2}, got {len(parameters)}")
                        valid_params = False
                    else:
                        # Check if each layer's shape matches
                        for i, (layer_weights, param_weights) in enumerate(
                                zip(self.model.coefs_, parameters[:n_layers])):
                            if layer_weights.shape != param_weights.shape:
                                print(f"Layer {i} shape mismatch: {layer_weights.shape} vs {param_weights.shape}")
                                valid_params = False
                                break

            # Only set parameters if dimensions match
            if valid_params:
                if self.model_type == "GNB":
                    self.model.var_ = parameters[0].copy()
                    self.model.theta_ = parameters[1].copy()

                elif self.model_type == "SGDC":
                    self.model.coef_ = parameters[0].copy()
                    self.model.intercept_ = parameters[1].copy()

                elif self.model_type == "MLPC":
                    n_layers = len(self.model.coefs_)
                    # Set weights
                    for i in range(n_layers):
                        self.model.coefs_[i] = parameters[i].copy()
                    # Set biases
                    for i in range(n_layers):
                        self.model.intercepts_[i] = parameters[n_layers + i].copy()

                print(f"Parameters successfully applied to model")
            else:
                print(f"Skipping parameter update due to dimension mismatch or uninitialized model")

        except Exception as e:
            print(f"Error setting parameters: {e}")
            import traceback
            traceback.print_exc()

    def simpleFit(self):
        """Perform initial model fitting to establish baseline performance"""
        # Model-specific random state settings
        if hasattr(self.model, 'random_state'):
            self.model.random_state = 42

        # Create a fresh scaler for the current data dimensionality
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(self.X_train)
        X_test_scaled = self.scaler.transform(self.X_test)

        # Include the extra class for unknowns in the classes parameter
        n_classes = len(self.target_encoder.classes_)
        all_classes = np.arange(n_classes + 1)  # +1 for unknown class

        try:
            print(f"Client {self.client_id}: Training initial model with {X_train_scaled.shape[1]} features")

            if hasattr(self.model, 'partial_fit') and callable(self.model.partial_fit):
                self.model.partial_fit(X_train_scaled, self.y_train, classes=all_classes)
            else:
                self.model.fit(X_train_scaled, self.y_train)

            # Mark model as initialized
            self.model_initialized = True

            # Calculate performance metrics
            train_pred = self.model.predict(X_train_scaled)
            test_pred = self.model.predict(X_test_scaled)

            res = {
                'test': f1_score(self.y_test, test_pred, average='weighted'),
                'train': f1_score(self.y_train, train_pred, average='weighted')
            }

            print(f"Client {self.client_id} initial fit - Train F1: {res['train']:.4f}, Test F1: {res['test']:.4f}")

            return res

        except Exception as e:
            print(f"Error in simpleFit: {e}")
            import traceback
            traceback.print_exc()

            # Return empty metrics
            return {
                'test': 0.0,
                'train': 0.0
            }

    def fit(self, parameters: List[np.ndarray], config: Dict[str, str]) -> Tuple[
        List[np.ndarray], int, Dict[str, float]]:
        """Train model on local data"""

        # Check if this is a feature selection round
        if config.get("feature_selection_round", "false").lower() == "true":
            try:
                # Perform feature selection - this will handle the separate models for before/after comparison
                binary_array, scores_array = self.perform_feature_selection()

                # Use the score values directly from perform_feature_selection
                # No need to recalculate here - that causes the dimension mismatch error

                # Return feature selection information - we don't return model parameters
                # because we're still in feature selection phase
                return [], len(self.y_train), {
                    "feature_selection": True,
                    "feature_mask": json.dumps(binary_array.tolist()),
                    "feature_scores": json.dumps(scores_array.tolist()),
                    "score_before_fs": f1_score(self.y_test, self.model.predict(self.scaler.transform(self.X_test)),
                                                average='weighted'),
                    "score_after_local_fs": f1_score(self.y_test,
                                                     self.model.predict(self.scaler.transform(self.X_test)),
                                                     average='weighted'),
                    "n_features_before": self.X_train_full.shape[1],
                    "n_features_after": self.X_train.shape[1]
                }
            except Exception as e:
                print(f"Error during feature selection: {e}")
                import traceback
                traceback.print_exc()

                # Return empty information with error
                return [], len(self.y_train), {
                    "feature_selection": True,
                    "error": str(e),
                    "n_features_before": self.X_train_full.shape[1] if hasattr(self, 'X_train_full') else 0
                }

        # Check if this is a round to apply the global feature mask
        elif config.get("apply_global_mask", "false").lower() == "true":
            try:
                # Get global mask from config
                if "global_feature_mask" in config:
                    global_mask = np.array(json.loads(config["global_feature_mask"]))

                    # Apply global feature mask and recreate the model
                    self.apply_global_feature_mask(global_mask)

                    # Create a new scaler for the reduced dimensions
                    self.scaler = StandardScaler()

                    # Initialize model with new feature dimensionality
                    n_classes = len(self.target_encoder.classes_)
                    all_classes = np.arange(n_classes + 1)  # +1 for unknown class

                    # Scale features with new feature dimensionality
                    X_train_scaled = self.scaler.fit_transform(self.X_train)
                    X_test_scaled = self.scaler.transform(self.X_test)

                    # Train the newly created model with the reduced feature set
                    if hasattr(self.model, 'partial_fit') and callable(self.model.partial_fit):
                        self.model.partial_fit(X_train_scaled, self.y_train, classes=all_classes)
                    else:
                        self.model.fit(X_train_scaled, self.y_train)

                    # Evaluate performance with global mask
                    score_with_global_mask = f1_score(self.y_test, self.model.predict(X_test_scaled),
                                                      average='weighted')

                    print(
                        f"Client {self.client_id}: Applied global feature mask, reduced to {self.X_train.shape[1]} features, F1 score: {score_with_global_mask:.4f}")

                    # Return metrics with applied global mask but NO MODEL PARAMETERS
                    # This ensures we don't try to aggregate models with different feature dimensions
                    return [], len(self.y_train), {
                        "applied_global_mask": True,
                        "score_with_global_mask": score_with_global_mask,
                        "n_features": self.X_train.shape[1]
                    }
            except Exception as e:
                print(f"Error applying global mask: {e}")
                import traceback
                traceback.print_exc()

                # Return empty information with error
                return [], len(self.y_train), {
                    "applied_global_mask": True,
                    "error": str(e)
                }

        # Normal training round - standard behavior
        try:
            # Try to set parameters if provided and compatible with current model
            if parameters and len(parameters) > 0:
                self.set_parameters(parameters)

            # Standard training procedure
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(self.X_train)

            # Print dimensionality for debugging
            print(f"Client {self.client_id}: Training model with {X_train_scaled.shape[1]} features")

            if hasattr(self.model, 'partial_fit') and callable(self.model.partial_fit):
                n_classes = len(self.target_encoder.classes_)
                all_classes = np.arange(n_classes + 1)  # +1 for unknown class
                self.model.partial_fit(X_train_scaled, self.y_train, classes=all_classes)
            else:
                self.model.fit(X_train_scaled, self.y_train)

            # Evaluate model
            X_test_scaled = self.scaler.transform(self.X_test)
            test_pred = self.model.predict(X_test_scaled)
            test_f1 = f1_score(self.y_test, test_pred, average='weighted')

            print(f"Client {self.client_id}: Training completed - Test F1: {test_f1:.4f}")

            # Return updated parameters with metrics
            return self.get_parameters({}), len(self.y_train), {"f1_score": float(test_f1)}

        except Exception as e:
            print(f"Error during fit: {e}")
            import traceback
            traceback.print_exc()

            # Return empty parameters and error message
            return [], len(self.y_train), {"error": str(e)}

    def get_parameters(self, config: Dict[str, str]) -> List[np.ndarray]:
        """Get model parameters to send to server"""
        try:
            if self.model_type == "GNB":
                if hasattr(self.model, 'var_') and hasattr(self.model, 'theta_'):
                    return [
                        self.model.var_.copy(),
                        self.model.theta_.copy(),
                    ]
                else:
                    # Model not fully initialized yet
                    print("GNB model not fully initialized, returning empty parameter list")
                    return []

            elif self.model_type == "SGDC":
                if hasattr(self.model, 'coef_') and hasattr(self.model, 'intercept_'):
                    return [
                        self.model.coef_.copy(),
                        self.model.intercept_.copy()
                    ]
                else:
                    # Model not fully initialized yet
                    print("SGDC model not fully initialized, returning empty parameter list")
                    return []

            elif self.model_type == "MLPC":
                # For MLPClassifier, we need to ensure the model has been fit first
                if hasattr(self.model, 'coefs_') and hasattr(self.model, 'intercepts_'):
                    weights = []
                    # Add all layer weights first
                    for layer_weights in self.model.coefs_:
                        weights.append(layer_weights.copy())
                    # Then add all biases
                    for layer_bias in self.model.intercepts_:
                        weights.append(layer_bias.copy())
                    return weights
                else:
                    # If the model hasn't been fit yet, train it on a small sample to initialize the weights
                    print("MLPC model not fully initialized, initializing with a mini-fit")

                    # Create a temporary scaler to avoid modifying the main one
                    temp_scaler = StandardScaler()
                    X_train_scaled = temp_scaler.fit_transform(self.X_train)

                    # Train the model with the first few samples to initialize weights
                    # Use a small subset to make this quick
                    sample_size = min(100, len(self.y_train))
                    X_sample = X_train_scaled[:sample_size]
                    y_sample = self.y_train[:sample_size]

                    # Fit the model to initialize weights
                    self.model.fit(X_sample, y_sample)

                    # Now extract the initialized weights
                    weights = []
                    for layer_weights in self.model.coefs_:
                        weights.append(layer_weights.copy())
                    for layer_bias in self.model.intercepts_:
                        weights.append(layer_bias.copy())
                    return weights
            else:
                print(f"Unsupported model type for parameters: {self.model_type}")
                return []

        except Exception as e:
            print(f"Error in get_parameters: {e}")
            import traceback
            traceback.print_exc()
            return []

    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, str]) -> Tuple[float, int, Dict[str, float]]:
        """Evaluate model on test data"""
        try:
            # Update model with server parameters
            self.set_parameters(parameters)

            # If model isn't initialized, try to initialize it
            if not self.model_initialized:
                print("Model not initialized for evaluation, initializing now...")
                self.initialize_model()

            # Scale features
            X_test_scaled = self.scaler.transform(self.X_test)

            # Evaluate model on test data
            y_pred = self.model.predict(X_test_scaled)
            loss = 1.0 - self.model.score(X_test_scaled, self.y_test)
            test_f1 = f1_score(self.y_test, y_pred, average='weighted')

            # Calculate model size
            model_size_bytes = self.calculate_model_size()

            print(f"Evaluation completed with Test F1 score: {test_f1:.4f}, Model size: {model_size_bytes} bytes")

            # Return test F1 as the primary metric and include model size
            return loss, len(self.y_test), {
                "f1_score": float(test_f1),
                "model_size_bytes": float(model_size_bytes),
                "n_features": self.X_train.shape[1]
            }

        except Exception as e:
            print(f"Error during evaluation: {e}")
            import traceback
            traceback.print_exc()

            # Return default metrics as fallback
            return 1.0, len(self.y_test), {"f1_score": 0.0, "error": str(e)}

    def calculate_model_size(self):
        """Calculate the size of the model parameters in bytes"""
        total_size = 0

        try:
            if self.model_type == "GNB":
                if hasattr(self.model, 'var_'):
                    total_size += self.model.var_.nbytes
                if hasattr(self.model, 'theta_'):
                    total_size += self.model.theta_.nbytes
                if hasattr(self.model, 'class_prior_'):
                    total_size += self.model.class_prior_.nbytes

            elif self.model_type == "SGDC":
                if hasattr(self.model, 'coef_'):
                    total_size += self.model.coef_.nbytes
                if hasattr(self.model, 'intercept_'):
                    total_size += self.model.intercept_.nbytes

            elif self.model_type == "MLPC":
                if hasattr(self.model, 'coefs_'):
                    for layer_weights in self.model.coefs_:
                        total_size += layer_weights.nbytes
                if hasattr(self.model, 'intercepts_'):
                    for layer_bias in self.model.intercepts_:
                        total_size += layer_bias.nbytes
        except Exception as e:
            print(f"Error calculating model size: {e}")
            import traceback
            traceback.print_exc()

        return total_size

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Flower client")
    parser.add_argument("--client-id", "--client_id", dest="client_id", type=int, required=True, help="Client ID")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--server", type=str, default="127.0.0.1:8080", help="Server address")
    parser.add_argument("--model", type=str, default="GNB",
                        choices=["GNB", "SGDC", "LogReg", "MLPC"],
                        help="Model type (default: GNB)")
    parser.add_argument("--random-state", "--random_state", dest="random_state", type=int, default=42,
                        help="Random state for reproducibility")
    parser.add_argument("--feature-selection", "--fs", dest="feature_selection", action="store_true",
                        help="Enable feature selection")
    parser.add_argument("--fs-method", dest="fs_method", type=str, default="impetus",
                        choices=["impetus", "lasso"],
                        help="Feature selection method (default: impetus)")

    # Print that arguments are being parsed and show command line args
    import sys
    print(f"Parsing arguments: {sys.argv}")

    # Try to parse known args first to debug
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"Warning: Unknown arguments: {unknown}")

    # Parse args fully
    args = parser.parse_args()

    # Check if dataset files exist
    dataset_dir = f"{DATA_PATH}{args.dataset}_federated"
    client_train_file = f"{dataset_dir}/{args.dataset}_client_{args.client_id}_train.csv"
    client_test_file = f"{dataset_dir}/{args.dataset}_client_{args.client_id}_test.csv"

    if not os.path.exists(client_train_file) or not os.path.exists(client_test_file):
        raise FileNotFoundError(f"Client data files not found for client {args.client_id}")

    # Create Flower client
    client = FlowerClient(
        client_id=args.client_id,
        dataset_name=args.dataset,
        model_type=args.model,
        perform_fs=args.feature_selection,
        fs_method=args.fs_method,
        random_state=args.random_state
    )

    # Print a message about starting the client with configuration
    print(f"Starting client {args.client_id} connection to {args.server}")
    print(f"Dataset: {args.dataset}, Model: {args.model}")
    print(f"Feature selection: {'Enabled' if args.feature_selection else 'Disabled'}")
    if args.feature_selection:
        print(f"Feature selection method: {args.fs_method}")
    print("=" * 50)

    # Start the client using the latest Flower API
    fl.client.start_numpy_client(server_address=args.server, client=client)

if __name__ == "__main__":
    main()