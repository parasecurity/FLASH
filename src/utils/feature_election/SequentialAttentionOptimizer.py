import optuna
import tensorflow as tf
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, f1_score
import logging

import optuna
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, f1_score
import logging

from utils.feature_election.sequential_attention import SequentialAttention


class SequentialAttentionOptimizer:
    def __init__(
            self,
            X,
            y,
            base_model,
            n_trials=100,
            cv_folds=4,
            random_state=42
    ):
        self.X = X
        self.y = y
        self.base_model = base_model
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.random_state = random_state

        # Create F1 scorer
        self.scorer = make_scorer(f1_score, average='weighted')

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def _evaluate_features(self, feature_indices):
        """Evaluate selected features using cross-validation with F1 score."""
        if len(feature_indices) == 0:
            return float('-inf')

        # Ensure feature_indices is a 1D numpy array
        if isinstance(feature_indices, tf.Tensor):
            feature_indices = feature_indices.numpy()
        feature_indices = np.asarray(feature_indices).ravel()

        # Handle both DataFrame and numpy array inputs
        if isinstance(self.X, pd.DataFrame):
            X_selected = self.X.iloc[:, feature_indices]
        else:
            X_selected = self.X[:, feature_indices]
        scores = cross_val_score(
            self.base_model,
            X_selected,
            self.y,
            cv=self.cv_folds,
            scoring=self.scorer,
            n_jobs=-1
        )
        return np.mean(scores)

    def objective(self, trial):
        """Optuna objective function using F1 score."""
        # Get parameters for SequentialAttention
        num_candidates_to_select_per_step = trial.suggest_int('num_candidates_to_select_per_step', 1, 5)

        # Ensure num_candidates_to_select is a multiple of num_candidates_to_select_per_step
        max_features = min(self.X.shape[1], int(0.5 * self.X.shape[1]))
        max_features = max_features - (max_features % num_candidates_to_select_per_step)

        num_candidates_to_select = trial.suggest_int(
            'num_candidates_to_select',
            num_candidates_to_select_per_step,
            max_features,
            step=num_candidates_to_select_per_step
        )

        params = {
            'num_candidates': self.X.shape[1],
            'num_candidates_to_select': num_candidates_to_select,
            'num_candidates_to_select_per_step': num_candidates_to_select_per_step,
            'start_percentage': trial.suggest_float('start_percentage', 0.1, 0.3),
            'stop_percentage': trial.suggest_float('stop_percentage', 0.7, 1.0)
        }

        try:
            # Initialize SequentialAttention
            attention = SequentialAttention(**params)

            # Run through selection process
            training_percentage = tf.Variable(attention._start_percentage, dtype=tf.float32)
            while training_percentage < attention._stop_percentage:
                weights = attention(training_percentage)
                training_percentage.assign_add(0.1)

            # Get final feature selection
            final_weights = attention(1.0)
            _, selected_indices = tf.math.top_k(
                final_weights,
                k=num_candidates_to_select
            )

            # Evaluate selected features using F1 score
            score = self._evaluate_features(selected_indices.numpy())

            # Add small penalty for number of features
            penalty_price = 0.0005
            feature_ratio = len(selected_indices) / self.X.shape[1]
            penalty = penalty_price * feature_ratio

            return score - penalty

        except Exception as e:
            self.logger.error(f"Trial failed with error: {str(e)}")
            return float('-inf')

    def optimize_and_select(self):
        """Run optimization process."""
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.random_state),  # Add sampler with seed
            pruner=optuna.pruners.SuccessiveHalvingPruner()
        )

        study.optimize(
            self.objective,
            n_trials=self.n_trials,
            show_progress_bar=True
        )

        # Get best parameters
        best_params = {
            'num_candidates': self.X.shape[1],
            **study.best_params
        }

        # Create final model with best parameters
        best_attention = SequentialAttention(**best_params)

        # Run final selection process
        training_percentage = tf.Variable(best_attention._start_percentage, dtype=tf.float32)
        while training_percentage < best_attention._stop_percentage:
            weights = best_attention(training_percentage)
            training_percentage.assign_add(0.1)

        # Get final weights and selected features
        final_weights = best_attention(1.0)
        final_selected = tf.where(final_weights > tf.reduce_mean(final_weights))[:, 0].numpy()
        final_score = self._evaluate_features(final_selected)

        self.results = {
            'best_parameters': best_params,
            'best_score': study.best_value,
            'final_f1_score': final_score,
            'attention_weights': final_weights.numpy(),
            'selected_features': final_selected
        }
        return self

    def align_arrays(self, total_num_of_features):
        weights = self.results['attention_weights']
        # Create zero-filled arrays for binary selection and weights
        binary_array = np.zeros(total_num_of_features, dtype=int)
        # weights_array = np.zeros(total_num_of_features, dtype=float)

        # Fill arrays using selected features and their corresponding weights
        selected_features = self.results['selected_features']
        binary_array[selected_features] = 1
        weights_array = weights  # Already aligned since attention weights are computed for all features

        # Normalize weights to [0, 1] range for consistency
        # if len(weights_array) > 0:  # Check if there are any weights
        #     weights_array = (weights_array - weights_array.min()) / (weights_array.max() - weights_array.min() + 1e-10)

        return binary_array, weights_array