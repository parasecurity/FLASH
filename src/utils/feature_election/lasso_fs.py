import numpy as np
from typing import List, Optional, Union

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Lasso
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, root_mean_squared_error
import optuna
# from optuna.pruners import SuccessiveHalvingPruner


class LassoFeatureSelector(BaseEstimator, TransformerMixin):
    """LASSO-based feature selection with optuna optimization and successive halving pruning."""

    def __init__(
            self,
            n_trials: int = 2000,
            timeout: int = 300,
            reduction_factor: int = 3,
            min_resource: int = 1,
            min_early_stopping_rate: int = 0,
            max_iter: int = 5000,
            n_splits: int = 5,
            random_state: Optional[int] = 42,
    ) -> None:
        self.n_trials = n_trials
        self.timeout = timeout
        self.reduction_factor = reduction_factor
        self.min_resource = min_resource
        self.min_early_stopping_rate = min_early_stopping_rate
        self.max_iter = max_iter
        self.n_splits = n_splits
        self.random_state = random_state  # Add this line
        self.selected_features_: Optional[np.ndarray] = None
        self.best_alpha_: Optional[float] = None
        self.scaler_: Optional[MinMaxScaler] = None
        self.rung_levels = int(np.log(n_trials) / np.log(reduction_factor))
        self.n_features_total_: Optional[int] = None

    def objective(self,
                  trial: optuna.Trial,
                  X: Union[np.ndarray, pd.DataFrame],
                  y: Union[np.ndarray, pd.Series]) -> float:
        """Improved objective function with k-fold cross-validation"""
        alpha: float = trial.suggest_float("alpha", 1e-8, 10.0, log=True)

        # Initialize cross-validation
        scores = []
        feature_counts = []
        # Replace this line in objective():
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

        # Convert pandas objects to numpy arrays for indexing
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        y_array = y.values if isinstance(y, pd.Series) else y

        # K-fold cross-validation in objective
        for train_idx, valid_idx in skf.split(X_array, y_array):
            X_train, X_valid = X_array[train_idx], X_array[valid_idx]
            y_train, y_valid = y_array[train_idx], y_array[valid_idx]

            # Train LASSO
            lasso = Lasso(alpha=alpha, random_state=42, max_iter=self.max_iter)
            lasso.fit(X_train, y_train)

            selected_features = np.abs(lasso.coef_) > 0
            num_selected = np.sum(selected_features)

            if num_selected > 0:
                # Use selected features
                X_valid_selected = X_valid[:, selected_features]
                X_train_selected = X_train[:, selected_features]

                # Retrain on selected features
                lasso_selected = Lasso(alpha=alpha, random_state=42, max_iter=self.max_iter)
                lasso_selected.fit(X_train_selected, y_train)
                y_valid_pred = lasso_selected.predict(X_valid_selected)

                scores.append(f1_score(y_valid, np.round(y_valid_pred), average='macro'))
                feature_counts.append(num_selected)
            else:
                scores.append(float('-inf'))
                feature_counts.append(0)

        # Average score across folds
        mean_score = np.mean(scores)
        mean_features = np.mean(feature_counts)

        # Calculate feature penalty
        feature_penalty = self._calculate_feature_ratio_penalty(mean_features)
        objective_value = mean_score - feature_penalty

        # Store metrics
        trial.set_user_attr("num_selected_features", mean_features)
        trial.set_user_attr("f1_score", mean_score)
        trial.set_user_attr("feature_penalty", feature_penalty)

        return objective_value

    def _calculate_feature_ratio_penalty(self, num_selected: float) -> float:
        """Calculate feature ratio penalty using logarithmic scaling."""
        if self.n_features_total_ is None:
            raise ValueError("n_features_total_ not set")

        if num_selected == 0:
            return float('inf')

        ratio = num_selected / self.n_features_total_
        min_ratio = 0.05  # Minimum acceptable ratio (5% of features)
        max_ratio = 0.5  # Maximum acceptable ratio (50% of features)

        # Base of the logarithm - affects how quickly penalty grows
        log_base = 2

        if ratio < min_ratio:
            penalty = np.log(min_ratio / ratio + 1) / np.log(log_base)
            return 5.0 * penalty
        elif ratio > max_ratio:
            penalty = np.log(ratio / max_ratio + 1) / np.log(log_base)
            return 0.5 * penalty

        return 0.0

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> 'LassoFeatureSelector':
        """Fit the feature selector."""
        # Convert to numpy array if necessary
        X_array = X.values if isinstance(X, pd.DataFrame) else X

        self.scaler_ = MinMaxScaler()
        X_scaled: np.ndarray = self.scaler_.fit_transform(X_array)
        self.n_features_total_ = X_array.shape[1]

        study = optuna.create_study(
            sampler=optuna.samplers.TPESampler(seed=self.random_state),  # Add sampler with seed
            pruner=optuna.pruners.SuccessiveHalvingPruner(),
            direction='maximize'
        )

        study.optimize(
            lambda trial: self.objective(trial, X_scaled, y),
            n_trials=self.n_trials,
            timeout=self.timeout
        )

        self.best_alpha_ = study.best_params['alpha']
        # Final feature selection with best alpha
        self.final_lasso = Lasso(alpha=self.best_alpha_, random_state=42, max_iter=self.max_iter)
        self.final_lasso.fit(X_scaled, y)
        self.selected_features_ = np.where(np.abs(self.final_lasso.coef_) > 0)[0]
        return self

    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
        """Transform X by selecting features."""
        if self.scaler_ is None or self.selected_features_ is None:
            raise ValueError("Transformer not fitted. Call 'fit' first.")

        if isinstance(X, pd.DataFrame):
            return X.iloc[:, self.selected_features_]
        elif isinstance(X, np.ndarray):
            return X[:, self.selected_features_]
        else:
            raise TypeError("Input must be a pandas DataFrame or numpy array")

    def align_arrays(self, total_num_of_features: int) -> tuple[np.ndarray, np.ndarray]:
        if self.scaler_ is None or not hasattr(self, 'selected_features_'):
            raise ValueError("Transformer not fitted. Call 'fit' first.")

        # Create binary array for selected features
        binary_array = np.zeros(total_num_of_features, dtype=int)
        binary_array[self.selected_features_] = 1

        # The absolute value is used as the magnitude denotes importance
        weights_array = np.abs(self.final_lasso.coef_)

        # Normalize weights to [0, 1] range for consistency
        # if len(weights_array) > 0:
        #     weights_array = (weights_array - weights_array.min()) / (weights_array.max() - weights_array.min() + 1e-10)
        # print(binary_array, weights_array)
        return binary_array, weights_array

    def get_support(self, indices: bool = False) -> Union[np.ndarray, List[int]]:
        """Get selected feature mask or indices."""
        if self.scaler_ is None:
            raise ValueError("Transformer not fitted. Call 'fit' first.")
        mask: np.ndarray = np.zeros(self.scaler_.n_features_in_, dtype=bool)
        mask[self.selected_features_] = True
        return mask if not indices else self.selected_features_