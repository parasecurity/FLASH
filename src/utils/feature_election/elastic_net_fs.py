import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import root_mean_squared_error
import optuna
from optuna.pruners import SuccessiveHalvingPruner


class ElasticNetFeatureSelector(BaseEstimator, TransformerMixin):
    """ElasticNet-based feature selection with optuna optimization and successive halving pruning."""

    def __init__(
            self,
            n_trials: int = 100,
            timeout: int = 60,
            feature_penalty: float = 0.01,
            reduction_factor: int = 3,
            min_resource: int = 1,
            min_early_stopping_rate: int = 0,
            max_iter: int = 2000,
            feature_threshold: float = 1e-5,
            l1_ratio_bounds: Tuple[float, float] = (0.1, 0.9)  # Bounds for l1_ratio
    ) -> None:
        self.n_trials = n_trials
        self.timeout = timeout
        self.feature_penalty = feature_penalty
        self.reduction_factor = reduction_factor
        self.min_resource = min_resource
        self.min_early_stopping_rate = min_early_stopping_rate
        self.max_iter = max_iter
        self.feature_threshold = feature_threshold
        self.l1_ratio_bounds = l1_ratio_bounds
        self.selected_features_: Optional[np.ndarray] = None
        self.best_alpha_: Optional[float] = None
        self.best_l1_ratio_: Optional[float] = None
        self.scaler_: Optional[MinMaxScaler] = None
        self.rung_levels = int(np.log(n_trials) / np.log(reduction_factor))
        self.n_features_total_: Optional[int] = None

    def _get_resource_allocation(self, trial):
        """Get resource allocation for the trial."""
        trials = trial.study.get_trials()
        current_trial = next(t for t in trials if t.number == trial.number)
        rung = current_trial.last_step if hasattr(current_trial, 'last_step') else None
        
        if rung is None:
            return self.min_resource / self.rung_levels
        return min(1.0, (rung + 1) / self.rung_levels)

    def _calculate_feature_ratio_penalty(self, num_selected: int) -> float:
        """Calculate penalty based on ratio of selected features."""
        if self.n_features_total_ is None:
            raise ValueError("n_features_total_ not set")
        
        if num_selected == 0:
            return float('inf')
            
        ratio = num_selected / self.n_features_total_
        min_ratio = 0.05  # At least 5% of features
        max_ratio = 0.7   # At most 50% of features
        
        if ratio < min_ratio:
            return 10.0 * (min_ratio - ratio)
        elif ratio > max_ratio:
            return 5.0 * (ratio - max_ratio)
        return 0.0

    def _calculate_stability_score(self, coef: np.ndarray) -> float:
        """Calculate stability score based on coefficient distribution."""
        nonzero_coef = coef[np.abs(coef) > self.feature_threshold]
        if len(nonzero_coef) == 0:
            return 0.0
        
        # Calculate coefficient of variation (normalized standard deviation)
        cv = np.std(np.abs(nonzero_coef)) / (np.mean(np.abs(nonzero_coef)) + 1e-10)
        return 1.0 / (1.0 + cv)  # Higher score for more stable coefficients

    def objective(self,
                  trial: optuna.Trial,
                  X: np.ndarray,
                  y: np.ndarray) -> float:
        """Objective function for Optuna optimization with balanced feature selection."""
        resource_fraction = self._get_resource_allocation(trial)
        n_samples = int(len(X_train) * resource_fraction)

        rng = np.random.RandomState(42 + trial.number)
        subset_indices = rng.choice(len(X_train), n_samples, replace=False)
        X_train_subset = X_train[subset_indices]
        y_train_subset = y_train[subset_indices]

        # Suggest both alpha and l1_ratio
        alpha = trial.suggest_float("alpha", 1e-10, 1.0, log=True)
        l1_ratio = trial.suggest_float("l1_ratio", *self.l1_ratio_bounds)

        # Train ElasticNet
        enet = ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            random_state=42,
            max_iter=self.max_iter
        )
        enet.fit(X_train_subset, y_train_subset)

        # Feature selection with threshold
        selected_features = np.abs(enet.coef_) > self.feature_threshold
        num_selected_features = np.sum(selected_features)

        # Calculate stability score
        stability_score = self._calculate_stability_score(enet.coef_)

        # Calculate RMSE using selected features
        X_valid_selected = X_valid[:, selected_features]
        X_train_subset_selected = X_train_subset[:, selected_features]


        # Calculate penalties
        feature_penalty = self._calculate_feature_ratio_penalty(num_selected_features)
        
        # Combine metrics with stability score
        objective_value = (
            rmse +
            self.feature_penalty * feature_penalty -
            0.1 * stability_score  # Encourage stability
        )

        # Report for pruning
        trial.report(objective_value, step=int(resource_fraction * 10))
        if trial.should_prune():
            raise optuna.TrialPruned()

        trial.set_user_attr("num_selected_features", num_selected_features)
        trial.set_user_attr("stability_score", stability_score)
        return objective_value

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'ElasticNetFeatureSelector':
        """Fit the feature selector with balanced optimization."""
        self.scaler_ = MinMaxScaler()
        X_scaled: np.ndarray = self.scaler_.fit_transform(X)
        self.n_features_total_ = X.shape[1]

        skf: StratifiedKFold = StratifiedKFold(n_splits=3, random_state=42, shuffle=True)
        train_idx, valid_idx = next(skf.split(X_scaled, y))

        X_train = X_scaled[train_idx]
        X_valid = X_scaled[valid_idx]
        y_train = y[train_idx]
        y_valid = y[valid_idx]

        pruner = SuccessiveHalvingPruner(
            reduction_factor=self.reduction_factor,
            min_resource=self.min_resource,
            min_early_stopping_rate=self.min_early_stopping_rate
        )

        study = optuna.create_study(direction='minimize', pruner=pruner)
        study.optimize(
            lambda trial: self.objective(trial, X_train, y_train, X_valid, y_valid),
            n_trials=self.n_trials,
            timeout=self.timeout
        )

        # Store best parameters
        self.best_alpha_ = study.best_params['alpha']
        self.best_l1_ratio_ = study.best_params['l1_ratio']

        # Final feature selection with best parameters
        final_enet = ElasticNet(
            alpha=self.best_alpha_,
            l1_ratio=self.best_l1_ratio_,
            random_state=42,
            max_iter=self.max_iter
        )
        final_enet.fit(X_scaled, y)
        self.selected_features_ = np.where(np.abs(final_enet.coef_) > 0)[0]

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.scaler_ is None:
            raise ValueError("Transformer not fitted. Call 'fit' first.")
        # X_scaled: np.ndarray = self.scaler_.transform(X)
        # return X_scaled[:, self.selected_features_]
        return X[:, self.selected_features_]

    def get_support(self, indices: bool = False) -> Union[np.ndarray, List[int]]:
        if self.scaler_ is None:
            raise ValueError("Transformer not fitted. Call 'fit' first.")
        mask: np.ndarray = np.zeros(self.scaler_.n_features_in_, dtype=bool)
        mask[self.selected_features_] = True
        return mask if not indices else self.selected_features_
