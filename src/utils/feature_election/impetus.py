from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import numpy as np
import pandas as pd
from PyImpetus import PPIMBC, PPIMBR
from typing import Optional, Union, List


class PyImpetusSelector(BaseEstimator, TransformerMixin):
    def __init__(
            self,
            task: str = 'classification',
            base_model: Optional[object] = None,
            p_val_thresh: float = 0.05,
            num_simul: int = 30,
            simul_size: float = 0.2,
            simul_type: int = 0,
            sig_test_type: str = "non-parametric",
            cv: Union[int, object] = 0,
            verbose: int = 2,
            random_state: Optional[int] = None,
            n_jobs: int = -1
    ):
        self.task = task
        self.base_model = base_model
        self.p_val_thresh = p_val_thresh
        self.num_simul = num_simul
        self.simul_size = simul_size
        self.simul_type = simul_type
        self.sig_test_type = sig_test_type
        self.cv = cv
        self.verbose = verbose
        self.random_state = random_state
        self.n_jobs = n_jobs

        if task not in ['classification', 'regression']:
            raise ValueError("Task must be either 'classification' or 'regression'")

        if self.base_model is None:
            if self.task == 'classification':
                self.base_model = DecisionTreeClassifier(random_state=random_state)
            else:
                self.base_model = DecisionTreeRegressor(random_state=random_state)

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> 'PyImpetusSelector':


        # Input validation
        # X, y = check_X_y(X, y, ensure_min_features=2)
        # Convert to np
        if isinstance(y, pd.Series):
            y = y.values
        if isinstance(X, pd.DataFrame):
            X = X.values

        # # Get unique labels
        # unique_labels = np.unique(y)
        # if len(unique_labels) < 2:
        #     raise ValueError(f"Need at least 2 labels, got {unique_labels}")

        self.n_features_in_ = X.shape[1]

        # Store original feature indices
        self.feature_indices_ = np.arange(X.shape[1])

        # Convert to pandas with consistent feature naming
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])

        # Store original column names if DataFrame
        # self.original_columns_ = X.columns.tolist()

        # Initialize appropriate PyImpetus model
        if self.task == 'classification':
            self.impetus_ = PPIMBC(
                model=self.base_model,
                p_val_thresh=self.p_val_thresh,
                num_simul=self.num_simul,
                simul_size=self.simul_size,
                simul_type=self.simul_type,
                sig_test_type=self.sig_test_type,
                cv=self.cv,
                verbose=self.verbose,
                random_state=self.random_state,
                n_jobs=self.n_jobs
            )
        else:
            self.impetus_ = PPIMBR(
                model=self.base_model,
                p_val_thresh=self.p_val_thresh,
                num_simul=self.num_simul,
                simul_size=self.simul_size,
                sig_test_type=self.sig_test_type,
                cv=self.cv,
                verbose=self.verbose,
                random_state=self.random_state,
                n_jobs=self.n_jobs
            )

        # Fit the model
        self.impetus_.fit(X, y)

        # Store selected features, importance scores, indices
        self.selected_features_ = self.impetus_.MB
        self.feature_importances_ = self.impetus_.feat_imp_scores
        self.selected_indices_ = [int(f.split('_')[1]) for f in self.selected_features_]

        # print("Selected features:", self.selected_features_)
        # print("Selected indices:", self.selected_indices_)
        # print("f importance:", self.feature_importances_)
        # print("f importance:", self.MB)
        return self

    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
        check_is_fitted(self)
        # Handle pandas DataFrame
        if isinstance(X, pd.DataFrame):
            return X.iloc[:, self.selected_indices_]
        # Handle numpy array
        elif isinstance(X, np.ndarray):
            return X[:, self.selected_indices_]
        else:
            raise TypeError("Input must be a pandas DataFrame or numpy array")

    def fit_transform(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> Union[
        np.ndarray, pd.DataFrame]:
        return self.fit(X, y).transform(X)

    # def get_feature_names_out(self, feature_names_in: Optional[List[str]] = None) -> np.ndarray:
    #     check_is_fitted(self)
    #     if feature_names_in is None:
    #         feature_names_in = self.feature_names_in_
    #     return np.array(feature_names_in)[self.selected_indices_]

    def plot_feature_importance(self):
        check_is_fitted(self)
        self.impetus_.feature_importance()

    def align_arrays(self, total_num_of_features):
        """
        Creates two aligned arrays from parallel indices and scores lists:
        1. A binary array with 1s at the specified indices
        2. A scores array with scores at the specified indices
        Both arrays are sorted by index position.
        """
        indices = self.selected_indices_
        scores = self.feature_importances_
        # Input validation
        if len(indices) != len(scores):
            raise ValueError("indices and scores must have the same length")

        # Sort both arrays based on indices
        sorted_pairs = sorted(zip(indices, scores))
        sorted_indices, sorted_scores = zip(*sorted_pairs)

        # Create zero-filled arrays
        binary_array = np.zeros(total_num_of_features, dtype=int)
        scores_array = np.zeros(total_num_of_features, dtype=float)

        # Fill arrays at specified indices
        for idx, score in zip(sorted_indices, sorted_scores):
            binary_array[idx] = 1
            scores_array[idx] = score

        return binary_array, scores_array