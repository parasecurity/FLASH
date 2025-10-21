import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif

class PyImpetusSelector:
    """
    Implementation of Impetus feature selection for classification tasks.
    This is a simplified version that ranks features based on mutual information
    and random forest feature importance.
    """
    
    def __init__(self, task='classification', n_estimators=100, random_state=42, verbose=False):
        self.task = task
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.verbose = verbose
        self.selected_features_mask = None
        self.feature_scores = None
        self.selected_indices = None
        self.threshold = 0.01  # Minimum score to keep a feature
        
    def fit(self, X, y):
        """Fit the feature selector to the data."""
        # Calculate mutual information
        n_features = X.shape[1]
        mi_scores = mutual_info_classif(X, y, random_state=self.random_state)
        
        # Normalize mutual information scores
        mi_scores = mi_scores / np.sum(mi_scores) if np.sum(mi_scores) > 0 else np.zeros(n_features)
        
        # Train a random forest classifier
        rf = RandomForestClassifier(
            n_estimators=self.n_estimators, 
            random_state=self.random_state
        )
        rf.fit(X, y)
        
        # Get feature importances from random forest
        rf_importances = rf.feature_importances_
        
        # Combine scores (simple average)
        combined_scores = (mi_scores + rf_importances) / 2
        
        # Select features above threshold
        self.feature_scores = combined_scores
        self.selected_indices = np.where(combined_scores > self.threshold)[0]
        self.selected_features_mask = np.zeros(n_features, dtype=bool)
        self.selected_features_mask[self.selected_indices] = True
        
        if self.verbose:
            print(f"PyImpetus selected {np.sum(self.selected_features_mask)} out of {n_features} features")
        
        return self
    
    def transform(self, X):
        """Transform data using selected features."""
        if self.selected_features_mask is None:
            raise ValueError("You must call fit before transform")
        
        if isinstance(X, np.ndarray):
            return X[:, self.selected_features_mask]
        else:
            # Assume it's a pandas DataFrame
            return X.iloc[:, self.selected_features_mask]
    
    def fit_transform(self, X, y):
        """Fit to data, then transform it."""
        return self.fit(X, y).transform(X)
    
    def align_arrays(self, n_features):
        """
        Create binary array and scores array for federated learning communication.
        
        Args:
            n_features: Total number of features in the dataset
            
        Returns:
            binary_array: Binary array indicating selected features (1) or not (0)
            scores_array: Array of feature importance scores
        """
        binary_array = np.zeros(n_features)
        scores_array = np.zeros(n_features)
        
        if self.selected_indices is not None:
            binary_array[self.selected_indices] = 1
            scores_array = self.feature_scores
            
        return binary_array, scores_array
