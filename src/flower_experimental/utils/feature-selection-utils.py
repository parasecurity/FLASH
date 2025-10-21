import numpy as np
import pandas as pd

def convert_to_dataframe_if_needed(X):
    """Convert numpy array to DataFrame if necessary"""
    if isinstance(X, np.ndarray):
        return pd.DataFrame(X)
    return X

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
