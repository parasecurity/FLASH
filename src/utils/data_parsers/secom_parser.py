import pandas as pd
import numpy as np
from datetime import datetime
# https://archive.ics.uci.edu/dataset/179/secom

def load_secom_dataset( verbose=False,
                        data_path='secom/secom.data',
                        labels_path='secom/secom_labels.data',
                        names_path='secom/secom.names'):
    """
    Load SECOM dataset from the three standard files:
    - secom.data: Main features data
    - secom_labels.data: Labels and timestamps
    - secom.names: Feature names (if available)

    Returns:
    - X: DataFrame with features
    - y: Series with labels (-1 for pass, 1 for fail)
    - timestamps: Series with timestamps (if available)
    """
    # Load main data
    try:
        X = pd.read_csv(data_path, delimiter=' ', header=None)
        print(f"Successfully loaded data with shape: {X.shape}")
    except Exception as e:
        print(f"Error loading data file: {e}")
        return None, None, None

    # Load labels
    try:
        labels_data = pd.read_csv(labels_path, delimiter=' ', header=None)
        print(f"Labels data shape: {labels_data.shape}")

        # Extract labels (first column)
        y = labels_data[0]  # First column is the label (-1: pass, 1: fail)

        # Handle timestamps if they exist
        if labels_data.shape[1] > 1:
            # If there are additional columns, combine them for timestamp
            timestamp_cols = labels_data.iloc[:, 1:]
            timestamps = pd.Series([' '.join(map(str, row)) for row in timestamp_cols.values])
        else:
            timestamps = pd.Series([None] * len(y))

    except Exception as e:
        print(f"Error loading labels file: {e}")
        return None, None, None

    # Try to load feature names if available
    try:
        with open(names_path, 'r') as f:
            feature_names = [line.strip() for line in f if line.strip()]
        if len(feature_names) == X.shape[1]:
            X.columns = feature_names
    except (FileNotFoundError, IOError):
        X.columns = [f'feature_{i}' for i in range(X.shape[1])]

    # Basic preprocessing
    # Replace any string NaN values with numpy NaN
    X = X.replace(['NA', 'nan', 'NaN'], np.nan)

    # Get info about missing values
    missing_stats = {
        'total_missing': X.isna().sum().sum(),
        'features_with_missing': X.columns[X.isna().any()].tolist()
    }

    # Fill missing values with median of each column
    X = X.fillna(X.median())

    if verbose:
        # Print dataset information
        print("\nDataset Statistics:")
        print(f"Number of samples: {len(X)}")
        print(f"Number of features: {X.shape[1]}")
        print(f"Number of fails: {sum(y == 1)}")
        print(f"Number of passes: {sum(y == -1)}")
        print(f"Missing values found: {missing_stats['total_missing']}")
        print(f"Features with missing values: {len(missing_stats['features_with_missing'])}")

    return X, y, timestamps

# from split_data import split_data
