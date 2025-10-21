import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

def load_cancer_rna_seq(data_path='cancerRNA/data.csv', labels_path='cancerRNA/labels.csv'):
    """
    Load and process the RNA-Seq cancer dataset.

    Parameters:
    -----------
    data_path : str
        Path to the gene expression data CSV file
    labels_path : str
        Path to the labels CSV file

    Returns:
    --------
    X : np.ndarray
        Gene expression data matrix
    y : np.ndarray
        Cancer type labels (encoded as integers)
    label_encoder : LabelEncoder
        Fitted label encoder for cancer types
    dataset_info : dict
        Dictionary containing dataset statistics and information
    """
    # Load the data
    print("Loading gene expression data...")
    X = pd.read_csv(data_path, index_col=0)  # Use first column as index

    print("Loading labels...")
    y = pd.read_csv(labels_path, index_col=0)  # Use first column as index

    print(f"\nInitial data shapes:")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

    # Basic data validation
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"Number of samples in data ({X.shape[0]}) and labels ({y.shape[0]}) don't match")

    # Convert labels to numeric using LabelEncoder
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y.iloc[:, 0])  # Assuming labels are in first column

    # Get dataset statistics
    dataset_info = {
        'n_samples': X.shape[0],
        'n_features': X.shape[1],
        'n_classes': len(label_encoder.classes_),
        'class_distribution': pd.Series(y_encoded).value_counts().to_dict(),
        'class_names': label_encoder.classes_.tolist(),
        'missing_values': X.isna().sum().sum(),
        'feature_stats': {
            'mean': X.mean().mean(),
            'std': X.std().mean(),
            'min': X.min().min(),
            'max': X.max().max()
        }
    }

    print("\nDataset Statistics:")
    print(f"Number of samples: {dataset_info['n_samples']}")
    print(f"Number of features (genes): {dataset_info['n_features']}")
    print(f"Number of classes: {dataset_info['n_classes']}")
    print("\nClass distribution:")
    for idx, count in dataset_info['class_distribution'].items():
        class_name = label_encoder.classes_[idx]
        print(f"{class_name}: {count} samples")

    # Print basic feature statistics
    print("\nFeature Statistics:")
    print(f"Mean expression value: {dataset_info['feature_stats']['mean']:.2f}")
    print(f"Standard deviation: {dataset_info['feature_stats']['std']:.2f}")
    print(f"Range: [{dataset_info['feature_stats']['min']:.2f}, {dataset_info['feature_stats']['max']:.2f}]")
    print(f"Missing values: {dataset_info['missing_values']}")

    # Convert X and y to numpy arrays for compatibility with sklearn
    X_numpy = X.to_numpy()
    y_numpy = y_encoded

    return X_numpy, y_numpy, label_encoder, dataset_info


def plot_class_distribution(y, label_encoder):
    """
    Plot the distribution of cancer types in the dataset.
    """
    plt.figure(figsize=(10, 6))
    # Updated countplot parameters to address deprecation warning
    sns.countplot(data=pd.DataFrame({'cancer_type': y}),
                 x='cancer_type',
                 hue='cancer_type',
                 palette='Set3',
                 legend=False)
    plt.xticks(range(len(label_encoder.classes_)), label_encoder.classes_, rotation=45)
    plt.title('Distribution of Cancer Types')
    plt.xlabel('Cancer Type')
    plt.ylabel('Number of Samples')
    plt.tight_layout()
    return plt.gcf()


def get_top_varying_genes(X, n_genes=50):
    """
    Get the top n genes with highest variance across samples.
    """
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)
    variances = X.var()
    top_genes = variances.nlargest(n_genes)
    return top_genes