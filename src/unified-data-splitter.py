import os
import pandas as pd
import numpy as np
from datasets import load_dataset
from sklearn.datasets import fetch_openml
from sklearn.model_selection import StratifiedKFold, train_test_split
from typing import Union, Tuple, List, Optional
from utils.data_parsers.cancerRNA_parser import load_cancer_rna_seq
from utils.data_parsers.secom_parser import load_secom_dataset
from utils.data_parsers.multi_dataset_loader import MicrobiomeMetabolomeDataLoader
from ucimlrepo import fetch_ucirepo

np.random.seed(42)
TUANDROMD_path = "TUANDROMD.csv"

class UnifiedDataSplitter:
    """
    A unified interface for splitting different types of datasets into client portions
    while maintaining stratification and handling different data formats.
    """
    def __init__(self):
        self.supported_loaders = {
            'secom': self._load_secom_data,
            'cancer_rna': self._load_cancer_rna_data,
            'microbiome': self._load_microbiome_data,
            'income': self._load_adult_sensus_income,
            'mushroom': self._load_mushroom,
            'heart_disease': self._load_heart_disease_kaggle,
            'breast_cancer': self._load_breast_cancer,
            'mnist': self._load_mnist,
            'phrasebank': self._load_phrasebank,
            'wine': self._load_wine,
            'TUANDROMD': self._load_tuandromd
            # 'heart_disease_kaggle': self._load_heart_disease_kaggle
        }

    def _load_heart_disease_kaggle(self) -> Tuple[pd.DataFrame, pd.Series]:
        # Load the dataset
        data = pd.read_csv("./downloaded_datasets/heart_2020_cleaned.csv")

        # Rename the target column to 'target' for internal processing
        data = data.rename(columns={'HeartDisease': 'target'})

        return data.drop(columns=['target']), data['target']

    def _load_phrasebank(self) -> Tuple[pd.DataFrame, pd.Series]:
        # Load the dataset
        dataset = load_dataset("financial_phrasebank", "sentences_allagree")

        # Convert to pandas DataFrame
        df = pd.DataFrame(dataset['train'])

        # Split into features and target
        X = pd.DataFrame(df['sentence'])
        y = pd.Series(df['label'], name='target')

        return X, y

    def _load_mnist(self):
        # Fetch dataset with as_frame=True to get pandas structures
        X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=True)

        # Ensure X is DataFrame (it should be already, but being defensive)
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X

        # Ensure y is Series (it should be already, but being defensive)
        y = pd.Series(y) if not isinstance(y, pd.Series) else y
        y.name = 'target'

        # Normalize the pixel values
        X = X / 255.0

        return X, y

    def _load_wine(self):
        # fetch dataset
        wine_quality = fetch_ucirepo(id=186)

        # data (as pandas dataframes)
        X = wine_quality.data.features
        y = wine_quality.data.targets

        # Handle potential multi-column case for targets
        y = y.iloc[:, 0] if y.shape[1] == 1 else y.iloc[:, 0]
        y.name = 'target'

        return X, y

    def _load_breast_cancer(self) -> Tuple[pd.DataFrame, pd.Series]:

        # fetch dataset
        breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17)

        X = breast_cancer_wisconsin_diagnostic.data.features
        y = breast_cancer_wisconsin_diagnostic.data.targets

        # y.name = 'target'
        y = y.iloc[:, 0] if y.shape[1] == 1 else y.iloc[:, 0]  # Take first column if multiple exist
        y.name = 'target'
        return X, y

    import pandas as pd
    from typing import Tuple

    def _load_tuandromd(self) -> Tuple[pd.DataFrame, pd.Series]:
        # Load the dataset
        data = pd.read_csv(TUANDROMD_path)

        # Remove rows with NaN values in the target column
        data = data.dropna(subset=[data.columns[-1]])

        # Separate features and target
        X = data.iloc[:, :-1]  # All columns except the last one as DataFrame
        y = data.iloc[:, -1]  # Last column as Series

        # Ensure y is a Series with name
        y = pd.Series(y, name='target')

        return X, y

    def _load_heart_disease(self):
        # fetch dataset
        heart = fetch_ucirepo(id=45)

        # data as pandas dataframes
        X = heart.data.features
        y = heart.data.targets

        # Handle potential multi-column case
        y = y.iloc[:, 0] if y.shape[1] == 1 else y.iloc[:, 0]
        y.name = 'target'

        return X, y

    def _load_adult_sensus_income(self) -> Tuple[pd.DataFrame, pd.Series]:
        # fetch dataset
        adult = fetch_ucirepo(id=2)

        # data (as pandas dataframes)
        X = adult.data.features
        y = adult.data.targets
        y = y.iloc[:, 0] if y.shape[1] == 1 else y.iloc[:, 0]  # Take first column if multiple exist
        y.name = 'target'
        return X, y

    def _load_mushroom(self) -> Tuple[pd.DataFrame, pd.Series]:
        from ucimlrepo import fetch_ucirepo

        # Fetch dataset
        mushroom = fetch_ucirepo(id=73)

        # Convert features to DataFrame
        X = pd.DataFrame(mushroom.data.features) if not isinstance(mushroom.data.features,
                                                                   pd.DataFrame) else mushroom.data.features

        # Convert targets to Series, handling potential multi-column case
        y_df = pd.DataFrame(mushroom.data.targets) if not isinstance(mushroom.data.targets,
                                                                     pd.DataFrame) else mushroom.data.targets
        y = y_df.iloc[:, 0] if y_df.shape[1] == 1 else y_df.iloc[:, 0]  # Take first column if multiple exist
        y.name = 'target'

        return X, y

    def _load_secom_data(self, data_path: str, labels_path: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Load SECOM dataset"""
        X, y, _ = load_secom_dataset(data_path=data_path, 
                                   labels_path=labels_path, 
                                   verbose=False)
        return X, y

    def _load_cancer_rna_data(self, data_path: str, labels_path: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Load Cancer RNA-Seq dataset"""
        X, y, _, _ = load_cancer_rna_seq(data_path=data_path, 
                                        labels_path=labels_path)
        return pd.DataFrame(X), pd.Series(y)

    def _load_microbiome_data(self, dataset_path: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Load Microbiome dataset"""
        loader = MicrobiomeMetabolomeDataLoader()
        data = loader.load_dataset(dataset_path)
        return data["features"], data["target"]

    def split_data(self,
                  dataset_type: str,
                  n_splits: int,
                  test_size: float,
                  split_size: Optional[List[float]] = None,
                  **dataset_kwargs) -> dict:
        """
        Split data into client portions with train-test splits.
        
        Args:
            dataset_type: Type of dataset ('secom', 'cancer_rna', or 'microbiome')
            n_splits: Number of clients
            test_size: Proportion of data for testing
            split_size: Optional list of split sizes for unbalanced splitting
            **dataset_kwargs: Dataset-specific loading parameters
            
        Returns:
            Dictionary containing client splits
        """
        if dataset_type not in self.supported_loaders:
            raise ValueError(f"Unsupported dataset type : {dataset_type}. Must be one of {list(self.supported_loaders.keys())}")

        # Load data using appropriate loader
        X, y = self.supported_loaders[dataset_type](**dataset_kwargs)
        
        # Initialize result dictionary
        client_splits = {}

        # Handle balanced vs unbalanced splitting
        if not split_size:
            # Equal client splitting using StratifiedKFold
            kfold = StratifiedKFold(n_splits=n_splits, shuffle=True)
            
            for client_id, (_, test_index) in enumerate(kfold.split(X, y)):
                client_portion = X.iloc[test_index].reset_index(drop=True)
                client_y = y.iloc[test_index] if isinstance(y, pd.Series) else y[test_index]
                
                # Create train-test split for client
                X_train, X_test, y_train, y_test = train_test_split(
                    client_portion, client_y,
                    test_size=test_size,
                    stratify=client_y
                )
                
                client_splits[f'client_{client_id}'] = {
                    'train': {'X': X_train, 'y': y_train},
                    'test': {'X': X_test, 'y': y_test}
                }
                
        else:
            # Unbalanced client splitting
            if len(split_size) != n_splits:
                raise ValueError('Number of splits should equal number of clients')
                
            data = X
            targets = y
            
            for client_id, client_size in enumerate(split_size):
                #need to scale the percentage due to loop data splitting.
                percentage_scaled = split_size[client_id] / sum(split_size[client_id: ])

                if percentage_scaled < 1:
                    # Split portion for current client
                    remain_portion, client_portion, remain_y, client_y = train_test_split(
                        data, targets,
                        test_size=percentage_scaled,
                        stratify=targets
                    )
                else:
                    remain_portion = None
                    client_portion = data
                    client_y = targets
                
                # Create train-test split for client
                X_train, X_test, y_train, y_test = train_test_split(
                    client_portion, client_y,
                    test_size=test_size,
                    stratify=client_y
                )
                
                client_splits[f'client_{client_id}'] = {
                    'train': {'X': X_train, 'y': y_train},
                    'test': {'X': X_test, 'y': y_test}
                }

                # Update remaining data for next iteration
                data = remain_portion
                targets = remain_y
                
        return client_splits

    def save_splits(self, client_splits: dict, output_dir: str, file_prefix: str = ''):
        """Save client splits to CSV files"""
        os.makedirs(output_dir, exist_ok=True)
        
        for client_id, splits in client_splits.items():
            # Save train data
            train_X = splits['train']['X']
            train_y = splits['train']['y']
            train_data = pd.concat([train_X, pd.Series(train_y, name='target')], axis=1)
            train_data.to_csv(os.path.join(output_dir, f'{file_prefix}{client_id}_train.csv'), index=False)
            
            # Save test data
            test_X = splits['test']['X']
            test_y = splits['test']['y']
            test_data = pd.concat([test_X, pd.Series(test_y, name='target')], axis=1)
            test_data.to_csv(os.path.join(output_dir, f'{file_prefix}{client_id}_test.csv'), index=False)


# Example usage
if __name__ == "__main__":
    splitter = UnifiedDataSplitter()

    # secom_splits = splitter.split_data(
    #     dataset_type='secom',
    #     n_splits=3,
    #     test_size=0.2,
    #     data_path='secom/secom.data',
    #     labels_path='secom/secom_labels.data'
    # )
    # cancer_splits = splitter.split_data(
    #     dataset_type='cancer_rna',
    #     n_splits=5,
    #     test_size=0.2,
    #     data_path='cancerRNA/data.csv',
    #     labels_path='cancerRNA/labels.csv',
    #     split_size=[0.3, 0.1, 0.1, 0.1, 0.4]
    # )
    # microbiome_splits = splitter.split_data(
    #     dataset_type='microbiome',
    #     n_splits=2,
    #     test_size=0.2,
    #     dataset_path='datasets/HE_INFANTS_MFGM_2019',
    #     split_size=[0.6, 0.4]
    # )

    # microbiome_splits = splitter.split_data(
    #     dataset_type='microbiome',
    #     n_splits=2,
    #     test_size=0.2,
    #     dataset_path='datasets/JACOBS_IBD_FAMILIES_2016',
    #     split_size=[0.5, 0.5]
    # )
    #
    income_splits = splitter.split_data(
        dataset_type='income',
        n_splits=3,
        test_size=0.2,
        split_size=[0.5, 0.3, 0.2]
    )

    mushroom_splits = splitter.split_data(
        dataset_type='mushroom',
        n_splits=3,
        test_size=0.2,
        split_size=[0.5, 0.3, 0.2]
    )

    # heart_disease_splits = splitter.split_data(
    #    dataset_type='heart_disease',
    #    n_splits=3,
    #    test_size=0.2,
    #    split_size=[0.5, 0.3, 0.2]
    # )

    breast_cancer_splits = splitter.split_data(
        dataset_type='breast_cancer',
        n_splits=3,
        test_size=0.2,
        split_size=[0.5, 0.3, 0.2]
    )
    #
    # heart_disease_splits = splitter.split_data(
    #     dataset_type='heart_disease',
    #     n_splits=3,
    #     test_size=0.2,
    #     split_size=[0.5, 0.3, 0.2]
    # )
    #
    TUANDROMD_splits = splitter.split_data(
        dataset_type='TUANDROMD',
        n_splits=3,
        test_size=0.2,
        split_size=[0.5, 0.3, 0.2]
    )

    # mnist_splits = splitter.split_data(
    #     dataset_type='mnist',
    #     n_splits=5,  # 5 clients as per the paper
    #     test_size=0.1,  # 90:10 train-test split
    #     split_size=[0.2, 0.2, 0.2, 0.2, 0.2]  # Equal distribution among clients
    # )

    # phrasebank_splits = splitter.split_data(
    #     dataset_type='phrasebank',
    #     n_splits=3,
    #     test_size=0.2,
    #     split_size=[0.5, 0.25, 0.25]
    # )
    # wine_splits = splitter.split_data(
    #     dataset_type='wine',
    #     n_splits=2,
    #     test_size=0.2,
    #     split_size=[0.5, 0.5]
    # )

    # Uncomment each process:
    splitter.save_splits(breast_cancer_splits, 'datasets/breastcancer_federated', 'breastcancer_')
    # splitter.save_splits(kaggle_heart_disease_splits, 'datasets/heartdisease_federated', 'heartdisease_')
    splitter.save_splits(mushroom_splits, 'datasets/mushroom_federated', 'mushroom_')
    splitter.save_splits(income_splits, 'datasets/income_federated', 'income_')
    splitter.save_splits(TUANDROMD_splits, 'datasets/TUANDROMD_federated', 'TUANDROMD_')
    # splitter.save_splits(heart_disease_splits, 'datasets/heartdisease_', 'heartdisease_')
    # splitter.save_splits(mnist_splits, 'datasets/mnist_federated', 'mnist_')
    # splitter.save_splits(secom_splits, 'output/secom', 'secom_')
    # splitter.save_splits(cancer_splits, 'datasets/cancer_rna_federated', 'cancer_rna_')
    # splitter.save_splits(microbiome_splits, 'datasets/infants_mfgm_federated', 'infants_mfgm_')
    # splitter.save_splits(microbiome_splits, 'datasets/ibd_families_federated', 'ibd_families_')
    # splitter.save_splits(mushroom_splits, 'datasets/mushroom_federated', 'mushroom_')
    # splitter.save_splits(heart_disease_splits, 'datasets/heart_disease_federated', 'heart_disease_')
    # splitter.save_splits(phrasebank_splits, 'datasets/phrasebank_federated', 'phrasebank_')
    # splitter.save_splits(wine_splits, 'datasets/wine_federated', 'wine_')
