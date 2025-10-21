import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report
# Import MultiTaskLasso instead of Lasso
from sklearn.linear_model import MultiTaskLasso
# Convert labels to one-hot encoded format
from sklearn.preprocessing import LabelBinarizer
# from sklearn.linear_model import Lasso
import json


class MicrobiomeMetabolomeDataLoader:
    def __init__(self):
        with open('dataset_info.json', 'r') as f:
            self.dataset_info = json.load(f)


    def load_dataset(self, dataset_path: str) -> Dict[str, Any]:
        """
        Load a specific dataset.

        Args:
        dataset_path (str): Path of the dataset to load.

        Returns:
        dict: A dictionary containing the loaded dataframes and series.
        """
        self.dataset_name = Path(dataset_path).name
        # data_path = self.data_dir / dataset_name
        # Load metadata
        metadata = pd.read_csv(Path(dataset_path) / "metadata.tsv", sep="\t")
        # print(f"Metadata columns: {metadata.columns.tolist()}")
        # Load metabolites data
        mtb = pd.read_csv(Path(dataset_path) / "mtb.tsv", sep="\t")
        # print(f"MTB columns: {mtb.columns.tolist()}")
        mtb_map = pd.read_csv(Path(dataset_path) / "mtb.map.tsv", sep="\t")
        # Load genera data
        genera = pd.read_csv(Path(dataset_path) / "genera.tsv", sep="\t")
        # print(f"Genera columns: {genera.columns.tolist()}")
        # Load species data if available
        species_path = Path(dataset_path) / "species.tsv"
        species = pd.read_csv(dataset_path, sep="\t") if species_path.exists() else None
        # if species is not None:
            # print(f"Species columns: {species.columns.tolist()}")

        # Prepare features
        features_mtb = mtb.set_index("Sample")
        features_genera = genera.set_index("Sample")
        features = pd.concat([features_mtb, features_genera], axis=1)

        if species is not None:
            features_species = species.set_index("Sample")
            features = pd.concat([features, features_species], axis=1)

        # Ensure the index (Sample) matches between features and metadata
        data = features.merge(metadata, left_index=True, right_on="Sample", how="inner")

        # Identify the target variable (assuming it's the Study.Group column)
        # target_column = "Study.Group"
        target_column = (self.dataset_info.get(self.dataset_name))['target_column']
        # print(target_column)
        if target_column not in data.columns:
            print(f"Warning: Target column '{target_column}' not found in the dataset. Using 'Study.Group' as default.")
            target_column = "Study.Group"

        # Drop metadata columns
        # columns_to_drop = ["Sample", "Subject", "Age", "Age.Units", "Gender", "BMI", "DOI", "Publication.Name", "Dataset"]
        columns_to_drop = ["Sample", "Subject", "Age.Units", "Gender", "BMI"]
        columns_to_drop = [col for col in columns_to_drop if col in data.columns]
        X = data.drop(columns=columns_to_drop + [target_column])
        y = data[target_column]

        return {
            "metadata": metadata,
            "mtb": mtb,
            "mtb_map": mtb_map,
            "genera": genera,
            "species": species,
            "features": X,
            "target": y
        }


    def preprocess_data(self, X, y, test_size_percentage=0.1):
        """
        Preprocess data ensuring consistent features across federated clients.
        """
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        print(f"Initial feature shape: {X.shape}")

        # Get feature columns
        columns = X.columns.tolist()
        id_columns = ['Sample']
        feature_cols = [col for col in columns if col not in id_columns]
        X = X[feature_cols]

        # Convert all columns to numeric
        # First, handle categorical columns by replacing them with numeric indicators
        categorical_data = X.select_dtypes(include=['object', 'category'])
        numeric_data = X.select_dtypes(include=['int64', 'float64'])
        #
        # print(f"Number of numeric features: {len(numeric_data.columns)}")
        # print(f"Number of categorical features: {len(categorical_data.columns)}")

        # Replace categorical values with numerical indicators
        for col in categorical_data.columns:
            # Replace with simple numerical indicators like 1, 2, 3...
            X[col] = pd.Categorical(X[col]).codes

        # Now all data should be numeric
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded,
            stratify=y_encoded,
            test_size=test_size_percentage,
            random_state=42
        )

        # Scale all features
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # print(f"Final feature shape: {X_train_scaled.shape}")

        return X_train_scaled, X_test_scaled, y_train, y_test, le

# Example usage
if __name__ == "__main__":
    loader = MicrobiomeMetabolomeDataLoader()

    # List of datasets to load
    datasets = [
        "ERAWIJANTARI_GASTRIC_CANCER_2020",
        "HE_INFANTS_MFGM_2019",
        "JACOBS_IBD_FAMILIES_2016",
        "KANG_AUTISM_2017",
        "KIM_ADENOMAS_2020",
        "MARS_IBS_2020",
        # "FRANZOSA_IBD_2019",
        # "YACHIDA_CRC_2019",
        # "KOSTIC_INFANTS_DIABETES_2015",
        # "iHMP_IBDMDB_2019",
        # "SINHA_CRC_2016",
        # "WANDRO_PRETERMS_2018",
        # "WANG_ESRD_2020"

    ]

    for dataset_name in datasets:
        print(f"\nLoading {dataset_name} dataset:")
        raw_data = loader.load_dataset(dataset_name)


        X = raw_data["features"]
        y = raw_data["target"]

        X_train_processed, X_test_processed, y_train, y_test, le = loader.preprocess_data(X, y)

        print(f"Number of training samples: {len(X_train_processed)}")
        print(f"Number of test samples: {len(X_test_processed)}")

        lb = LabelBinarizer()
        y_train_onehot = lb.fit_transform(y_train)
        y_test_onehot = lb.transform(y_test)
        # Try different alpha values
        alpha_values = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
        for alpha in alpha_values:
            print(f"\nTrying alpha = {alpha}")

            # Use MultiTaskLasso for multiclass
            mtlasso = MultiTaskLasso(alpha=alpha, random_state=42)
            mtlasso.fit(X_train_processed, y_train_onehot)

            # Select features where coefficients are non-zero for any class
            selected_features = np.where(np.any(mtlasso.coef_ != 0, axis=0))[0]

            if len(selected_features) == 0:
                print("No features selected. Skipping this alpha value.")
                continue

            # Select features for both train and test sets
            X_train_selected = X_train_processed[:, selected_features]
            X_test_selected = X_test_processed[:, selected_features]

            # Make predictions
            y_test_pred = mtlasso.predict(X_test_processed)

            # Convert predictions back to original label format
            y_test_pred_labels = lb.inverse_transform(y_test_pred)

            # Evaluate using F1 score with 'weighted' average for multiclass
            test_f1 = f1_score(y_test, y_test_pred_labels, average='weighted')

            print(f"Number of selected features: {len(selected_features)}")
            print(f"Test F1 Score: {test_f1}")

            # Random Forest (this part remains mostly the same as it already handles multiclass)
            rfc = RandomForestClassifier(random_state=42)
            rfc.fit(X_train_selected, y_train)

            y_test_pred = rfc.predict(X_test_selected)
            f1 = f1_score(y_test, y_test_pred, average='weighted')

            print(f"Random Forest Test F1 Score: {f1}")
            print("\nClassification Report:")
            print(classification_report(y_test, y_test_pred))

        # Compare with Random Forest using all features
        rfc_all_features = RandomForestClassifier(random_state=42)
        rfc_all_features.fit(X_train_processed, y_train)
        y_test_pred_all_features = rfc_all_features.predict(X_test_processed)
        f1_all_features = f1_score(y_test, y_test_pred_all_features, average='weighted')

        print("\nRandom Forest with all features:")
        print(f"Test F1 Score: {f1_all_features}")
