import os
import pandas as pd
import numpy as np
from typing import List, Tuple, Union, Optional, Dict, Set
from sklearn.preprocessing import StandardScaler
import itertools


class ConsistentFeatureAugmentor:
    """
    A utility class for augmenting features in dataset files produced by UnifiedDataSplitter.
    This class ensures the same features are added consistently across all client splits,
    with built-in feature consistency checks to prevent mismatch errors.
    """

    def __init__(self, random_seed: int = 42):
        """
        Initialize the feature augmentor.

        Args:
            random_seed: Seed for reproducibility
        """
        self.random_seed = random_seed
        np.random.seed(random_seed)

        # Track all features created during augmentation
        self.all_features = set()

    def discover_features(self, input_dir: str) -> Tuple[List[str], List[str]]:
        """
        Discover all available features across client splits and identify numeric ones.

        Args:
            input_dir: Directory containing CSV files from UnifiedDataSplitter

        Returns:
            Tuple of (all_features, numeric_features)
        """
        all_features = set()
        numeric_features = set()

        # Scan the first file to get initial feature list
        first_file = None
        for filename in os.listdir(input_dir):
            if filename.endswith('.csv'):
                first_file = os.path.join(input_dir, filename)
                break

        if not first_file:
            raise ValueError(f"No CSV files found in {input_dir}")

        # Read first file
        df = pd.read_csv(first_file)

        # Remove target if it exists
        if 'target' in df.columns:
            df = df.drop(columns=['target'])

        # Get feature names
        all_features.update(df.columns)

        # Get numeric features
        for col in df.columns:
            if np.issubdtype(df[col].dtype, np.number):
                numeric_features.add(col)
        print(f"Number of numeric features: {len(numeric_features)}")
        # Convert to sorted lists for consistent ordering
        return sorted(list(all_features)), sorted(list(numeric_features))

    def add_interaction_features(self,
                                 all_client_dfs: List[pd.DataFrame],
                                 feature_pairs: List[Tuple[str, str]],
                                 operations: List[str] = ['multiply'],
                                 prefix: str = "interact") -> List[pd.DataFrame]:
        """
        Add the same interaction features to all client DataFrames.

        Args:
            all_client_dfs: List of client DataFrames
            feature_pairs: List of tuples containing pairs of column names to create interactions from
            operations: List of operations to perform ('multiply', 'add', 'subtract', 'divide')
            prefix: Prefix for the new feature names

        Returns:
            List of DataFrames with added interaction features
        """
        # Create a copy of all DataFrames
        augmented_dfs = [df.copy() for df in all_client_dfs]

        # List to store which feature pairs were successfully processed
        processed_pairs = []
        # Track all new features created
        new_features = set()

        # Print warning if no feature pairs to process
        if not feature_pairs:
            print("Warning: No feature pairs provided for interaction features")
            return augmented_dfs

        print(f"Processing {len(feature_pairs)} feature pairs for interactions:")
        for pair in feature_pairs:
            print(f"  - {pair[0]} × {pair[1]}")

        operations_map = {
            'multiply': lambda x, y: x * y,
            'add': lambda x, y: x + y,
            'subtract': lambda x, y: x - y,
            'divide': lambda x, y: x / (y + 1e-10)  # Adding small value to avoid division by zero
        }

        for feat1, feat2 in feature_pairs:
            # Check if features exist in all DataFrames
            if not all(feat1 in df.columns and feat2 in df.columns for df in all_client_dfs):
                print(f"Skipping interaction: {feat1} or {feat2} not found in all clients")
                continue

            # Skip if any feature is categorical/non-numeric in any DataFrame
            if not all(np.issubdtype(df[feat1].dtype, np.number) and np.issubdtype(df[feat2].dtype, np.number)
                       for df in all_client_dfs):
                print(f"Skipping non-numeric interaction between '{feat1}' and '{feat2}'")
                continue

            processed_pairs.append((feat1, feat2))

            for op in operations:
                if op not in operations_map:
                    print(f"Skipping unknown operation: {op}")
                    continue

                new_col_name = f"{prefix}_{op}_{feat1}_{feat2}"
                new_features.add(new_col_name)

                # Apply the same operation to each client
                for i, df in enumerate(all_client_dfs):
                    augmented_dfs[i][new_col_name] = operations_map[op](df[feat1], df[feat2])

                print(f"Created interaction feature '{new_col_name}'")

        # Record how many feature pairs we actually processed
        print(f"Processed {len(processed_pairs)} feature pairs out of {len(feature_pairs)} requested")

        # Update the global feature set
        self.all_features.update(new_features)

        return augmented_dfs

    def collect_all_features(self, dfs: List[pd.DataFrame]) -> Set[str]:
        """
        Collect all feature names from a list of DataFrames.

        Args:
            dfs: List of DataFrames

        Returns:
            Set of all unique feature names
        """
        all_cols = set()
        for df in dfs:
            cols = set(df.columns)
            if 'target' in cols:
                cols.remove('target')
            all_cols.update(cols)
        return all_cols

    def ensure_feature_consistency(self, train_dfs: List[pd.DataFrame], test_dfs: List[pd.DataFrame]) -> Tuple[
        List[pd.DataFrame], List[pd.DataFrame]]:
        """
        Ensure all DataFrames have the same features.

        Args:
            train_dfs: List of training DataFrames
            test_dfs: List of test DataFrames

        Returns:
            Tuple of (consistent_train_dfs, consistent_test_dfs)
        """
        # Collect all features from all DataFrames
        all_training_features = self.collect_all_features(train_dfs)
        all_testing_features = self.collect_all_features(test_dfs)

        # Combine all features
        all_features = all_training_features.union(all_testing_features).union(self.all_features)
        all_features_sorted = sorted(list(all_features))

        print(f"Ensuring feature consistency across all datasets...")
        print(f"Total unique features across all clients: {len(all_features)}")

        # Check for feature mismatches
        missing_in_train = all_features - all_training_features
        missing_in_test = all_features - all_testing_features

        if missing_in_train:
            print(f"Adding {len(missing_in_train)} missing features to training data")

        if missing_in_test:
            print(f"Adding {len(missing_in_test)} missing features to testing data")

        # Create consistent versions of all DataFrames
        consistent_train_dfs = []
        for df in train_dfs:
            # Make a copy to avoid modifying the original
            new_df = df.copy()

            # Extract target if present
            target = None
            if 'target' in new_df.columns:
                target = new_df['target']
                new_df = new_df.drop(columns=['target'])

            # Add missing columns with zeros
            for col in all_features:
                if col not in new_df.columns:
                    new_df[col] = 0

            # Ensure consistent column order
            new_df = new_df[all_features_sorted]

            # Re-add target if it existed
            if target is not None:
                new_df['target'] = target

            consistent_train_dfs.append(new_df)

        consistent_test_dfs = []
        for df in test_dfs:
            # Make a copy to avoid modifying the original
            new_df = df.copy()

            # Extract target if present
            target = None
            if 'target' in new_df.columns:
                target = new_df['target']
                new_df = new_df.drop(columns=['target'])

            # Add missing columns with zeros
            for col in all_features:
                if col not in new_df.columns:
                    new_df[col] = 0

            # Ensure consistent column order
            new_df = new_df[all_features_sorted]

            # Re-add target if it existed
            if target is not None:
                new_df['target'] = target

            consistent_test_dfs.append(new_df)

        print(f"All datasets now have consistent feature sets with {len(all_features)} features")

        return consistent_train_dfs, consistent_test_dfs

    def generate_feature_pairs(self, numeric_features, max_pairs=100):
        """
        Generate feature pairs for interaction based on the number of numeric features.

        Args:
            numeric_features: List of numeric feature names
            max_pairs: Maximum number of feature pairs to generate

        Returns:
            List of feature pairs (tuples)
        """
        # If fewer than 100 numeric features, generate all possible combinations
        if len(numeric_features) < 15:  # This will generate less than 105 pairs (14*15/2)
            feature_pairs = list(itertools.combinations(numeric_features, 2))
            print(f"Generated all possible {len(feature_pairs)} feature pairs")

            # If there are too many, take only the first max_pairs
            if len(feature_pairs) > max_pairs:
                feature_pairs = feature_pairs[:max_pairs]
                print(f"Limited to {max_pairs} feature pairs")

        else:
            # With many features, select pairs more strategically
            feature_pairs = []
            pairs_added = 0

            # First add pairs of consecutive features (assuming features might be related)
            for i in range(len(numeric_features) - 1):
                if pairs_added < max_pairs:
                    feature_pairs.append((numeric_features[i], numeric_features[i + 1]))
                    pairs_added += 1
                else:
                    break

            # Then add some pairs of features with larger gaps
            if pairs_added < max_pairs:
                skip_size = min(5, len(numeric_features) // 10)
                for i in range(0, len(numeric_features) - skip_size, skip_size):
                    if pairs_added < max_pairs:
                        feature_pairs.append((numeric_features[i], numeric_features[i + skip_size]))
                        pairs_added += 1
                    else:
                        break

            # Finally, add some random pairs if we still have room
            remaining_pairs = max_pairs - pairs_added
            if remaining_pairs > 0:
                np.random.seed(self.random_seed)
                random_indices = list(range(len(numeric_features)))
                np.random.shuffle(random_indices)

                for i in range(0, min(len(random_indices) - 1, 2 * remaining_pairs), 2):
                    if pairs_added < max_pairs:
                        idx1, idx2 = random_indices[i], random_indices[i + 1]
                        pair = (numeric_features[idx1], numeric_features[idx2])
                        if pair not in feature_pairs and (pair[1], pair[0]) not in feature_pairs:
                            feature_pairs.append(pair)
                            pairs_added += 1
                    else:
                        break

            print(f"Generated {len(feature_pairs)} strategic feature pairs")

        return feature_pairs

    def process_directory(self,
                          input_dir: str,
                          output_dir: Optional[str] = None,
                          max_interaction_pairs: int = 100):
        """
        Process all CSV files in a directory, ensuring consistent feature augmentation across clients.

        Args:
            input_dir: Directory containing CSV files from UnifiedDataSplitter
            output_dir: Directory to save augmented files (if None, will use input_dir + "augmented")
            operations: List of operations to perform on feature pairs
            max_interaction_pairs: Maximum number of feature interaction pairs to create
        """
        # Reset the all_features tracker for this run
        self.all_features = set()

        # Extract the base dataset name from the input directory
        base_name = os.path.basename(input_dir)
        if "_federated" in base_name:
            base_name = base_name.replace("_federated", "")

        # If output_dir is not specified, create it by appending "augmented" to the input directory name
        if output_dir is None:
            # Create output directory name
            output_dir = os.path.join(os.path.dirname(input_dir), f"{base_name}augmented_federated")

        os.makedirs(output_dir, exist_ok=True)

        print(f"Processing dataset: {base_name}")
        print(f"Output directory: {output_dir}")
        print(f"Output files will use naming pattern: {base_name}augmented_client_N_train/test.csv")

        # Discover all features and numeric features
        all_features, numeric_features = self.discover_features(input_dir)
        print(f"Discovered {len(all_features)} total features")
        print(f"Discovered {len(numeric_features)} numeric features")

        # Add original features to the tracked set
        self.all_features.update(all_features)

        # Generate feature pairs based on numeric features
        feature_pairs = self.generate_feature_pairs(numeric_features, max_interaction_pairs)
        print(f"Using {len(feature_pairs)} feature pairs for interaction")

        # Group files by client to ensure we process train and test files together
        client_files = {}
        all_files = []

        # First collect all CSV files
        for filename in os.listdir(input_dir):
            if not filename.endswith('.csv'):
                continue

            all_files.append(filename)

            # Extract client ID from filename (e.g., "client_0_train.csv" -> "client_0")
            if '_train.csv' in filename:
                client_id = filename.replace('_train.csv', '')
                train_file = filename
                test_file = filename.replace('_train.csv', '_test.csv')

                # Only include if both train and test files exist
                if os.path.exists(os.path.join(input_dir, test_file)):
                    client_files[client_id] = {
                        'train': train_file,
                        'test': test_file
                    }

        # Create a mapping of client IDs to their filenames for consistent naming
        client_id_map = {}
        for filename in all_files:
            if not ('_train.csv' in filename or '_test.csv' in filename):
                continue

            if '_client_' in filename:
                parts = filename.split('_client_')
                if len(parts) > 1:
                    # Extract client number
                    client_part = parts[1].split('_')[0]
                    # Store just the client number
                    client_id_map[filename] = client_part

        print(f"Found {len(client_files)} client pairs (train+test)")

        # Process train files consistently
        train_dfs = []
        train_targets = []
        train_filenames = []
        test_dfs = []
        test_targets = []
        test_filenames = []

        # First load all train files
        for client_id, files in client_files.items():
            # Load train file
            train_file = os.path.join(input_dir, files['train'])
            train_df = pd.read_csv(train_file)

            # Separate target if it exists
            if 'target' in train_df.columns:
                train_targets.append(train_df['target'])
                train_dfs.append(train_df.drop(columns=['target']))
            else:
                train_targets.append(None)
                train_dfs.append(train_df)

            train_filenames.append(files['train'])

            # Load test file
            test_file = os.path.join(input_dir, files['test'])
            test_df = pd.read_csv(test_file)

            # Separate target if it exists
            if 'target' in test_df.columns:
                test_targets.append(test_df['target'])
                test_dfs.append(test_df.drop(columns=['target']))
            else:
                test_targets.append(None)
                test_dfs.append(test_df)

            test_filenames.append(files['test'])

        # Apply feature augmentation to all train files together
        augmented_train_dfs = train_dfs

        # Add interaction features with multiplication only
        if feature_pairs:
            augmented_train_dfs = self.add_interaction_features(
                augmented_train_dfs,
                feature_pairs,
                operations=['multiply']
            )

        # Apply the same transformations to test data
        augmented_test_dfs = []
        for i, test_df in enumerate(test_dfs):
            # Get corresponding train df
            train_df = train_dfs[i]
            augmented_train_df = augmented_train_dfs[i]

            # Get new columns added during augmentation
            new_columns = [col for col in augmented_train_df.columns if col not in train_df.columns]

            # Create a copy of the test df
            augmented_test_df = test_df.copy()

            # Apply each transformation
            for new_col in new_columns:
                if new_col.startswith('interact_'):
                    # For interaction features, extract operation and features
                    parts = new_col.split('_')
                    op = parts[1]
                    feat1 = parts[2]
                    feat2 = '_'.join(parts[3:])

                    operations_map = {
                        'multiply': lambda x, y: x * y,
                        'add': lambda x, y: x + y,
                        'subtract': lambda x, y: x - y,
                        'divide': lambda x, y: x / (y + 1e-10)
                    }

                    if op in operations_map and feat1 in test_df.columns and feat2 in test_df.columns:
                        augmented_test_df[new_col] = operations_map[op](test_df[feat1], test_df[feat2])
                    else:
                        # Set to zero if operation can't be performed
                        augmented_test_df[new_col] = 0

            augmented_test_dfs.append(augmented_test_df)

        # Ensure feature consistency between train and test sets
        print("\nChecking feature consistency between train and test sets...")
        consistent_train_dfs, consistent_test_dfs = self.ensure_feature_consistency(
            augmented_train_dfs, augmented_test_dfs
        )

        # Save augmented train files
        for i, df in enumerate(consistent_train_dfs):
            # Re-add target if it existed
            if train_targets[i] is not None:
                df['target'] = train_targets[i]

            # Get original filename
            original_filename = train_filenames[i]

            # Extract client number
            client_number = "0"  # Default if not found
            if original_filename in client_id_map:
                client_number = client_id_map[original_filename]
            elif '_client_' in original_filename:
                # Fallback extraction method
                parts = original_filename.split('_client_')
                if len(parts) > 1:
                    client_number = parts[1].split('_')[0]

            # Create the EXACT filename format needed
            new_filename = f"{base_name}augmented_client_{client_number}_train.csv"

            # Debug print to see what we're creating
            print(f"Original: {original_filename} → New: {new_filename}")

            # Save with the exact filename format
            output_path = os.path.join(output_dir, new_filename)
            df.to_csv(output_path, index=False)
            print(
                f"Saved augmented train file with {len(df.columns) - (1 if 'target' in df.columns else 0)} features to {output_path}")

        # Save augmented test files
        for i, df in enumerate(consistent_test_dfs):
            # Re-add target if it existed
            if test_targets[i] is not None:
                df['target'] = test_targets[i]

            # Get original filename
            original_filename = test_filenames[i]

            # Extract client number
            client_number = "0"  # Default if not found
            if original_filename in client_id_map:
                client_number = client_id_map[original_filename]
            elif '_client_' in original_filename:
                # Fallback extraction method
                parts = original_filename.split('_client_')
                if len(parts) > 1:
                    client_number = parts[1].split('_')[0]

            # Create the EXACT filename format needed
            new_filename = f"{base_name}augmented_client_{client_number}_test.csv"

            # Debug print to see what we're creating
            print(f"Original: {original_filename} → New: {new_filename}")

            # Save with the exact filename format
            output_path = os.path.join(output_dir, new_filename)
            df.to_csv(output_path, index=False)
            print(
                f"Saved augmented test file with {len(df.columns) - (1 if 'target' in df.columns else 0)} features to {output_path}")

        print(f"\nAll files processed and feature consistency ensured.")
        print(f"Original features: {len(all_features)}")
        print(f"Augmented features: {len(self.all_features)}")
        print(
            f"Total features in final datasets: {len(consistent_train_dfs[0].columns) - (1 if 'target' in consistent_train_dfs[0].columns else 0)}")


# Example usage
if __name__ == "__main__":
    augmentor = ConsistentFeatureAugmentor(random_seed=42)

    # Process dataset with feature interactions (multiplication only)
    augmentor.process_directory(
        input_dir="datasets/TUANDROMD_federated",
        max_interaction_pairs=5000
    )
    augmentor.process_directory(
        input_dir="datasets/breastcancer_federated",
        max_interaction_pairs=5000
    )
    # augmentor.process_directory(
    #     input_dir="datasets/heartdisease_federated",
    #     max_interaction_pairs=5000
    # )
    # augmentor.process_directory(
    #     input_dir="datasets/mushroom_federated",
    #     max_interaction_pairs=5000
    # )
    # augmentor.process_directory(
    #     input_dir="datasets/income_federated",
    #     max_interaction_pairs=5000
    # )
