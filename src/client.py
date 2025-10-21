# Copyright 2025 Ioannis Christofilogiannis, Georgios Valavanis,
# Alexander Shevtsov, Ioannis Lamprou and Sotiris Ioannidis
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Cython imports to use C++ infrastructure
import joblib
from FL_cpp_client import py_fl_client
from sklearn.ensemble import RandomForestClassifier
# Machine Learning client imports

from sklearn.metrics import recall_score, f1_score
from joblib import dump, load
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.svm import SVC

# Common helper methods for server and client
from utils.helper import *

# ML
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
# FS
from utils.feature_election.elastic_net_fs import ElasticNetFeatureSelector
from utils.feature_election.impetus import PyImpetusSelector
from utils.feature_election.lasso_fs import LassoFeatureSelector
from utils.feature_election.SequentialAttentionOptimizer import SequentialAttentionOptimizer
from sklearn.model_selection import StratifiedKFold

import argparse
import numpy as np

# For reproducible experiments
np.random.seed(42)
from utils.BufferIO import BufferIO

# SUPPORTED_MODELS = ['GNB', 'SGDC', 'MLPC', 'LogReg']
# SUPPORTED_FS_METHODS = ['impetus', 'lasso', 'elastic_net', 'sequential', 'none']
DATA_PATH = './datasets/'
"""Check if execution is running in docker or not. If running in docker, output directory is /app/data/ else it is ./experiments_data/"""
EXPR_OUTPUT_DIR = '/app/data/' if os.getcwd().startswith('/app') else './logs/'

class EncryptionKeys:
    def __init__(self):
        # https://www.ibm.com/docs/en/zos/2.5.0?topic=pdk-using-rsa-public-keys-protect-keys-sent-between-systems
        # Encryption keys struct
        # Encrypt data and generate RSA keys
        self.client_private_key, self.client_public_key = generate_rsa_keys()
        self.server_public_key = None
        self.symmetric_key = None

class ClientML:

    def __init__(self, dataset_name, max_models=500):

        self.scaler = None
        self.initial_f1 = None
        self.sample_weights = None
        self.perform_FS = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        #self.DATA_PATH = "./datasets/"
        self.model = None
        self.model_type = None
        self.dataset_name = dataset_name
        self._make_experiment_dir()

        # TODO Used in Hyperparameter fine tuning
        self.X_val = None
        self.y_val = None
        self.X_h_train = None
        self.y_h_train = None
        self.k_folds = None
        self.max_models = max_models
        self.TIME_PATH = os.path.join(self.experiment_dir, 'Time', '')
        self.DATA_PATH = os.path.join(self.experiment_dir, 'Data', '')
        self.TEMP_PATH = "./Temp/"
        self.stored_models = []
        self.ml_mode = None
        # Check whether directory exists
        os.makedirs(self.TEMP_PATH, exist_ok=True)
        file_path = os.path.join(self.TEMP_PATH, 'latest_csv_path')
        with open(file_path, 'wb') as file:
            dump(self.dataset_name, file)

    # def load_data(self):
    #     #  Create output directory for experiments
    #     # self._make_experiment_dir(dataset_name, given_type)
    #     path_to_dataset = f"{DATA_PATH}{self.dataset_name}_federated/{self.dataset_name}_client_{Client_instance.mid}"
    #     # Load training data
    #     train_df = pd.read_csv(path_to_dataset+"_train.csv", header=0)
    #     # TODO temp solution
    #     train_df = train_df.dropna()
    #     self.y_train = train_df['target'].copy()
    #     self.X_train = train_df.drop(['target'], axis=1)
    #
    #     # Load test data
    #     test_df = pd.read_csv(path_to_dataset+"_test.csv", header=0)
    #     # TODO temp solution
    #     test_df = test_df.dropna()
    #     self.y_test = test_df['target'].copy()
    #     self.X_test = test_df.drop(['target'], axis=1)

    def load_data(self):
        path_to_dataset = f"{DATA_PATH}{self.dataset_name}_federated/{self.dataset_name}_client_{Client_instance.mid}"

        # Load data with a fixed index to ensure consistent ordering
        train_df = pd.read_csv(path_to_dataset + "_train.csv", header=0)
        test_df = pd.read_csv(path_to_dataset + "_test.csv", header=0)

        # Sort by index before dropping NA to ensure consistent row ordering
        train_df = train_df.sort_index().dropna()
        test_df = test_df.sort_index().dropna()

        # Prepare target encoder and transform
        self.target_encoder = LabelEncoder()
        # Sort unique values before fitting to ensure consistent encoding
        unique_targets = sorted(train_df['target'].unique())
        self.target_encoder.fit(unique_targets)
        self.y_train = self.target_encoder.transform(train_df['target'])

        # Transform test data, mapping unknown values to a new class
        self.y_test = np.array([
            self.target_encoder.transform([val])[0] if val in self.target_encoder.classes_
            else -1 for val in test_df['target']
        ])

        # Get categorical columns in a deterministic order
        categorical_columns = sorted(train_df.select_dtypes(include=['object', 'category']).columns)
        categorical_columns = [col for col in categorical_columns if col != 'target']

        # Initialize feature encoders dictionary
        self.feature_encoders = {}

        # Process each categorical feature
        X_train = train_df.drop('target', axis=1)
        X_test = test_df.drop('target', axis=1)

        for col in categorical_columns:
            encoder = LabelEncoder()
            # Fit on training data only
            encoder.fit(X_train[col])
            n_classes_feat = len(encoder.classes_)
            self.feature_encoders[col] = encoder

            # Transform training data
            X_train[col] = encoder.transform(X_train[col])
            # Transform test data, mapping unknown values to a new class
            X_test[col] = np.array([
                encoder.transform([val])[0] if val in encoder.classes_
                else n_classes_feat for val in X_test[col]
            ])

        self.X_train = X_train
        self.X_test = X_test
        self.sample_weights = compute_sample_weight(class_weight="balanced", y=self.y_train)

    # def load_data(self):
    #     path_to_dataset = f"{DATA_PATH}{self.dataset_name}_federated/{self.dataset_name}_client_{Client_instance.mid}"
    #
    #     # Load data
    #     train_df = pd.read_csv(path_to_dataset + "_train.csv", header=0)
    #     test_df = pd.read_csv(path_to_dataset + "_test.csv", header=0)
    #
    #     # Drop NaN values
    #     train_df = train_df.dropna()
    #     test_df = test_df.dropna()
    #
    #     # Prepare target and features
    #     self.y_train = train_df['target']
    #     self.y_test = test_df['target']
    #     self.X_train = train_df.drop(['target'], axis=1)
    #     self.X_test = test_df.drop(['target'], axis=1)
    #
    #     # Handle categorical features and target
    #     categorical_columns = self.X_train.select_dtypes(include=['object', 'category']).columns
    #     self.categorical_encoder = LabelEncoder()
    #
    #     # Encode target
    #     all_targets = pd.concat([self.y_train, self.y_test])
    #     self.y_train = self.categorical_encoder.fit_transform(all_targets)[:len(train_df)]
    #     self.y_test = self.categorical_encoder.transform(self.y_test)
    #
    #     # Encode each categorical feature
    #     for col in categorical_columns:
    #         # Combine train and test data for the column to capture all categories
    #         all_values = pd.concat([self.X_train[col], self.X_test[col]])
    #         # Fit and transform
    #         encoded_values = self.categorical_encoder.fit_transform(all_values)
    #         # Split back to train and test
    #         self.X_train[col] = encoded_values[:len(self.X_train)]
    #         self.X_test[col] = encoded_values[len(self.X_train):]
    #


    def _make_experiment_dir(self, given_type=None):
        timestamp = os.environ.get('EXPERIMENT_TIME',
                                   datetime.now().strftime('%Y%m%d_%H%M%S'))

        # if given_type == 'sgd':
        #     experiment_dir = os.path.join(f'{EXPR_OUTPUT_DIR}client{Client_instance.mid+1}', f"{timestamp}_{dataset_name}_{given_type}_{str(self.blend_factor).replace('.', '_')}_{self.method}")
        # else:
        experiment_dir = os.path.join(f'{EXPR_OUTPUT_DIR}client{Client_instance.mid + 1}',f"{timestamp}_{self.dataset_name}")
        os.makedirs(experiment_dir, exist_ok=True)
        self.experiment_dir = experiment_dir

    def _initialize_selector(self, selector_name):
        match selector_name:
            case 'lasso':
                return LassoFeatureSelector(n_trials=150, random_state=None)
            case 'elastic_net':
                return ElasticNetFeatureSelector()
            case 'sequential':
                return None
            case 'impetus':
                return PyImpetusSelector(task='classification', verbose=False)
            case _:
                raise ValueError(f"Unknown feature selector: {selector_name}")

    def _get_rf_classifier(self, n_features):
        # Used for feature selection optimization
        # Creates a new scaled Random Forest instance that scales to the number of features
        # Scale min_samples_split and min_samples_leaf with log of n_features
        min_samples_factor = max(2, int(np.log2(n_features)))
        # Scale max_features with sqrt of n_features

        return RandomForestClassifier(
            n_estimators=max(100, int(np.log2(n_features))),
            max_features=max(1, int(np.sqrt(n_features))),
            min_samples_split=min_samples_factor,
            min_samples_leaf=max(1, min_samples_factor // 2),
            random_state=42,
            n_jobs=-1
        )


    def feature_selection(self, method='lasso'):
        selector = self._initialize_selector(method)
        # Special case for Sequential Attention
        if method == 'sequential':
            base_model = self._get_rf_classifier(self.X_train.shape[1])
            optimizer = SequentialAttentionOptimizer(
                X=self.X_train,
                y=self.y_train,
                base_model=base_model,
                n_trials=200
            )
            results = optimizer.optimize_and_select()
            # Keep selected features in separate table
            self.X_train_full = self.X_train
            self.X_test_full = self.X_test
            self.X_train = self.X_train.iloc[:, results.results['selected_features']]
            self.X_test = self.X_test.iloc[:, results.results['selected_features']]
        else:
            results = selector.fit(self.X_train, self.y_train)
            # Keep selected features in separate table
            self.X_train_full = self.X_train
            self.X_test_full = self.X_test
            self.X_train = selector.transform(self.X_train)
            self.X_test = selector.transform(self.X_test)
        # Get selector preferences
        match method:
            case 'impetus'|'lasso'|'elastic_net':
                binary_array, scores_array = selector.align_arrays(self.X_train_full.shape[1])
            case 'sequential':
                binary_array, scores_array = optimizer.align_arrays(self.X_train_full.shape[1])
            case _:
                raise ValueError("Invalid selector, preferences cannot be saved")
        # Keep local selected features
        self.binary_array = binary_array
        return binary_array, scores_array


    def transform_global_mask(self, binary_array):
        selected_columns = self.X_train_full.columns[binary_array == 1]
        self.X_train = self.X_train_full[selected_columns]
        self.X_test = self.X_test_full[selected_columns]

    def transform_local_mask(self):
        # Reuse global mask method for stored binary array
        self.transform_global_mask(self.binary_array)

    def useBestModel(self, params):
        """Use model indicated by the best_index in params
        Arguments:
             params """
        self.model = self.model(**self.stored_models[params["best_index"]])

        # Check whether directory exists
        os.makedirs(self.TEMP_PATH, exist_ok=True)
        file_path = os.path.join(self.TEMP_PATH, 'best_model')
        # Dump the best model to the Temp folder for future usage
        with open(file_path, 'wb') as file:
            dump(self.model, file)

        # Scale the clients data
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)

        return

    def create_models(self, combinations=None, model="GNB"):
        """Generates models based on combinations and model and saves them to stored_models list
        Arguments:
            combinations {list} -- List of combinations
            model {str} -- Model name
        """
        params = dict()
        if model == "GNB":
            self.model = GaussianNB
        elif model == "SGDC":
            self.model = SGDClassifier
            params['warm_start'] = True
            # Add default params (empty dict)
            self.stored_models.append({'warm_start': True})
        elif model == "LogReg":
            self.model = LogisticRegression
            params['warm_start'] = True
            # Add default params (empty dict)
            self.stored_models.append({'warm_start': True})
        elif model == "MLPC":
            self.model = MLPClassifier
            params['warm_start'] = True
            # Add default params (empty dict)
            self.stored_models.append({'warm_start': True})

        else:
            raise Exception("Model type not supported")

        #  Combinations is used for hyperparameter finetuning
        if combinations is not None:
            #create model params (dict) for every model combination in combinations dict
            for param_dict in combinations:
                #print(param_dict, type(param_dict))
                for pname, pvalue in param_dict.items():
                    params[pname] = pvalue
                self.stored_models.append(params)


        return

    def create_model(self, model="GNB"):
        """Generates model with default parameters
        Arguments:
            model {str} -- Model name
        """
        if model == "GNB":
            self.model = GaussianNB()
        elif model == "SGDC":
            self.model = SGDClassifier(
                loss='log_loss',
                penalty='l2',
                alpha=0.01,  # Same as before
                max_iter=1000,
                tol=1e-3,
                learning_rate='adaptive',
                eta0=0.005,
                power_t=0.25,
                warm_start=True,
                average=False
            )
        elif model == "LogReg":
            self.model = LogisticRegression()
        elif model == "MLPC":
            self.model = MLPClassifier()
        else:
            raise Exception("Model type not supported")


    def load_best_model(self):
        """Used in unweighted aggregation, skips the hyperparameter tuning phase and directly loads the best model (used in weighted aggregation)
        Arguments:
            params {dict} -- Model parameters
        Returns:
            performance {dict} -- Performance metrics
            params["model"] {str} -- Model type name"""
        self.model = load(f"{self.TEMP_PATH}best_model")

        # Parameters read, no need to keep buffer
        # ClientIO.aggregationBuffer.clear()

        #Initialize model and fit
        performance = self.updateModel_ML(None)
        return performance

    def scale_data(self):
        """Scales the clients data"""
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)

    def train_model(self):

        res = dict()

        """Trains the model using the current training data"""
        self.create_model(model=Client_instance.server_config['ml-model'])

        # Model-specific random state settings
        if hasattr(self.model, 'random_state'):
            self.model.random_state = 42

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(self.X_train)
        X_test_scaled = scaler.transform(self.X_test)

        # Include the extra class for unknowns in the classes parameter
        n_classes = len(self.target_encoder.classes_)
        all_classes = np.arange(n_classes + 1)  # +1 for unknown class

        if hasattr(self.model, 'partial_fit') and callable(self.model.partial_fit):
            self.model.partial_fit(X_train_scaled, self.y_train, classes=all_classes)
        else:
            self.model.fit(X_train_scaled, self.y_train)

        res['test'] = self.getModelPerformance(X_test_scaled, self.y_test, model=self.model)
        res['train'] = self.getModelPerformance(X_train_scaled, self.y_train, model=self.model)

        return res

    def packParams_FS(self, selected_features, feature_preference, num_of_samples, score_initial, score_fs):
        # Setup params dictionary of selected features
        params = {
            "selected_features": selected_features,
            "feature_preference": feature_preference,
            "num_of_samples": num_of_samples,
            "initial_score": score_initial,
            "fs_score": score_fs
        }
        return params

    def packParams_ML(self, scores,  state, model_type, model=None):
        params = dict()
        params["num_of_samples"] = self.X_train.shape[0]
        params["val_perf"] = 0.0 if 'val' not in scores else scores['val']
        params["train_perf"] = 0.0 if 'train' not in scores else scores['train']
        params["test_perf"] = 0.0 if 'test' not in scores else scores['test']
        params["state"] = state
        params["model"] = model_type

        model_to_use = model if model is not None else self.model

        # if voting phase
        if state == 'model_hp_voting':
            pass

        elif model_type == "SGDC" or model_type == "LogReg":

            params["model_params"] = {
                "coef_": model_to_use.coef_,
                "intercept_": model_to_use.intercept_,
            }
        elif model_type == "MLPC":
            params["model_params"] = {
                "coefs_": model_to_use.coefs_,
                "intercepts_": model_to_use.intercepts_,
            }
        elif model_type == "GNB":
            params["model_params"] = {
                "var_": model_to_use.var_,
                "theta_": model_to_use.theta_
            }
        else:
            raise ValueError(f"Unsupported model: {model_type}")

        return params

    def getModelPerformance(self, X, y, model):
        """Returns the f1 of the model used
        Arguments:
            X {array-like} -- test data
            y {array-like} -- test labels
            model {object} -- model to use
        Returns:
            float -- f1 score"""

        # Calculate F1 score
        return {'f1': f1_score(y, model.predict(X), average='weighted'), 'y_true': y, 'y_pred': model.predict(X)}

    # Update model with params occured during aggregation
    def updateModel_ML(self, params, X_train=None, X_test=None, y_train=None, y_test=None, model=None):
        """ Update model with the aggregated params sent from the server
        Arguments:
            params {dict} -- params sent from the server
            X_train {np.ndarray} -- training data
            X_test {np.ndarray} -- testing data
            y_train {np.ndarray} -- training labels
            y_test {np.ndarray} -- testing labels
            model {Model} -- model to be updated
        Returns:
            res {dict} -- result of the update
        """
        model = model if model is not None else self.model
        temp_f1 =0
        i = 0

        # If no data provided use the stored data (normal update phase) else used the data provided (hyperparameter tune phase)
        if X_train is None or y_train is None:
            X_train, y_train, X_test, y_test, fine_tune = self.X_train, self.y_train, self.X_test, self.y_test, False
        else:
            X_train, y_train, X_test, y_test, fine_tune = X_train, y_train, X_test, y_test, True



        # Fit the model with the client data. In case of aggregation round and if partial_fit function exist in the model
        if Client_instance.aggregationRound != 1 or fine_tune:
            if fine_tune or self.initial_f1 is None or self.sample_weights is None:
                initial_f1 = self.getModelPerformance(X_test, y_test, model=model)['f1']
                sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)
            else:
                initial_f1 = self.initial_f1
                sample_weights = self.sample_weights

            # Update model with new params
            for pname in params["model_params"]:
                setattr(model, pname, params["model_params"][pname])

            if hasattr(model, 'partial_fit') and callable(model.partial_fit):
                # Train-Update model based on received params
                model.partial_fit(X_train, y_train)
                # if fine_tune:
                #     if params['model'] != "MLPC":
                #         # Train-Update model based on received params
                #         model.partial_fit(X_train, y_train, sample_weight=sample_weights)
                #         # print("sample weights used")
                #     else:
                #         # Train-Update model based on received params
                #         model.partial_fit(X_train, y_train)
                #         # print("sample weights not used")
                # else:
                #     while temp_f1 <= initial_f1 and i < 100:
                #
                #         if params['model'] != "MLPC":
                #             # Train-Update model based on received params
                #             model.partial_fit(X_train, y_train, sample_weight=sample_weights)
                #             # print("sample weights used")
                #         else:
                #             # Train-Update model based on received params
                #             model.partial_fit(X_train, y_train)
                #             # print("sample weights not used")
                #
                #         temp_f1 = self.getModelPerformance(X_test, y_test, model=model)['f1']
                #         i += 1

            else:
                model.fit(X_train, y_train)

        else:
            # Train the model without the aggregation parameters
            model.fit(X_train, y_train)
            self.initial_f1 = self.getModelPerformance(X_test, y_test, model=model)['f1']
        # Get new performance after partial fit
        res = {'test': self.getModelPerformance(X_test, y_test, model=model)}
        if not fine_tune:
            res['train'] = self.getModelPerformance(X_train, y_train, model=model)

        return res

    def updateModel_FL(self, params):
        # Update model with new params
        for pname in params["model_params"]:
            setattr(self.model, pname, params["model_params"][pname])
        temp_f1 =0
        i = 0
        if hasattr(self.model, 'partial_fit') and callable(self.model.partial_fit):
            self.model.partial_fit(self.X_train, self.y_train)
            # while temp_f1 <= self.initial_f1 and i < 100:
            #
            #     if params['model'] != "MLPC":
            #         # Train-Update model based on received params
            #         self.model.partial_fit(self.X_train, self.y_train, sample_weight=self.sample_weights)
            #         # print("sample weights used")
            #     else:
            #         # Train-Update model based on received params
            #         self.model.partial_fit(self.X_train, self.y_train)
            #         # print("sample weights not used")
            #
            #     temp_f1 = self.getModelPerformance(self.X_test, self.y_test, model=self.model)['f1']
            #     i += 1
        else:
            self.model.fit(self.X_train, self.y_train)

        # Get new performance after partial fit
        res = {'test': self.getModelPerformance(self.X_test, self.y_test, model=self.model)}
        res['train'] = self.getModelPerformance(self.X_train, self.y_train, model=self.model)

        return res


    def k_fold_loop(self):
        """Hyperparameter tuning all stored models using StratifiedKFold
            Arguments:
                 n_splits {int} -- Number of folds
                 max_models {int} -- Maximum number of models
            Returns:
                f1s {list} -- Validation F1-scores of each model"""
        skf = StratifiedKFold(n_splits=self.k_folds, shuffle=True, random_state=42)
        scores = [0] * len(self.stored_models)
        X = self.X_train
        y = self.y_train

        for train_index, validation_index in skf.split(X, y):
            # Handle both DataFrame and numpy array for X
            if isinstance(X, pd.DataFrame):
                KFOLD_X_train = X.iloc[train_index]
                KFOLD_X_val = X.iloc[validation_index]
            else:
                KFOLD_X_train = X[train_index]
                KFOLD_X_val = X[validation_index]

            # Handle both Series and numpy array for y
            if isinstance(y, pd.Series):
                KFOLD_y_train = y.iloc[train_index]
                KFOLD_y_val = y.iloc[validation_index]
            else:
                KFOLD_y_train = y[train_index]
                KFOLD_y_val = y[validation_index]

            scaler = StandardScaler()
            if isinstance(KFOLD_X_train, pd.DataFrame):
                KFOLD_X_train = pd.DataFrame(
                    scaler.fit_transform(KFOLD_X_train),
                    columns=KFOLD_X_train.columns,
                    index=KFOLD_X_train.index
                )
                KFOLD_X_val = pd.DataFrame(
                    scaler.transform(KFOLD_X_val),
                    columns=KFOLD_X_val.columns,
                    index=KFOLD_X_val.index
                )
            else:
                KFOLD_X_train = scaler.fit_transform(KFOLD_X_train)
                KFOLD_X_val = scaler.transform(KFOLD_X_val)


            # limit models used for hyperparameter tune
            model_limiter = 0

            for index, params in enumerate(self.stored_models):

                if model_limiter >= self.max_models:
                    break
                model = self.model(**params)
                # Model-specific random state settings
                if hasattr(model, 'random_state'):
                    model.random_state = 42

                model.fit(KFOLD_X_train, KFOLD_y_train)

                params = self.packParams_ML(scores={}, state='model_hp_aggr',
                                            model_type=self.model_type, model=model)
                # aggregate and update model
                ClientIO.send(data=params)
                # Notify C++ that the params are ready
                Client_instance.fl_client.setParamsReady(Client_instance.stop_rounds)

                # Get new params from the aggregation buffer and wait if they are not ready
                agg_params = ClientIO.receive()
                print("Aggregation Round" + str(Client_instance.aggregationRound) + ": Decision to stop loop:" + str(
                    Client_instance.stop_rounds))
                # Parameters read, no need to keep buffer
                # ClientIO.aggregationBuffer.clear()
                # Fit the aggregated parameters
                res = self.updateModel_ML(params=agg_params, X_train=KFOLD_X_train, X_test=KFOLD_X_val,
                                               y_train=KFOLD_y_train, y_test=KFOLD_y_val,
                                               model=model)

                scores[index] += res['test']['f1'] / self.k_folds  # Store the f1 scores of the Validation set
                model_limiter += 1
        return scores

    def model_fine_tune(self):
        """Initiates hyperparameter tuning procedure and sends final model scores to server for voting to start
        Arguments:
            model_combs {dict} -- Model combinations generated by the server
        Returns:
            performance {dict} -- Performance metrics
            params["model"] {str} -- Model type name"""

        #Read model combinations from server
        model_combs = ClientIO.receive()
        # Parameters read, no need to keep buffer
        # ClientIO.aggregationBuffer.clear()
        # create models from model_combs
        self.create_models(model_combs['combinations'], model_combs['model'])
        """-----------------------------------------------------------------------------------------------------------------
                                                     Hyper-Parameter Fine tuning 
        -----------------------------------------------------------------------------------------------------------------"""
        # hyperparameter fine tune those models
        print("Starting Hyperparameter Tuning")
        self.model_type = model_combs['model']
        #Perform hyperparameter fine tune using stratified kfold
        f1_list = self.k_fold_loop()
        print(f"f1_list: {f1_list}")

        params = self.packParams_ML(scores={'val': f1_list}, state='model_hp_voting', model_type=model_combs['model'])
        # send all model f1s (we send weights later)
        ClientIO.send(data=params)
        # Notify C++ that the params are ready
        Client_instance.fl_client.setParamsReady(Client_instance.stop_rounds)
        #Read new paramaters from the aggregation buffer
        params = ClientIO.receive()
        #Use model chosen by server
        self.useBestModel(params)

        # update stop rounds
        Client_instance.stop_rounds = params["stop_sig"]
        print(
            "Round should be 0  :" + str(Client_instance.aggregationRound) + ": Decision to stop loop:" + str(Client_instance.stop_rounds))
        # Initialize model and fit
        performance_dict = self.updateModel_ML(params)
        return performance_dict, params["model"]

    def perform_htune(self):

        if self.ml_mode == 'w':
            # Write dummy params and send to server (so that server thread can continue)
            ClientIO.send(data={'send model combs': 'ready'})
            # Notify C++ that the params are ready
            Client_instance.fl_client.setParamsReady(Client_instance.stop_rounds)

            return self.model_fine_tune()
        elif self.ml_mode == 'uw':

            return self.load_best_model()
        else:
            raise ValueError(f"Unknown ML mode {self.ml_mode}")



Keys_instance = EncryptionKeys()
ClientIO =  BufferIO(role="client")

Client_instance = None
ClientML_instance = None#ClientML()

def resolve_hostname(hostname) -> str | None:
    try:
        return socket.gethostbyname(hostname)
    except socket.gaierror:
        print(f"Error: Could not resolve hostname {hostname}")
        return None

class Client:
    # Aggregation round global variable
    aggregationRound = 0
    stop_rounds = False

    def __init__(self,ip_address=None, hostname=None, portnum=None):
        # Validate input arguments and get valid IP address
        valid_args, ip_address_valid = Client.validate_args(ip_address, hostname, portnum)
        print(f"PARTICIPATE ENTERED: {ip_address_valid}")

        # Init client with validated IP, port number, and Cython buffers as parameters
        self.fl_client = py_fl_client(ip_address_valid, portnum, ClientIO.paramsBuffer, ClientIO.aggregationBuffer)
        self.mid = self.fl_client.getMachineID()
        self.server_config = None
        self.sequence = None

    def start_federated_learning_thread(self):
        # Participate in Federated Learning using separate C++ thread
        self.fl_client.participate()
        self.aggregationRound = 1


    # @staticmethod
    def handle_encryption(self):
        # 1. Share public key
        ClientIO.send(data = {'key':Keys_instance.client_public_key})
        # ClientIO.writeKeyToBuffer(Keys_instance.client_public_key)             # Write public key to paramsBuffer
        self.fl_client.setParamsReady(False)                                            # Reset C++ params flag
        Keys_instance.server_public_key = ClientIO.receive()["key"]     # Get server public key from params
        # ClientIO.aggregationBuffer.clear()

        # 2. Read server encrypted symmetric key
        ClientIO.send(data = {'dummy':'yes'})
        self.fl_client.setParamsReady(False)                                            # Reset C++ params flag
        Keys_instance.symmetric_key = ClientIO.receive(rsa_key = Keys_instance.client_private_key)    # Get server symmetric key
        # print(f"Symmetric key: {Keys_instance.symmetric_key}")
        # ClientIO.aggregationBuffer.clear()
        """Get server configuration"""
        ClientIO.send(data = {'dummy':'yes'})  # Write encrypted dummy message to paramsBuffer
        self.fl_client.setParamsReady(False)
        self.server_config = ClientIO.receive()
        # ClientIO.aggregationBuffer.clear()

        print(f"Server configuration: {self.server_config}")
        ClientML_instance.method = self.server_config['fs-model'] if self.server_config.get('fs-model') != 'none' else None
        self.perform_feature_election = self.server_config['fs-model'] != 'none'
        self.perform_hyperparameter_tuning = self.server_config['tuning']
        # ClientML_instance.blend_factor = self.server_config['blend-factor'] if self.server_config.get('blend-factor') is not None else None
        ClientML_instance.load_data()
        ClientML_instance.ml_tuning = self.server_config['ml-model']
        ClientML_instance.k_folds = self.server_config['k-folds'] if self.server_config.get('k-folds') is not None else None
        ClientML_instance.model_type = self.server_config['ml-model']
        ClientML_instance.ml_mode = self.server_config['ml-mode']
        # self.sequence = self.server_config['sequence']
        score = ClientML_instance.train_model()
        # save results from before running feature selection
        save_var_to_json(score, 'initial_training_results.json', ClientML_instance.experiment_dir)
        return score

    def perform_fs(self, score_before_fs):
        # self.method = given_method
        binary_array, scores_array = ClientML_instance.feature_selection(ClientML_instance.method)
        # Evaluate local FS performance
        ClientML_instance.transform_local_mask()
        # ClientML_instance.updateModel(fs_params)
        score_after_fs = ClientML_instance.train_model()
        # Pack params to be sent to the server
        fs_params = ClientML_instance.packParams_FS(binary_array, scores_array, ClientML_instance.X_train.shape[0],
                                                    score_initial=score_before_fs['test']['f1'], score_fs=score_after_fs['test']['f1'])
        # score = ClientML_instance.getModelPerformance()
        ClientIO.send(data=fs_params)
        # Notify C++ that the params are ready
        self.fl_client.setParamsReady(self.stop_rounds)
        # Get new params from the aggregation buffer and wait if they are not ready
        params = ClientIO.receive()
        print("FS Round " + str(self.aggregationRound) + ": Decision to stop loop:" + str(self.stop_rounds))
        # Parameters read, no need to keep buffer
        if "selected_features" in params:
            global_mask = params["selected_features"]
            # print(global_mask)
        else:
            raise ValueError("Invalid params received after FS")
        # Use the received feature mask to train the model with selected features
        ClientML_instance.transform_global_mask(global_mask)
        # ClientML_instance.model = None
        feature_election_score = ClientML_instance.train_model()
        # send results after running feature election
        ClientIO.send(data=feature_election_score)
        self.fl_client.setParamsReady(self.stop_rounds)
        ClientIO.receive()
        # Increase aggregation rounds
        # self.aggregationRound += 1
        return feature_election_score


    def perform_rounds(self, score):
        # Loop to perform FL rounds
        while not self.stop_rounds:
            params = ClientML_instance.packParams_ML(scores=score, state='', model_type=ClientML_instance.model_type)
            # Write new params to paramsBuffer
            ClientIO.send(data=params)
            # Notify C++ that the params are ready
            self.fl_client.setParamsReady(self.stop_rounds)
            # Get new params from the aggregation buffer and wait if they are not ready
            params = ClientIO.receive()

            #Check stop sig
            self.stop_rounds = params["stop_sig"]
            print("Round " + str(self.aggregationRound) + ": Decision to stop loop:" + str(self.stop_rounds))

            # Partially fit the aggregated parameters
            score = ClientML_instance.updateModel_FL(params)
            # save results
            save_var_to_json(score, f'round_{self.aggregationRound}_results.json', ClientML_instance.experiment_dir)

            # If server closes aggregation notify client thread to exit
            if self.stop_rounds:
                self.fl_client.setParamsReady(self.stop_rounds)

            # Increase aggregation rounds
            self.aggregationRound += 1
        return f1

    def finalize_model(self, f1):
        # Get final model performance and store the model to disk
        print("Performance metrics on beginning.")
        ClientML_instance.getModelPerformance(ClientML_instance.X_test, ClientML_instance.y_test, ClientML_instance.model)
        dump(ClientML_instance.model, f'{ClientML_instance.experiment_dir}/model_{ClientML_instance.model_type}_{self.mid}')
        print("Final model saved to disk...")
        return 0

    def validate_args(ip_address, hostname, portnum) -> tuple[bool, None | str]:
        if (ip_address is not None or hostname is not None) and portnum is not None:
            valid = True
            if ip_address is None and hostname is not None:
                # If booth are given, IP address takes priority over hostname
                ip_address = resolve_hostname(hostname)
        elif ip_address is None and hostname is None:
            print("You need to provide an IP address or hostname.")
            valid = False
        elif portnum is None:
            print("You need to provide port number.")
            valid = False
        return valid, ip_address

""" Valavanis - Argparser to take machine id from terminal params """
parser = argparse.ArgumentParser(description="Specify parameters.")
parser.add_argument('--ip', type=str, required=False, help='Server IP address')
parser.add_argument('--host', type=str, required=False, help='Server Hostname')
parser.add_argument('--port', type=int, required=True, help='Server port number')
parser.add_argument('--dataset', type=str, required=True, help='Dataset ID to use', default='default')


args = parser.parse_args()

if __name__ == "__main__":
    """Check if the provided dataset folder exists"""
    if not os.path.exists(f"{DATA_PATH}{args.dataset}_federated") or len([ filename for filename in os.listdir(f"{DATA_PATH}{args.dataset}_federated") if filename.startswith(f'{args.dataset}_client_')]) == 0:
        raise Exception('Provided dataset folder is not empty or not exists. Please provide a valid dataset folder.' +
                        f'\n\tProvided folder: {DATA_PATH}{args.dataset}_federated' +
                        f'\n\tLooking for files in format: {args.dataset}_client_CLIENTID_train.csv')
    # if not (args.freedom <= 1 and args.freedom >= 0):
    #     raise Exception('Freedom value is not in range [0.0, 1.0].')

    """Create client instance and start federated learning thread"""

    Client_instance = Client(args.ip, args.host, args.port)

    ClientML_instance = ClientML(dataset_name=args.dataset)

    Client_instance.start_federated_learning_thread()

    """ Establish encrypted communication """
    f1 = Client_instance.handle_encryption()

    """ Perform feature selection """
    if Client_instance.perform_feature_election:
        f1 = Client_instance.perform_fs(f1)
    """ Perform hyperparameter tuning """
    if Client_instance.perform_hyperparameter_tuning:
        f1, _ = ClientML_instance.perform_htune()
    else:
        ClientML_instance.scale_data()
        ClientML_instance.initial_f1 = ClientML_instance.getModelPerformance(ClientML_instance.X_test, ClientML_instance.y_test, model=ClientML_instance.model)['f1']
    """ if neither fs or tuning happens, train initial model to get f1 score """
    # if not Client_instance.perform_feature_election and not Client_instance.perform_hyperparameter_tuning:
    #     f1 = ClientML_instance.train_model()

    """Run federated learning rounds"""
    f1 = Client_instance.perform_rounds(f1)

    """Finalize model and store model to disk"""
    result = Client_instance.finalize_model(f1)
