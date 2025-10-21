import argparse
import csv

# Cython magic:
from FL_cpp_server import py_fl_server
from sklearn.metrics import f1_score

# Common helper methods for server and client
from utils.helper import *
from utils.BufferIO import BufferIO
from ModelGenerator import ModelGenerator as MG
from joblib import load
import datetime
"""
Server_instance Create instance
    |-> self.fl_server.run()#C++ server start 
---------
ServerAggregator Create instance 
---------
Server_instance.start_server()

"""

class EncryptionKeys:
    def __init__(self):
        """ Encryption keys struct """
        # Public Keys of Clients
        self.client_keys = None
        self.symmetric_keys = None
        self.server_public_key = None
        self.server_private_key = None


Keys_instance = EncryptionKeys()
ServerIO =  BufferIO(role="server")
Server_instance = None
Aggregator = None
"""Check if execution is running in docker or not. If running in docker, output directory is /app/data/ else it is ./experiments_data/"""
EXPR_OUTPUT_DIR = '/app/data/' if os.getcwd().startswith('/app') else './logs/run_server/'

SUPPORTED_MODELS = ['GNB', 'SGDC', 'MLPC', 'LogReg']
SUPPORTED_FS_METHODS = ['impetus', 'lasso', 'elastic_net', 'sequential', 'none']

class Server:
    def __init__(self, tuning=False, portnum=8080, numOfClients = 3, numOfIterations = 10, model = None,
                 fe_weighted = None, ml_tuning=None, freedom_rate=None,
                 fe_model = None , k_folds=0, combs=1, ml_mode='w', csv_folder=None, fedprox=False, mu=0.01):

        self.feature_election_scores = None
        self.client_initial_scores = None
        self.fs_scores = None

        # ServerIO.init_buffers()
        self.received_param_sizes = {i: [] for i in range(numOfClients)}  # Track sizes per client
        self.sent_param_sizes = []
        self.feature_election = fe_model if fe_model != 'none' else None
        self.perform_feature_election = fe_model != 'none'
        self.freedom_rate = freedom_rate
        self.ml_tuning = ml_tuning
        self.ml_model = model
        self.fe_weighted = fe_weighted
        self.ml_combs = combs
        self.ml_mode = ml_mode
        self.csv_folder = csv_folder
        self.tuning = tuning
        self.rounds = 0
        self.config = {
                        'fs-model': fe_model,
                        'ml-model': self.ml_model,
                        'tuning': tuning,
                        'k-folds': k_folds,
                        'ml-mode': self.ml_mode
                       }
        self.combinations = None
        self.stop_rounds = False
        self.numberOfClients = numOfClients
        self.numberOfIterations = numOfIterations
        self.timestamp = os.environ.get('EXPERIMENT_TIME',
                                   datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
        self.fedprox = fedprox
        self.mu = mu

        print("Starting server at:", self.timestamp)

        # Setup experiments directory
        self.experiment_dir = os.path.join(EXPR_OUTPUT_DIR , f"{self.timestamp}_{model}_{self.feature_election}_{self.ml_tuning}")
        os.makedirs(self.experiment_dir, exist_ok=True)
        config_dict = self.config.copy()
        config_dict['freedom_rate'] = freedom_rate
        # write config to file
        config_file_path = os.path.join(self.experiment_dir, 'server_config.json')
        with open(config_file_path, 'w') as f:
            json.dump(config_dict, f)

        self.CSV_PATH = './CSV'
        self.TEMP_PATH = './Temp'

        # Initialize server object with given parameters
        self.fl_server = py_fl_server(portnum,
                                      (self.numberOfClients - 1),
                                      ServerIO.outputBuffer,
                                      ServerIO.inputBuffers)

        # Start server listening thread
        self.fl_server.run()

    def get_params_size(self, params):
        """Calculate actual size of parameters in bytes, handling different data types appropriately.

        Args:
            params (dict): Parameter dictionary containing model parameters

        Returns:
            int: Total size in bytes
        """

        def get_size_recursive(obj):
            if obj is None:
                return 0

            # Handle numpy arrays directly
            if isinstance(obj, np.ndarray):
                return obj.nbytes

            # Handle lists/tuples of numpy arrays (like MLPC coefs/intercepts)
            if isinstance(obj, (list, tuple)):
                return sum(get_size_recursive(item) for item in obj)

            # Handle dictionaries (including nested)
            if isinstance(obj, dict):
                return sum(get_size_recursive(value) for value in obj.values())

            # Handle basic numeric types
            if isinstance(obj, (int, float, bool)):
                return obj.__sizeof__()

            # Handle strings
            if isinstance(obj, str):
                return len(obj.encode('utf-8'))

            # For any other type, try to get size directly or estimate
            try:
                return obj.__sizeof__()
            except (AttributeError, TypeError):
                try:
                    return len(str(obj).encode('utf-8'))
                except:
                    return 0

        # Get total size starting from the model_params
        if "model_params" not in params:
            return 0

        total_size = get_size_recursive(params["model_params"])

        # Add logging for debugging
        if "model" in params and params["model"] == "MLPC":
            # print(f"Parameter Sizes Breakdown:")
            for param_name, param_value in params["model_params"].items():
                param_size = get_size_recursive(param_value)
                print(f"  {param_name}: {param_size} bytes")

        return total_size

    def save_results(self):
        # TODO Save hyperparameter tuning stuff
        self.save_data_to_csv(self.ml_model,self.experiment_dir)
        self.save_fs_data(self.experiment_dir)
        # Save stats to JSON file
        self.save_parameter_stats()

    def save_fs_data(self, base_path):
        fs_filename = 'feature_election_results.csv'
        fs_filepath = os.path.join(base_path, fs_filename)
        # Initialize lists to store scores for averaging
        initial_scores = []
        local_fs_scores = []
        federated_fs_scores = []
        with open(fs_filepath, 'w', newline='') as csvfile:
            # Log FS data
            if self.perform_feature_election:
                writer = csv.writer(csvfile)
                writer.writerow(['Client', 'Initial F1', 'Local FS', 'Feature Election'])

                # Write individual client scores
                for client_id in range(self.numberOfClients):
                    client_fs_data = {
                        'initial_training': self.client_initial_scores[client_id].get('test', None),
                        'local_fs': self.fs_scores[client_id].get('test', None),
                        'federated_fs': self.feature_election_scores[client_id].get('test', None)
                    }
                    # Append non-None scores for averaging
                    if client_fs_data['initial_training'] is not None:
                        initial_scores.append(client_fs_data['initial_training'])
                    if client_fs_data['local_fs'] is not None:
                        local_fs_scores.append(client_fs_data['local_fs'])
                    if client_fs_data['federated_fs'] is not None:
                        federated_fs_scores.append(client_fs_data['federated_fs'])

                    writer.writerow([
                        f'Client {client_id + 1}',
                        client_fs_data['initial_training'],
                        client_fs_data['local_fs'],
                        client_fs_data['federated_fs']
                    ])

                # Calculate averages, handling empty lists
                avg_initial = sum(initial_scores) / len(initial_scores) if initial_scores else None
                avg_local_fs = sum(local_fs_scores) / len(local_fs_scores) if local_fs_scores else None
                avg_federated_fs = sum(federated_fs_scores) / len(federated_fs_scores) if federated_fs_scores else None

                # Add a blank row for readability
                writer.writerow([])

                # Write averages row
                writer.writerow([
                    'Average',
                    avg_initial,
                    avg_local_fs,
                    avg_federated_fs
                ])

            else:
                self.client_initial_scores = [None]* self.numberOfClients
                writer = csv.writer(csvfile)
                writer.writerow(['Client', 'Initial F1'])
                # Write individual client scores
                for client_id in range(self.numberOfClients):
                    client_fs_data = {
                        'initial_training': self.client_initial_scores[client_id]
                    }
                    # Append non-None scores for averaging
                    if client_fs_data['initial_training'] is not None:
                        initial_scores.append(client_fs_data['initial_training'])
                    writer.writerow([
                        f'Client {client_id + 1}',
                        client_fs_data['initial_training'],
                    ])
                # Calculate averages, handling empty lists
                avg_initial = sum(initial_scores) / len(initial_scores) if initial_scores else None
                writer.writerow([])

                # Write averages row
                writer.writerow([
                    'Average',
                    avg_initial,
                ])


    def save_data_to_csv(self, model, csv_folder=None):
        """
        Saves all stored data for each client and the aggregations to CSV files.
        Arguments:
              model {str} -- Model name
              csv_folder {str} -- Folder where to save data (optional)
        """
        # if csv_folder is None:
        #     # Ensure the directory exists
        #     base_path = f'{self.CSV_PATH}/{model}_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}'
        # else:
        #
        base_path = os.path.join(csv_folder,"CSV")

        #if folder does not exist, create it before saving the data
        os.makedirs(base_path, exist_ok=True)

        log_filename = f'log.txt'
        log_filepath = os.path.join(base_path, log_filename)

        best_model_path = f"{self.TEMP_PATH}/best_model"

        if os.path.exists(best_model_path):
            best_model = load(best_model_path)
        else:
            best_model = None

        if not os.path.exists(log_filepath):
            with open(log_filepath, "w") as file:
                file.write(f"Dataset: {load(f'{self.TEMP_PATH}/latest_csv_path')}\n"
                           f"Clients: {self.numberOfClients}\n"
                           f"LR: {Aggregator.LR}\n"
                           f"Iterations arg: {self.numberOfIterations}\n"
                           f"Actual aggregation rounds number: {self.rounds}\n"
                           f"Feature Election method: {self.feature_election}\n"
                           f"Freedom rate: {self.freedom_rate}\n"
                           f"Model Name: {model}\n"
                           # f"Best model: {vars(load(f"{self.TEMP_PATH}/best_model"))}\n"
                           f"Combinations generated: {self.combinations}\n")
                if best_model is not None:
                    file.write(f"Best model: {vars(best_model)}\n")


        # Save clients and aggregation data to CSV
        if self.ml_mode == 'w':
            # Save client data to CSV (train and test f1s)
            for client_id in range(self.numberOfClients):
                client_filename = f'{model}_client_{client_id}_data.csv'
                client_filepath = os.path.join(base_path, client_filename)
                with open(client_filepath, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['Round', 'F1 Score'])
                    writer.writerows(zip(Aggregator.client_data[client_id]['rounds'], Aggregator.client_data[client_id]['f1s']))
                print(f'Saved client {client_id} data to {client_filename}')

                train_client_filename = f'{model}_train_client_{client_id}_data.csv'
                train_client_filepath = os.path.join(base_path, train_client_filename)
                with open(train_client_filepath, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['Round', 'F1 Score'])
                    writer.writerows(
                        zip(Aggregator.train_client_data[client_id]['rounds'], Aggregator.train_client_data[client_id]['f1s']))
                print(f'Saved train client {client_id} data to {train_client_filename}')

            # Save aggregation data to CSV (weighted or unweighted)
            w_aggregation_filename = f'{model}_weighted_aggregation_data.csv'
            w_aggregation_filepath = os.path.join(base_path, w_aggregation_filename)
            with open(w_aggregation_filepath, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Round', 'F1 Score'])
                writer.writerows(zip(Aggregator.w_aggregation_data['rounds'], Aggregator.w_aggregation_data['f1s']))
            print(f'Saved weighted aggregation data to {w_aggregation_filename}')
        elif self.ml_mode == 'uw':
            uw_aggregation_filename = f'{model}_unweighted_aggregation_data.csv'
            uw_aggregation_filepath = os.path.join(base_path, uw_aggregation_filename)
            with open(uw_aggregation_filepath, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Round', 'F1 Score'])
                writer.writerows(zip(Aggregator.uw_aggregation_data['rounds'], Aggregator.uw_aggregation_data['f1s']))
            print(f'Saved unweighted aggregation data to {uw_aggregation_filename}')
        else:
            raise ValueError(f"Unkown mode {self.ml_mode}")

    def save_parameter_stats(self):
        """Save parameter size statistics"""
        stats = {
            'overall': {
                'total_rounds': self.rounds,
                'timestamp': self.timestamp
            }
        }

        total_received = 0
        total_sent = sum(self.sent_param_sizes)  # Total sent across all rounds

        # Calculate overall statistics
        stats['overall'].update({
            'total_bytes_received': total_received,
            'total_bytes_sent': total_sent,

        })
        save_var_to_json(stats, 'parameter_size_stats.json', self.experiment_dir)

    def perform_ml_tuning(self):
        if self.ml_mode == 'w':
            #Init ModelGenerator
            ModelGenerator = MG(self.ml_combs)
            combination_dict, model_type = ModelGenerator.generate_combinations(self.ml_model)
            self.combinations = combination_dict
            # Send random generated model combinations to clients
            # Create params dict
            params = {
                'combinations': combination_dict,
                'model': model_type,
                'mode': self.ml_mode
            }
            #Read dummy params from client (so server can send new data)
            ServerIO.receive(max_id=self.numberOfClients)

            #Send params to clients
            ServerIO.send(data = params)
            # Clear input parameter buffers, in order to reuse on next iteration
            # ServerIO.clientDataBuffers.clear_all_buffers()
            # Notify the handler threads that the aggregated data is ready to be sent to the client
            self.fl_server.aggregationDone(self.stop_rounds)

        else:
            print(f'ML Tuning is in {self.ml_mode} mode')
            return

    def start_server(self, mode='weighted'):
        self.exchange_public_keys()
        self.handle_symmetric_keys()

        # Feature Election Algorithm - here
        """In case the FL should be executed with Feature Selection function that execute FS"""
        # Find most important features to keep across all clients. Send the selected features to all clients.
        if self.perform_feature_election:
            self.perform_fe(mode)


        """After the feature selection, we should proceed into federated ml fine-tuning of models."""
        # request from all clients train on their data and aggregate their parameters and send them back to clients.
        if self.tuning:
            self.perform_ml_tuning()

        """Final model evaluation. Execute model over the testing sets."""
        # Request all clients to predict their testing sets.

        self.aggregation_loop()
        self.save_results()
        self.fl_server.allDone()


    def exchange_public_keys(self):
        # Encryption:
        # 1. Public key exchange:
        # Generate RSA keys
        Keys_instance.server_private_key, Keys_instance.server_public_key = generate_rsa_keys()
        # Receive client key/secret in params
        client_keys_dict = ServerIO.receive(max_id=self.numberOfClients)
        Keys_instance.client_keys = [entry['key'] for entry in client_keys_dict]
        # Send key/secret to client as aggregated data
        ServerIO.send(data={'key': Keys_instance.server_public_key}, max_id=self.numberOfClients)
        # Clear key buffers, in order to reuse on next iteration
        # ServerIO.clientDataBuffers.clear_all_buffers()
        self.fl_server.aggregationDone(False)


    def handle_symmetric_keys(self):
        # 2. Generate and send encrypted symmetric keys:
        # Dummy
        ServerIO.receive(max_id=self.numberOfClients)
        Keys_instance.symmetric_keys = [generate_aes_key() for _ in range(self.numberOfClients)]

        # Send can be called without keys and both send and receive without number of clients, after this call
        ServerIO.setKeys(Keys_instance.symmetric_keys, self.numberOfClients)
        # Send key/secret to client as aggregated data
        ServerIO.send(max_id=self.numberOfClients, data=Keys_instance.symmetric_keys, rsa_key=Keys_instance.client_keys)

        self.fl_server.aggregationDone(False)
        # Dummy
        ServerIO.receive()
        # TODO change
        # self.fl_server.aggregationDone(False)
        # Propagate the global training/execution configuration to all clients
        # TODO HERE
        ServerIO.send(data=self.config)
        self.fl_server.aggregationDone(False)

    def perform_fe(self, set_mode):
        # print("Election reached.....")
        # Read buffers to get params
        params = ServerIO.receive(max_id=self.numberOfClients)
        # Perform Feature Election
        Aggregator.feature_election(self.numberOfClients, params, freedom_rate=self.freedom_rate, mode=set_mode)
        # Clear input parameter buffers, in order to reuse for main loop
        # ServerIO.clientDataBuffers.clear_all_buffers()
        # Notify the handler threads that the global feature mask is ready to be sent to the client
        # TODO check TRUE/FALSE
        self.fl_server.aggregationDone(False)


    def aggregation_loop(self):
        # Loop the process if stop_rounds is False
        self.rounds = 0
        while not self.stop_rounds:
            #self.rounds = self.rounds + 1
            # Read buffers to get params
            params = ServerIO.receive(max_id=self.numberOfClients)
            # Track received parameter sizes for each client
            for client_id in range(self.numberOfClients):
                if client_id < len(params):  # Check if client params exist
                    size = self.get_params_size(params[client_id])
                    self.received_param_sizes[client_id].append(size)

            # Aggregate the given parameters
            Aggregator.aggregate_model(self.numberOfClients, params)

            # Clear input parameter buffers, in order to reuse on next iteration
            # ServerIO.clientDataBuffers.clear_all_buffers()
            # Notify the handler threads that the aggregated data is ready to be sent to the client
            self.fl_server.aggregationDone(self.stop_rounds)
        self.save_results()
        # self.save_fs_data(self.experiment_dir)


class ServerAggregator:
    def __init__(self, federated_fs=True, mode='w', lr=0.0):
        self.LR = lr
        self.global_params = list()
        self.federated_fs = federated_fs
        self.prevScore = None
        self.newScore = None
        # Performance threshold to stop aggregation rounds
        self.thr = 0.055
        self.model_type = None
        self.local_optimal = 0
        self.mode = mode
        self.client_data = {client_id: {'rounds': [], 'f1s': []} for client_id in range(Server_instance.numberOfClients)}
        self.train_client_data = {client_id: {'rounds': [], 'f1s': []} for client_id in range(Server_instance.numberOfClients)}
        self.w_aggregation_data = {'rounds': [], 'f1s': []}
        self.uw_aggregation_data = {'rounds': [], 'f1s': []}

    @staticmethod
    def _get_features_intersection(masks: np.ndarray) -> np.ndarray:
        return np.all(masks == 1, axis=0)
    @staticmethod
    def _get_features_union(masks: np.ndarray) -> np.ndarray:
        return np.any(masks == 1, axis=0)

    @staticmethod
    def _get_features_difference(masks: np.ndarray) -> np.ndarray:
        union = ServerAggregator._get_features_union(masks)
        intersection = ServerAggregator._get_features_intersection(masks)
        return union & ~intersection

    def _log_election_scores(self, params):
        num_clients = len(params)
        # Log f1 performances of clients in server
        # Server_instance.feature_election_scores = np.zeros(num_clients, dtype=int)
        Server_instance.feature_election_scores = [None] * num_clients
        for i in range(num_clients):
            Server_instance.feature_election_scores[i] = params[i]

    """
    mask = [ 0     X   0     0    X]
    probabilities = [[0.1,  0.9, 0.3, 0.0, 0.0]
                     [0.05, 0.1, 0.0, 0.0, 0.0]
                     [0.00, 0.01, 0.1, 0.2, 0.0]]
    -> Scale each row of V
    -> Make 0.00 coef for features in intersection of clients
    -> Multiply by the scale factor of samples v * w
    -> New V = sum of rows of V
    -> select freedom_rate of highest scores features in V
    Corner cases:
        |-freedom_rate == 0.0 no need to compute anything in v since we took only intersection based on mask from clients (masks intersection)
        |-freedom_rate == 1.0 no need to compute anything in v since we tool all selected features from each client (masks union)
    Freedom_rate: percentage of features select by clients that don't belong to intersection but belongs to union (union - intersection).
                We select TOP freedom_rate percentage of these features.
                (example: if we have total union of 80 features where only 20 in intersection with freedom_rate of 0.5 so we will also select 50% of (80-20)=60 features -> 30 features + 20 from intersection )
    """
    def feature_election(self, max_id, params, freedom_rate, mode='uniform'):
        if mode not in ['weighted', 'uniform']:
            raise ValueError(f'Mode:{mode} is invalid')
        # Reset the aggregation data buffer before writing
        # ServerIO.aggregationDatabuffer.clear_all_buffers()
        # Calculate total number of features and samples
        total_samples = sum(params[i]["num_of_samples"] for i in range(len(params)))
        # Initialize arrays
        num_clients = len(params)
        w = np.zeros(len(params))
        num_features = len(params[0]["feature_preference"])

        # Log f1 performances of clients in server
        # self.client_initial_scores = np.zeros(num_clients, dtype=int)
        # self.fs_scores = np.zeros(num_clients, dtype=int)
        Server_instance.client_initial_scores = [None] * num_clients
        Server_instance.fs_scores = [None] * num_clients
        for i in range(num_clients):
            Server_instance.client_initial_scores[i] = params[i]["initial_score"]
            Server_instance.fs_scores[i] = params[i]["fs_score"]

        masks = np.zeros((num_clients, num_features), dtype=int)
        scores = np.zeros((num_clients, num_features), dtype=float)
        # weighted_scores = np.zeros((num_clients, num_features), dtype=float)
        for i in range(num_clients):
            # Get client preference mask from params
            scores[i] = params[i]["feature_preference"]
            w[i] = (params[i]["num_of_samples"]) / total_samples
            # weighted_scores[i] = w[i] * scores[i]
            # Get binary client selection mask from params
            masks[i] = params[i]["selected_features"]
            save_var_to_json(masks[i],f"client_{i}_binary_mask.json",Server_instance.experiment_dir)
            save_var_to_json(scores[i],f"client_{i}_score_mask.json",Server_instance.experiment_dir)

        masks = np.array(masks)
        scores = np.array(scores)


        intersection_mask = ServerAggregator._get_features_intersection(masks)
        union_mask = ServerAggregator._get_features_union(masks)
        # Handle edge cases
        if freedom_rate == 0:
            G = intersection_mask
        elif freedom_rate == 1:
            G = union_mask
        else:
            # Get features in (union - intersection)
            difference_mask = self._get_features_difference(masks)
            # Scale scores and zero out non-selected features for each client
            scaled_scores = []
            for client_index, (client_mask, client_scores) in enumerate(zip(masks, scores)):
                # First scale all selected features for the iteration client
                scaled_row = np.zeros_like(client_scores)
                # Find the selected features of each client
                selected_features = client_mask == 1

                #  If there are selected features
                if np.any(selected_features):
                    selected_scores = client_scores[selected_features]
                    max_score = np.max(selected_scores)
                    min_score = np.min(selected_scores)
                    range_score = max_score - min_score

                    if range_score > 0:
                        scaled_row[selected_features] = (client_scores[selected_features] - min_score) / range_score
                    # If all preferences are zero
                    else:
                        scaled_row[selected_features] = 1.0

                # Then zero out the intersection features
                scaled_row[intersection_mask] = 0.0
                if mode == 'weighted':
                    weighted_row = w[client_index]* scaled_row
                    # Append processed row to dictionary
                    scaled_scores.append(weighted_row)
                else:
                    scaled_scores.append(scaled_row)
            # Aggregate scaled scores from all clients
            aggregated_scores = np.sum(scaled_scores, axis=0)

            # Calculate number of additional features to select from the difference mask
            n_additional = int(np.ceil(np.sum(difference_mask) * freedom_rate))

            # Select top features from difference set
            diff_scores = aggregated_scores[difference_mask]
            if len(diff_scores) > 0:
                threshold = np.sort(diff_scores)[-min(n_additional, len(diff_scores))]
                selected_difference = difference_mask & (aggregated_scores >= threshold)
                # Combine intersection and selected difference features
                G = intersection_mask | selected_difference
            else:
                # selected_difference = np.zeros_like(difference_mask)
                G = intersection_mask
        save_var_to_json(G, f"selected_features_after_election.json", Server_instance.experiment_dir)
        # Return selected feature vector
        selected_features_params = {}
        selected_features_params["num_of_samples"] = total_samples
        selected_features_params["selected_features"] = G
        selected_features_params["feature_preference"] = G
        ServerIO.send(data=selected_features_params)
        Server_instance.fl_server.aggregationDone(False)
        # Receive Feature Election scores from each client
        feature_election_score_params = ServerIO.receive()
        self._log_election_scores(feature_election_score_params)
        ServerIO.send(data={"dummy":True})


    def checkScoreDiff(self):
        """Check whether threshold is broken
        Returns:
            False: |newScore - prevScore| > thr
            True: |newScore - prevScore| < thr"""
        if self.newScore is None or self.prevScore is None:
            return False
        else:
            #Do we want the abs??
            return abs(self.newScore - self.prevScore) < self.thr

    def update_data(self, params):
        """
        Updates client and aggregation data (f1 per round)
        Arguments:
            params {dict} -- Dictionary of model parameters of clients
        """
        round_num = Server_instance.rounds
        Server_instance.rounds += 1

        if self.mode == 'w':
            # Update each client's data
            for client_id in range(Server_instance.numberOfClients):
                self.client_data[client_id]['rounds'].append(round_num)
                self.client_data[client_id]['f1s'].append(params[client_id]['test_perf']['f1'])

                self.train_client_data[client_id]['rounds'].append(round_num)
                self.train_client_data[client_id]['f1s'].append(params[client_id]['train_perf']['f1'])

            # Update weighted aggregation data
            self.w_aggregation_data['rounds'].append(round_num)
            self.w_aggregation_data['f1s'].append(self.newScore)
        elif self.mode == 'uw':
            # Update unweighted aggregation data
            self.uw_aggregation_data['rounds'].append(round_num)
            self.uw_aggregation_data['f1s'].append(self.newScore)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def compute_total_f1(self, params):
        y_pred_full = list()
        y_true_full = list()

        #print(f"test_perf: {params[0]['test_perf']}")

        for i in range(len(params)):
            y_pred_full.append(params[i]['test_perf']['y_pred'])
            y_true_full.append(params[i]['test_perf']['y_true'])
        y_pred = np.concatenate(y_pred_full)
        y_true = np.concatenate(y_true_full)

        return f1_score(y_true, y_pred, average='binary')

    def aggregate_model(self, max_id, params):
        # Reset the aggregation data buffer before writing
        # ServerIO.aggregationDatabuffer.clear_all_buffers()

        total_samples = sum(params[i]["num_of_samples"] for i in range(len(params)))
        aggregated_params = dict()

        # Check if it is voting phase
        if params[0]["state"] == 'model_hp_voting':
            # voting, each vote has weight analogous to num of samples
            votes = list()
            print(params)
            for param in params:
                for index, score in enumerate(param["val_perf"]):
                    if index >= len(votes):
                        votes.append(score * (param["num_of_samples"] / total_samples))
                    else:
                        votes[index] = score * (param["num_of_samples"] / total_samples)
            # find best model:
            max_vote = max(votes)
            max_index = votes.index(max_vote)
            aggregated_params["best_index"] = max_index
        # Aggregation phase
        else:
            global_model_params = params[0]["model_params"]
            #self.newScore = self.compute_total_f1(params) if params[0]["state"] != 'model_hp_aggr' else 0.0
            self.newScore = sum(params[i]["test_perf"]["f1"] * params[i]["num_of_samples"] for i in range(len(params))) / total_samples if params[0]["state"] != 'model_hp_aggr' else 0.0

            # update local optimal
            if self.newScore > self.local_optimal and self.newScore != 0:
                self.local_optimal = self.newScore

            # Check whether the aggregation should continue
            Server_instance.stop_rounds = (Server_instance.rounds >= Server_instance.numberOfIterations)

            # Update plot with new weighted f1
            if params[0]['state'] != 'model_hp_aggr':
                self.update_data(params)

            # Set previous score as current, for the next comparison
            self.prevScore = self.newScore

            if self.mode == 'w':
                print("Weighted average score: " + str(self.newScore))
            else:
                print("Unweighted average score: " + str(self.newScore))

            aggregated_params["model_params"] = dict()

            #iterations counter
            iter = 0

            aggregated_params["model_params"] = dict()
            # for every param check whether the param value types are supported and then aggregate
            for param_name, param_value in params[0]["model_params"].items():
                # print(f"Param: {param_name} Type: {type(param_value)}")
                if isinstance(param_value, np.ndarray) or isinstance(param_value, float) or isinstance(param_value, int):
                    if Server_instance.fedprox:
                        # Calculate the proximal term for FedProx
                        prox_term = sum(
                            (params[i]["num_of_samples"] / total_samples) *
                            (params[i]["model_params"][param_name] - global_model_params[param_name])
                            for i in range(len(params))
                        )
                        # Apply the proximal term to the global model
                        aggregated_params["model_params"][param_name] = global_model_params[
                                                                            param_name] + Server_instance.mu * prox_term
                    else:
                        # Standard FedAvg for numpy arrays and floats
                        param_name_avg = sum(
                            params[i]["model_params"][param_name] * params[i]["num_of_samples"] for i in
                            range(len(params))) / total_samples \
                            if self.mode == 'w' else sum(
                            params[i]["model_params"][param_name] for i in range(len(params))) / len(params)

                        if params[0]['state'] != 'model_hp_aggr' and self.LR > 0:
                            if self.global_params is not None and len(self.global_params) == len(
                                    params[0]["model_params"]):
                                # print(f"Update of global params iteration: {iter}, Global params: {self.global_params}")
                                self.global_params[iter] = param_name_avg * self.LR + self.global_params[iter]  # / 2
                            else:
                                # print(f"Initialization of global params iteration: {iter}, Global params: {self.global_params}")
                                self.global_params.append(param_name_avg)

                            aggregated_params["model_params"][param_name] = self.global_params[iter]
                            iter += 1

                        if self.LR <= 0 :
                            aggregated_params["model_params"][param_name] = param_name_avg
                            # print(f"No use of global params mode {self.mode}")

                elif isinstance(param_value, list) and (isinstance(param_value[0], np.ndarray)):
                    # Initialize the param with zeros of the same shape
                    param_name_avg = [np.zeros_like(params[0]["model_params"][param_name][layer]) for layer in
                                      range(len(params[0]["model_params"][param_name]))]

                    # Process each client's parameters
                    for i in range(len(params)):
                        num_samples = params[i]["num_of_samples"]

                        for layer in range(len(params[0]["model_params"][param_name])):
                            if Server_instance.fedprox:
                                # Accumulate FedProx proximal term
                                param_name_avg[layer] += (num_samples / total_samples) * (
                                        params[i]["model_params"][param_name][layer] - global_model_params[param_name][
                                    layer]
                                )
                            else:
                                # Standard FedAvg accumulation
                                if self.mode == 'w':
                                    param_name_avg[layer] += num_samples * params[i]["model_params"][param_name][layer]
                                else:
                                    param_name_avg[layer] += params[i]["model_params"][param_name][layer]

                    # Apply appropriate final step based on algorithm
                    if Server_instance.fedprox:
                        # For FedProx, apply the proximal term to global model
                        param_name_avg = [
                            global_model_params[param_name][layer] + Server_instance.mu * param_name_avg[layer]
                            for layer in range(len(param_value))]
                    else:
                        # For standard FedAvg, normalize by weights
                        param_name_avg = [pname / total_samples for pname in param_name_avg] if self.mode == 'w' \
                            else [pname / len(params) for pname in param_name_avg]

                        if params[0]['state'] != 'model_hp_aggr' and self.LR > 0:
                            if self.global_params is not None and len(self.global_params) == len(
                                    params[0]["model_params"]):
                                # print(f"Update of global params iteration: {iter}, Global params: {self.global_params}")
                                for layer in range(len(param_name_avg)):
                                    self.global_params[iter][layer] = param_name_avg[layer] * self.LR + \
                                                                      self.global_params[iter][layer]  # / 2
                            else:
                                # print(f"Initialization of global params iteration: {iter}, Global params: {self.global_params}")
                                self.global_params.append(param_name_avg)

                            aggregated_params["model_params"][param_name] = self.global_params[iter]
                            iter += 1
                        if self.LR <= 0:
                            aggregated_params["model_params"][param_name] = param_name_avg
                            # print(f"No use of global params mode {self.mode}")
                else:
                    raise ValueError(f"Unsupported model param type {type(param_value)}")

            if params[0]['state'] != "model_hp_aggr":
                Server_instance.rounds += 1

        aggregated_params["stop_sig"] = Server_instance.stop_rounds
        aggregated_params["model"] = params[0]['model']

        print("Decision to stop: " + str(Server_instance.stop_rounds))

        # Track sent parameter sizes (aggregated params)
        Server_instance.sent_param_sizes.append(Server_instance.get_params_size(aggregated_params))
        ServerIO.send(data=aggregated_params)

def create_parser():
    parser = argparse.ArgumentParser(description="Give model parameters.")

    # Required arguments for all operations
    required_group = parser.add_argument_group('Required Arguments')
    required_group.add_argument('--port', type=int, default=8080,
                                help='Server port number')
    required_group.add_argument('--clients', type=int, required=True,
                                help='Number of Clients')
    required_group.add_argument('--model', dest='model', required=True,
                                choices=SUPPORTED_MODELS,
                                help=f'Select machine learning model from list: {SUPPORTED_MODELS}')

    # Common optional arguments
    common_group = parser.add_argument_group('Common Optional Arguments')
    common_group.add_argument('--iter', type=int, default=15,
                              help='Number of Iterations')
    common_group.add_argument('--csv_folder', type=str, default=None,
                              help='Folder name, where result csv\'s will be stored.')
    common_group.add_argument('--fedprox', action='store_true', help='Use FedProx Aggregation')
    common_group.add_argument('--mu', type=float, default=0.01, help='FedProx Proximal Term Strength')

    # Feature Selection related arguments
    fs_group = parser.add_argument_group('Feature Selection Arguments')
    fs_group.add_argument('--fe-method', dest='fe_method',
                          choices=SUPPORTED_FS_METHODS,
                          help=f'Select feature election method from list: {SUPPORTED_FS_METHODS}')
    fs_group.add_argument('--fe_weighted', dest='fe_weighted',
                          action='store_true', default=False,
                          help='Flag value to selected weighted averaging during feature selection.')
    fs_group.add_argument('--freedom', dest='freedom',
                          type=float, default=0.1,
                          help='Degree of freedom for feature selection of non-overlapping features. '
                               'Provide a float value in range : [0.0, 1.0]. Default value is 0.1.')

    # Fine-tuning related arguments
    tuning_group = parser.add_argument_group('Fine-tuning Arguments')
    tuning_group.add_argument('--tuning', dest='tuning',
                              action='store_true', default=False,
                              help='Enable model fine-tuning')
    tuning_group.add_argument('--k_folds', type=int, default=3,
                              help='Number of k-folds to be used for cross validation during '
                                   'the ml model fine-tuning. Default value is 0.')
    tuning_group.add_argument('--combs', type=int, default=1,
                              help='Number of combinations per target parameter that server '
                                   'should generate when ml_tuning is enabled.')
    tuning_group.add_argument('--ml_mode', type=str, default='w',
                              choices=['w', 'uw'],
                              help='Aggregation mode when ml_tuning is enabled (w=weighted or u=unweighted).')
    tuning_group.add_argument('--lr', type=float, default='0.0',
                              help='Learning Rate for residual path.')


    return parser


def validate_args(args):
    """Validate argument combinations and values."""
    if args.tuning:
        if args.k_folds <= 0:
            raise ValueError("k_folds must be greater than 0 when tuning is enabled")
        if args.combs < 1:
            raise ValueError("combs must be at least 1 when tuning is enabled")

    if args.fe_method:
        if not (0.0 <= args.freedom <= 1.0):
            raise ValueError("freedom must be between 0.0 and 1.0")


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    validate_args(args)
    #create server instance
    Server_instance = Server(portnum=args.port, numOfClients = args.clients, numOfIterations = args.iter,
                             freedom_rate=args.freedom, model = args.model, tuning=args.tuning,
                             fe_weighted=args.fe_weighted,
                             k_folds=args.k_folds, combs=args.combs, ml_mode=args.ml_mode,
                             fe_model=args.fe_method, csv_folder=args.csv_folder, fedprox=args.fedprox, mu=args.mu)


    # Server_instance = Server(args.p, args.c, args.i, freedom_rate, perform_fe=False)
    Aggregator = ServerAggregator(mode=args.ml_mode, lr=args.lr)

    # Run main function with given args
    #Server.host(args.p, args.c, args.i)
    Server_instance.start_server(mode='weighted' if args.fe_weighted else 'uniform')