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

import json
import os
import random
from joblib import dump


class ModelGenerator:
    random.seed(42)
    def __init__(self, combinations, TEMP_PATH='./Temp'):
        self.TEMP_PATH = TEMP_PATH
        self.config_path = './config.json'
        self.combinations = combinations

    def send_model_combinations(self, model_type:str):
        """Generates random model combinations (of type model_type) for clients by reading config.json file
        Arguments: model_type {str} -- Type of model
        """
        # read config file
        config_params = self.read_config_params(model_type)
        # generate combinations
        print(f"config_params: {config_params}")
        combinations, model = self.generate_combinations(config_params, model_type)
        combinations_dict = {"combinations": combinations, "model": model}

        #Check whether directory exists
        os.makedirs(self.TEMP_PATH, exist_ok=True)
        file_path = os.path.join(self.TEMP_PATH, 'combinations')
        # Dump the dictionary to the file
        with open(file_path, 'wb') as file:
            dump(combinations_dict, file)
        print(f"combinations: {combinations}")

        return combinations, model

    def read_config_params(self, modeltype):
        """Reads model configurations from json"""
        with open(self.config_path, 'r') as file:
            params = json.load(file)
        try:
            return params[modeltype]
        except KeyError:
            raise KeyError(f"{modeltype} not found as key in config")
    def generate_combinations(self, model):
        """Generates random model combinations
        Arguments:
            config_params {dict} -- Dictionary of model parameters
            model -- model name"""
        config_params = self.read_config_params(model)

        if model == "SGDC":
            return self.generate_SGDC_combinations(config_params)
        elif model == "LogReg":
            return self.generate_LogReg_combinations(config_params)
        elif model == "MLPC":
            return self.generate_MLPC_combinations(config_params)
        elif model == "GNB":
            return self.generate_GNB_combinations(config_params)
        else:
            raise ValueError(f"Model {model} not implemented")



    def generate_GNB_combinations(self, params):
        """Generates random GNB combinations from config.json. Generates self.combinations for every var_smoothing value.
        Arguments:
            params {dict} -- Dictionary of GNB parameters
        Returns:
            combinations {dict} -- Dictionary of GNB combinations
            {str} -- Model type name"""
        combinations = list()

        for var_smoothing in params['var_smoothing']:

            selected_combinations = set()

            while len(selected_combinations) < self.combinations:
                random_combination = {
                    "var_smoothing": var_smoothing
                }
                selected_combinations.add(frozenset(random_combination.items()))

            # Convert frozensets back to dicts for final output
            combinations.extend(dict(combination) for combination in selected_combinations)

        return combinations, "GNB"


    def generate_SGDC_combinations(self, params):
        """Generates random SGDC combinations from config.json. Generates self.combinations for every loss type.
        Arguments:
            params {dict} -- Dictionary of SGDC parameters
        Returns:
            combinations {dict} -- Dictionary of SGDC combinations
            {str} -- Model type name"""
        combinations = list()

        for loss in params['loss']:

            selected_combinations = set()  # Use set to avoid duplicates

            while len(selected_combinations) < self.combinations:
                random_combination = {
                    "penalty": random.choice(params['penalty']),
                    "alpha": random.choice(params['alpha']),
                    "l1_ratio": random.choice(params['l1_ratio']),
                    "fit_intercept": random.choice(params['fit_intercept']),
                    "max_iter": random.choice(params['max_iter']),
                    "tol": random.choice(params['tol']),
                    "shuffle": random.choice(params['shuffle']),
                    "epsilon": random.choice(params['epsilon']),
                    "learning_rate": random.choice(params['learning_rate']),
                    "average": random.choice(params['average']),
                    "eta0": random.choice(params['eta0']),
                    "loss": loss
                }

                # Convert dict to frozenset of items for uniqueness
                selected_combinations.add(frozenset(random_combination.items()))

            # Convert frozensets back to dicts for final output
            combinations.extend(dict(combination) for combination in selected_combinations)

        return combinations, "SGDC"




    def generate_LogReg_combinations(self, params):
        """Generates random LogReg combinations from config.json. Generates self.combinations for every solver type.
        Arguments:
            params {dict} -- Dictionary of LogReg parameters
        Returns:
            combinations {dict} -- Dictionary of LogReg combinations
            {str} -- Model type name"""
        combinations = list()

        for solver in params['solver']:

            selected_combinations = set()  # Use set to avoid duplicates

            while len(selected_combinations) < self.combinations:

                if len(selected_combinations) >= 18 and solver == "lbfgs":
                    break
                elif len(selected_combinations) >= 45 and solver == "saga":
                    break

                if solver == 'saga':
                    allowed_penalty = params['penalty']
                else:
                    allowed_penalty = [None, "l2"]

                penalty = random.choice(allowed_penalty)

                if penalty != 'elasticnet':
                    l1_ratio = None
                else:
                    l1_ratio = random.choice(params['l1_ratio'])

                random_combination = {
                    "penalty": random.choice(allowed_penalty),
                    "l1_ratio": l1_ratio,
                    "fit_intercept": random.choice(params['fit_intercept']),
                    "max_iter": random.choice(params['max_iter']),
                    "tol": random.choice(params['tol']),
                    "C": random.choice(params['C']),
                    "intercept_scaling": random.choice(params['intercept_scaling']),
                    "class_weight": random.choice(params['class_weight']),
                    "solver": solver
                }
                selected_combinations.add(frozenset(random_combination.items()))

            # Convert frozensets back to dicts for final output
            combinations.extend(dict(combination) for combination in selected_combinations)

        return combinations, "LogReg"


    def generate_MLPC_combinations(self, params):
        """Generates random MLPC combinations from config.json. Generates self.combinations for every activation type.
        Arguments:
            params {dict} -- Dictionary of MLPC parameters
        Returns:
            combinations {dict} -- Dictionary of MLPC combinations
            {str} -- Model type name"""
        combinations = list()

        for activation in params['activation']:
            selected_combinations = set()

            while len(selected_combinations) < self.combinations:
                random_combination = {
                    "hidden_layer_sizes": tuple(random.choice(params['hidden_layer_sizes'])),
                    "solver": random.choice(params['solver']),
                    "alpha": random.choice(params['alpha']),
                    "learning_rate": random.choice(params['learning_rate']),
                    "learning_rate_init": random.choice(params['learning_rate_init']),
                    "power_t": random.choice(params['power_t']),
                    "max_iter": random.choice(params['max_iter']),
                    "shuffle": random.choice(params['shuffle']),
                    "tol": random.choice(params['tol']),
                    "momentum": random.choice(params['momentum']),
                    "nesterovs_momentum": random.choice(params['nesterovs_momentum']),
                    "validation_fraction": random.choice(params['validation_fraction']),
                    "max_fun": random.choice(params['max_fun']),
                    "beta_1": random.choice(params['beta_1']),
                    "beta_2": random.choice(params['beta_2']),
                    "epsilon": random.choice(params['epsilon']),
                    "activation": activation
                }
                selected_combinations.add(frozenset(random_combination.items()))

            # Convert frozensets back to dicts for final output
            combinations.extend(dict(combination) for combination in selected_combinations)

        return combinations, "MLPC"