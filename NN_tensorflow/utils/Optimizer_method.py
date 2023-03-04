# ******************************************************************************************************************** #
# ********************************************* #   CENTRALESUPELEC   # ********************************************** #
# ******************************************************************************************************************** #
# Project : deepl
# File    : Optimizer_method.py
# PATH    : NN_tensorflow
# Author  : trisr
# Date    : 18/11/2022
# Description :
"""




"""
# Last commit ID   :
# Last commit date :
# ******************************************************************************************************************** #
# ******************************************************************************************************************** #
# ******************************************************************************************************************** #


# ******************************************************************************************************************** #
# Importations
import tensorflow as tf
import json

# ******************************************************************************************************************** #
# Configuration
dict_params_opt = {}
dict_params_opt["Adam"] = ["name", "learning_rate", "decay", "beta_1", "beta_2", "epsilon", "amsgrad"]
dict_params_opt["Adadelta"] = ["name", "learning_rate", "decay", "rho", "epsilon"]
dict_params_opt["Adagrad"] = ["name", "learning_rate", "decay", "initial_accumulator_value", "epsilon"]
dict_params_opt["Adamax"] = ["name", "learning_rate", "decay", "beta_1", "beta_2", "epsilon"]
dict_params_opt["Ftrl"] = [
    "name",
    "learning_rate",
    "decay",
    "initial_accumulator_value",
    "learning_rate_power",
    "l1_regularization_strength",
    "l2_regularization_strength",
    "beta",
    "l2_shrinkage_regularization_strength",
]
dict_params_opt["Nadam"] = ["name", "learning_rate", "decay", "beta_1", "beta_2", "epsilon"]
dict_params_opt["RMSprop"] = ["name", "learning_rate", "decay", "rho", "momentum", "epsilon", "centered"]
dict_params_opt["SGD"] = ["name", "learning_rate", "decay", "momentum", "nesterov"]

# ******************************************************************************************************************** #
# Function definition
def optimizer_Ftrl(params):
    return tf.keras.optimizers.Ftrl(**params)


def optimizer_Adamax(params):
    return tf.keras.optimizers.Adamax(**params)


def optimizer_Adadelta(params):
    return tf.keras.optimizers.Adadelta(**params)


def optimizer_Adam(params):
    return tf.keras.optimizers.Adam(**params)


def optimizer_Adagrad(params):
    return tf.keras.optimizers.Adagrad(**params)


def optimizer_Nadam(params):
    return tf.keras.optimizers.Nadam(**params)


def optimizer_RMSprop(params):
    return tf.keras.optimizers.RMSprop(**params)


def optimizer_SGD(params):
    return tf.keras.optimizers.SGD(**params)


# ******************************************************************************************************************** #
# Vectorization

OPTIMIZERS = {name[(len("optimizer_")) :]: value for name, value in globals().items() if name.startswith("optimizer_")}

# ******************************************************************************************************************** #
# Class definition
class tf_optimizer:
    """


    
    """

    def __init__(self, name=None, type=None, params={}):
        self.name = name
        self.type = type
        self.params = params

    def update(self, name=None, type=None, params=None):
        if name:
            self.name = name
        if type:
            self.type = type
        if params:
            self.params = params

    def update_parameters(self, new_parameters=None, new_key=None, new_value=None):
        if new_key and new_value:
            self.params[new_key] = new_value
        if new_parameters:
            for key, value in new_parameters.items():
                self.params[key] = value

    def test_valid_parameters(self):
        for key in self.params.keys():
            assert key in dict_params_opt[self.type], "{} isn't a valid parameter for this optimizer"

    def build(self):
        self.test_valid_parameters()
        self.optimizer = OPTIMIZERS[self.type](self.params)

    def __str__(self):
        output = "   Optimizer {} ".format(self.name) + "\n"
        output = output + "-" * 80 + "\n"
        output = output + "   Type   : {} ".format(self.type) + "\n"
        output = output + "   Params : {} ".format(str(self.params)) + "\n"
        output = output + "-" * 80 + "\n"
        return output

    def to_json(self, path):
        params = {}
        params["name"] = self.name
        params["type"] = self.type
        params["params"] = self.params.copy()
        with open(path, "w") as write_file:
            json.dump(params, write_file, indent=4)

    def from_json(self, path):
        file = open(path)
        params = json.load(file)
        self.name = params.get("name")
        self.type = params.get("type")
        self.params = params.get("params")


# ******************************************************************************************************************** #
# Main
if __name__ == "__main__":
    optimizer = tf_optimizer(name="Adam", type="Adam", params={"learning_rate": 0.1})
    optimizer.update(type="Nadam")
    optimizer.test_valid_parameters()
    optimizer.build()
    optimizer.to_json("Opt.json")
    optimizer.from_json("Opt.json")
    print(optimizer)
