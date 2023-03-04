# ******************************************************************************************************************** #
# ********************************************* #   CENTRALESUPELEC   # ********************************************** #
# ******************************************************************************************************************** #
# Project : deepl
# File    : Loss_method.py
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
from tensorflow.keras import losses
import json

# ******************************************************************************************************************** #
# Configuration
dict_params_loss = {}
dict_params_loss["BinaryCrossentropy"] = ["reduction", "name", "from_logits", "label_smoothing", "axis"]
dict_params_loss["CategoricalCrossentropy"] = ["reduction", "name", "from_logits", "label_smoothing", "axis"]
dict_params_loss["CategoricalHinge"] = ["reduction", "name"]
dict_params_loss["CosineSimilarity"] = ["reduction", "name", "axis"]
dict_params_loss["Hinge"] = ["reduction", "name"]
dict_params_loss["Huber"] = ["reduction", "name", "delta"]
dict_params_loss["KLDivergence"] = ["reduction", "name"]
dict_params_loss["LogCosh"] = ["reduction", "name"]
dict_params_loss["Loss"] = ["reduction", "name"]
dict_params_loss["MeanAbsoluteError"] = ["reduction", "name"]
dict_params_loss["MeanAbsolutePercentageError"] = ["reduction", "name"]
dict_params_loss["MeanSquaredError"] = ["reduction", "name"]
dict_params_loss["MeanSquaredLogarithmicError"] = ["reduction", "name"]
dict_params_loss["Poisson"] = ["reduction", "name"]
dict_params_loss["SparseCategoricalCrossentropy"] = ["reduction", "name", "from_logits"]
dict_params_loss["SquaredHinge"] = ["reduction", "name"]
dict_params_loss["categorical_hinge"] = []
dict_params_loss["huber"] = []
dict_params_loss["cosine_similarity"] = ["axis"]
dict_params_loss["huber"] = ["delta"]

# ******************************************************************************************************************** #
# Function definition
def loss_BinaryCrossentropy(params):
    return losses.BinaryCrossentropy(**params)


def loss_CategoricalCrossentropy(params):
    return losses.CategoricalCrossentropy(**params)


def loss_CategoricalHinge(params):
    return losses.CategoricalHinge(**params)


def loss_CosineSimilarity(params):
    return losses.CosineSimilarity(**params)


def loss_Hinge(params):
    return losses.Hinge(**params)


def loss_Huber(params):
    return losses.Huber(**params)


def loss_KLDivergence(params):
    return losses.KLDivergence(**params)


def loss_LogCosh(params):
    return losses.LogCosh(**params)


def loss_Loss(params):
    return losses.Loss(**params)


def loss_MeanAbsoluteError(params):
    return losses.MeanAbsoluteError(**params)


def loss_MeanAbsolutePercentageError(params):
    return losses.MeanAbsolutePercentageError(**params)


def loss_MeanSquaredError(params):
    return losses.MeanSquaredError(**params)


def loss_MeanSquaredLogarithmicError(params):
    return losses.MeanSquaredLogarithmicError(**params)


def loss_Poisson(params):
    return losses.Poisson(**params)


def loss_SparseCategoricalCrossentropy(params):
    return losses.SparseCategoricalCrossentropy(**params)


def loss_SquaredHinge(params):
    return losses.SquaredHinge(**params)


def loss_categorical_hinge(params):
    return losses.categorical_hinge(**params)


def loss_huber(params):
    return losses.huber(**params)


def loss_cosine_similarity(params):
    return losses.cosine_similarity(**params)


def loss_michel(params):
    def loss(xpred, ytrue):
        pass

    return loss


# ******************************************************************************************************************** #
# Vectorization

LOSS = {name[(len("loss_")) :]: value for name, value in globals().items() if name.startswith("loss_")}

# ******************************************************************************************************************** #
# Class definition
class tf_loss:
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
            assert key in dict_params_loss[self.type], "{} isn't a valid parameter for this optimizer"

    def build(self):
        self.test_valid_parameters()
        self.loss = LOSS[self.type](self.params)

    def __str__(self):
        output = "   Loss {} ".format(self.name) + "\n"
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
    loss = tf_loss(name="BN", type="BinaryCrossentropy", params={"from_logits": False})
    loss.update(name="BinaryCross")
    loss.test_valid_parameters()
    loss.build()
    loss.to_json("Loss.json")
    loss.from_json("Loss.json")
    print(loss)
