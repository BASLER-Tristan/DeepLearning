# ******************************************************************************************************************** #
# ********************************************* #   CENTRALESUPELEC   # ********************************************** #
# ******************************************************************************************************************** #
# Project : deepl
# File    : Callback_Method.py
# PATH    : NN_tensorflow/utils
# Author  : trisr
# Date    : 28/02/2023
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
import os
import json
import copy
# ******************************************************************************************************************** #
# Function definition

def CSV_logger_build(root):
    def CSV_logger(optimizer,loss,metrics,fit_params,dataset):
        os.makedirs(os.path.join(root, "results"), exist_ok=True)
        csv_logger = tf.keras.callbacks.CSVLogger(
            os.path.join(
                root, "results/training_{}.csv".format(len(os.listdir(os.path.join(root, "results"))))
            )
        )
        return csv_logger
    return CSV_logger

def HP_saver_build(root):
    def hp_saver(optimizer,loss,metrics,fit_params,dataset):
        os.makedirs(os.path.join(root, "HP"), exist_ok=True)
        optimizer_parameters = {}
        optimizer_parameters["Type"]=optimizer.type
        optimizer_parameters["Parameters"]=optimizer.params

        loss_parameters = {}
        loss_parameters["Type"]=loss.type
        loss_parameters["Parameters"]=loss.params
        
        metrics_parameters = {}
        metrics_parameters["params"]=metrics.params

        fit_parameters = {}
        fit_parameters["params"]=fit_params.fit_params

        dataset_parameters = {}
        if isinstance(dataset.xparams, dict):
            dataset_parameters["XPARAMS"] = copy.deepcopy(dataset.xparams)
        if isinstance(dataset.xpreprocess, dict):
            dataset_parameters["XPREPROCESS"] = copy.deepcopy(dataset.xpreprocess)
        if isinstance(dataset.yparams, dict):
            dataset_parameters["YPARAMS"] = copy.deepcopy(dataset.yparams)
        if isinstance(dataset.ypreprocess, dict):
            dataset_parameters["YPREPROCESS"] = copy.deepcopy(dataset.ypreprocess)
        for key in dataset_parameters:
            try:
                for key2 in dataset_parameters[key]:
                    for key3 in dataset_parameters[key][key2]:
                        if callable(dataset_parameters[key][key2][key3]):
                            dataset_parameters[key][key2][key3] = dataset_parameters[key][key2][key3].__doc__
            except:
                pass
            
        params = {}
        params["OPT"] = optimizer_parameters
        params["LOSS"] = loss_parameters
        params["METRICS"] = metrics_parameters
        params["FIT"] = fit_parameters
        params["DATASET"] = dataset_parameters
        path = os.path.join(
            root, "HP/HP_{}.csv".format(len(os.listdir(os.path.join(root, "HP"))))
        )
        with open(path, "w") as write_file:
            json.dump(params, write_file, indent=4)

    return hp_saver

BUILDERS = {name[: -(len("_build"))]: value for name, value in globals().items() if name.endswith("_build")}

# ******************************************************************************************************************** #
# Class definition
class tf_callback:
    """



        """

    def __init__(self, name=None, params={}):
        self.name = name
        self.params = params
        self.callbacks = []
    def update(self, name=None, params=None):
        if name:
            self.name = name
        if params:
            self.params = params

    def update_parameters(self, new_parameters=None, new_key=None, new_value=None):
        if new_key and new_value:
            self.params[new_key] = new_value
        if new_parameters:
            for key, value in new_parameters.items():
                self.params[key] = value

    def build(self,optimizer,loss,metrics,fit_params,dataset):
        for key in self.params:
            self.callbacks.append(BUILDERS[key](self.params[key])(optimizer,loss,metrics,fit_params,dataset))

    def to_json(self, path):
        with open(path, "w") as write_file:
            json.dump(self.params, write_file, indent=4)

    def from_json(self, path):
        file = open(path)
        self.params = json.load(file)

    def __str__(self):
        text = ""
        for key in self.params:
            text = text +("""
            {} : {},
            """.format(key,self.params[key])
                  )
        return text

# ******************************************************************************************************************** #
# Main
if __name__ == "__main__":
    test_callback = tf_callback(
        name   = "Test",
        params = {
            "CSV_LOGGER" : "TEST/",
            "HP_saver": "TEST/",
        }
    )
    print(test_callback)
    test_callback.to_json("Test_callback.json")
