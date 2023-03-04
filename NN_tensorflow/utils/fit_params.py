# ******************************************************************************************************************** #
# ********************************************* #   CENTRALESUPELEC   # ********************************************** #
# ******************************************************************************************************************** #
# Project : deepl
# File    : fit_params.py
# PATH    : NN_tensorflow
# Author  : trisr
# Date    : 06/12/2022
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
import json
import pandas as pd

# ******************************************************************************************************************** #
# class definition
class tf_FP:
    """



    Returns
    -------

    """

    def __init__(self):
        self.fit_params = {
            "x": None,
            "y": None,
            "batch_size": None,
            "epochs": 1,
            "verbose": "auto",
            "callbacks": None,
            "validation_split": 0.0,
            "validation_data": None,
            "shuffle": True,
            "class_weight": None,
            "sample_weight": None,
            "initial_epoch": 0,
            "steps_per_epoch": None,
            "validation_steps": None,
            "validation_batch_size": None,
            "validation_freq": 1,
            "max_queue_size": 10,
            "workers": 1,
            "use_multiprocessing": False,
        }

    def from_json(self, path):
        file = open(path)
        self.fit_params = json.load(file)

    def get_params(self):
        params = {}
        for key, value in self.fit_params.items():
            if not (value is None):
                params[key] = value
        return params

    def to_json(self, path):
        with open(path, "w") as write_file:
            json.dump(self.fit_params, write_file, indent=4)

    def from_CSV(self, path):
        pass

    def to_CSV(self, path):
        df = pd.DataFrame.from_dict(self.fit_params, orient="index")
        df.to_csv(path)

    def from_excel(self, path):
        pass

    def update_params(self, key=None, value=None):
        if key:
            self.fit_params[key] = value

    def update(self, new_dict, full=False):
        if full:
            self.fit_params = new_dict
        else:
            for key, value in new_dict.items():
                self.fit_params[key] = value

    def __str__(self):
        pass


# ******************************************************************************************************************** #
# Main
if __name__ == "__main__":
    LP = tf_FP()
    LP.to_JSON("Fit_params_example.json")
    LP.to_CSV("Fit_params_example.csv")

    LP1 = tf_FP()
    LP1.from_JSON("Fit_params_example.json")
    LP.get_params()
    print(LP1.fit_params)
