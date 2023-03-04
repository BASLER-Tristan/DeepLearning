# ******************************************************************************************************************** #
# ********************************************* #   CENTRALESUPELEC   # ********************************************** #
# ******************************************************************************************************************** #
# Project : deepl
# File    : loading_item_method.py
# PATH    : TF_input_pipeline/utils
# Author  : trisr
# Date    : 06/01/2023
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
import os
import numpy as np
import pandas as pd
import h5py
import pandas as pd

# ******************************************************************************************************************** #
# Function definition


def open_files_hdf5(path):
    file = h5py.File(path, "r")
    return file


def loading_HDF5(path):
    file = h5py.File(path, "r")
    name = path.split("\\")[-1]
    name = name.split(".")[0]
    return file


def loading_npz(params):
    file = np.load(params["PATH"])
    data = np.array(file[params["COL"]])
    return data


def loading_csv(params):
    file = pd.read_csv(params["PATH"])
    data = np.array(file[params["COL"]])
    return data


# ******************************************************************************************************************** #
# Configuration
loading = {name[(len("loading_")) :]: value for name, value in globals().items() if name.startswith("loading_")}
