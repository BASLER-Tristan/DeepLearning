# ******************************************************************************************************************** #
# ********************************************* #   CENTRALESUPELEC   # ********************************************** #
# ******************************************************************************************************************** #
# Project : deepl
# File    : loading_method.py
# PATH    : TF_input_pipeline
# Author  : trisr
# Date    : 13/12/2022
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
import tensorflow as tf

# ******************************************************************************************************************** #
# Function definition
def open_files_hdf5(path):
    file = h5py.File(path, "r")
    return file


def loading_HDF5(params):
    data = []
    index = []
    if os.path.isdir(params["PATH"]):
        for path in os.listdir(params["PATH"]):
            if path.endswith(".hdf5"):
                file = open_files_hdf5(params["PATH"] + path)
                data.append(file)
                index.append(path[:-5])
    else:
        file = open_files_hdf5(params["PATH"])
        data.append(file)
        index = params["PATH"][:-5]
    return data, index


def loading_csv(params):
    data = []
    index = []
    if os.path.isdir(params["PATH"]):
        for path in os.listdir(params["PATH"]):
            if path.endswith(".csv"):
                file = pd.read_csv(params["PATH"] + path)[params["COL"]]
                data.append(file)
                index.append(path[:-4])
    else:
        file = pd.read_csv(params["PATH"])
        data = np.array(file[params["COL"]])
        index = list(file[params["INDEX"]])
    return data, index


def loading_npz(params):
    data = []
    index = []
    if os.path.isdir(params["PATH"]):
        for path in os.listdir(params["PATH"]):
            if path.endswith(".npz"):
                file = np.load(params["PATH"] + path)[params["COL"]]
                data.append(file)
                index.append(path.split(".")[0])
    else:
        file = np.load(params["PATH"])
        data = np.array(file[params["COL"]])
        index = list(file[params["INDEX"]])
    return data, index


def loading_TFRecord(params):
    tfrecord_dataset = tf.data.TFRecordDataset(params["PATH"])
    dataset = tfrecord_dataset.map(params["parse_function"])
    dataset = dataset.shuffle(int(params["SHUFFLE_BUFFER_SIZE"])).batch(int(params["BATCH_SIZE"]))
    return dataset, None


LOADING = {name[(len("loading_")) :]: value for name, value in globals().items() if name.startswith("loading_")}

# ******************************************************************************************************************** #
# Main
if __name__ == "__main__":
    xparams = {
        "XPATH_TRAIN": r"C:\Users\trisr\Desktop\deepl\train",
        "XKEY_TRAIN": None,
        "XPATH_TEST": None,
        "XKEY_TEST": None,
        "yPATH_TRAIN": r"C:\Users\trisr\Desktop\deepl\train",
        "yKEY_TRAIN": None,
        "yPATH_TEST": None,
        "yKEY_TEST": None,
    }
