# ******************************************************************************************************************** #
# ********************************************* #   CENTRALESUPELEC   # ********************************************** #
# ******************************************************************************************************************** #
# Project : deepl
# File    : data_splitter.py
# PATH    :
# Author  : trisr
# Date    : 30/01/2023
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
import numpy as np
import pandas as pd
import random
import os
import tqdm

# ******************************************************************************************************************** #
# Configuration
size_test = 100
proportion_test = None

data_path = "D:/dataset_inter_scaled/"
label_path = "C:/Users/trisr/Desktop/deepl/DATA/train_labels.csv"

# ******************************************************************************************************************** #
# Main
if __name__ == "__main__":
    df = pd.read_csv(label_path)
    xtest0 = list(df[df["target"] == 0]["id"])
    xtest1 = list(df[df["target"] == 1]["id"])
    random.shuffle(xtest0)
    xtest0 = xtest0[:50]
    random.shuffle(xtest1)
    xtest1 = xtest1[:50]
    os.makedirs(data_path + "/test/", exist_ok=True)
    os.makedirs(data_path + "/train/", exist_ok=True)
    for file in tqdm.tqdm(os.listdir(data_path)):
        if file.split(".")[-1] == "npz":
            array = np.load(data_path + file)["xtrain"]
            name = file.split(".")[0]
            label = df[df["id"] == name]["target"].iloc[0]
            if name in xtest1 or name in xtest0:
                np.savez(data_path + "/test/" + file, xtrain=array, index=name, y=label)
            else:
                np.savez(data_path + "/train/" + file, xtrain=array, index=name, y=label)

    random.shuffle(xtest0)
    sxtrain0 = xtest0[10:20]
    sxtest0 = xtest0[:10]
    sxtrain1 = xtest1[10:20]
    sxtest1 = xtest1[:10]
    os.makedirs(data_path + "/small/test/", exist_ok=True)
    os.makedirs(data_path + "/small/train/", exist_ok=True)
    for file in tqdm.tqdm(os.listdir(data_path)):
        if file.split(".")[-1] == "npz":
            name = file.split(".")[0]
            if name in sxtest1 or name in sxtest0:
                array = np.load(data_path + file)["xtrain"]
                label = df[df["id"] == name]["target"].iloc[0]
                np.savez(data_path + "/small/test/" + file, xtrain=array, index=name, y=label)
            elif name in sxtrain1 or name in sxtrain0:
                array = np.load(data_path + file)["xtrain"]
                label = df[df["id"] == name]["target"].iloc[0]
                np.savez(data_path + "/small/train/" + file, xtrain=array, index=name, y=label)
