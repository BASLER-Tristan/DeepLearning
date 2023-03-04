# ******************************************************************************************************************** #
# ********************************************* #   CENTRALESUPELEC   # ********************************************** #
# ******************************************************************************************************************** #
# Project : deepl
# File    : PipeClass.py
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
import tensorflow as tf
import json
from .utils import *
import os
import numpy as np
import git
import tqdm
import copy

# ******************************************************************************************************************** #
# Class definition
class TF_Dataset:
    """

    """

    def __init__(
        self, root=None, xparams=None, xpreprocess=None, yparams=None, ypreprocess=None,
    ):
        """
        Initialization of the tf_dataset class

        Parameters
        ----------
        root        :
        xparams     :
        xpreprocess :
        yparams     :
        ypreprocess :
        """
        self.root = root
        self.xparams = xparams
        self.xpreprocess = xpreprocess
        self.yparams = yparams
        self.ypreprocess = ypreprocess
        self.tf_data_test = None

    def update(
        self,
        xparams=None,
        root=None,
        xpreprocess=None,
        yparams=None,
        ypreprocess=None,
        specfication=None,
        key=None,
        keyparams=None,
        paramsvalue=None,
    ):
        """
        Update the parameters gived in inputs.

        Parameters
        ----------
        xparams
        root
        xpreprocess
        yparams
        ypreprocess

        Returns
        -------

        """
        if xparams:
            self.xparams = xparams
        if root:
            self.root = root
        if xpreprocess:
            self.xpreprocess = xpreprocess
        if yparams:
            self.yparams = yparams
        if xpreprocess:
            self.ypreprocess = ypreprocess

        if specfication == "xparams":
            self.xparams[key][keyparams] = paramsvalue
        if specfication == "xpreprocess":
            self.xpreprocess[key][keyparams] = paramsvalue
        if specfication == "yparams":
            self.yparams[key][keyparams] = paramsvalue
        if specfication == "ypreprocess":
            self.ypreprocess[key][keyparams] = paramsvalue

    def build(self):
        """
        Load the dataset from the path in xparams and yparams
        Apply the preprocessing function

        if mode == "full", load every data and apply preprocessing to every of them
        if mode == "item", load item by item, apply preprocessing by item and save the result in
            the folder self.xparams["TRAIN3/"TEST"]["PATH"] with the name corresponding of the opened file.

        Create
        -------
            self.XDATA_train contains the train input data
            self.Xindex_train contains the name of the train input data
                self.XDATA_train = [1,2,3]
                self.Xindex_train = "input_1034"

            self.XDATA_test contains test data (if existing)
            self.Xindex_test contains test data index (if existing)

            Same for self.yDATA_train, self.yindex_train, self.yDATA_test and self.yindex_test.
        """
        if self.xparams["TRAIN"]["METHOD"] == "full":
            print("XTrain Loading")
            self.XDATA_train, self.Xindex_train = LOADING[self.xparams["TRAIN"]["MODE"]](params=self.xparams["TRAIN"])

            if "TEST" in self.xparams.keys():
                print("Xtest loading")
                self.XDATA_test, self.Xindex_test = LOADING[self.xparams["TEST"]["MODE"]](params=self.xparams["TEST"])
            else:
                self.XDATA_test, self.Xindex_test = None, None
            self.xprocess(method=self.xparams["TRAIN"]["METHOD"], path=None)

        elif self.xparams["TRAIN"]["METHOD"] == "item":
            assert self.xparams["TRAIN"]["OUTPUT"], "Need to a folder to save the preprocessed data, but path is None"
            print("XTrain Loading")
            self.XDATA_train = os.listdir(self.xparams["TRAIN"]["PATH"])
            self.xprocess(method=self.xparams["TRAIN"]["METHOD"], path=self.xparams["TRAIN"]["OUTPUT"])

            if "TEST" in self.xparams.keys():
                print("Xtest Loading")
                assert self.xparams["TEST"][
                    "OUTPUT"
                ], "Need to a folder to save the preprocessed data, but path is None"
                self.XDATA_test = os.listdir(self.xparams["TEST"]["PATH"])
                self.xprocess(method=self.xparams["TEST"]["METHOD"], path=self.xparams["TEST"]["OUTPUT"])
        if self.yparams:
            if "TRAIN" in self.yparams.keys():
                if self.yparams.get(["TRAIN"])["METHOD"] == "full":
                    print("yTrain Loading")
                    self.yDATA_train, self.yindex_train = LOADING[self.yparams["TRAIN"]["MODE"]](
                        params=self.yparams["TRAIN"]
                    )
                    self.yprocess(method=self.yparams["TRAIN"]["METHOD"], path=None)

                elif self.yparams["TRAIN"]["METHOD"] == "item":
                    assert self.yparams["TRAIN"][
                        "OUTPUT"
                    ], "Need to a folder to save the preprocessed data, but path is None"
                    print("yTrain Loading")
                    for file in tqdm.tqdm(os.listdir(self.yparams["TRAIN"]["PATH"])):
                        self.ytrain, self.yname = loading[self.yparams["TRAIN"]["MODE"]](
                            self.yparams["TRAIN"]["PATH"] + file
                        )
                    self.yprocess(method=self.yparams["TRAIN"]["METHOD"], path=self.yparams["TRAIN"]["OUTPUT"])

            if "TEST" in self.yparams.keys():
                print("yTest Loading")
                if self.yparams["TEST"]["METHOD"] == "full":
                    self.yDATA_test, self.yindex_test = LOADING[self.yparams["TEST"]["MODE"]](
                        params=self.yparams["TEST"]
                    )
                elif self.yparams["TEST"]["METHOD"] == "item":
                    print("yTest Loading")
                    for file in tqdm.tqdm(os.listdir(self.yparams["TEST"]["PATH"])):
                        self.ytrain, self.yname = loading[self.yparams["TEST"]["MODE"]](
                            self.yparams["TEST"]["PATH"] + file
                        )
                        assert self.yparams["TEST"][
                            "OUTPUT"
                        ], "Need to a folder to save the preprocessed data, but path is None"
                        self.yprocess(method=self.yparams["TEST"]["METHOD"], path=self.yparams["TEST"]["OUTPUT"])

                else:
                    self.yDATA_test, self.yindex_test = None, None

    def filter(self):
        if isinstance(self.XDATA_train, tf.data.Dataset):
            pass
        else:
            print("XFilter")
            self.index_train = [x for x in self.Xindex_train if self.yindex_train]
            clean_XDATA_train = []
            clean_yDATA_train = []
            for i in range(len(self.XDATA_train)):
                if self.Xindex_train[i] in self.index_train:
                    clean_XDATA_train.append([self.Xindex_train[i], self.XDATA_train[i]])

            for i in range(len(self.yDATA_train)):
                if self.yindex_train[i] in self.index_train:
                    clean_yDATA_train.append([self.yindex_train[i], self.yDATA_train[i]])

            del self.XDATA_train, self.Xindex_train
            del self.yDATA_train, self.yindex_train

            clean_XDATA_train = [x for i, x in sorted(clean_XDATA_train)]
            clean_yDATA_train = [x for i, x in sorted(clean_yDATA_train)]
            clean_XDATA_train = np.array(clean_XDATA_train[0:10], dtype=np.float32)
            clean_yDATA_train = np.array(clean_yDATA_train[0:10], dtype=np.float32)
            self.XDATA_train = clean_XDATA_train
            self.yDATA_train = clean_yDATA_train
            self.shape = np.shape(self.XDATA_train[0])

            if self.Xindex_test:
                print("Test Filter")
                self.index_test = [x for x in self.Xindex_test if self.yindex_test]
                clean_XDATA_test = []
                clean_yDATA_test = []
                for i in range(len(self.XDATA_test)):
                    if self.Xindex_test[i] in self.index_test:
                        clean_XDATA_test.append(self.XDATA_test[i])
                        clean_yDATA_test.append(self.yDATA_test[i])

                del self.XDATA_test, self.Xindex_test
                del self.yDATA_test, self.yindex_test
                clean_XDATA_test = [x for x, i in sorted(zip(self.index_test, clean_XDATA_test))]
                clean_yDATA_test = [x for x, i in sorted(zip(self.index_test, clean_yDATA_test))]
                clean_XDATA_test = np.array(clean_XDATA_test, dtype=np.float16)
                clean_yDATA_test = np.array(clean_yDATA_test, dtype=np.float16)
                self.XDATA_test = clean_XDATA_test
                self.yDATA_test = clean_yDATA_test

    def tf_transform(self, shuffle=None, batch_size=None):
        if isinstance(self.XDATA_train, tf.data.Dataset):
            self.tf_data = self.XDATA_train
        else:
            print("Tf Transform")
            self.tf_data = tf.data.Dataset.from_tensor_slices((self.XDATA_train, self.yDATA_train))
            self.tf_data = self.tf_data.shuffle(shuffle).batch(batch_size)
        if self.XDATA_test or self.yDATA_test:
            if isinstance(self.XDATA_test, tf.data.Dataset):
                self.tf_data_test = self.XDATA_test
            else:
                print("Tf Transform")
                self.tf_data_test = tf.data.Dataset.from_tensor_slices((self.XDATA_test, self.yDATA_test))
                self.tf_data_test = self.tf_data_test.shuffle(shuffle).batch(batch_size)

    def xprocess(self, method, path):
        if method == "full":
            if self.xpreprocess:
                print("X Preprocessing")
                for key in tqdm.tqdm(list(self.xpreprocess.keys())):
                    preprocessing = PREPROCESS_FUNCTION[key](train_data=self.XDATA_train, params=self.xpreprocess[key])
                    self.XDATA_train = preprocessing(self.XDATA_train)
                    if self.XDATA_test:
                        self.XDATA_test = preprocessing(self.XDATA_test)

        elif method == "item":
            if self.xpreprocess:
                preprocessing = []
                for key in self.xpreprocess.keys():
                    preprocessing.append(
                        preprocess_function[key](
                            list_data=self.XDATA_train,
                            open_data=loading[self.xparams["TRAIN"]["MODE"]],
                            root=self.xparams["TRAIN"]["PATH"],
                            params=self.xpreprocess[key],
                        )
                    )
                os.makedirs(path, exist_ok=True)
                for file in tqdm.tqdm(self.XDATA_train):
                    xtrain = loading[self.xparams["TRAIN"]["MODE"]](self.xparams["TRAIN"]["PATH"] + file)
                    for func in preprocessing:
                        xtrain, file = func(xtrain, file)
                    np.savez(path + file, xtrain=xtrain, xindex=file.split(".")[0])

    def yprocess(self, method, path):
        if method == "full":
            if self.ypreprocess:
                print("yPreprocessing")
                for key in tqdm.tqdm(list(self.ypreprocess.keys())):
                    preprocessing = PREPROCESS_FUNCTION[key](train_data=self.yDATA_train, params=self.ypreprocess[key])
                    self.yDATA_train = preprocessing(self.yDATA_train)
                    if self.yDATA_test:
                        self.yDATA_test = preprocessing(self.yDATA_test)

        elif method == "item":
            if self.ypreprocess:
                preprocessing = []
                for key in self.ypreprocess.keys():
                    preprocessing.append(
                        preprocess_function[key](
                            list_data=self.yDATA_train,
                            open_data=loading[self.yparams["TRAIN"]["MODE"]],
                            root=self.yparams["TRAIN"]["PATH"],
                            yvalue=[self.yDATA_train, self.yindex_train],
                            params=self.ypreprocess[key],
                        )
                    )

                for file in tqdm.tqdm(self.yDATA_train):
                    ytrain = loading[self.yparams["TRAIN"]["MODE"]](self.yparams["TRAIN"]["PATH"] + file)
                    for func in preprocessing:
                        ytrain, file = func(ytrain, file)

                    np.savez(path + file, ytrain=ytrain, yindex=file.split(".")[0])

    def to_json(self, path):
        params = {}
        if isinstance(self.xparams, dict):
            params["XPARAMS"] = copy.deepcopy(self.xparams)
        if isinstance(self.xpreprocess, dict):
            params["XPREPROCESS"] = copy.deepcopy(self.xpreprocess)
        if isinstance(self.yparams, dict):
            params["YPARAMS"] = copy.deepcopy(self.yparams)
        if isinstance(self.ypreprocess, dict):
            params["YPREPROCESS"] = copy.deepcopy(self.ypreprocess)
        for key in params:
            try:
                for key2 in params[key]:
                    for key3 in params[key][key2]:
                        if callable(params[key][key2][key3]):
                            params[key][key2][key3] = params[key][key2][key3].__doc__
            except:
                pass
        with open(path, "w") as write_file:
            json.dump(params, write_file, indent=4)

    def from_json(self, path):
        file = open(path)
        params = json.load(file)
        self.xparams = params.get("XPARAMS")
        self.xpreprocess = params.get("XPREPROCESS")
        self.yparams = params.get("YPARAMS")
        self.ypreprocess = params.get("YPREPROCESS")

    def save_database(self, path, key=None):
        if "xtrain" in key:
            print("Saving Xtrain")
            os.makedirs(path, exist_ok=True)
            np.savez(path + "XTRAIN.npz", x=self.XDATA_train, index=self.Xindex_train)

        if "xtest" in key:
            print("Saving Xtest")
            os.makedirs(path, exist_ok=True)
            np.savez(path + "Xtest.npz", x=self.XDATA_test, index=self.Xindex_test)

        if "ytrain" in key:
            print("Saving ytrain")
            os.makedirs(path, exist_ok=True)
            np.savez(path + "yTRAIN.npz", x=self.yDATA_train, index=self.yindex_train)

        if "ytest" in key:
            print("Saving ytest")
            os.makedirs(path, exist_ok=True)
            np.savez(path + "ytest.npz", x=self.yDATA_test, index=self.yindex_test)


# ******************************************************************************************************************** #
# Configuration
def get_git_root(path=os.getcwd()):
    git_repo = git.Repo(path, search_parent_directories=True)
    git_root = git_repo.git.rev_parse("--show-toplevel")
    return git_root


git_root = get_git_root()

# ******************************************************************************************************************** #
# Main
if __name__ == "__main__":

    xparams_train = {
        "PATH": git_root + "/dataset_padded/",
        "MODE": "npz",
        "COL": "x",
        "METHOD": "full",
        "OUTPUT": git_root + "/dataset/",
    }
    xparams_test = {
        "PATH": git_root + "/train/",
        "MODE": "HDF5",
        "METHOD": "full",
        "OUTPUT": git_root + "/dataset_padded/",
    }
    yparams_train = {
        "PATH": git_root + "/train/train_labels.csv",
        "MODE": "csv",
        "COL": "target",
        "INDEX": "id",
        "METHOD": "full",
    }

    yparams_test = {
        "PATH": git_root + "/train/train_labels.csv",
        "MODE": "csv",
        "COL": "target",
        "INDEX": "id",
        "METHOD": "full",
    }
    xparams = {}
    yparams = {}
    xparams["TRAIN"] = xparams_train
    xparams["TEST"] = xparams_test
    yparams["TRAIN"] = yparams_train
    yparams["TEST"] = yparams_test
    xpreprocess = {"downsampling": {"SIZE": 1024}}
    ypreprocess = {"function_name": {"SIZE": 1024}}

    dataset = TF_Dataset(xparams=xparams, yparams=yparams, xpreprocess=xpreprocess, ypreprocess=None)
    # dataset.build()
    # dataset.filter()
    dataset.to_json("Example_params.json")

    dataset1 = TF_Dataset()
    dataset1.from_json("Example_params.json")
