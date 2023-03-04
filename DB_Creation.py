#!/usr/bin/env python
# coding: utf-8

import os
from NN_tensorflow import TF_input_pipeline

import git


def get_git_root(path=os.getcwd()):
    git_repo = git.Repo(path, search_parent_directories=True)
    git_root = git_repo.git.rev_parse("--show-toplevel")
    return git_root


git_root = get_git_root()

dataset = TF_input_pipeline.TF_Dataset()
dataset.from_json("DCGW_dataset_params.json")
dataset.build()
# dataset.save_database(path="D:/dataset_padded/", key=["xtrain"])
print("Finish")
