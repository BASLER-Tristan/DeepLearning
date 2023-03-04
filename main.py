# ******************************************************************************************************************** #
# ********************************************* #   CENTRALESUPELEC   # ********************************************** #
# ******************************************************************************************************************** #
# Project : deepl
# File    : main.py
# PATH    :
# Author  : trisr
# Date    : 22/12/2022
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
import NN_tensorflow
from NN_tensorflow import TF_input_pipeline
import tensorflow as tf
import numpy as np

# ******************************************************************************************************************** #
# Function definition
def parse_function(example_proto):
    """
    TEST

    Parameters
    ----------
    example_proto

    Returns
    -------

    """
    features = {"array": tf.io.FixedLenFeature([], tf.string), "label": tf.io.FixedLenFeature([], tf.float32)}
    parsed_features = tf.io.parse_single_example(example_proto, features)
    array = tf.io.decode_raw(parsed_features["array"], np.float64)
    array = tf.reshape(array, [4843, 360, 4])
    label = tf.reshape(parsed_features["label"], [1])
    label = tf.cast(label, tf.float64)
    return array, label


def parse_function_inter(example_proto):
    """
    TEST

    Parameters
    ----------
    example_proto

    Returns
    -------

    """
    features = {"array": tf.io.FixedLenFeature([], tf.string), "label": tf.io.FixedLenFeature([], tf.float32)}
    parsed_features = tf.io.parse_single_example(example_proto, features)
    array = tf.io.decode_raw(parsed_features["array"], np.float32)
    array = tf.reshape(array, [1024, 360, 4])
    label = tf.reshape(parsed_features["label"], [1])
    label = tf.cast(label, tf.float64)
    return array, label


# ******************************************************************************************************************** #
# Configuration


# ******************************************************************************************************************** #
# Main
if __name__ == "__main__":
    model = NN_tensorflow.TF_Model(name="Test Model")
    model.from_json("Archi_small/")
    model.dataset.update(
        specfication="xparams", key="TRAIN", keyparams="parse_function", paramsvalue=parse_function,
    )
    model.dataset.update(
        specfication="xparams", key="TEST", keyparams="parse_function", paramsvalue=parse_function,
    )
    model.fit_params.update_params(
        key="class_weight", value={0: 5, 1: 1},
    )
    model.build(inputs_shape=[4843, 360, 4])
    model.compile()
    # print(model)
    print(model.model.summary())
    model.fit()
    print(model.model.predict(model.dataset.XDATA_test))
    # print(model.dataset.XDATA_test)
    # tfrecord_dataset = tf.data.TFRecordDataset("D:/tfrecord_small.tfrecord")
    # dataset = tfrecord_dataset.map(parse_function)
    # print("loaded")
    i = 0
    for element in model.dataset.XDATA_test.as_numpy_iterator():
        # i = i + 1
        # if i > 5:
        #     break
        print(model.model.predict((element[0])))
        print(element[1])
        print()
        # print(np.shape(element[0]))
    i = 0
    for element in model.dataset.XDATA_train.as_numpy_iterator():
        i = i + 1
        if i > 5:
            break
        print(model.model.predict((element[0])))
        print(element[1])
        print()
    # i = 0
    # for element in model.dataset.XDATA_test.as_numpy_iterator():
    #     i = i + 1
    #     if i > 1:
    #         break
    #     print(element)
    #     print(np.sum(element[0] == 0))
    #     print(np.shape(element[0]))
    #     print(np.shape(element[1]))
