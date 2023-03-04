# ******************************************************************************************************************** #
# ********************************************* #   CENTRALESUPELEC   # ********************************************** #
# ******************************************************************************************************************** #
# Project : deepl
# File    : preprocessing_item.py
# PATH    : TF_input_pipeline/utils
# Author  : trisr
# Date    : 06/01/2023
# Description :
"""




"""
import os.path

# Last commit ID   :
# Last commit date :
# ******************************************************************************************************************** #
# ******************************************************************************************************************** #
# ******************************************************************************************************************** #


# ******************************************************************************************************************** #
# Importations
import numpy as np
import tensorflow as tf
from scipy import interpolate

# ******************************************************************************************************************** #
# Function definition
def interpolation(list_data, open_data, root="", params=None):
    def convert_seq(seq):
        rep = seq - seq[0]
        return rep

    def calculate_size(xdata):
        key = list(xdata.keys())[0]
        sizes = []
        sizes.append(len(np.real(np.array(xdata[key]["H1"]["SFTs"]))))
        sizes.append(len(np.real(np.array(xdata[key]["L1"]["SFTs"]))))
        return [sizes]

    def mask_creation(size, mean_size):
        return [i * int(mean_size / size) for i in range(size)]

    def interpolate_timeseries(y, index, mask, type):
        f = interpolate.interp1d(index, y, kind=type)
        return f(mask)

    def x_procress(x, mask, params):
        key = list(x.keys())[0]
        x0 = np.real(np.array(x[key]["H1"]["SFTs"]))
        x1 = np.imag(np.array(x[key]["H1"]["SFTs"]))
        seq01 = convert_seq((x[key]["H1"]["timestamps_GPS"]))
        x2 = np.real(np.array(x[key]["L1"]["SFTs"]))
        x3 = np.imag(np.array(x[key]["L1"]["SFTs"]))
        seq23 = convert_seq((x[key]["L1"]["timestamps_GPS"]))

        x0 = interpolate_timeseries(x0, seq01, mask, params["TYPE"])
        x0 = np.transpose(x0)
        x1 = interpolate_timeseries(x1, seq01, mask, params["TYPE"])
        x1 = np.transpose(x1)

        x2 = interpolate_timeseries(x2, seq23, mask, params["TYPE"])
        x2 = np.transpose(x2)
        x3 = interpolate_timeseries(x3, seq23, mask, params["TYPE"])
        x3 = np.transpose(x3)
        return (x0, x1, x2, x3, key)

    sizes = []
    for path in list_data:
        xdata = open_data(root + path)
        sizes = sizes + calculate_size(xdata)
    mean_size = np.mean(sizes)
    mask = mask_creation(params["SIZE"], mean_size)

    def preprocessing(data, file):
        file = file.split(".")[0]
        file = file + ".npz"
        x_map = np.zeros((params["SIZE"], 360, 4), dtype=np.float32)
        (x0, x1, x2, x3, key) = x_procress(data, mask, params)
        x_map[:, :, 0] = x0
        x_map[:, :, 1] = x1
        x_map[:, :, 2] = x2
        x_map[:, :, 3] = x3
        return x_map, file

    return preprocessing


def padding_process(list_data, open_data, root="", params=None):
    def padding_zero_2D(x_array, length):
        n = len(x_array)
        p = len(x_array[0])
        rep = np.zeros((n, length))
        for i in range(n):
            rep[i][:p] = x_array[i][:p]
        return rep

    def calculate_max_length(xdata, result):

        key = list(xdata.keys())[0]
        size1 = np.array(xdata[key]["H1"]["SFTs"]).shape[1]
        size2 = np.array(xdata[key]["L1"]["SFTs"]).shape[1]
        return max([size1, size2, result])

    def x_procress(x, length):
        key = list(x.keys())[0]
        x0 = np.real(np.array(x[key]["H1"]["SFTs"]))
        x1 = np.imag(np.array(x[key]["H1"]["SFTs"]))
        x2 = np.real(np.array(x[key]["L1"]["SFTs"]))
        x3 = np.imag(np.array(x[key]["L1"]["SFTs"]))
        x0 = padding_zero_2D(x0, length)
        x1 = padding_zero_2D(x1, length)
        x2 = padding_zero_2D(x2, length)
        x3 = padding_zero_2D(x3, length)
        x0 = np.transpose(x0)
        x1 = np.transpose(x1)
        x2 = np.transpose(x2)
        x3 = np.transpose(x3)

        return (x0, x1, x2, x3, key)

    length = 0
    for path in list_data:
        xdata = open_data(root + path)
        length = calculate_max_length(xdata, length)

    def preprocessing(x, file):
        file = file.split(".")[0]
        file = file + ".npz"
        x_map = np.zeros((length, 360, 4))
        (x0, x1, x2, x3, key) = x_procress(x, length)
        x_map[:, :, 0] = x0
        x_map[:, :, 1] = x1
        x_map[:, :, 2] = x2
        x_map[:, :, 3] = x3
        return x_map, file

    return preprocessing


def rescaling_min_max(list_data, open_data, root="", params=None):
    def max_calculation(x, prev_max):
        key = list(x.keys())[0]
        x0 = np.real(np.array(x[key]["H1"]["SFTs"]))
        x1 = np.imag(np.array(x[key]["H1"]["SFTs"]))
        x2 = np.real(np.array(x[key]["L1"]["SFTs"]))
        x3 = np.imag(np.array(x[key]["L1"]["SFTs"]))
        prev_max[0] = max(np.max(x0), prev_min[0])
        prev_max[1] = max(np.max(x1), prev_min[1])
        prev_max[2] = max(np.max(x2), prev_min[2])
        prev_max[3] = max(np.max(x3), prev_min[3])
        return prev_max

    def min_calculation(x, prev_min):
        key = list(x.keys())[0]
        x0 = np.real(np.array(x[key]["H1"]["SFTs"]))
        x1 = np.imag(np.array(x[key]["H1"]["SFTs"]))
        x2 = np.real(np.array(x[key]["L1"]["SFTs"]))
        x3 = np.imag(np.array(x[key]["L1"]["SFTs"]))
        prev_min[0] = min(np.min(x0), prev_min[0])
        prev_min[1] = min(np.min(x1), prev_min[1])
        prev_min[2] = min(np.min(x2), prev_min[2])
        prev_min[3] = min(np.min(x3), prev_min[3])
        return prev_min

    prev_min = [np.inf] * 4
    prev_max = [-np.inf] * 4
    for data in list_data:
        xdata = open_data(root + data)
        prev_min = min_calculation(xdata, prev_min)
        prev_max = max_calculation(xdata, prev_max)

    def preprocessing(x, file):
        rep = np.copy(x)
        for i in range(np.shape(x)[2]):
            rep[:, :, i] = (rep[:, :, i] - prev_min[i]) / (prev_max[i] - prev_min[i])
        return rep, file

    return preprocessing


def list2tfrecord(list_data_path, open_data, root="", yvalue=None, params=None):
    writer = tf.io.TFRecordWriter(params["WRITER"])
    # yvalue = {x: y for x, y in yvalue}

    def preprocessing(x, file):
        y = 1
        # Convert the np.array to bytes
        x_bytes = x.tostring()
        # Create a tf.train.Example object
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "x": tf.train.Feature(bytes_list=tf.train.BytesList(value=[x_bytes])),
                    "y": tf.train.Feature(int64_list=tf.train.Int64List(value=[y])),
                }
            )
        )
        # Write the example to the TFRecord file
        writer.write(example.SerializeToString())
        return 1, 1

    return preprocessing


# ******************************************************************************************************************** #
# Configuration
preprocess_function = {name[(len("")) :]: value for name, value in globals().items() if name.startswith("")}
