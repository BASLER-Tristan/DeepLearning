# ******************************************************************************************************************** #
# ********************************************* #   CENTRALESUPELEC   # ********************************************** #
# ******************************************************************************************************************** #
# Project : deepl
# File    : preprocessing.py
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
import numpy as np
import tensorflow as tf
import datetime
from scipy import interpolate
import os
import tqdm

# ******************************************************************************************************************** #
# Function definition
def HDF5_GW_detection(train_data, params):
    def HDF5_GW_detection_process(data):
        rep_data = []
        for input in data:
            input = input[list(input.keys())[0]]
            img = np.empty((2, 360, 128), dtype=np.float32)

            for ch, s in enumerate(["H1", "L1"]):
                a = input[s]["SFTs"][:, :4096] * 1e22  # Fourier coefficient complex64
                if a.shape != (360, 4096):
                    append = False
                    break
                append = True
                p = a.real ** 2 + a.imag ** 2  # power
                p /= np.mean(p)  # normalize
                p = np.mean(p.reshape(360, 128, 32), axis=2)  # compress 4096 -> 128

                img[ch] = p
            if append:
                rep_data.append(img)
        rep_data = np.array(rep_data)
        return rep_data

    return HDF5_GW_detection_process


def HDF5_GW_detection_label(train_data, params):
    def HDF5_GW_label(data):
        return list(data[0]["target"])

    return HDF5_GW_label


def type_transform(train_data, params):
    def convert(data):
        data = tf.cast(data, params)
        return data

    return convert


def padding_process(train_data, params):
    def padding_zero_2D(x_array, length):
        n = len(x_array)
        rep = np.zeros((n, length))
        for i in range(n):
            p = len(x_array[i])
            rep[0][:p] = x_array[i]
        return rep

    def calculate_max_length(data):
        size1 = []
        size2 = []
        for x in data:
            key = list(x.keys())[0]
            size1.append(np.array(x[key]["H1"]["SFTs"]).shape)
            size2.append(np.array(x[key]["L1"]["SFTs"]).shape)
        return max(np.max(size1), np.max(size2))

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

    def preprocessing(data):
        length = calculate_max_length(data)
        rep_dataset = []
        for i in range(len(data)):
            x = data[i]
            x_map = np.zeros((length, 360, 4))
            (x0, x1, x2, x3, key) = x_procress(x, length)
            x_map[:, :, 0] = x0
            x_map[:, :, 1] = x1
            x_map[:, :, 2] = x2
            x_map[:, :, 3] = x3
            rep_dataset.append(x_map)
        return rep_dataset

    return preprocessing


def downsampling(train_data, params):
    def convert_timestamp(seq):
        rep = []
        for time in np.array(seq):
            rep.append(datetime.datetime.fromtimestamp(time / 1e3))
        rep = np.array(rep)
        return rep

    def calculate_step(train_data, size):
        reptime1 = []
        reptime2 = []
        for x in train_data:
            key = list(x.keys())[0]
            rep = []
            for time in np.array(x[key]["H1"]["timestamps_GPS"]):
                rep.append(datetime.datetime.fromtimestamp(time / 1e3))
            rep = np.array(rep)
            reptime1.append(rep)

            rep = []
            for time in np.array(x[key]["L1"]["timestamps_GPS"]):
                rep.append(datetime.datetime.fromtimestamp(time / 1e3))
            rep = np.array(rep)
            reptime2.append(rep)

        freq1 = []
        for x in reptime1:
            freq1.append((x[len(x) - 1] - x[0]) / size)

        freq2 = []
        for x in reptime2:
            freq2.append((x[len(x) - 1] - x[0]) / size)
        step1 = np.mean(freq1)
        step2 = np.mean(freq2)
        step = np.mean([step1, step2])
        return step

    def mask_creation(size, step):
        return [i * step for i in range(size)]

    def downsample_index(timestamp, mask):
        timestamp = timestamp - timestamp[0]
        index = []
        for value in mask:
            diff = abs(timestamp - value)
            ind = np.argmin(diff)
            index.append(ind)
        return index

    def extract_downsampling(x, downsample_index):
        rep = []
        downsample_index = np.sort(downsample_index)
        for value in x:
            row = []
            for ind in downsample_index:
                row.append(value[ind])
            rep.append(row)
        rep = np.array(rep)
        return rep

    def x_procress(x, mask):
        key = list(x.keys())[0]
        x0 = np.real(np.array(x[key]["H1"]["SFTs"]))
        x1 = np.imag(np.array(x[key]["H1"]["SFTs"]))
        x2 = np.real(np.array(x[key]["L1"]["SFTs"]))
        x3 = np.imag(np.array(x[key]["L1"]["SFTs"]))

        time0_1 = convert_timestamp(x[key]["H1"]["timestamps_GPS"])
        index0_1 = downsample_index(time0_1, mask)
        time2_3 = convert_timestamp(x[key]["L1"]["timestamps_GPS"])
        index2_3 = downsample_index(time2_3, mask)
        x0 = extract_downsampling(x0, index0_1)
        x0 = np.transpose(x0)
        x1 = extract_downsampling(x1, index0_1)
        x1 = np.transpose(x1)
        x2 = extract_downsampling(x2, index2_3)
        x2 = np.transpose(x2)
        x3 = extract_downsampling(x3, index2_3)
        x3 = np.transpose(x3)
        return (x0, x1, x2, x3, key)

    def preprocessing(data):
        step = calculate_step(train_data, params["SIZE"])
        mask = mask_creation(params["SIZE"], step)
        rep_dataset = []
        # os.makedirs(params["PATH"], exist_ok=True)
        for i in range(len(data)):
            x = data[i]
            x_map = np.zeros((params["SIZE"], 360, 4), dtype=np.float32)
            (x0, x1, x2, x3, key) = x_procress(x, mask)
            x_map[:, :, 0] = x0
            x_map[:, :, 1] = x1
            x_map[:, :, 2] = x2
            x_map[:, :, 3] = x3
            rep_dataset.append(x_map)
        return rep_dataset

    return preprocessing


def agglomeration(train_data, params):
    def convert_timestamp(seq):
        rep = []
        for time in np.array(seq):
            rep.append(datetime.datetime.fromtimestamp(time / 1e3))
        rep = np.array(rep)
        return rep

    def calculate_step(train_data, size):
        reptime1 = []
        reptime2 = []
        for x in train_data:
            key = list(x.keys())[0]
            rep = []
            for time in np.array(x[key]["H1"]["timestamps_GPS"]):
                rep.append(datetime.datetime.fromtimestamp(time / 1e3))
            rep = np.array(rep)
            reptime1.append(rep)

            rep = []
            for time in np.array(x[key]["L1"]["timestamps_GPS"]):
                rep.append(datetime.datetime.fromtimestamp(time / 1e3))
            rep = np.array(rep)
            reptime2.append(rep)

        freq1 = []
        for x in reptime1:
            freq1.append((x[len(x) - 1] - x[0]) / size)

        freq2 = []
        for x in reptime2:
            freq2.append((x[len(x) - 1] - x[0]) / size)
        step1 = np.mean(freq1)
        step2 = np.mean(freq2)
        step = np.mean([step1, step2])
        return step

    def mask_creation(size, step):
        return [i * step for i in range(size)]

    def downsample_index(timestamp, mask):
        timestamp = timestamp - timestamp[0]
        index = []
        for value in mask:
            diff = abs(timestamp - value)
            ind = np.argmin(diff)
            index.append(ind)
        return index

    def compact_timeseries(x, downsample_index, func):
        rep = []
        downsample_index = np.sort(downsample_index)
        for value in x:
            row = []
            for i in range(len(downsample_index) - 1):
                row.append(func(value[downsample_index[i] : (downsample_index[i + 1])]))
            row.append(func(value[downsample_index[-1] :]))
            rep.append(np.array(row))
        rep = np.array(rep)
        return rep

    def x_procress(x, mask, params):
        key = list(x.keys())[0]
        x0 = np.real(np.array(x[key]["H1"]["SFTs"]))
        x1 = np.imag(np.array(x[key]["H1"]["SFTs"]))
        x2 = np.real(np.array(x[key]["L1"]["SFTs"]))
        x3 = np.imag(np.array(x[key]["L1"]["SFTs"]))

        time0_1 = convert_timestamp(x[key]["H1"]["timestamps_GPS"])
        index0_1 = downsample_index(time0_1, mask)
        time2_3 = convert_timestamp(x[key]["L1"]["timestamps_GPS"])
        index2_3 = downsample_index(time2_3, mask)
        x0 = compact_timeseries(x0, index0_1, params["FUNC"])
        x0 = np.transpose(x0)
        x1 = compact_timeseries(x1, index0_1, params["FUNC"])
        x1 = np.transpose(x1)
        x2 = compact_timeseries(x2, index2_3, params["FUNC"])
        x2 = np.transpose(x2)
        x3 = compact_timeseries(x3, index2_3, params["FUNC"])
        x3 = np.transpose(x3)
        return (x0, x1, x2, x3, key)

    def preprocessing(data):
        step = calculate_step(train_data, params["SIZE"])
        mask = mask_creation(params["SIZE"], step)
        rep_dataset = []
        # os.makedirs(params["PATH"], exist_ok=True)
        for i in range(len(data)):
            x = data[i]
            x_map = np.zeros((params["SIZE"], 360, 4), dtype=np.float32)
            (x0, x1, x2, x3, key) = x_procress(x, mask, params)
            x_map[:, :, 0] = x0
            x_map[:, :, 1] = x1
            x_map[:, :, 2] = x2
            x_map[:, :, 3] = x3
            rep_dataset.append(x_map)
        return rep_dataset

    return preprocessing


def interpolation(train_data, params):
    def convert_seq(seq):
        rep = seq - seq[0]
        return rep

    def calculate_mean_size(train_data):
        sizes = []
        for x in train_data:
            key = list(x.keys())[0]
            sizes.append(len(np.real(np.array(x[key]["H1"]["SFTs"]))))
            sizes.append(len(np.real(np.array(x[key]["L1"]["SFTs"]))))
        mean_size = np.mean(sizes)
        return mean_size

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

    def preprocessing(data):
        mean_size = calculate_mean_size(train_data)
        mask = mask_creation(params["SIZE"], mean_size)
        rep_dataset = []
        # os.makedirs(params["PATH"], exist_ok=True)
        for i in tqdm.tqdm([j for j in range(len(data))]):
            x = data[i]
            x_map = np.zeros((params["SIZE"], 360, 4), dtype=np.float32)
            (x0, x1, x2, x3, key) = x_procress(x, mask, params)
            x_map[:, :, 0] = x0
            x_map[:, :, 1] = x1
            x_map[:, :, 2] = x2
            x_map[:, :, 3] = x3
            # print(np.shape(x_map))
            rep_dataset.append(x_map)
        return rep_dataset

    return preprocessing


# ******************************************************************************************************************** #
# Configuration


# ******************************************************************************************************************** #
# Main
PREPROCESS_FUNCTION = {name[(len("")) :]: value for name, value in globals().items() if name.startswith("")}
