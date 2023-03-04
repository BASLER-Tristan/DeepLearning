import tensorflow as tf
import pandas as pd
import os
import numpy as np
import tqdm

paths = [
    "D:/dataset_inter_scaled/test/",
    "D:/dataset_inter_scaled/train/",
    "D:/dataset_inter_scaled/small/test/",
    "D:/dataset_inter_scaled/small/train/",
]
df = pd.read_csv("DATA/train_labels.csv")


def tf_creation(data_path):
    # Create a TFRecordWriter
    print(data_path)
    writer = tf.io.TFRecordWriter(data_path + "tfrecord.tfrecord")
    i = 0
    # Iterate over the list of arrays and labels
    for file in tqdm.tqdm(os.listdir(data_path)):
        if file.split(".")[-1] == "npz":
            array = np.load(path + file)["xtrain"]
            # Convert the array to a bytes-like object
            array_bytes = array.tobytes()
            label = np.load(path + file)["y"]
            # Create features containing the array and label
            feature = {
                "array": tf.train.Feature(bytes_list=tf.train.BytesList(value=[array_bytes])),
                "label": tf.train.Feature(float_list=tf.train.FloatList(value=[label])),
            }

            # Create a Example message
            example = tf.train.Example(features=tf.train.Features(feature=feature))

            # Serialize the example message and write it to the TFRecord file
            writer.write(example.SerializeToString())

    # Close the writer
    writer.close()


def parse_function(example_proto):
    features = {"array": tf.io.FixedLenFeature([], tf.string), "label": tf.io.FixedLenFeature([], tf.float32)}
    parsed_features = tf.io.parse_single_example(example_proto, features)
    array = tf.io.decode_raw(parsed_features["array"], np.float32)
    array = tf.reshape(array, [1024, 360, 4])
    label = tf.reshape(parsed_features["label"], [1])
    label = tf.cast(label, tf.float64)
    return array, label


if __name__ == "__main__":
    for path in paths:
        tf_creation(path)

    tfrecord_dataset = tf.data.TFRecordDataset("D:/dataset_inter_scaled/train/tfrecord.tfrecord")
    dataset = tfrecord_dataset.map(parse_function)
    print("loaded")

    for element in dataset.as_numpy_iterator():
        print(np.shape(element[1]))
        print(type(element[1]))
        print(np.max(element[0]))
        print(np.mean(element[0]))
        # print(np.shape(element[0]))
