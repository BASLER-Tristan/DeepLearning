# ******************************************************************************************************************** #
# ********************************************* #   CENTRALESUPELEC   # ********************************************** #
# ******************************************************************************************************************** #
# Project : deepl
# File    : ModelClass.py
# PATH    : NN_tensorflow
# Author  : trisr
# Date    : 17/11/2022
# Description :
"""




"""
import os

# Last commit ID   :
# Last commit date :
# ******************************************************************************************************************** #
# ******************************************************************************************************************** #
# ******************************************************************************************************************** #


# ******************************************************************************************************************** #
# Importations
import tensorflow as tf
import json

if __name__ == "__main__":
    from utils import *
    from TF_input_pipeline import TF_Dataset
else:
    from .utils import *
    from .TF_input_pipeline import TF_Dataset


# ******************************************************************************************************************** #
# ModelClass definition
class TF_Model:
    """



    Attributes
    ----------
    name          :
        str, Name of the following model
    model         :
        tf.model, tensorflow model can be defined or fit after
    optimizer     :
        tf_optimizer, translates the parameter for the optimizer
    loss          :
        tf_loss, translates the parameter for the loss
    metrics       :
        tf_metrics, translates the parameter for the metrics
    layers        :
        tf_layer, translates the parameter for the layers
    dataset       :
        TF_Dataset, translates the parameter for the dataset
    fit_params    :
        tf_FP, translates the fitting parameters
    built         :
        bool, translates if the model is built
    compiled      :
        bool, translates if the model is compiled
    dataset_build :
        bool, translates if the dataset is built.
        To be fit a model need to be build and compiled and the dataset need to be built.

    """

    def __init__(
        self,
        name=None,
        model=None,
        optimizer=None,
        loss=None,
        metrics=None,
        callbacks=None,
        layers=None,
        dataset=None,
        fit_params=None,
    ):
        """
        Initialize the various parameters for the model class

        Parameters
        ----------
        name       : str, Name of the following model
        model      : tf.model, tensorflow model can be defined or fit after
        optimizer  : tf_optimizer, translates the parameter for the optimizer
        loss       : tf_loss, translates the parameter for the loss
        metrics    : tf_metrics, translates the parameter for the metrics
        layers     : tf_layer, translates the parameter for the layers
        dataset    : TF_Dataset, translates the parameter for the dataset
        fit_params : tf_FP, translates the fitting parameters
        """
        if name:
            self.name = name
        else:
            self.name = None

        if model:
            self.model = model
        else:
            self.model = None

        if optimizer:
            self.optimizer = optimizer
        else:
            self.optimizer = tf_optimizer()

        if loss:
            self.loss = loss
        else:
            self.loss = tf_loss()

        if metrics:
            self.metrics = metrics
        else:
            self.metrics = tf_metrics()

        if callbacks:
            self.callbacks = callbacks
        else:
            self.callbacks = tf_callback()
        if layers:
            self.layers = layers
        else:
            self.layers = []

        if dataset:
            self.dataset = dataset
        else:
            self.dataset = TF_Dataset()

        if fit_params:
            self.fit_params = fit_params
        else:
            self.fit_params = tf_FP()

        self.built = False
        self.compiled = False
        self.dataset_build = False

    # **************************************************************************************************************** #
    # ************************************************* LAYERS SECTION *********************************************** #
    # **************************************************************************************************************** #
    def add_layer(self, layer=None, layer_type=None, layer_config=None):
        """

        See document of tf_layer in the README in utils

        Parameters
        ----------
        layer        : tf_layer class, pre-initialized
        layer_type   : type of the layer
        layer_config : params of the layer

        Add a layer in the model
        -------

        """
        if layer:
            if layer.name:
                self.layers.append(layer)
            else:
                layer.update(name=len(self.layers) + 1)
                self.layers.append(layer)
        elif layer_type and layer_config:
            self.layers.append(tf_layer(name=(len(self.layers) + 1)), layer_type=layer_type, layer_config=layer_config)

    def remove_layer(self, layer_id):
        if layer_id:
            self.layers.pop(layer_id)
            for i in range(layer_id, len(self.layers)):
                self.layers[i].update(name=(self.layers[i].name - 1))

    def modify_layer(self, layer_id=None, new_layer=None, new_type=None, new_config=None):
        if layer_id:
            if new_layer:
                self.layers[layer_id] = new_layer
            elif new_type and new_config:
                self.layers[layer_id] = tf_layer(
                    name=self.layers[layer_id].name, layer_type=new_type, layer_config=new_config
                )

    def layers_from_json(self, path):
        file = open(path)
        layers = json.load(file)

        def filter(dict):
            dict2 = dict.copy()
            new_dict = {}
            for key, value in dict["layer_config"].items():
                if value:
                    new_dict[key] = value
            dict2["layer_config"] = new_dict
            return dict2

        for i in range(len(layers)):
            params = filter(layers[f"layer {i}"])
            self.add_layer(layer=tf_layer(**params))

    def layers_to_json(self, path):
        params = {}
        i = 0
        for layer in self.layers:
            params[f"layer {i}"] = layer.get_params()
            i = i + 1
        with open(path, "w") as write_file:
            json.dump(params, write_file, indent=4)

    # **************************************************************************************************************** #
    # ************************************************ DATASET SECTION *********************************************** #
    # **************************************************************************************************************** #

    def dataset_from_json(self, path):
        self.dataset.from_json(path)

    def dataset_to_json(self, path):
        self.dataset.to_json(path)

    # **************************************************************************************************************** #
    # ************************************************ METRICS SECTION *********************************************** #
    # **************************************************************************************************************** #
    def add_metrics(self, Metrics):
        self.metrics = Metrics

    def update_metrics(self, Metrics):
        self.metrics = Metrics

    def metrics_to_json(self, path):
        self.metrics.to_json(path)

    def metrics_from_json(self, path):
        self.metrics.from_json(path)

    # **************************************************************************************************************** #
    # ************************************************** LOSS SECTION ************************************************ #
    # **************************************************************************************************************** #
    def add_loss(self, Loss):
        self.loss = Loss

    def update_loss(self, Loss):
        if Loss:
            self.loss = Loss

    def loss_to_json(self, path):
        self.loss.to_json(path)

    def loss_from_json(self, path):
        self.loss.from_json(path)

    # **************************************************************************************************************** #
    # ************************************************ OPTIMIZER SECTION ********************************************* #
    # **************************************************************************************************************** #
    def add_optimizer(self, Optimizer):
        self.optimizer = Optimizer

    def update_optimizer(self, Optimizer):
        if Optimizer:
            self.optimizer = Optimizer

    def optimizer_to_json(self, path):
        self.optimizer.to_json(path)

    def optimizer_from_json(self, path):
        self.optimizer.from_json(path)

    # **************************************************************************************************************** #
    # ************************************************ OPTIMIZER SECTION ********************************************* #
    # **************************************************************************************************************** #
    def add_callbacks(self, Callbacks):
        self.callbacks = Callbacks

    def update_optimizer(self, Callbacks):
        if Callbacks:
            self.callbacks = Callbacks

    def callbacks_to_json(self, path):
        self.callbacks.to_json(path)

    def callbacks_from_json(self, path):
        self.callbacks.from_json(path)

    # **************************************************************************************************************** #
    # ************************************************* BUILD SECTION ************************************************ #
    # **************************************************************************************************************** #
    def build(self, inputs=None, inputs_shape=None):
        self.built = True
        if inputs:
            self.build_layer(inputs=inputs)
        elif inputs_shape:
            inputs = tf.keras.Input(inputs_shape)
            self.build_layer(inputs=inputs)
        else:
            self.build_dataset()
            inputs = tf.keras.Input(self.dataset.shape)
            self.build_layer(inputs=inputs)
        if self.loss:
            self.build_loss()
        else:
            print("No Loss described, the model isn't built.")
            self.built = False
        if self.optimizer:
            self.build_optimizer()
        else:
            print("No optimizer described, the model isn't built.")
            self.built = False
        if self.metrics:
            self.build_metrics()
        else:
            print("No Loss described, the model isn't built.")
            self.built = False
        if self.callbacks:
            self.build_callbacks()

    def build_layer(self, inputs):
        outputs = tf.identity(inputs)
        for layer in self.layers:
            outputs = layer.build(outputs)
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)

    def build_loss(self):
        self.loss.build()

    def build_metrics(self):
        self.metrics.build()

    def build_optimizer(self):
        self.optimizer.build()

    def build_callbacks(self):
        self.callbacks.build(self.optimizer, self.loss, self.metrics, self.fit_params, self.dataset)

    def build_dataset(self, shuffle=None, batch_size=None):
        self.dataset.build()
        self.dataset.filter()
        self.dataset.tf_transform(shuffle, batch_size)
        self.dataset_build = True

    def sequential_built(self):
        layers = []
        for layer in self.layers():
            layers.append(layer)

    def compile(self):
        if self.built:
            self.model.compile(optimizer=self.optimizer.optimizer, loss=self.loss.loss, metrics=self.metrics.metrics)
            self.compiled = True

    # **************************************************************************************************************** #
    # ************************************************** FIT SECTION ************************************************* #
    # **************************************************************************************************************** #

    def update_params_fit(self, fit_params=None):
        if fit_params:
            self.fit_params = fit_params

    def fit_params_from_json(self, path):
        self.fit_params.from_json(path)

    def fit_params_to_json(self, path):
        self.fit_params.to_json(path)

    def fit(self):
        os.makedirs(os.path.join(self.root, "results"), exist_ok=True)
        csv_logger = tf.keras.callbacks.CSVLogger(
            os.path.join(
                self.root, "results/training_{}.csv".format(len(os.listdir(os.path.join(self.root, "results"))))
            )
        )
        if self.compiled and self.dataset_build:
            dataset = self.dataset.tf_data
            if self.dataset.tf_data_test:
                self.model.fit(
                    dataset,
                    validation_data=self.dataset.tf_data_test,
                    **self.fit_params.get_params(),
                    callbacks=[csv_logger],
                )
            else:
                self.model.fit(dataset, **self.fit_params.get_params(), callbacks=[csv_logger])
        else:
            self.compile()
            self.build_dataset(shuffle=100, batch_size=self.fit_params.fit_params["batch_size"])
            dataset = self.dataset.tf_data
            if self.dataset.tf_data_test:
                self.model.fit(
                    dataset,
                    validation_data=self.dataset.tf_data_test,
                    **self.fit_params.get_params(),
                    callbacks=[csv_logger],
                )
            else:
                self.model.fit(dataset, **self.fit_params.get_params(), callbacks=[csv_logger])

    # **************************************************************************************************************** #
    # *********************************************** LOAD/SAVE SECTION ********************************************** #
    # **************************************************************************************************************** #
    def to_json(self, path):
        os.makedirs(path, exist_ok=False)
        self.layers_to_json(os.path.join(path, "layers.json"))
        self.dataset_to_json(os.path.join(path, "dataset.json"))
        self.metrics_to_json(os.path.join(path, "metrics.json"))
        self.loss_to_json(os.path.join(path, "loss.json"))
        self.optimizer_to_json(os.path.join(path, "opt.json"))
        self.fit_params_to_json(os.path.join(path, "fit_params.json"))
        if self.callbacks:
            self.callbacks_to_json(os.path.join(path, "Callbacks.json"))

    def from_json(self, path):
        self.layers_from_json(os.path.join(path, "layers.json"))
        self.dataset_from_json(os.path.join(path, "dataset.json"))
        self.metrics_from_json(os.path.join(path, "metrics.json"))
        self.loss_from_json(os.path.join(path, "loss.json"))
        self.optimizer_from_json(os.path.join(path, "opt.json"))
        self.fit_params_from_json(os.path.join(path, "fit_params.json"))
        if os.path.isfile(os.path.join(path, "Callbacks.json")):
            self.callbacks_from_json(os.path.join(path, "Callbacks.json"))
        self.root = path

    # **************************************************************************************************************** #
    # ********************************************* HYPERPARAMS SECTION ********************************************** #
    # **************************************************************************************************************** #
    def hyper_params_update(self):
        pass

    # **************************************************************************************************************** #
    # ************************************************* PLOT SECTION ************************************************* #
    # **************************************************************************************************************** #

    def __str__(self):
        output = "Name     : {} ".format(self.name) + "\n"
        output = output + "Build    : {}".format(self.built) + "\n"
        output = output + "Compiled : {}".format(self.compiled) + "\n"
        output = output + "-" * 80 + "\n"
        if self.optimizer:
            output = output + str(self.optimizer)
        else:
            output = output + ("No optimizer defined")
        output = output + "-" * 80 + "\n"
        if self.loss:
            output = output + str(self.loss)
        else:
            output = output + ("No Loss defined")
        output = output + "-" * 80 + "\n" + "\n"
        output = output + "-" * 80 + "\n"
        output = output + "INTERN STRUCTURE :" + "\n"
        for layer in self.layers:
            output = output + str(layer)
        output = output + "-" * 80 + "\n"
        return output

    def print_intern_structure(self):
        output = "-" * 80 + "\n"
        output = output + "INTERN STRUCTURE :" + "\n"
        for layer in self.layers:
            output = output + str(layer)
        output = output + "-" * 80 + "\n"
        print(output)

    def print_optimizer(self):
        if self.optimizer:
            print(self.optimizer)
        else:
            print("No optimizer defined")

    def print_loss(self):
        if self.loss:
            print(self.loss)
        else:
            print("No loss defined")


# ******************************************************************************************************************** #
# Main

if __name__ == "__main__":
    from utils import *

    model = TF_Model(name="Test Model")
    model.layers_from_json("example_layers.json")
    model.add_optimizer(tf_optimizer(name="Adam", type="Adam", params={"learning_rate": 0.1}))
    model.add_loss(tf_loss(name="BN", type="BinaryCrossentropy", params={"from_logits": False}))
    model.add_metrics(tf_metrics(params={"Accuracy": {"name": "Acc"},}))
    input_shape = (None, 2048, 2048, 3)
    inputs = tf.keras.Input(input_shape)
    model.build(inputs)
    model.compile()
    print(model)
    model.layers_to_json("example_layers.json")
    # model.print_intern_structure()
    # model.print_optimizer()
