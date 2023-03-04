# ******************************************************************************************************************** #
# ********************************************* #   CENTRALESUPELEC   # ********************************************** #
# ******************************************************************************************************************** #
# Project : deepl
# File    : Metric_method.py
# PATH    : NN_tensorflow/utils
# Author  : trisr
# Date    : 10/01/2023
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
import json
import tensorflow.keras.metrics as metrics

# ******************************************************************************************************************** #
# Configuration
dict_needed_params = {}
dict_needed_params["AUC"] = []
dict_needed_params["Accuracy"] = []
dict_needed_params["BinaryAccuracy"] = []
dict_needed_params["BinaryCrossentropy"] = []
dict_needed_params["BinaryIoU"] = []
dict_needed_params["CategoricalAccuracy"] = []
dict_needed_params["CategoricalCrossentropy"] = []
dict_needed_params["CategoricalHinge"] = []
dict_needed_params["CosineSimilarity"] = []
dict_needed_params["FalseNegatives"] = []
dict_needed_params["FalsePositives"] = []
dict_needed_params["Hinge"] = []
dict_needed_params["IoU"] = ["num_classes", "target_class_ids"]
dict_needed_params["KLDivergence"] = []
dict_needed_params["LogCoshError"] = []
dict_needed_params["MeanAbsoluteError"] = []
dict_needed_params["MeanAbsolutePercentageError"] = []
dict_needed_params["MeanIoU"] = ["num_classes"]
dict_needed_params["MeanRelativeError"] = ["normalizer"]
dict_needed_params["MeanSquaredError"] = []
dict_needed_params["MeanSquaredLogarithmicError"] = []
dict_needed_params["MeanTensor"] = []
dict_needed_params["OneHotIoU"] = ["num_classes", "target_class_ids"]
dict_needed_params["OneHotMeanIoU"] = ["num_classes"]
dict_needed_params["Poisson"] = []
dict_needed_params["Precision"] = []
dict_needed_params["PrecisionAtRecall"] = ["recall"]
dict_needed_params["Recall"] = []
dict_needed_params["RecallAtPrecision"] = ["precision"]
dict_needed_params["RootMeanSquaredError"] = []
dict_needed_params["SensitivityAtSpecificity"] = ["specificity"]
dict_needed_params["SparseCategoricalAccuracy"] = []
dict_needed_params["SparseCategoricalCrossentropy"] = []
dict_needed_params["SparseTopKCategoricalAccuracy"] = []
dict_needed_params["SpecificityAtSensitivity"] = ["sensitivity"]
dict_needed_params["SquaredHinge"] = []
dict_needed_params["Sum"] = []
dict_needed_params["TopKCategoricalAccuracy"] = []
dict_needed_params["TrueNegatives"] = []
dict_needed_params["TruePositives"] = []
dict_needed_params["DCGW"] = []

dict_params_metrics = {}
dict_params_metrics["AUC"] = metrics.AUC().get_config().keys()
dict_params_metrics["Accuracy"] = metrics.Accuracy().get_config().keys()
dict_params_metrics["BinaryAccuracy"] = metrics.BinaryAccuracy().get_config().keys()
dict_params_metrics["BinaryCrossentropy"] = metrics.BinaryCrossentropy().get_config().keys()
dict_params_metrics["BinaryIoU"] = metrics.BinaryIoU().get_config().keys()
dict_params_metrics["CategoricalAccuracy"] = metrics.CategoricalAccuracy().get_config().keys()
dict_params_metrics["CategoricalCrossentropy"] = metrics.CategoricalCrossentropy().get_config().keys()
dict_params_metrics["CategoricalHinge"] = metrics.CategoricalHinge().get_config().keys()
dict_params_metrics["CosineSimilarity"] = metrics.CosineSimilarity().get_config().keys()
dict_params_metrics["FalseNegatives"] = metrics.FalseNegatives().get_config().keys()
dict_params_metrics["FalsePositives"] = metrics.FalsePositives().get_config().keys()
dict_params_metrics["Hinge"] = metrics.Hinge().get_config().keys()
dict_params_metrics["IoU"] = metrics.IoU(num_classes=2, target_class_ids=[0]).get_config().keys()
dict_params_metrics["KLDivergence"] = metrics.KLDivergence().get_config().keys()
dict_params_metrics["LogCoshError"] = metrics.LogCoshError().get_config().keys()
dict_params_metrics["MeanAbsoluteError"] = metrics.MeanAbsoluteError().get_config().keys()
dict_params_metrics["MeanAbsolutePercentageError"] = metrics.MeanAbsolutePercentageError().get_config().keys()
dict_params_metrics["MeanIoU"] = metrics.MeanIoU(2).get_config().keys()
dict_params_metrics["MeanRelativeError"] = metrics.MeanRelativeError(normalizer=[1, 3, 2, 3]).get_config().keys()
dict_params_metrics["MeanSquaredError"] = metrics.MeanSquaredError().get_config().keys()
dict_params_metrics["MeanSquaredLogarithmicError"] = metrics.MeanSquaredLogarithmicError().get_config().keys()
dict_params_metrics["MeanTensor"] = metrics.MeanTensor().get_config().keys()
dict_params_metrics["OneHotIoU"] = metrics.OneHotIoU(num_classes=3, target_class_ids=[0, 2]).get_config().keys()
dict_params_metrics["OneHotMeanIoU"] = metrics.OneHotMeanIoU(num_classes=3).get_config().keys()
dict_params_metrics["Poisson"] = metrics.Poisson().get_config().keys()
dict_params_metrics["Precision"] = metrics.Precision().get_config().keys()
dict_params_metrics["PrecisionAtRecall"] = metrics.PrecisionAtRecall(recall=0.8).get_config().keys()
dict_params_metrics["Recall"] = metrics.Recall().get_config().keys()
dict_params_metrics["RecallAtPrecision"] = metrics.RecallAtPrecision(precision=0.8).get_config().keys()
dict_params_metrics["RootMeanSquaredError"] = metrics.RootMeanSquaredError().get_config().keys()
dict_params_metrics["SensitivityAtSpecificity"] = metrics.SensitivityAtSpecificity(specificity=0.5).get_config().keys()
dict_params_metrics["SparseCategoricalAccuracy"] = metrics.SparseCategoricalAccuracy().get_config().keys()
dict_params_metrics["SparseCategoricalCrossentropy"] = metrics.SparseCategoricalCrossentropy().get_config().keys()
dict_params_metrics["SparseTopKCategoricalAccuracy"] = metrics.SparseTopKCategoricalAccuracy().get_config().keys()
dict_params_metrics["SpecificityAtSensitivity"] = metrics.SpecificityAtSensitivity(sensitivity=0.5).get_config().keys()
dict_params_metrics["SquaredHinge"] = metrics.SquaredHinge().get_config().keys()
dict_params_metrics["Sum"] = metrics.Sum().get_config().keys()
dict_params_metrics["TopKCategoricalAccuracy"] = metrics.TopKCategoricalAccuracy().get_config().keys()
dict_params_metrics["TrueNegatives"] = metrics.TrueNegatives().get_config().keys()
dict_params_metrics["TruePositives"] = metrics.TruePositives().get_config().keys()
dict_params_metrics["DCGW"] = []

# ******************************************************************************************************************** #
# Function definition


def AUC_build(config_metrics):
    """

    Parameters
    ----------
    self.config_metrics

    Inplace function, build a metric AUC with the config_metric parameters for a tensorflow model.
    -------

    """
    metric = metrics.AUC(**config_metrics)
    return metric


def Accuracy_build(config_metrics):
    """

    Parameters
    ----------
    self.config_metrics

    Inplace function, build a metric Accuracy with the config_metric parameters for a tensorflow model.
    -------

    """
    metric = metrics.Accuracy(**config_metrics)
    return metric


def BinaryAccuracy_build(config_metrics):
    """

    Parameters
    ----------
    self.config_metrics

    Inplace function, build a metric BinaryAccuracy with the config_metric parameters for a tensorflow model.
    -------

    """
    metric = metrics.BinaryAccuracy(**config_metrics)
    return metric


def BinaryCrossentropy_build(config_metrics):
    """

    Parameters
    ----------
    self.config_metrics

    Inplace function, build a metric BinaryCrossentropy with the config_metric parameters for a tensorflow model.
    -------

    """
    metric = metrics.BinaryCrossentropy(**config_metrics)
    return metric


def BinaryIoU_build(config_metrics):
    """

    Parameters
    ----------
    self.config_metrics

    Inplace function, build a metric BinaryIoU with the config_metric parameters for a tensorflow model.
    -------

    """
    metric = metrics.BinaryIoU(**config_metrics)
    return metric


def CategoricalAccuracy_build(config_metrics):
    """

    Parameters
    ----------
    self.config_metrics

    Inplace function, build a metric CategoricalAccuracy with the config_metric parameters for a tensorflow model.
    -------

    """
    metric = metrics.CategoricalAccuracy(**config_metrics)
    return metric


def CategoricalCrossentropy_build(config_metrics):
    """

    Parameters
    ----------
    self.config_metrics

    Inplace function, build a metric CategoricalCrossentropy with the config_metric parameters for a tensorflow model.
    -------

    """
    metric = metrics.CategoricalCrossentropy(**config_metrics)
    return metric


def CategoricalHinge_build(config_metrics):
    """

    Parameters
    ----------
    self.config_metrics

    Inplace function, build a metric CategoricalHinge with the config_metric parameters for a tensorflow model.
    -------

    """
    metric = metrics.CategoricalHinge(**config_metrics)
    return metric


def CosineSimilarity_build(config_metrics):
    """

    Parameters
    ----------
    self.config_metrics

    Inplace function, build a metric CosineSimilarity with the config_metric parameters for a tensorflow model.
    -------

    """
    metric = metrics.CosineSimilarity(**config_metrics)
    return metric


def FalseNegatives_build(config_metrics):
    """

    Parameters
    ----------
    self.config_metrics

    Inplace function, build a metric FalseNegatives with the config_metric parameters for a tensorflow model.
    -------

    """
    metric = metrics.FalseNegatives(**config_metrics)
    return metric


def FalsePositives_build(config_metrics):
    """

    Parameters
    ----------
    self.config_metrics

    Inplace function, build a metric FalsePositives with the config_metric parameters for a tensorflow model.
    -------

    """
    metric = metrics.FalsePositives(**config_metrics)
    return metric


def Hinge_build(config_metrics):
    """

    Parameters
    ----------
    self.config_metrics

    Inplace function, build a metric Hinge with the config_metric parameters for a tensorflow model.
    -------

    """
    metric = metrics.Hinge(**config_metrics)
    return metric


def IoU_build(config_metrics):
    """

    Parameters
    ----------
    self.config_metrics

    Inplace function, build a metric IoU with the config_metric parameters for a tensorflow model.
    -------

    """
    metric = metrics.IoU(**config_metrics)
    return metric


def KLDivergence_build(config_metrics):
    """

    Parameters
    ----------
    self.config_metrics

    Inplace function, build a metric KLDivergence with the config_metric parameters for a tensorflow model.
    -------

    """
    metric = metrics.KLDivergence(**config_metrics)
    return metric


def LogCoshError_build(config_metrics):
    """

    Parameters
    ----------
    self.config_metrics

    Inplace function, build a metric LogCoshError with the config_metric parameters for a tensorflow model.
    -------

    """
    metric = metrics.LogCoshError(**config_metrics)
    return metric


def MeanAbsoluteError_build(config_metrics):
    """

    Parameters
    ----------
    self.config_metrics

    Inplace function, build a metric MeanAbsoluteError with the config_metric parameters for a tensorflow model.
    -------

    """
    metric = metrics.MeanAbsoluteError(**config_metrics)
    return metric


def MeanAbsolutePercentageError_build(config_metrics):
    """

    Parameters
    ----------
    self.config_metrics

    Inplace function, build a metric MeanAbsolutePercentageError with the config_metric parameters for a tensorflow model.
    -------

    """
    metric = metrics.MeanAbsolutePercentageError(**config_metrics)
    return metric


def MeanIoU_build(config_metrics):
    """

    Parameters
    ----------
    self.config_metrics

    Inplace function, build a metric MeanIoU with the config_metric parameters for a tensorflow model.
    -------

    """
    metric = metrics.MeanIoU(**config_metrics)
    return metric


def MeanRelativeError_build(config_metrics):
    """

    Parameters
    ----------
    self.config_metrics

    Inplace function, build a metric MeanRelativeError with the config_metric parameters for a tensorflow model.
    -------

    """
    metric = metrics.MeanRelativeError(**config_metrics)
    return metric


def MeanSquaredError_build(config_metrics):
    """

    Parameters
    ----------
    self.config_metrics

    Inplace function, build a metric MeanSquaredError with the config_metric parameters for a tensorflow model.
    -------

    """
    metric = metrics.MeanSquaredError(**config_metrics)
    return metric


def MeanSquaredLogarithmicError_build(config_metrics):
    """

    Parameters
    ----------
    self.config_metrics

    Inplace function, build a metric MeanSquaredLogarithmicError with the config_metric parameters for a tensorflow model.
    -------

    """
    metric = metrics.MeanSquaredLogarithmicError(**config_metrics)
    return metric


def MeanTensor_build(config_metrics):
    """

    Parameters
    ----------
    self.config_metrics

    Inplace function, build a metric MeanTensor with the config_metric parameters for a tensorflow model.
    -------

    """
    metric = metrics.MeanTensor(**config_metrics)
    return metric


def OneHotIoU_build(config_metrics):
    """

    Parameters
    ----------
    self.config_metrics

    Inplace function, build a metric OneHotIoU with the config_metric parameters for a tensorflow model.
    -------

    """
    metric = metrics.OneHotIoU(**config_metrics)
    return metric


def OneHotMeanIoU_build(config_metrics):
    """

    Parameters
    ----------
    self.config_metrics

    Inplace function, build a metric OneHotMeanIoU with the config_metric parameters for a tensorflow model.
    -------

    """
    metric = metrics.OneHotMeanIoU(**config_metrics)
    return metric


def Poisson_build(config_metrics):
    """

    Parameters
    ----------
    self.config_metrics

    Inplace function, build a metric Poisson with the config_metric parameters for a tensorflow model.
    -------

    """
    metric = metrics.Poisson(**config_metrics)
    return metric


def Precision_build(config_metrics):
    """

    Parameters
    ----------
    self.config_metrics

    Inplace function, build a metric Precision with the config_metric parameters for a tensorflow model.
    -------

    """
    metric = metrics.Precision(**config_metrics)
    return metric


def PrecisionAtRecall_build(config_metrics):
    """

    Parameters
    ----------
    self.config_metrics

    Inplace function, build a metric PrecisionAtRecall with the config_metric parameters for a tensorflow model.
    -------

    """
    metric = metrics.PrecisionAtRecall(**config_metrics)
    return metric


def Recall_build(config_metrics):
    """

    Parameters
    ----------
    self.config_metrics

    Inplace function, build a metric Recall with the config_metric parameters for a tensorflow model.
    -------

    """
    metric = metrics.Recall(**config_metrics)
    return metric


def RecallAtPrecision_build(config_metrics):
    """

    Parameters
    ----------
    self.config_metrics

    Inplace function, build a metric RecallAtPrecision with the config_metric parameters for a tensorflow model.
    -------

    """
    metric = metrics.RecallAtPrecision(**config_metrics)
    return metric


def RootMeanSquaredError_build(config_metrics):
    """

    Parameters
    ----------
    self.config_metrics

    Inplace function, build a metric RootMeanSquaredError with the config_metric parameters for a tensorflow model.
    -------

    """
    metric = metrics.RootMeanSquaredError(**config_metrics)
    return metric


def SensitivityAtSpecificity_build(config_metrics):
    """

    Parameters
    ----------
    self.config_metrics

    Inplace function, build a metric SensitivityAtSpecificity with the config_metric parameters for a tensorflow model.
    -------

    """
    metric = metrics.SensitivityAtSpecificity(**config_metrics)
    return metric


def SparseCategoricalAccuracy_build(config_metrics):
    """

    Parameters
    ----------
    self.config_metrics

    Inplace function, build a metric SparseCategoricalAccuracy with the config_metric parameters for a tensorflow model.
    -------

    """
    metric = metrics.SparseCategoricalAccuracy(**config_metrics)
    return metric


def SparseCategoricalCrossentropy_build(config_metrics):
    """

    Parameters
    ----------
    self.config_metrics

    Inplace function, build a metric SparseCategoricalCrossentropy with the config_metric parameters for a tensorflow model.
    -------

    """
    metric = metrics.SparseCategoricalCrossentropy(**config_metrics)
    return metric


def SparseTopKCategoricalAccuracy_build(config_metrics):
    """

    Parameters
    ----------
    self.config_metrics

    Inplace function, build a metric SparseTopKCategoricalAccuracy with the config_metric parameters for a tensorflow model.
    -------

    """
    metric = metrics.SparseTopKCategoricalAccuracy(**config_metrics)
    return metric


def SpecificityAtSensitivity_build(config_metrics):
    """

    Parameters
    ----------
    self.config_metrics

    Inplace function, build a metric SpecificityAtSensitivity with the config_metric parameters for a tensorflow model.
    -------

    """
    metric = metrics.SpecificityAtSensitivity(**config_metrics)
    return metric


def SquaredHinge_build(config_metrics):
    """

    Parameters
    ----------
    self.config_metrics

    Inplace function, build a metric SquaredHinge with the config_metric parameters for a tensorflow model.
    -------

    """
    metric = metrics.SquaredHinge(**config_metrics)
    return metric


def Sum_build(config_metrics):
    """

    Parameters
    ----------
    self.config_metrics

    Inplace function, build a metric Sum with the config_metric parameters for a tensorflow model.
    -------

    """
    metric = metrics.Sum(**config_metrics)
    return metric


def TopKCategoricalAccuracy_build(config_metrics):
    """

    Parameters
    ----------
    self.config_metrics

    Inplace function, build a metric TopKCategoricalAccuracy with the config_metric parameters for a tensorflow model.
    -------

    """
    metric = metrics.TopKCategoricalAccuracy(**config_metrics)
    return metric


def TrueNegatives_build(config_metrics):
    """

    Parameters
    ----------
    self.config_metrics

    Inplace function, build a metric TrueNegatives with the config_metric parameters for a tensorflow model.
    -------

    """
    metric = metrics.TrueNegatives(**config_metrics)
    return metric


def TruePositives_build(config_metrics):
    """

    Parameters
    ----------
    self.config_metrics

    Inplace function, build a metric TruePositives with the config_metric parameters for a tensorflow model.
    -------

    """
    metric = metrics.TruePositives(**config_metrics)
    return metric


def DCGW_build(params):
    def metrics_function(ypred, ytrue):
        def positif(y):
            return int(y >= 0)

        metrics_value = metrics.MeanAbsoluteError(ypred, ytrue) * positif(ytrue)
        return metrics_value

    return metrics_function


METRICS = {name[: -(len("_build"))]: value for name, value in globals().items() if name.endswith("_build")}

# ******************************************************************************************************************** #
# Class Definition


class tf_metrics:
    """


    """

    def __init__(self, params=None):
        self.params = params

    def update(self, params=None):
        if params:
            self.params = params

    def update_parameters(self, optimizer=None, key=None, value=None):
        if key and value and optimizer:
            self.params[optimizer][key] = value

    def test_valid_parameters(self):
        for metric in self.params.keys():
            for key in self.params[metric].keys():
                assert key in dict_params_metrics[metric], "{} isn't a valid parameter for this metric"

    def build(self):
        self.test_valid_parameters()
        self.metrics = []
        for metric in self.params.keys():
            self.metrics.append(METRICS[metric](self.params[metric]))

    def to_json(self, path):
        with open(path, "w") as write_file:
            json.dump(self.params, write_file, indent=4)

    def from_json(self, path):
        file = open(path)
        self.params = json.load(file)


# ******************************************************************************************************************** #
# Main
if __name__ == "__main__":
    Metrics = tf_Metrics(params={"Accuracy": {"name": "Acc"},})
    Metrics.build()
    Metrics.to_JSON(path="Example_Metrics.json")
    print("Done")
