# ******************************************************************************************************************** #
# ********************************************* #   CENTRALESUPELEC   # ********************************************** #
# ******************************************************************************************************************** #
# Project : deepl
# File    : Layer_Method.py
# PATH    : NN_tensorflow
# Author  : trisr
# Date    : 17/11/2022
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
from tensorflow.keras import layers
from .Config_params import dict_needed_params, dict_params_layers


# ****************************** ************************************************************************************** #
# Building function definition


def AbstractRNNCell_build(inputs, config_layers):
    """

    Parameters
    ----------
    self
    config_layers

    Inplace function, add a layer conv3D in model.
    -------

    """
    outputs = layers.AbstractRNNCell(**config_layers)(inputs)
    return outputs


def Activation_build(inputs, config_layers):
    """

    Parameters
    ----------
    self
    config_layers

    Inplace function, add a layer conv3D in model.
    -------

    """
    outputs = layers.Activation(**config_layers)(inputs)
    return outputs


def ActivityRegularization_build(inputs, config_layers):
    """

    Parameters
    ----------
    self
    config_layers

    Inplace function, add a layer conv3D in model.
    -------

    """
    outputs = layers.ActivityRegularization(**config_layers)(inputs)
    return outputs


def Add_build(inputs, config_layers):
    """

    Parameters
    ----------
    self
    config_layers

    Inplace function, add a layer conv3D in model.
    -------

    """
    outputs = layers.Add(**config_layers)(inputs)
    return outputs


def AdditiveAttention_build(inputs, config_layers):
    """

    Parameters
    ----------
    self
    config_layers

    Inplace function, add a layer conv3D in model.
    -------

    """
    outputs = layers.AdditiveAttention(**config_layers)(inputs)
    return outputs


def AlphaDropout_build(inputs, config_layers):
    """

    Parameters
    ----------
    self
    config_layers

    Inplace function, add a layer conv3D in model.
    -------

    """
    outputs = layers.AlphaDropout(**config_layers)(inputs)
    return outputs


def Attention_build(inputs, config_layers):
    """

    Parameters
    ----------
    self
    config_layers

    Inplace function, add a layer conv3D in model.
    -------

    """
    outputs = layers.Attention(**config_layers)(inputs)
    return outputs


def Average_build(inputs, config_layers):
    """

    Parameters
    ----------
    self
    config_layers

    Inplace function, add a layer conv3D in model.
    -------

    """
    outputs = layers.Average(**config_layers)(inputs)
    return outputs


def AveragePooling1D_build(inputs, config_layers):
    """

    Parameters
    ----------
    self
    config_layers

    Inplace function, add a layer conv3D in model.
    -------

    """
    outputs = layers.AveragePooling1D(**config_layers)(inputs)
    return outputs


def AveragePooling2D_build(inputs, config_layers):
    """

    Parameters
    ----------
    self
    config_layers

    Inplace function, add a layer conv3D in model.
    -------

    """
    outputs = layers.AveragePooling2D(**config_layers)(inputs)
    return outputs


def AveragePooling3D_build(inputs, config_layers):
    """

    Parameters
    ----------
    self
    config_layers

    Inplace function, add a layer conv3D in model.
    -------

    """
    outputs = layers.AveragePooling3D(**config_layers)(inputs)
    return outputs


def AvgPool1D_build(inputs, config_layers):
    """

    Parameters
    ----------
    self
    config_layers

    Inplace function, add a layer conv3D in model.
    -------

    """
    outputs = layers.AvgPool1D(**config_layers)(inputs)
    return outputs


def AvgPool2D_build(inputs, config_layers):
    """

    Parameters
    ----------
    self
    config_layers

    Inplace function, add a layer conv3D in model.
    -------

    """
    outputs = layers.AvgPool2D(**config_layers)(inputs)
    return outputs


def AvgPool3D_build(inputs, config_layers):
    """

    Parameters
    ----------
    self
    config_layers

    Inplace function, add a layer conv3D in model.
    -------

    """
    outputs = layers.AvgPool3D(**config_layers)(inputs)
    return outputs


def BatchNormalization_build(inputs, config_layers):
    """

    Parameters
    ----------
    self
    config_layers

    Inplace function, add a layer conv3D in model.
    -------

    """
    outputs = layers.BatchNormalization(**config_layers)(inputs)
    return outputs


def CategoryEncoding_build(inputs, config_layers):
    """

    Parameters
    ----------
    self
    config_layers

    Inplace function, add a layer conv3D in model.
    -------

    """
    outputs = layers.CategoryEncoding(**config_layers)(inputs)
    return outputs


def CenterCrop_build(inputs, config_layers):
    """

    Parameters
    ----------
    self
    config_layers

    Inplace function, add a layer conv3D in model.
    -------

    """
    outputs = layers.CenterCrop(**config_layers)(inputs)
    return outputs


def Concatenate_build(inputs, config_layers):
    """

    Parameters
    ----------
    self
    config_layers

    Inplace function, add a layer conv3D in model.
    -------

    """
    outputs = layers.Concatenate(**config_layers)(inputs)
    return outputs


def Conv1D_build(inputs, config_layers):
    """

    Parameters
    ----------
    self
    config_layers

    Inplace function, add a layer conv3D in model.
    -------

    """
    outputs = layers.Conv1D(**config_layers)(inputs)
    return outputs


def Conv1DTranspose_build(inputs, config_layers):
    """

    Parameters
    ----------
    self
    config_layers

    Inplace function, add a layer conv3D in model.
    -------

    """
    outputs = layers.Conv1DTranspose(**config_layers)(inputs)
    return outputs


def Conv2D_build(inputs, config_layers):
    """

    Parameters
    ----------
    self
    config_layers

    Inplace function, add a layer conv3D in model.
    -------

    """
    outputs = layers.Conv2D(**config_layers)(inputs)
    return outputs


def Conv2DTranspose_build(inputs, config_layers):
    """

    Parameters
    ----------
    self
    config_layers

    Inplace function, add a layer conv3D in model.
    -------

    """
    outputs = layers.Conv2DTranspose(**config_layers)(inputs)
    return outputs


def Conv3D_build(inputs, config_layers):
    """

    Parameters
    ----------
    self
    config_layers

    Inplace function, add a layer conv3D in model.
    -------

    """
    outputs = layers.Conv3D(**config_layers)(inputs)
    return outputs


def Conv3DTranspose_build(inputs, config_layers):
    """

    Parameters
    ----------
    self
    config_layers

    Inplace function, add a layer conv3D in model.
    -------

    """
    outputs = layers.Conv3DTranspose(**config_layers)(inputs)
    return outputs


def ConvLSTM1D_build(inputs, config_layers):
    """

    Parameters
    ----------
    self
    config_layers

    Inplace function, add a layer conv3D in model.
    -------

    """
    outputs = layers.ConvLSTM1D(**config_layers)(inputs)
    return outputs


def ConvLSTM2D_build(inputs, config_layers):
    """

    Parameters
    ----------
    self
    config_layers

    Inplace function, add a layer conv3D in model.
    -------

    """
    outputs = layers.ConvLSTM2D(**config_layers)(inputs)
    return outputs


def ConvLSTM3D_build(inputs, config_layers):
    """

    Parameters
    ----------
    self
    config_layers

    Inplace function, add a layer conv3D in model.
    -------

    """
    outputs = layers.ConvLSTM3D(**config_layers)(inputs)
    return outputs


def Convolution1D_build(inputs, config_layers):
    """

    Parameters
    ----------
    self
    config_layers

    Inplace function, add a layer conv3D in model.
    -------

    """
    outputs = layers.Convolution1D(**config_layers)(inputs)
    return outputs


def Convolution1DTranspose_build(inputs, config_layers):
    """

    Parameters
    ----------
    self
    config_layers

    Inplace function, add a layer conv3D in model.
    -------

    """
    outputs = layers.Convolution1DTranspose(**config_layers)(inputs)
    return outputs


def Convolution2D_build(inputs, config_layers):
    """

    Parameters
    ----------
    self
    config_layers

    Inplace function, add a layer conv3D in model.
    -------

    """
    outputs = layers.Convolution2D(**config_layers)(inputs)
    return outputs


def Convolution2DTranspose_build(inputs, config_layers):
    """

    Parameters
    ----------
    self
    config_layers

    Inplace function, add a layer conv3D in model.
    -------

    """
    outputs = layers.Convolution2DTranspose(**config_layers)(inputs)
    return outputs


def Convolution3D_build(inputs, config_layers):
    """

    Parameters
    ----------
    self
    config_layers

    Inplace function, add a layer conv3D in model.
    -------

    """
    outputs = layers.Convolution3D(**config_layers)(inputs)
    return outputs


def Convolution3DTranspose_build(inputs, config_layers):
    """

    Parameters
    ----------
    self
    config_layers

    Inplace function, add a layer conv3D in model.
    -------

    """
    outputs = layers.Convolution3DTranspose(**config_layers)(inputs)
    return outputs


def Cropping1D_build(inputs, config_layers):
    """

    Parameters
    ----------
    self
    config_layers

    Inplace function, add a layer conv3D in model.
    -------

    """
    outputs = layers.Cropping1D(**config_layers)(inputs)
    return outputs


def Cropping2D_build(inputs, config_layers):
    """

    Parameters
    ----------
    self
    config_layers

    Inplace function, add a layer conv3D in model.
    -------

    """
    outputs = layers.Cropping2D(**config_layers)(inputs)
    return outputs


def Cropping3D_build(inputs, config_layers):
    """

    Parameters
    ----------
    self
    config_layers

    Inplace function, add a layer conv3D in model.
    -------

    """
    outputs = layers.Cropping3D(**config_layers)(inputs)
    return outputs


def Dense_build(inputs, config_layers):
    """

    Parameters
    ----------
    self
    config_layers

    Inplace function, add a layer conv3D in model.
    -------

    """
    outputs = layers.Dense(**config_layers)(inputs)
    return outputs


def DepthwiseConv1D_build(inputs, config_layers):
    """

    Parameters
    ----------
    self
    config_layers

    Inplace function, add a layer conv3D in model.
    -------

    """
    outputs = layers.DepthwiseConv1D(**config_layers)(inputs)
    return outputs


def DepthwiseConv2D_build(inputs, config_layers):
    """

    Parameters
    ----------
    self
    config_layers

    Inplace function, add a layer conv3D in model.
    -------

    """
    outputs = layers.DepthwiseConv2D(**config_layers)(inputs)
    return outputs


def Discretization_build(inputs, config_layers):
    """

    Parameters
    ----------
    self
    config_layers

    Inplace function, add a layer conv3D in model.
    -------

    """
    outputs = layers.Discretization(**config_layers)(inputs)
    return outputs


def Dot_build(inputs, config_layers):
    """

    Parameters
    ----------
    self
    config_layers

    Inplace function, add a layer conv3D in model.
    -------

    """
    outputs = layers.Dot(**config_layers)(inputs)
    return outputs


def Dropout_build(inputs, config_layers):
    """

    Parameters
    ----------
    self
    config_layers

    Inplace function, add a layer conv3D in model.
    -------

    """
    outputs = layers.Dropout(**config_layers)(inputs)
    return outputs


def ELU_build(inputs, config_layers):
    """

    Parameters
    ----------
    self
    config_layers

    Inplace function, add a layer conv3D in model.
    -------

    """
    outputs = layers.ELU(**config_layers)(inputs)
    return outputs


def EinsumDense_build(inputs, config_layers):
    """

    Parameters
    ----------
    self
    config_layers

    Inplace function, add a layer conv3D in model.
    -------

    """
    outputs = layers.EinsumDense(**config_layers)(inputs)
    return outputs


def Embedding_build(inputs, config_layers):
    """

    Parameters
    ----------
    self
    config_layers

    Inplace function, add a layer conv3D in model.
    -------

    """
    outputs = layers.Embedding(**config_layers)(inputs)
    return outputs


def Flatten_build(inputs, config_layers):
    """

    Parameters
    ----------
    self
    config_layers

    Inplace function, add a layer conv3D in model.
    -------

    """
    outputs = layers.Flatten(**config_layers)(inputs)
    return outputs


def GRU_build(inputs, config_layers):
    """

    Parameters
    ----------
    self
    config_layers

    Inplace function, add a layer conv3D in model.
    -------

    """
    outputs = layers.GRU(**config_layers)(inputs)
    return outputs


def GRUCell_build(inputs, config_layers):
    """

    Parameters
    ----------
    self
    config_layers

    Inplace function, add a layer conv3D in model.
    -------

    """
    outputs = layers.GRUCell(**config_layers)(inputs)
    return outputs


def GaussianDropout_build(inputs, config_layers):
    """

    Parameters
    ----------
    self
    config_layers

    Inplace function, add a layer conv3D in model.
    -------

    """
    outputs = layers.GaussianDropout(**config_layers)(inputs)
    return outputs


def GaussianNoise_build(inputs, config_layers):
    """

    Parameters
    ----------
    self
    config_layers

    Inplace function, add a layer conv3D in model.
    -------

    """
    outputs = layers.GaussianNoise(**config_layers)(inputs)
    return outputs


def GlobalAveragePooling1D_build(inputs, config_layers):
    """

    Parameters
    ----------
    self
    config_layers

    Inplace function, add a layer conv3D in model.
    -------

    """
    outputs = layers.GlobalAveragePooling1D(**config_layers)(inputs)
    return outputs


def GlobalAveragePooling2D_build(inputs, config_layers):
    """

    Parameters
    ----------
    self
    config_layers

    Inplace function, add a layer conv3D in model.
    -------

    """
    outputs = layers.GlobalAveragePooling2D(**config_layers)(inputs)
    return outputs


def GlobalAveragePooling3D_build(inputs, config_layers):
    """

    Parameters
    ----------
    self
    config_layers

    Inplace function, add a layer conv3D in model.
    -------

    """
    outputs = layers.GlobalAveragePooling3D(**config_layers)(inputs)
    return outputs


def GlobalAvgPool1D_build(inputs, config_layers):
    """

    Parameters
    ----------
    self
    config_layers

    Inplace function, add a layer conv3D in model.
    -------

    """
    outputs = layers.GlobalAvgPool1D(**config_layers)(inputs)
    return outputs


def GlobalAvgPool2D_build(inputs, config_layers):
    """

    Parameters
    ----------
    self
    config_layers

    Inplace function, add a layer conv3D in model.
    -------

    """
    outputs = layers.GlobalAvgPool2D(**config_layers)(inputs)
    return outputs


def GlobalAvgPool3D_build(inputs, config_layers):
    """

    Parameters
    ----------
    self
    config_layers

    Inplace function, add a layer conv3D in model.
    -------

    """
    outputs = layers.GlobalAvgPool3D(**config_layers)(inputs)
    return outputs


def GlobalMaxPool1D_build(inputs, config_layers):
    """

    Parameters
    ----------
    self
    config_layers

    Inplace function, add a layer conv3D in model.
    -------

    """
    outputs = layers.GlobalMaxPool1D(**config_layers)(inputs)
    return outputs


def GlobalMaxPool2D_build(inputs, config_layers):
    """

    Parameters
    ----------
    self
    config_layers

    Inplace function, add a layer conv3D in model.
    -------

    """
    outputs = layers.GlobalMaxPool2D(**config_layers)(inputs)
    return outputs


def GlobalMaxPool3D_build(inputs, config_layers):
    """

    Parameters
    ----------
    self
    config_layers

    Inplace function, add a layer conv3D in model.
    -------

    """
    outputs = layers.GlobalMaxPool3D(**config_layers)(inputs)
    return outputs


def GlobalMaxPooling1D_build(inputs, config_layers):
    """

    Parameters
    ----------
    self
    config_layers

    Inplace function, add a layer conv3D in model.
    -------

    """
    outputs = layers.GlobalMaxPooling1D(**config_layers)(inputs)
    return outputs


def GlobalMaxPooling2D_build(inputs, config_layers):
    """

    Parameters
    ----------
    self
    config_layers

    Inplace function, add a layer conv3D in model.
    -------

    """
    outputs = layers.GlobalMaxPooling2D(**config_layers)(inputs)
    return outputs


def GlobalMaxPooling3D_build(inputs, config_layers):
    """

    Parameters
    ----------
    self
    config_layers

    Inplace function, add a layer conv3D in model.
    -------

    """
    outputs = layers.GlobalMaxPooling3D(**config_layers)(inputs)
    return outputs


def Hashing_build(inputs, config_layers):
    """

    Parameters
    ----------
    self
    config_layers

    Inplace function, add a layer conv3D in model.
    -------

    """
    outputs = layers.Hashing(**config_layers)(inputs)
    return outputs


def Input_build(inputs, config_layers):
    """

    Parameters
    ----------
    self
    config_layers

    Inplace function, add a layer conv3D in model.
    -------

    """
    outputs = layers.Input(**config_layers)(inputs)
    return outputs


def InputLayer_build(inputs, config_layers):
    """

    Parameters
    ----------
    self
    config_layers

    Inplace function, add a layer conv3D in model.
    -------

    """
    outputs = layers.InputLayer(**config_layers)(inputs)
    return outputs


def InputSpec_build(inputs, config_layers):
    """

    Parameters
    ----------
    self
    config_layers

    Inplace function, add a layer conv3D in model.
    -------

    """
    outputs = layers.InputSpec(**config_layers)(inputs)
    return outputs


def IntegerLookup_build(inputs, config_layers):
    """

    Parameters
    ----------
    self
    config_layers

    Inplace function, add a layer conv3D in model.
    -------

    """
    outputs = layers.IntegerLookup(**config_layers)(inputs)
    return outputs


def LSTM_build(inputs, config_layers):
    """

    Parameters
    ----------
    self
    config_layers

    Inplace function, add a layer conv3D in model.
    -------

    """
    outputs = layers.LSTM(**config_layers)(inputs)
    return outputs


def LSTMCell_build(inputs, config_layers):
    """

    Parameters
    ----------
    self
    config_layers

    Inplace function, add a layer conv3D in model.
    -------

    """
    outputs = layers.LSTMCell(**config_layers)(inputs)
    return outputs


def Lambda_build(inputs, config_layers):
    """

    Parameters
    ----------
    self
    config_layers

    Inplace function, add a layer conv3D in model.
    -------

    """
    outputs = layers.Lambda(**config_layers)(inputs)
    return outputs


def Layer_build(inputs, config_layers):
    """

    Parameters
    ----------
    self
    config_layers

    Inplace function, add a layer conv3D in model.
    -------

    """
    outputs = layers.Layer(**config_layers)(inputs)
    return outputs


def LayerNormalization_build(inputs, config_layers):
    """

    Parameters
    ----------
    self
    config_layers

    Inplace function, add a layer conv3D in model.
    -------

    """
    outputs = layers.LayerNormalization(**config_layers)(inputs)
    return outputs


def LeakyReLU_build(inputs, config_layers):
    """

    Parameters
    ----------
    self
    config_layers

    Inplace function, add a layer conv3D in model.
    -------

    """
    outputs = layers.LeakyReLU(**config_layers)(inputs)
    return outputs


def LocallyConnected1D_build(inputs, config_layers):
    """

    Parameters
    ----------
    self
    config_layers

    Inplace function, add a layer conv3D in model.
    -------

    """
    outputs = layers.LocallyConnected1D(**config_layers)(inputs)
    return outputs


def LocallyConnected2D_build(inputs, config_layers):
    """

    Parameters
    ----------
    self
    config_layers

    Inplace function, add a layer conv3D in model.
    -------

    """
    outputs = layers.LocallyConnected2D(**config_layers)(inputs)
    return outputs


def Masking_build(inputs, config_layers):
    """

    Parameters
    ----------
    self
    config_layers

    Inplace function, add a layer conv3D in model.
    -------

    """
    outputs = layers.Masking(**config_layers)(inputs)
    return outputs


def MaxPool1D_build(inputs, config_layers):
    """

    Parameters
    ----------
    self
    config_layers

    Inplace function, add a layer conv3D in model.
    -------

    """
    outputs = layers.MaxPool1D(**config_layers)(inputs)
    return outputs


def MaxPool2D_build(inputs, config_layers):
    """

    Parameters
    ----------
    self
    config_layers

    Inplace function, add a layer conv3D in model.
    -------

    """
    outputs = layers.MaxPool2D(**config_layers)(inputs)
    return outputs


def MaxPool3D_build(inputs, config_layers):
    """

    Parameters
    ----------
    self
    config_layers

    Inplace function, add a layer conv3D in model.
    -------

    """
    outputs = layers.MaxPool3D(**config_layers)(inputs)
    return outputs


def MaxPooling1D_build(inputs, config_layers):
    """

    Parameters
    ----------
    self
    config_layers

    Inplace function, add a layer conv3D in model.
    -------

    """
    outputs = layers.MaxPooling1D(**config_layers)(inputs)
    return outputs


def MaxPooling2D_build(inputs, config_layers):
    """

    Parameters
    ----------
    self
    config_layers

    Inplace function, add a layer conv3D in model.
    -------

    """
    outputs = layers.MaxPooling2D(**config_layers)(inputs)
    return outputs


def MaxPooling3D_build(inputs, config_layers):
    """

    Parameters
    ----------
    self
    config_layers

    Inplace function, add a layer conv3D in model.
    -------

    """
    outputs = layers.MaxPooling3D(**config_layers)(inputs)
    return outputs


def Maximum_build(inputs, config_layers):
    """

    Parameters
    ----------
    self
    config_layers

    Inplace function, add a layer conv3D in model.
    -------

    """
    outputs = layers.Maximum(**config_layers)(inputs)
    return outputs


def Minimum_build(inputs, config_layers):
    """

    Parameters
    ----------
    self
    config_layers

    Inplace function, add a layer conv3D in model.
    -------

    """
    outputs = layers.Minimum(**config_layers)(inputs)
    return outputs


def MultiHeadAttention_build(inputs, config_layers):
    """

    Parameters
    ----------
    self
    config_layers

    Inplace function, add a layer conv3D in model.
    -------

    """
    outputs = layers.MultiHeadAttention(**config_layers)(inputs)
    return outputs


def Multiply_build(inputs, config_layers):
    """

    Parameters
    ----------
    self
    config_layers

    Inplace function, add a layer conv3D in model.
    -------

    """
    outputs = layers.Multiply(**config_layers)(inputs)
    return outputs


def Normalization_build(inputs, config_layers):
    """

    Parameters
    ----------
    self
    config_layers

    Inplace function, add a layer conv3D in model.
    -------

    """
    outputs = layers.Normalization(**config_layers)(inputs)
    return outputs


def Out_build(inputs, config_layers):
    """

    Parameters
    ----------
    self
    config_layers

    Inplace function, add a layer conv3D in model.
    -------

    """
    outputs = layers.Out(**config_layers)(inputs)
    return outputs


def PReLU_build(inputs, config_layers):
    """

    Parameters
    ----------
    self
    config_layers

    Inplace function, add a layer conv3D in model.
    -------

    """
    outputs = layers.PReLU(**config_layers)(inputs)
    return outputs


def Permute_build(inputs, config_layers):
    """

    Parameters
    ----------
    self
    config_layers

    Inplace function, add a layer conv3D in model.
    -------

    """
    outputs = layers.Permute(**config_layers)(inputs)
    return outputs


def RandomBrightness_build(inputs, config_layers):
    """

    Parameters
    ----------
    self
    config_layers

    Inplace function, add a layer conv3D in model.
    -------

    """
    outputs = layers.RandomBrightness(**config_layers)(inputs)
    return outputs


def RandomContrast_build(inputs, config_layers):
    """

    Parameters
    ----------
    self
    config_layers

    Inplace function, add a layer conv3D in model.
    -------

    """
    outputs = layers.RandomContrast(**config_layers)(inputs)
    return outputs


def RandomCrop_build(inputs, config_layers):
    """

    Parameters
    ----------
    self
    config_layers

    Inplace function, add a layer conv3D in model.
    -------

    """
    outputs = layers.RandomCrop(**config_layers)(inputs)
    return outputs


def RandomFlip_build(inputs, config_layers):
    """

    Parameters
    ----------
    self
    config_layers

    Inplace function, add a layer conv3D in model.
    -------

    """
    outputs = layers.RandomFlip(**config_layers)(inputs)
    return outputs


def RandomHeight_build(inputs, config_layers):
    """

    Parameters
    ----------
    self
    config_layers

    Inplace function, add a layer conv3D in model.
    -------

    """
    outputs = layers.RandomHeight(**config_layers)(inputs)
    return outputs


def RandomRotation_build(inputs, config_layers):
    """

    Parameters
    ----------
    self
    config_layers

    Inplace function, add a layer conv3D in model.
    -------

    """
    outputs = layers.RandomRotation(**config_layers)(inputs)
    return outputs


def RandomTranslation_build(inputs, config_layers):
    """

    Parameters
    ----------
    self
    config_layers

    Inplace function, add a layer conv3D in model.
    -------

    """
    outputs = layers.RandomTranslation(**config_layers)(inputs)
    return outputs


def RandomWidth_build(inputs, config_layers):
    """

    Parameters
    ----------
    self
    config_layers

    Inplace function, add a layer conv3D in model.
    -------

    """
    outputs = layers.RandomWidth(**config_layers)(inputs)
    return outputs


def RandomZoom_build(inputs, config_layers):
    """

    Parameters
    ----------
    self
    config_layers

    Inplace function, add a layer conv3D in model.
    -------

    """
    outputs = layers.RandomZoom(**config_layers)(inputs)
    return outputs


def ReLU_build(inputs, config_layers):
    """

    Parameters
    ----------
    self
    config_layers

    Inplace function, add a layer conv3D in model.
    -------

    """
    outputs = layers.ReLU(**config_layers)(inputs)
    return outputs


def RepeatVector_build(inputs, config_layers):
    """

    Parameters
    ----------
    self
    config_layers

    Inplace function, add a layer conv3D in model.
    -------

    """
    outputs = layers.RepeatVector(**config_layers)(inputs)
    return outputs


def Rescaling_build(inputs, config_layers):
    """

    Parameters
    ----------
    self
    config_layers

    Inplace function, add a layer conv3D in model.
    -------

    """
    outputs = layers.Rescaling(**config_layers)(inputs)
    return outputs


def Reshape_build(inputs, config_layers):
    """

    Parameters
    ----------
    self
    config_layers

    Inplace function, add a layer conv3D in model.
    -------

    """
    outputs = layers.Reshape(**config_layers)(inputs)
    return outputs


def Resizing_build(inputs, config_layers):
    """

    Parameters
    ----------
    self
    config_layers

    Inplace function, add a layer conv3D in model.
    -------

    """
    outputs = layers.Resizing(**config_layers)(inputs)
    return outputs


def SeparableConv1D_build(inputs, config_layers):
    """

    Parameters
    ----------
    self
    config_layers

    Inplace function, add a layer conv3D in model.
    -------

    """
    outputs = layers.SeparableConv1D(**config_layers)(inputs)
    return outputs


def SeparableConv2D_build(inputs, config_layers):
    """

    Parameters
    ----------
    self
    config_layers

    Inplace function, add a layer conv3D in model.
    -------

    """
    outputs = layers.SeparableConv2D(**config_layers)(inputs)
    return outputs


def SeparableConvolution1D_build(inputs, config_layers):
    """

    Parameters
    ----------
    self
    config_layers

    Inplace function, add a layer conv3D in model.
    -------

    """
    outputs = layers.SeparableConvolution1D(**config_layers)(inputs)
    return outputs


def SeparableConvolution2D_build(inputs, config_layers):
    """

    Parameters
    ----------
    self
    config_layers

    Inplace function, add a layer conv3D in model.
    -------

    """
    outputs = layers.SeparableConvolution2D(**config_layers)(inputs)
    return outputs


def SimpleRNN_build(inputs, config_layers):
    """

    Parameters
    ----------
    self
    config_layers

    Inplace function, add a layer conv3D in model.
    -------

    """
    outputs = layers.SimpleRNN(**config_layers)(inputs)
    return outputs


def SimpleRNNCell_build(inputs, config_layers):
    """

    Parameters
    ----------
    self
    config_layers

    Inplace function, add a layer conv3D in model.
    -------

    """
    outputs = layers.SimpleRNNCell(**config_layers)(inputs)
    return outputs


def Softmax_build(inputs, config_layers):
    """

    Parameters
    ----------
    self
    config_layers

    Inplace function, add a layer conv3D in model.
    -------

    """
    outputs = layers.Softmax(**config_layers)(inputs)
    return outputs


def SpatialDropout1D_build(inputs, config_layers):
    """

    Parameters
    ----------
    self
    config_layers

    Inplace function, add a layer conv3D in model.
    -------

    """
    outputs = layers.SpatialDropout1D(**config_layers)(inputs)
    return outputs


def SpatialDropout2D_build(inputs, config_layers):
    """

    Parameters
    ----------
    self
    config_layers

    Inplace function, add a layer conv3D in model.
    -------

    """
    outputs = layers.SpatialDropout2D(**config_layers)(inputs)
    return outputs


def SpatialDropout3D_build(inputs, config_layers):
    """

    Parameters
    ----------
    self
    config_layers

    Inplace function, add a layer conv3D in model.
    -------

    """
    outputs = layers.SpatialDropout3D(**config_layers)(inputs)
    return outputs


def StackedRNNCells_build(inputs, config_layers):
    """

    Parameters
    ----------
    self
    config_layers

    Inplace function, add a layer conv3D in model.
    -------

    """
    outputs = layers.StackedRNNCells(**config_layers)(inputs)
    return outputs


def StringLookup_build(inputs, config_layers):
    """

    Parameters
    ----------
    self
    config_layers

    Inplace function, add a layer conv3D in model.
    -------

    """
    outputs = layers.StringLookup(**config_layers)(inputs)
    return outputs


def Subtract_build(inputs, config_layers):
    """

    Parameters
    ----------
    self
    config_layers

    Inplace function, add a layer conv3D in model.
    -------

    """
    outputs = layers.Subtract(**config_layers)(inputs)
    return outputs


def TextVectorization_build(inputs, config_layers):
    """

    Parameters
    ----------
    self
    config_layers

    Inplace function, add a layer conv3D in model.
    -------

    """
    outputs = layers.TextVectorization(**config_layers)(inputs)
    return outputs


def ThresholdedReLU_build(inputs, config_layers):
    """

    Parameters
    ----------
    self
    config_layers

    Inplace function, add a layer conv3D in model.
    -------

    """
    outputs = layers.ThresholdedReLU(**config_layers)(inputs)
    return outputs


def UnitNormalization_build(inputs, config_layers):
    """

    Parameters
    ----------
    self
    config_layers

    Inplace function, add a layer conv3D in model.
    -------

    """
    outputs = layers.UnitNormalization(**config_layers)(inputs)
    return outputs


def UpSampling1D_build(inputs, config_layers):
    """

    Parameters
    ----------
    self
    config_layers

    Inplace function, add a layer conv3D in model.
    -------

    """
    outputs = layers.UpSampling1D(**config_layers)(inputs)
    return outputs


def UpSampling2D_build(inputs, config_layers):
    """

    Parameters
    ----------
    self
    config_layers

    Inplace function, add a layer conv3D in model.
    -------

    """
    outputs = layers.UpSampling2D(**config_layers)(inputs)
    return outputs


def UpSampling3D_build(inputs, config_layers):
    """

    Parameters
    ----------
    self
    config_layers

    Inplace function, add a layer conv3D in model.
    -------

    """
    outputs = layers.UpSampling3D(**config_layers)(inputs)
    return outputs


def ZeroPadding1D_build(inputs, config_layers):
    """

    Parameters
    ----------
    self
    config_layers

    Inplace function, add a layer conv3D in model.
    -------

    """
    outputs = layers.ZeroPadding1D(**config_layers)(inputs)
    return outputs


def ZeroPadding2D_build(inputs, config_layers):
    """

    Parameters
    ----------
    self
    config_layers

    Inplace function, add a layer conv3D in model.
    -------

    """
    outputs = layers.ZeroPadding2D(**config_layers)(inputs)
    return outputs


def ZeroPadding3D_build(inputs, config_layers):
    """

    Parameters
    ----------
    self
    config_layers

    Inplace function, add a layer conv3D in model.
    -------

    """
    outputs = layers.ZeroPadding3D(**config_layers)(inputs)
    return outputs


def add_build(inputs, config_layers):
    """

    Parameters
    ----------
    self
    config_layers

    Inplace function, add a layer conv3D in model.
    -------

    """
    outputs = layers.add(**config_layers)(inputs)
    return outputs


def average_build(inputs, config_layers):
    """

    Parameters
    ----------
    self
    config_layers

    Inplace function, add a layer conv3D in model.
    -------

    """
    outputs = layers.average(**config_layers)(inputs)
    return outputs


def concatenate_build(inputs, config_layers):
    """

    Parameters
    ----------
    self
    config_layers

    Inplace function, add a layer conv3D in model.
    -------

    """
    outputs = layers.concatenate(**config_layers)(inputs)
    return outputs


def maximum_build(inputs, config_layers):
    """

    Parameters
    ----------
    self
    config_layers

    Inplace function, add a layer conv3D in model.
    -------

    """
    outputs = layers.maximum(**config_layers)(inputs)
    return outputs


def minimum_build(inputs, config_layers):
    """

    Parameters
    ----------
    self
    config_layers

    Inplace function, add a layer conv3D in model.
    -------

    """
    outputs = layers.minimum(**config_layers)(inputs)
    return outputs


def multiply_build(inputs, config_layers):
    """

    Parameters
    ----------
    self
    config_layers

    Inplace function, add a layer conv3D in model.
    -------

    """
    outputs = layers.multiply(**config_layers)(inputs)
    return outputs


# ******************************************************************************************************************** #
# Vectorization
BUILDERS = {name[: -(len("_build"))]: value for name, value in globals().items() if name.endswith("_build")}

# ******************************************************************************************************************** #
# Class function


class tf_layer:
    """
    
    
    """

    def __init__(self, name=None, layer_type=None, layer_config=None):
        """
        

        Parameters
        ----------
        layer_type : TYPE, optional
            DESCRIPTION. The default is None.
        layer_config : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
        self.name = name
        self.layer_type = layer_type
        self.layer_config = layer_config
        self.output_shape = None

    def update(self, name=None, layer_type=None, layer_config=None):
        """
        

        Parameters
        ----------
        layer_type : TYPE, optional
            DESCRIPTION. The default is None.
        layer_config : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
        if name:
            self.name = name
        if layer_type:
            self.layer_type = layer_type
        if layer_config:
            self.layer_config = layer_config

    def build(self, inputs):
        """
        

        Parameters
        ----------
        inputs : TYPE
            DESCRIPTION.

        Returns
        -------
        outputs : TYPE
            DESCRIPTION.

        """
        self.test()
        outputs = BUILDERS[self.layer_type](inputs, self.layer_config)
        self.output_shape = outputs.shape
        return outputs

    def test_params(self):
        """
        

        Returns
        -------
        None.

        """
        for key in self.layer_config:
            assert key in dict_params_layers[self.layer_type], "Params {} not supported".format(key)

    def test_layer_type(self):
        """
        

        Returns
        -------
        None.

        """
        assert self.layer_type in BUILDERS, "Layer type {} not supported.".format(self.layer_type)

    def needed_params(self):
        """
        

        Returns
        -------
        None.

        """
        for key in dict_needed_params[self.layer_type]:
            assert key in self.layer_config, " Params {} is needed".format(key)

    def test(self):
        """
        

        Returns
        -------
        None.

        """
        self.test_layer_type()
        self.test_params()

    def get_params(self):
        global_params = dict_params_layers[self.layer_type]
        config = {}
        for key in global_params:
            if key in self.layer_config:
                config[key] = self.layer_config[key]
            else:
                config[key] = None
        params = {"name": self.name, "layer_type": self.layer_type, "layer_config": config}
        return params

    def __str__(self):
        """
        

        Returns
        -------
        output : TYPE
            DESCRIPTION.

        """
        output = "   Layer {} ".format(self.name) + "\n"
        output = output + "-" * 80 + "\n"
        max_length = max(len(str(self.layer_type)), len("output shape"))
        output = output + "   {}".format(self.layer_type) + " " * (max_length - len(str(self.layer_type)) + 3) + ": "
        for key in self.layer_config.keys():
            output = output + "{} : {}".format(key, self.layer_config[key]) + "\n"
            output = output + " " * (max_length + 8)
        output = output + str(self.layer_config)
        if self.output_shape:
            output = (
                output
                + "\n"
                + "   output shape"
                + " " * (max_length - len("output shape") + 3)
                + ": {}".format(str(self.output_shape))
            )
        output = output + "\n" + "-" * 80 + "\n"
        return output


# ******************************************************************************************************************** #
# Main
if __name__ == "__main__":
    layer = tf_layer(name="Test_layer", layer_type="Conv1D", layer_config={"filters": 3, "kernel_size": 3})
    layer.test()
    input_shape = (4, 2048, 2048, 3)
    inputs = tf.random.normal(input_shape)
    outputs = layer.build(inputs)
    print(layer)
