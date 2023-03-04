# ******************************************************************************************************************** #
# ********************************************* #   CENTRALESUPELEC   # ********************************************** #
# ******************************************************************************************************************** #
# Project : deepl
# File    : Config_Param.py
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
from tensorflow.keras import layers

# ****************************** ************************************************************************************** #
# Building function definition

# ******************************************************************************************************************** #
# Configuration
dict_needed_params = {}
dict_needed_params["AbstractRNNCell"] = []
dict_needed_params["Activation"] = ["activation"]
dict_needed_params["ActivityRegularization"] = []
dict_needed_params["Add"] = []
dict_needed_params["AdditiveAttention"] = []
dict_needed_params["AlphaDropout"] = ["rate"]
dict_needed_params["Attention"] = []
dict_needed_params["Average"] = []
dict_needed_params["AveragePooling1D"] = []
dict_needed_params["AveragePooling2D"] = []
dict_needed_params["AveragePooling3D"] = []
dict_needed_params["AvgPool1D"] = []
dict_needed_params["AvgPool2D"] = []
dict_needed_params["AvgPool3D"] = []
dict_needed_params["BatchNormalization"] = []
dict_needed_params["CategoryEncoding"] = ["num_tokens", "output_mode"]
dict_needed_params["CenterCrop"] = ["width", "heigth"]
dict_needed_params["Concatenate"] = []
dict_needed_params["Conv1D"] = ["filters", "kernel_size"]
dict_needed_params["Conv1DTranspose"] = ["filters", "kernel_size"]
dict_needed_params["Conv2D"] = ["filters", "kernel_size"]
dict_needed_params["Conv2DTranspose"] = ["filters", "kernel_size"]
dict_needed_params["Conv3D"] = ["filters", "kernel_size"]
dict_needed_params["Conv3DTranspose"] = ["filters", "kernel_size"]
dict_needed_params["ConvLSTM1D"] = ["filters", "kernel_size"]
dict_needed_params["ConvLSTM2D"] = ["filters", "kernel_size"]
dict_needed_params["ConvLSTM3D"] = ["filters", "kernel_size"]
dict_needed_params["Convolution1D"] = ["filters", "kernel_size"]
dict_needed_params["Convolution1DTranspose"] = ["filters", "kernel_size"]
dict_needed_params["Convolution2D"] = ["filters", "kernel_size"]
dict_needed_params["Convolution2DTranspose"] = ["filters", "kernel_size"]
dict_needed_params["Convolution3D"] = ["filters", "kernel_size"]
dict_needed_params["Convolution3DTranspose"] = ["filters", "kernel_size"]
dict_needed_params["Cropping1D"] = []
dict_needed_params["Cropping2D"] = []
dict_needed_params["Cropping3D"] = []
dict_needed_params["Dense"] = ["units"]
dict_needed_params["DepthwiseConv1D"] = ["kernel_size"]
dict_needed_params["DepthwiseConv2D"] = ["kernel_size"]
dict_needed_params["Discretization"] = []
dict_needed_params["Dot"] = ["axes"]
dict_needed_params["Dropout"] = ["rate"]
dict_needed_params["ELU"] = []
dict_needed_params["EinsumDense"] = ["equation", "output_shape"]
dict_needed_params["Embedding"] = ["input_dim", "output_dim"]

dict_needed_params["Embedding"] = ["input_dim", "output_dim"]
dict_needed_params["Flatten"] = []
dict_needed_params["GRU"] = ["units"]
dict_needed_params["GRUCell"] = ["units"]
dict_needed_params["GaussianDropout"] = ["rate"]
dict_needed_params["GaussianNoise"] = ["stddev"]
dict_needed_params["GlobalAveragePooling1D"] = []
dict_needed_params["GlobalAveragePooling2D"] = []
dict_needed_params["GlobalAveragePooling3D"] = []
dict_needed_params["GlobalAvgPool1D"] = []
dict_needed_params["GlobalAvgPool2D"] = []
dict_needed_params["GlobalAvgPool3D"] = []
dict_needed_params["GlobalMaxPool1D"] = []
dict_needed_params["GlobalMaxPool2D"] = []
dict_needed_params["GlobalMaxPool3D"] = []
dict_needed_params["GlobalMaxPooling1D"] = []
dict_needed_params["GlobalMaxPooling2D"] = []
dict_needed_params["GlobalMaxPooling3D"] = []
dict_needed_params["Hashing"] = ["num_bins"]
dict_needed_params["Input"] = []
dict_needed_params["InputLayer"] = []
dict_needed_params["InputSpec"] = []
dict_needed_params["IntegerLookup"] = []
dict_needed_params["LSTM"] = ["units"]
dict_needed_params["LSTMCell"] = ["units"]

dict_needed_params["Lambda"] = ["function"]
dict_needed_params["Layer"] = []
dict_needed_params["LayerNormalization"] = []
dict_needed_params["LeakyReLU"] = []
dict_needed_params["LocallyConnected1D"] = ["filters", "kernel_size"]
dict_needed_params["LocallyConnected2D"] = ["filters", "kernel_size"]
dict_needed_params["Masking"] = []
dict_needed_params["MaxPool1D"] = []
dict_needed_params["MaxPool2D"] = []
dict_needed_params["MaxPool3D"] = []
dict_needed_params["MaxPooling1D"] = []
dict_needed_params["MaxPooling2D"] = []
dict_needed_params["MaxPooling3D"] = []
dict_needed_params["Maximum"] = []
dict_needed_params["Minimum"] = []

dict_needed_params["MultiHeadAttention"] = ["num_heads", "key_dim"]
dict_needed_params["Multiply"] = []
dict_needed_params["Normalization"] = []
dict_needed_params["Out"] = []
dict_needed_params["PReLU"] = []
dict_needed_params["Permute"] = ["dims"]
dict_needed_params["RandomBrightness"] = ["factor"]
dict_needed_params["RandomContrast"] = ["factor"]
dict_needed_params["RandomCrop"] = ["height", "width"]
dict_needed_params["RandomFlip"] = []
dict_needed_params["RandomHeight"] = ["factor"]
dict_needed_params["RandomRotation"] = ["factor"]
dict_needed_params["RandomTranslation"] = ["height_factor", "width_factor"]
dict_needed_params["RandomWidth"] = ["factor"]
dict_needed_params["RandomZoom"] = ["height_factor"]
dict_needed_params["ReLU"] = []
dict_needed_params["RepeatVector"] = ["n"]
dict_needed_params["Rescaling"] = ["scale"]
dict_needed_params["Reshape"] = ["target_shape"]
dict_needed_params["Resizing"] = ["height", "width"]

dict_needed_params["SeparableConv1D"] = ["filters", "kernel_size"]
dict_needed_params["SeparableConv2D"] = ["filters", "kernel_size"]
dict_needed_params["SeparableConvolution1D"] = ["filters", "kernel_size"]
dict_needed_params["SeparableConvolution2D"] = ["filters", "kernel_size"]
dict_needed_params["SimpleRNN"] = ["units"]
dict_needed_params["SimpleRNNCell"] = ["units"]
dict_needed_params["Softmax"] = []
dict_needed_params["SpatialDropout1D"] = ["rate"]
dict_needed_params["SpatialDropout2D"] = ["rate"]
dict_needed_params["SpatialDropout3D"] = ["rate"]
dict_needed_params["StackedRNNCells"] = []
dict_needed_params["StringLookup"] = []
dict_needed_params["Subtract"] = []
dict_needed_params["TextVectorization"] = []
dict_needed_params["ThresholdedReLU"] = []
dict_needed_params["UnitNormalization"] = []
dict_needed_params["UpSampling1D"] = []
dict_needed_params["UpSampling2D"] = []
dict_needed_params["UpSampling3D"] = []
dict_needed_params["ZeroPadding1D"] = []
dict_needed_params["ZeroPadding2D"] = []
dict_needed_params["ZeroPadding3D"] = []
dict_needed_params["add"] = ["inputs"]
dict_needed_params["average"] = ["inputs"]
dict_needed_params["concatenate"] = ["inputs"]
dict_needed_params["maximum"] = ["inputs"]
dict_needed_params["minimum"] = ["inputs"]
dict_needed_params["multiply"] = ["inputs"]


dict_params_layers = {}
dict_params_layers["AbstractRNNCell"] = layers.AbstractRNNCell().get_config().keys()
dict_params_layers["Activation"] = layers.Activation(activation="relu").get_config().keys()
dict_params_layers["ActivityRegularization"] = layers.ActivityRegularization().get_config().keys()
dict_params_layers["Add"] = layers.Add().get_config().keys()
dict_params_layers["AdditiveAttention"] = layers.AdditiveAttention().get_config().keys()
dict_params_layers["AlphaDropout"] = layers.AlphaDropout(rate=0.1).get_config().keys()
dict_params_layers["Attention"] = layers.Attention().get_config().keys()
dict_params_layers["Average"] = layers.Average().get_config().keys()
dict_params_layers["AveragePooling1D"] = layers.AveragePooling1D().get_config().keys()
dict_params_layers["AveragePooling2D"] = layers.AveragePooling2D().get_config().keys()
dict_params_layers["AveragePooling3D"] = layers.AveragePooling3D().get_config().keys()
dict_params_layers["AvgPool1D"] = layers.AvgPool1D().get_config().keys()
dict_params_layers["AvgPool2D"] = layers.AvgPool2D().get_config().keys()
dict_params_layers["AvgPool3D"] = layers.AvgPool3D().get_config().keys()
dict_params_layers["BatchNormalization"] = layers.BatchNormalization().get_config().keys()
dict_params_layers["CategoryEncoding"] = (
    layers.CategoryEncoding(num_tokens=4, output_mode="one_hot").get_config().keys()
)
dict_params_layers["CenterCrop"] = layers.CenterCrop(height=[0, 1], width=[0, 1]).get_config().keys()
dict_params_layers["Concatenate"] = layers.Concatenate().get_config().keys()
dict_params_layers["Conv1D"] = layers.Conv1D(filters=1, kernel_size=1).get_config().keys()
dict_params_layers["Conv1DTranspose"] = layers.Conv1DTranspose(filters=1, kernel_size=1).get_config().keys()
dict_params_layers["Conv2D"] = layers.Conv2D(filters=1, kernel_size=1).get_config().keys()
dict_params_layers["Conv2DTranspose"] = layers.Conv2DTranspose(filters=1, kernel_size=1).get_config().keys()
dict_params_layers["Conv3D"] = layers.Conv3D(filters=1, kernel_size=1).get_config().keys()
dict_params_layers["Conv3DTranspose"] = layers.Conv3DTranspose(filters=1, kernel_size=1).get_config().keys()
dict_params_layers["ConvLSTM1D"] = layers.ConvLSTM1D(filters=1, kernel_size=1).get_config().keys()
dict_params_layers["ConvLSTM2D"] = layers.ConvLSTM2D(filters=1, kernel_size=1).get_config().keys()
dict_params_layers["ConvLSTM3D"] = layers.ConvLSTM3D(filters=1, kernel_size=1).get_config().keys()
dict_params_layers["Convolution1D"] = layers.Convolution1D(filters=1, kernel_size=1).get_config().keys()
dict_params_layers["Convolution1DTranspose"] = (
    layers.Convolution1DTranspose(filters=1, kernel_size=1).get_config().keys()
)
dict_params_layers["Convolution2D"] = layers.Convolution2D(filters=1, kernel_size=1).get_config().keys()
dict_params_layers["Convolution2DTranspose"] = (
    layers.Convolution2DTranspose(filters=1, kernel_size=1).get_config().keys()
)
dict_params_layers["Convolution3D"] = layers.Convolution3D(filters=1, kernel_size=1).get_config().keys()
dict_params_layers["Convolution3DTranspose"] = (
    layers.Convolution3DTranspose(filters=1, kernel_size=1).get_config().keys()
)
dict_params_layers["Cropping1D"] = layers.Cropping1D().get_config().keys()
dict_params_layers["Cropping2D"] = layers.Cropping2D().get_config().keys()
dict_params_layers["Cropping3D"] = layers.Cropping3D().get_config().keys()
dict_params_layers["Dense"] = layers.Dense(units=1).get_config().keys()
dict_params_layers["DepthwiseConv1D"] = layers.DepthwiseConv1D(kernel_size=1).get_config().keys()
dict_params_layers["DepthwiseConv2D"] = layers.DepthwiseConv2D(kernel_size=1).get_config().keys()
dict_params_layers["Discretization"] = layers.Discretization().get_config().keys()
dict_params_layers["Dot"] = layers.Dot(axes=(1, 2)).get_config().keys()
dict_params_layers["Dropout"] = layers.Dropout(rate=0.1).get_config().keys()
dict_params_layers["ELU"] = layers.ELU().get_config().keys()
dict_params_layers["Embedding"] = layers.Embedding(input_dim=4, output_dim=1).get_config().keys()
dict_params_layers["Flatten"] = layers.Flatten().get_config().keys()
dict_params_layers["GRU"] = layers.GRU(units=4).get_config().keys()
dict_params_layers["GRUCell"] = layers.GRUCell(units=4).get_config().keys()
dict_params_layers["GaussianDropout"] = layers.GaussianDropout(rate=0.1).get_config().keys()
dict_params_layers["GaussianNoise"] = layers.GaussianNoise(stddev=1).get_config().keys()
dict_params_layers["GlobalAveragePooling1D"] = layers.GlobalAveragePooling1D().get_config().keys()
dict_params_layers["GlobalAveragePooling2D"] = layers.GlobalAveragePooling2D().get_config().keys()
dict_params_layers["GlobalAveragePooling3D"] = layers.GlobalAveragePooling3D().get_config().keys()
dict_params_layers["GlobalAvgPool1D"] = layers.GlobalAvgPool1D().get_config().keys()
dict_params_layers["GlobalAvgPool2D"] = layers.GlobalAvgPool2D().get_config().keys()
dict_params_layers["GlobalAvgPool3D"] = layers.GlobalAvgPool3D().get_config().keys()
dict_params_layers["GlobalMaxPool1D"] = layers.GlobalMaxPool1D().get_config().keys()
dict_params_layers["GlobalMaxPool2D"] = layers.GlobalMaxPool2D().get_config().keys()
dict_params_layers["GlobalMaxPool3D"] = layers.GlobalMaxPool3D().get_config().keys()
dict_params_layers["GlobalMaxPooling1D"] = layers.GlobalMaxPooling1D().get_config().keys()
dict_params_layers["GlobalMaxPooling2D"] = layers.GlobalMaxPooling2D().get_config().keys()
dict_params_layers["GlobalMaxPooling3D"] = layers.GlobalMaxPooling3D().get_config().keys()
dict_params_layers["Hashing"] = layers.Hashing(num_bins=1).get_config().keys()
dict_params_layers["InputLayer"] = layers.InputLayer().get_config().keys()
dict_params_layers["InputSpec"] = layers.InputSpec().get_config().keys()
dict_params_layers["IntegerLookup"] = layers.IntegerLookup().get_config().keys()
dict_params_layers["LSTM"] = layers.LSTM(units=1).get_config().keys()

dict_params_layers["LSTMCell"] = layers.LSTMCell(units=1).get_config().keys()
dict_params_layers["Lambda"] = layers.Lambda(function=lambda x: x * scale).get_config().keys()
dict_params_layers["Layer"] = layers.Layer().get_config().keys()
dict_params_layers["LayerNormalization"] = layers.LayerNormalization().get_config().keys()
dict_params_layers["LeakyReLU"] = layers.LeakyReLU().get_config().keys()
dict_params_layers["LocallyConnected1D"] = layers.LocallyConnected1D(filters=1, kernel_size=1).get_config().keys()
dict_params_layers["LocallyConnected2D"] = layers.LocallyConnected2D(filters=1, kernel_size=1).get_config().keys()
dict_params_layers["Masking"] = layers.Masking().get_config().keys()
dict_params_layers["MaxPool1D"] = layers.MaxPool1D().get_config().keys()
dict_params_layers["MaxPool2D"] = layers.MaxPool2D().get_config().keys()
dict_params_layers["MaxPool3D"] = layers.MaxPool3D().get_config().keys()
dict_params_layers["MaxPooling1D"] = layers.MaxPooling1D().get_config().keys()
dict_params_layers["MaxPooling2D"] = layers.MaxPooling2D().get_config().keys()
dict_params_layers["MaxPooling3D"] = layers.MaxPooling3D().get_config().keys()
dict_params_layers["Maximum"] = layers.Maximum().get_config().keys()
dict_params_layers["Minimum"] = layers.Minimum().get_config().keys()
dict_params_layers["MultiHeadAttention"] = layers.MultiHeadAttention(num_heads=2, key_dim=2).get_config().keys()
dict_params_layers["Multiply"] = layers.Multiply().get_config().keys()
dict_params_layers["Normalization"] = layers.Normalization().get_config().keys()
dict_params_layers["PReLU"] = layers.PReLU().get_config().keys()
dict_params_layers["Permute"] = layers.Permute(dims=(2, 1)).get_config().keys()
dict_params_layers["RandomBrightness"] = layers.RandomBrightness(factor=1).get_config().keys()
dict_params_layers["RandomContrast"] = layers.RandomContrast(factor=1).get_config().keys()
dict_params_layers["RandomCrop"] = layers.RandomCrop(height=[0, 1], width=[0, 1]).get_config().keys()
dict_params_layers["RandomFlip"] = layers.RandomFlip().get_config().keys()
dict_params_layers["RandomHeight"] = layers.RandomHeight(factor=1).get_config().keys()
dict_params_layers["RandomRotation"] = layers.RandomRotation(factor=1).get_config().keys()
dict_params_layers["RandomTranslation"] = layers.RandomTranslation(width_factor=1, height_factor=1).get_config().keys()
dict_params_layers["RandomWidth"] = layers.RandomWidth(factor=1).get_config().keys()
dict_params_layers["RandomZoom"] = layers.RandomZoom(height_factor=1).get_config().keys()
dict_params_layers["ReLU"] = layers.ReLU().get_config().keys()
dict_params_layers["RepeatVector"] = layers.RepeatVector(n=2).get_config().keys()
dict_params_layers["Rescaling"] = layers.Rescaling(scale=1).get_config().keys()
dict_params_layers["Reshape"] = layers.Reshape(target_shape=(3, 4)).get_config().keys()
dict_params_layers["Resizing"] = layers.Resizing(height=1, width=1,).get_config().keys()
dict_params_layers["SeparableConv1D"] = layers.SeparableConv1D(filters=1, kernel_size=1).get_config().keys()
dict_params_layers["SeparableConv2D"] = layers.SeparableConv2D(filters=1, kernel_size=1).get_config().keys()
dict_params_layers["SeparableConvolution1D"] = (
    layers.SeparableConvolution1D(filters=1, kernel_size=1).get_config().keys()
)
dict_params_layers["SeparableConvolution2D"] = (
    layers.SeparableConvolution2D(filters=1, kernel_size=1).get_config().keys()
)
dict_params_layers["SimpleRNN"] = layers.SimpleRNN(units=1).get_config().keys()
dict_params_layers["SimpleRNNCell"] = layers.SimpleRNNCell(units=1).get_config().keys()
dict_params_layers["Softmax"] = layers.Softmax().get_config().keys()
dict_params_layers["SpatialDropout1D"] = layers.SpatialDropout1D(rate=0.1).get_config().keys()
dict_params_layers["SpatialDropout2D"] = layers.SpatialDropout2D(rate=0.1).get_config().keys()
dict_params_layers["SpatialDropout3D"] = layers.SpatialDropout3D(rate=0.1).get_config().keys()
dict_params_layers["StringLookup"] = layers.StringLookup().get_config().keys()
dict_params_layers["Subtract"] = layers.Subtract().get_config().keys()
dict_params_layers["TextVectorization"] = layers.TextVectorization().get_config().keys()
dict_params_layers["ThresholdedReLU"] = layers.ThresholdedReLU().get_config().keys()
dict_params_layers["UnitNormalization"] = layers.UnitNormalization().get_config().keys()
dict_params_layers["UpSampling1D"] = layers.UpSampling1D().get_config().keys()
dict_params_layers["UpSampling2D"] = layers.UpSampling2D().get_config().keys()
dict_params_layers["UpSampling3D"] = layers.UpSampling3D().get_config().keys()
dict_params_layers["ZeroPadding1D"] = layers.ZeroPadding1D().get_config().keys()
dict_params_layers["ZeroPadding2D"] = layers.ZeroPadding2D().get_config().keys()
dict_params_layers["ZeroPadding3D"] = layers.ZeroPadding3D().get_config().keys()
dict_params_layers["add"] = []
dict_params_layers["average"] = []
dict_params_layers["concatenate"] = []
dict_params_layers["maximum"] = []
dict_params_layers["minimum"] = []
dict_params_layers["multiply"] = []

# ******************************************************************************************************************** #
# Main
if __name__ == "__main__":
    pass
