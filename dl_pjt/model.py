from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import GlobalAveragePooling2D

import numpy as np


class Facial_Kepoints_Detect:
    def __init__(
        self,
        input_size,
        output_size,
        init_conv_filters,
        kernel_size=(3, 3),
        conv_stride=(1, 1),
        pool_size=(2, 2),
        pool_stride=(2, 2),
    ):
        last_size = int(input_size[0] / 4) * int(input_size[1] / 4)
        last_layer = last_size * init_conv_filters * 4
        self.__layers = Sequential(
            [
                Conv2D(
                    input_shape=input_size,
                    filters=init_conv_filters,
                    kernel_size=kernel_size,
                    strides=conv_stride,
                    padding="same",
                    activation="relu",
                ),
                BatchNormalization(),
                Conv2D(
                    filters=init_conv_filters * 2,
                    kernel_size=kernel_size,
                    strides=conv_stride,
                    padding="same",
                    activation="relu",
                ),
                BatchNormalization(),
                MaxPool2D(pool_size=pool_size, strides=pool_stride),
                Conv2D(
                    filters=init_conv_filters * 3,
                    kernel_size=kernel_size,
                    strides=conv_stride,
                    padding="same",
                    activation="relu",
                ),
                BatchNormalization(),
                Conv2D(
                    filters=init_conv_filters * 4,
                    kernel_size=kernel_size,
                    strides=conv_stride,
                    padding="same",
                    activation="relu",
                ),
                BatchNormalization(),
                MaxPool2D(pool_size=pool_size, strides=pool_stride),
                GlobalAveragePooling2D(),
                Dense(last_layer, activation="relu"),
                Dense(int(last_layer * 0.5), activation="relu"),
                Dense(output_size, activation="relu"),
            ]
        )
        self.__layers.summary()

    def get_model(self):
        return self.__layers

    def set_loaded_model(self, model_file_name):
        base_path = "models/"
        self.__layers.load_weights(base_path + model_file_name)
        return
