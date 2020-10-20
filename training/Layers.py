import numpy as np
import tensorflow as tf
import tensorflow.keras as keras


class DownBlock(keras.layers.Layer):

    """ Implements encoding block for U-Net
        - nc: number of channels in block
        - pool_strides: number of pooling strides e.g. (2, 2, 1) """

    def __init__(self, nc, pool_strides):
        super(DownBlock, self).__init__(self)

        self.conv1 = keras.layers.Conv3D(nc, (3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu')
        self.conv2 = keras.layers.Conv3D(nc, (3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu')
        self.pool = keras.layers.MaxPool3D((2, 2, 2), strides=pool_strides, padding='same')
        # Consider group normalisation
        # Consider pool -> conv3
    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return self.pool(x), x


class UpBlock(keras.layers.Layer):

    """ Implements encoding block for U-Net
        - nc: number of channels in block
        - tconv_strides: number of transpose conv  strides e.g. (2x2x1) """
    
    def __init__(self, nc, tconv_strides):
        super(UpBlock, self).__init__(self)

        self.tconv = keras.layers.Conv3DTranspose(nc, (2, 2, 2), strides=tconv_strides, padding='same', activation='relu')
        self.conv1 = keras.layers.Conv3D(nc, (3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu')
        self.conv2 = keras.layers.Conv3D(nc, (3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu')
    
    def call(self, x, skip):
        x = self.tconv(x)
        x = keras.layers.concatenate([x, skip], axis=4)
        x = self.conv1(x)
        return self.conv2(x)