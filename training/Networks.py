import tensorflow as tf
import tensorflow.keras as keras

from Transformation import affineTransform


def dnResNetBlock(nc, inputlayer):
    conv1 = keras.layers.Conv3D(nc, (3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu')(inputlayer)
    conv2 = keras.layers.Conv3D(nc, (3, 3, 3), strides=(2, 2, 2), padding='same', activation='relu')(conv1)
    return conv2


def upResNetBlock(nc, inputlayer, skip):
    tconv = keras.layers.Conv3DTranspose(nc, (3, 3, 3), strides=(2, 2, 2), padding='same', activation='relu')(inputlayer)
    upconv = keras.layers.Conv3D(nc, (3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu')(tconv + skip)
    return upconv


def STNModule(inputlayer):
    conv1 = keras.layers.Conv3D(8, (3, 3, 3), strides=(2, 2, 2), padding='same', activation='relu')(inputlayer)
    conv2 = keras.layers.Conv3D(8, (3, 3, 3), strides=(2, 2, 2), padding='same', activation='relu')(conv1)
    flatlayer = keras.layers.Flatten()(conv2)
    dense1 = keras.layers.Dense(50, activation='relu')(flatlayer)
    dense2 = keras.layers.Dense(6, activation='relu')(dense1)
    return dense2


def UNetGen():
    inputlayer = keras.layers.Input(shape=(256, 256, 8, 1, ))
    transform = STNModule(inputlayer)
    dnresnet1 = dnResNetBlock(32, inputlayer)
    dnresnet2 = dnResNetBlock(64, dnresnet1)
    upresnet2 = upResNetBlock(32, dnresnet2, dnresnet1)
    upresnet1 = upResNetBlock(1, upresnet2, inputlayer)
    outputlayer = upresnet1
    return keras.Model(inputs=inputlayer, outputs=outputlayer)

