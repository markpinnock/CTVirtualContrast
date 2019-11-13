import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

from utils.Transformation import affineTransformation


def dnResNetBlock(nc, inputlayer):
    conv1 = keras.layers.Conv3D(nc, (3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu')(inputlayer)
    conv2 = keras.layers.Conv3D(nc, (3, 3, 3), strides=(2, 2, 2), padding='same', activation='relu')(conv1)
    return conv2


def upResNetBlock(nc, inputlayer, skip):
    tconv = keras.layers.Conv3DTranspose(nc, (3, 3, 3), strides=(2, 2, 2), padding='same', activation='relu')(inputlayer)
    upconv = keras.layers.Conv3D(nc, (3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu')(tconv + skip)
    return upconv


def STNModule(inputlayer):
    conv1 = keras.layers.Conv3D(32, (3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu')(inputlayer)
    BN1 = tf.nn.relu(keras.layers.BatchNormalization()(conv1))
    conv2 = keras.layers.Conv3D(32, (3, 3, 3), strides=(2, 2, 2), padding='same', activation='relu')(BN1)
    BN2 = tf.nn.relu(keras.layers.BatchNormalization()(conv2))
    conv3 = keras.layers.Conv3D(32, (3, 3, 3), strides=(2, 2, 2), padding='same', activation='relu')(BN2)
    BN3 = tf.nn.relu(keras.layers.BatchNormalization()(conv3))
    flatlayer = keras.layers.Flatten()(BN3)
    dense1 = keras.layers.Dense(50, activation='relu')(flatlayer)
    dense2 = keras.layers.Dense(12, activation='relu')(dense1)
    return dense2


def UNetGen(mb_size):
    inputlayer = keras.layers.Input(shape=(128, 128, 8, 1, ))
    thetas = STNModule(inputlayer)
    trans_input = affineTransformation(inputlayer, thetas, mb_size)
    dnresnet1 = dnResNetBlock(32, trans_input)
    dnresnet2 = dnResNetBlock(64, dnresnet1)
    dnresnet3 = dnResNetBlock(128, dnresnet2)
    upresnet3 = upResNetBlock(64, dnresnet3, dnresnet2)
    upresnet2 = upResNetBlock(32, upresnet3, dnresnet1)
    upresnet1 = upResNetBlock(1, upresnet2, inputlayer)
    outputlayer = upresnet1
    return keras.Model(inputs=inputlayer, outputs=[outputlayer, thetas])

