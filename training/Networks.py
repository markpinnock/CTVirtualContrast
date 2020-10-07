import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

from utils.Transformation import affineTransformation


def dnResNetBlock(nc, inputlayer, pool_strides):
    conv1 = keras.layers.Conv3D(nc, (3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu')(inputlayer)
    BN1 = keras.layers.BatchNormalization()(conv1)
    conv2 = keras.layers.Conv3D(nc, (3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu')(BN1)
    BN2 = keras.layers.BatchNormalization()(conv2)
    pool = keras.layers.MaxPool3D((2, 2, 2), strides=pool_strides, padding='same')(conv2)
    return BN2, pool


def upResNetBlock(nc, inputlayer, skip, tconv_strides):
    tconv = keras.layers.Conv3DTranspose(nc, (3, 3, 3), strides=tconv_strides, padding='same', activation='relu')(inputlayer)
    BN1 = keras.layers.BatchNormalization()(tconv)
    concat = keras.layers.concatenate([BN1, skip], axis=4)
    conv1 = keras.layers.Conv3D(nc, (3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu')(concat)
    BN2 = keras.layers.BatchNormalization()(conv1)
    conv2 = keras.layers.Conv3D(nc, (3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu')(BN2)
    BN3 = keras.layers.BatchNormalization()(conv2)
    return BN3


def UNetGen(mb_size):
    input_layer = keras.layers.Input(shape=(256, 256, 3, 1, ))

    skip1, dnres1 = dnResNetBlock(16, input_layer, (2, 2, 1))
    skip2, dnres2 = dnResNetBlock(32, dnres1, (2, 2, 1))
    skip3, dnres3 = dnResNetBlock(64, dnres2, (2, 2, 1))
    skip4, dnres4 = dnResNetBlock(128, dnres3, (2, 2, 1))

    dn5 = keras.layers.Conv3D(256, (3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu')(dnres4)
    BN = keras.layers.BatchNormalization()(dn5)

    upres4 = upResNetBlock(128, BN, skip4, (2, 2, 1))
    upres3 = upResNetBlock(64, upres4, skip3, (2, 2, 1))
    upres2 = upResNetBlock(32, upres3, skip2, (2, 2, 1))
    upres1 = upResNetBlock(16, upres2, skip1, (2, 2, 1))

    output_layer = keras.layers.Conv3D(1, (1, 1, 1), strides=(1, 1, 1), padding='same', activation='relu')(upres1)

    return keras.Model(inputs=input_layer, outputs=output_layer)
