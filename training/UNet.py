import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

from utils.Transformation import affineTransformation


class DownBlock(keras.layers.Layer):
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


class UNet(keras.Model):
    def __init__(self, nc, optimiser):
        super(UNet, self).__init__(self)
        self.optimiser = optimiser
        self.loss = keras.losses.MeanSquaredError()
        self.metric = keras.metrics.MeanSquaredError()

        self.down1 = DownBlock(nc, (2, 2, 2))
        self.down2 = DownBlock(nc * 2, (2, 2, 2))
        self.down3 = DownBlock(nc * 4, (2, 2, 1))
        self.down4 = keras.layers.Conv3D(nc * 8, (3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu')
        self.up1 = UpBlock(nc * 4, (2, 2, 1))
        self.up2 = UpBlock(nc * 2, (2, 2, 2))
        self.up3 = UpBlock(nc, (2, 2, 2))
        self.out = keras.layers.Conv3D(1, (1, 1, 1), strides=(1, 1, 1), padding='same', activation='relu')

    def call(self, x):
        x, skip1 = self.down1(x)
        x, skip2 = self.down2(x)
        x, skip3 = self.down3(x)
        x = self.down4(x)
        x = self.up1(x, skip3)
        x = self.up2(x, skip2)
        x = self.up3(x, skip1)
        return self.out(x)
    
    def train_step(self, data):
        imgs, labels = data

        with tf.GradientTape() as tape:
            predictions = self(imgs, training=True)
            loss = self.loss(labels, predictions)
        
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimiser.apply_gradients(zip(gradients, self.trainable_variables))
        self.metric.update_state(labels, predictions)
    
    def test_step(self, data):
        imgs, labels = data
        predictions = self(imgs, training=False)
        self.metric.update_state(labels, predictions)
