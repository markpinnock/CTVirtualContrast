import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

from Layers import DownBlock, UpBlock


class UNet(keras.Model):

    """ Implements U-Net
        - nc: number of channels in first layer
        - optimiser: of type e.g. keras.optimizers.Adam """
    
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
        self.out = keras.layers.Conv3D(1, (1, 1, 1), strides=(1, 1, 1), padding='same', activation='linear')

    def call(self, x):
        h, skip1 = self.down1(x)
        h, skip2 = self.down2(h)
        h, skip3 = self.down3(h)
        h = self.down4(h)
        h = self.up1(h, skip3)
        h = self.up2(h, skip2)
        h = self.up3(h, skip1)
        return self.out(h)
    
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
