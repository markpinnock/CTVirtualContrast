import numpy as np
import sys
import tensorflow as tf
import tensorflow.keras as keras

sys.path.append("..")

from Layers import DownBlock, UpBlock
from utils.Losses import FocalLoss, FocalMetric


class UNet(keras.Model):

    """ Implements U-Net
        - nc: number of channels in first layer
        - lambda_: hyperparameter for focal loss on range [0, 1]
        - optimiser: of type e.g. keras.optimizers.Adam """
    
    def __init__(self, nc, lambda_, optimiser):
        super(UNet, self).__init__(self)
        self.optimiser = optimiser
        # self.loss = keras.losses.MeanSquaredError()
        self.loss = FocalLoss(lambda_)
        # self.metric = keras.metrics.MeanSquaredError()
        self.metric = FocalMetric(lambda_)

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
    
    def compile(self, optimiser, loss, metrics):
        raise NotImplementedError
    
    def train_step(self, data):
        imgs, labels, segs = data

        with tf.GradientTape() as tape:
            predictions = self(imgs, training=True)
            loss = self.loss(labels, predictions, segs)
        
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimiser.apply_gradients(zip(gradients, self.trainable_variables))
        self.metric.update_state(labels, predictions, segs)
    
    def test_step(self, data):
        imgs, labels = data
        predictions = self(imgs, training=False)
        self.metric.update_state(labels, predictions)
