import numpy as np
import sys
import tensorflow as tf
import tensorflow.keras as keras

sys.path.append("..")

from networks.Layers import DownBlock, UpBlock
from utils.Losses import FocalLoss, FocalMetric, DiceLoss, DiceMetric


class UNet(keras.Model):
    
    def __init__(self, config):
        super().__init__(name="UNet")
        self.optimiser = keras.optimizers.Adam(config["HYPERPARAMS"]["ETA"], 0.5, 0.999, name="opt")

        if config["HYPERPARAMS"]["MU"] > 0.0:
            self.loss = FocalLoss(config["HYPERPARAMS"]["MU"], loss_fn=config["HYPERPARAMS"]["LOSS"])
        elif config["HYPERPARAMS"]["MU"] == 0.0 and config["HYPERPARAMS"]["LOSS"] == "mae":
            self.loss = keras.losses.MeanAbsoluteError()
        elif config["HYPERPARAMS"]["MU"] == 0.0 and config["HYPERPARAMS"]["LOSS"] == "mse":
            self.loss = keras.losses.MeanSquaredError()
        else:
            raise ValueError

        self.metric = FocalMetric(loss_fn=config["HYPERPARAMS"]["LOSS"])
        nc = config["HYPERPARAMS"]["NF"]

        # TODO: arbitrary img dims and layers
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
    
    @tf.function
    def train_step(self, data):
        NCE, ACE, seg, _ = data

        with tf.GradientTape() as tape:
            pred = self(NCE, training=True)
            loss = self.loss(ACE, pred)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimiser.apply_gradients(zip(gradients, self.trainable_variables))
        self.metric.update_state(ACE, pred, seg)
    
    @tf.function
    def val_step(self, data):
        NCE, ACE, seg, _ = data
        
        pred = self(NCE, training=False)
        self.metric.update_state(ACE, pred, seg)
