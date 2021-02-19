import matplotlib.pyplot as plt
import numpy as np
import sys
import tensorflow as tf
import tensorflow.keras as keras

sys.path.append("..")

from networks.Layers import DownBlock, UpBlock
from utils.Losses import FocalLoss, FocalMetric, DiceLoss, DiceMetric
from utils.UNetAug import affine_transformation, TransMatGen
from utils.Transformation import affineTransformation


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
        self.bn = config["HYPERPARAMS"]["BATCHNORM"]

        # Data augmentation class
        if config["HYPERPARAMS"]["AUGMENT"]:
            self.DataAug = TransMatGen()
        else:
            self.DataAug = None

        # TODO: arbitrary img dims and layers
        self.down1 = DownBlock(nc, (2, 2, 2), self.bn)
        self.down2 = DownBlock(nc * 2, (2, 2, 2), self.bn)
        self.down3 = DownBlock(nc * 4, (2, 2, 1), self.bn)
        self.down4 = keras.layers.Conv3D(nc * 8, (3, 3, 3), strides=(1, 1, 1), padding='same', activation='linear')
        if self.bn: self.batchnorm = keras.layers.BatchNormalization()
        self.up1 = UpBlock(nc * 4, (2, 2, 1), self.bn)
        self.up2 = UpBlock(nc * 2, (2, 2, 2), self.bn)
        self.up3 = UpBlock(nc, (2, 2, 2), self.bn)
        self.out = keras.layers.Conv3D(1, (1, 1, 1), strides=(1, 1, 1), padding='same', activation='linear')

    def call(self, x, training):
        h, skip1 = self.down1(x, training)
        h, skip2 = self.down2(h, training)
        h, skip3 = self.down3(h, training)

        if self.bn:
            h = tf.nn.relu(self.batchnorm(self.down4(h), training))
        else:
            h = tf.nn.relu(self.down4(h))

        h = self.up1(h, skip3, training)
        h = self.up2(h, skip2, training)
        h = self.up3(h, skip1, training)
        
        return self.out(h)
    
    def compile(self, optimiser, loss, metrics):
        raise NotImplementedError
    
    @tf.function
    def train_step(self, data, transformed=0):
        NCE, ACE, seg, _ = data

        if self.DataAug and not transformed:
            trans_mat = self.DataAug.transMatGen(NCE.shape[0])
            NCE = affine_transformation(NCE, trans_mat)
            ACE = affine_transformation(ACE, trans_mat)
            seg = affine_transformation(seg, trans_mat)

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

#-------------------------------------------------------------------------

class CropUNet(UNet):

    def __init__(self, config):
        super().__init__(config)
        self.img_dims = config["EXPT"]["IMG_DIMS"]

    def crop_ROI(self, source, target, mask, coords):

        """ Crops images to ROI centred around coords """

        CROP_HEIGHT = self.img_dims[1]
        CROP_WIDTH = self.img_dims[0]
        MB_SIZE = source.shape[0]
        XY_DIMS = source.shape[1:3]
        IMG_DEPTH = source.shape[3]

        # NB: note x and y coords are swapped!
        y_coord = tf.reshape(tf.cast(coords[:, 0], tf.int32), [MB_SIZE, 1, 1, 1])
        x_coord = tf.reshape(tf.cast(coords[:, 1], tf.int32), [MB_SIZE, 1, 1, 1])

        N, X, Y, Z = tf.meshgrid(tf.range(MB_SIZE), tf.range(CROP_WIDTH, dtype=tf.int32), tf.range(CROP_HEIGHT, dtype=tf.int32), tf.range(IMG_DEPTH), indexing='ij')
        X = X + x_coord - CROP_WIDTH // 2
        Y = Y + y_coord - CROP_HEIGHT // 2

        idx_grid = tf.stack([X, Y, Z], axis=-1)
        idx_grid = tf.reshape(idx_grid, [MB_SIZE, CROP_WIDTH * CROP_HEIGHT * IMG_DEPTH, 3])
        # TODO: prevent idx_grid extending past borders
        source = tf.gather_nd(source, idx_grid, batch_dims=1)
        source = tf.reshape(source, [MB_SIZE, CROP_HEIGHT, CROP_WIDTH, IMG_DEPTH, 1])
        target = tf.gather_nd(target, idx_grid, batch_dims=1)
        target = tf.reshape(target, [MB_SIZE, CROP_HEIGHT, CROP_WIDTH, IMG_DEPTH, 1])
        mask = tf.gather_nd(mask, idx_grid, batch_dims=1)
        mask = tf.reshape(mask, [MB_SIZE, CROP_HEIGHT, CROP_WIDTH, IMG_DEPTH, 1])

        return source, target, mask 

    @tf.function
    def train_step(self, data):
        source, target, mask, coords = data
        transformed = 0
    
        if self.DataAug:
            # TODO: transform coords as well as image
            trans_mat = self.DataAug.transMatGen(source.shape[0])
            source = affine_transformation(source, trans_mat)
            target = affine_transformation(target, trans_mat)
            mask = affine_transformation(mask, trans_mat)
            transformed = 1
        
        for i in range(coords.shape[1]):
            # Crop ROI
            source, target, mask = self.crop_ROI(source, target, mask, coords[:, i, :])
            super().train_step((source, target, mask, None), transformed)

    @tf.function
    def val_step(self, data):
        source, target, mask, coords = data
        
        for i in range(coords.shape[1]):
            # Crop ROI
            source, target, mask = self.crop_ROI(source, target, mask, coords[:, i, :])
            super().val_step((source, target, mask, None))
