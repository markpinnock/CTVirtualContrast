import numpy as np
import tensorflow as tf

from .Pix2Pix import Discriminator, Generator, HyperGenerator
from .STN import SpatialTransformer
from syntheticcontrast_v02.utils.augmentation import DiffAug, StdAug
from syntheticcontrast_v02.utils.losses import (
    minimax_D, minimax_G, L1, wasserstein_D, wasserstein_G, gradient_penalty, FocalLoss, FocalMetric)


#-------------------------------------------------------------------------
""" Wrapper for standard Pix2pix GAN """

class GAN(tf.keras.Model):

    def __init__(self, config, name="GAN"):
        super().__init__(name=name)
        self.initialiser = tf.keras.initializers.RandomNormal(0, 0.02)
        self.config = config
        self.lambda_ = config["hyperparameters"]["lambda"]
        self.mb_size = config["expt"]["mb_size"]

        if len(config["data"]["segs"]) > 0:
            self.d_in_ch = 3
        else:
            self.d_in_ch = 2

        self.img_dims = config["hyperparameters"]["img_dims"]

        # Set up augmentation
        if config["augmentation"]["type"] == "standard":
            self.Aug = StdAug(config=config)
        elif config["augmentation"]["type"] == "differentiable":
            self.Aug = DiffAug({"colour": True, "translation": True, "cutout": True})
        else:
            self.Aug = None

        # Initialise generator and discriminators
        self.generator_init(config["hyperparameters"])
        self.discriminator_init(config["hyperparameters"])

        # Spatial transformer if necessary
        if config["hyperparameters"]["stn_layers"] > 0:
            self.STN = SpatialTransformer(config=config)
        else:
            self.STN = None

    def generator_init(self, config):
        # Check generator output dims match input
        G_input_size = [1] + self.img_dims + [1]
        self.Generator = Generator(self.initialiser, config, name="generator")
        assert self.Generator.build_model(tf.zeros(G_input_size)) == G_input_size

    def discriminator_init(self, config):
        # Get discriminator patch size
        D_input_size = [1] + self.img_dims + [self.d_in_ch]
        self.Discriminator = Discriminator(self.initialiser, config, name="discriminator")
        self.patch_size = self.Discriminator.build_model(tf.zeros(D_input_size))

    def compile(self, g_optimiser, d_optimiser):
        self.g_optimiser = g_optimiser
        self.d_optimiser = d_optimiser

        self.d_loss = minimax_D
        self.g_loss = minimax_G
        
        if self.config["expt"]["focal"]:
            self.L1_loss = FocalLoss(self.config["hyperparameters"]["mu"], name="FocalLoss")
        else:
            self.L1_loss = L1

        # Set up metrics
        self.d_metric = tf.keras.metrics.Mean(name="d_metric")
        self.g_metric = tf.keras.metrics.Mean(name="g_metric")

        if len(self.config["data"]["segs"]) > 0:
            self.train_L1_metric = FocalMetric(name="train_L1")
            self.val_L1_metric = FocalMetric(name="val_L1")

        else:
            self.train_L1_metric = tf.keras.metrics.Mean(name="train_L1")
            self.val_L1_metric = tf.keras.metrics.Mean(name="val_L1")

        if self.STN:
            self.s_optimiser = tf.keras.optimizers.Adam(self.config["hyperparameters"]["stn_eta"])
    
    def summary(self):
        source = tf.keras.Input(shape=self.img_dims + [1])
        outputs = self.Generator.call(source)
        print("===========================================================")
        print("Generator")
        print("===========================================================")
        tf.keras.Model(inputs=source, outputs=outputs).summary()
        source = tf.keras.Input(shape=self.img_dims + [1])
        if self.STN: target = self.STN.call(source, source)
        outputs = self.Discriminator.call(tf.concat([source] * self.d_in_ch, axis=4))
        print("===========================================================")
        print("Discriminator")
        print("===========================================================")
        tf.keras.Model(inputs=source, outputs=outputs).summary()
    
    def generator_step(self, g_source, g_real_target, g_seg=None):

        """ Generator training """

        # Get gradients from discriminator predictions of generated fake images and update weights
        with tf.GradientTape() as g_tape:
            if self.STN:
                g_real_target, g_seg = self.STN(source=g_source, target=g_real_target, seg=g_seg, training=True)

            g_fake_target = self.Generator(g_source)

            # Calculate L1 before augmentation
            if g_seg is not None:
                g_L1 = self.L1_loss(g_real_target, g_fake_target, g_seg)
            else:
                g_L1 = self.L1_loss(g_real_target, g_fake_target)

            if g_seg is not None:
                self.train_L1_metric.update_state(g_real_target, g_fake_target, g_seg)
            else:
                self.train_L1_metric.update_state(g_L1)
            
            if self.Aug:
                imgs, g_seg = self.Aug(imgs=[g_source, g_fake_target], seg=g_seg)
                g_source, g_fake_target = imgs

            if g_seg is not None:
                g_fake_in = tf.concat([g_source, g_fake_target, g_seg], axis=4, name="g_fake_concat")
            else:
                g_fake_in = tf.concat([g_source, g_fake_target], axis=4, name="g_fake_concat")

            g_pred_fake = self.Discriminator(g_fake_in)
            g_loss = self.g_loss(g_pred_fake)
            g_total_loss = g_loss + self.lambda_ * g_L1

        if self.STN:
            gen_grads = len(self.STN.trainable_variables)
            g_grads = g_tape.gradient(g_total_loss, self.STN.trainable_variables + self.Generator.trainable_variables)
            self.s_optimiser.apply_gradients(zip(g_grads[0:gen_grads], self.STN.trainable_variables))
            self.g_optimiser.apply_gradients(zip(g_grads[gen_grads:], self.Generator.trainable_variables))
  
        else:
            g_grads = g_tape.gradient(g_total_loss, self.Generator.trainable_variables)
            self.g_optimiser.apply_gradients(zip(g_grads, self.Generator.trainable_variables))

        # Update metric
        self.g_metric.update_state(g_loss)

    def discriminator_step(self, d_source, d_real_target, d_fake_target, d_seg=None):

        """ Discriminator training """

        # Spatial transformer if required
        if self.STN:
            d_real_target, d_seg = self.STN(source=d_source, target=d_real_target, seg=d_seg, training=True)

        # DiffAug if required
        if self.Aug:
            imgs, d_seg = self.Aug(imgs=[d_source, d_real_target, d_fake_target], seg=d_seg)
            d_source, d_real_target, d_fake_target = imgs

        if d_seg is not None:
            d_fake_in = tf.concat([d_source, d_fake_target, d_seg], axis=4, name="d_fake_concat")
            d_real_in = tf.concat([d_source, d_real_target, d_seg], axis=4, name="d_real_concat")
        else:
            d_fake_in = tf.concat([d_source, d_fake_target], axis=4, name="d_fake_concat")
            d_real_in = tf.concat([d_source, d_real_target], axis=4, name="d_real_concat")

        # Get gradients from discriminator predictions and update weights
        with tf.GradientTape() as d_tape:
            d_pred_fake = self.Discriminator(d_fake_in)
            d_pred_real = self.Discriminator(d_real_in)
            d_loss = self.d_loss(d_pred_real, d_pred_fake)
        
        d_grads = d_tape.gradient(d_loss, self.Discriminator.trainable_variables)
        self.d_optimiser.apply_gradients(zip(d_grads, self.Discriminator.trainable_variables))

        # Update metrics
        self.d_metric.update_state(d_loss)
    
    @tf.function
    def train_step(self, source, target, seg=None, source_time=None, seg_time=None):

        """ Expects data in order 'source, target' or 'source, target, segmentations'"""

        # Select minibatch of images and masks for generator training
        g_source = source[0:self.mb_size, :, :, :, :]
        g_real_target = target[0:self.mb_size, :, :, :, :]

        if seg is not None:
            g_seg = seg[0:self.mb_size, :, :, :, :]
        else:
            g_seg = None

        # Select minibatch of real images and generate fake images for discriminator run
        # If not enough different images for both generator and discriminator, share minibatch
        if self.mb_size == source.shape[0]:
            d_source = source[0:self.mb_size, :, :, :, :]
            d_fake_target = self.Generator(d_source)
            d_real_target = target[0:self.mb_size, :, :, :, :]

            if seg is not None:
                d_seg = seg[0:self.mb_size, :, :, :, :]
            else:
                d_seg = None

        else:
            d_source = source[self.mb_size:, :, :, :, :]
            d_real_target = target[self.mb_size:, :, :, :, :]
            d_fake_target = self.Generator(d_source)

            if seg is not None:
                d_seg = seg[self.mb_size:, :, :, :, :]
            else:
                d_seg = None
        
        self.discriminator_step(d_source, d_real_target, d_fake_target, d_seg)
        self.generator_step(g_source, g_real_target, g_seg)


    @tf.function
    def test_step(self, source, target, seg=None):
        if self.STN:
            target, seg = self.STN(source=source, target=target, seg=seg, training=False)

        g_fake = self.Generator(source)
        g_L1 = L1(target, g_fake)

        if seg is not None:
            self.val_L1_metric.update_state(target, g_fake, seg)
        else:
            self.val_L1_metric.update_state(g_L1)


#-------------------------------------------------------------------------
""" Wrapper for Pix2pix GAN with HyperNetwork """

class HyperGAN(GAN):

    """ GAN class using HyperNetwork for generator """

    def __init__(self, config, name="HyperGAN"):
        super().__init__(config, name=name)

    def generator_init(self, config):
        # Check generator output dims match input
        G_input_size = [1] + config["img_dims"] + [1]
        self.Generator = HyperGenerator(self.initialiser, config, name="generator")
        assert self.Generator.build_model(tf.zeros(G_input_size)) == G_input_size

    def summary(self):
        source = tf.keras.Input(shape=self.img_dims + [1])
        outputs = self.Generator.call(source)
        print("===========================================================")
        print(f"Generator: {np.sum([np.prod(v.shape) for v in self.Generator.trainable_variables])}")
        print("===========================================================")
        tf.keras.Model(inputs=source, outputs=outputs).summary()
        source = tf.keras.Input(shape=self.img_dims + [1])
        if self.STN: target = self.STN.call(source, source)
        outputs = self.Discriminator.call(tf.concat([source] * self.d_in_ch, axis=4))
        print("===========================================================")
        vs = 0
        print(f"Discriminator: {np.sum([np.prod(v.shape) for v in self.Discriminator.trainable_variables])}")
        print("===========================================================")
        tf.keras.Model(inputs=source, outputs=outputs).summary()
