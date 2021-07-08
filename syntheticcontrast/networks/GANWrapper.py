import matplotlib.pyplot as plt
import sys
import tensorflow as tf

from .Pix2Pix import Discriminator, Generator
from .STN import SpatialTransformer
from utils.dataloader import DiffAug
from utils.losses_v01 import Loss, FocalMetric
from utils.losses_v02 import (
    minimax_D, minimax_G, L1, wasserstein_D, wasserstein_G, gradient_penalty)


#-------------------------------------------------------------------------
""" Wrapper for standard Pix2pix GAN """

class GAN(tf.keras.Model):

    """ GAN class
        - config: configuration json
        - GAN_type: 'original', 'least_square', 'wasserstein' or 'wasserstein-GP' """

    def __init__(self, config, GAN_type="original", name="GAN"):
        super().__init__(name=name)
        self.initialiser = tf.keras.initializers.RandomNormal(0, 0.02)
        self.lambda_ = config["HYPERPARAMS"]["LAMBDA"]

        if len(config["DATA"]["SEGS"]) > 0:
            self.d_in_ch = 3
        else:
            self.d_in_ch = 2

        self.img_dims = config["DATA"]["IMG_DIMS"]
        self.n_critic = config["HYPERPARAMS"]["N_CRITIC"]
        self.mb_size = config["HYPERPARAMS"]["MB_SIZE"]

        # Set up augmentation
        if config["HYPERPARAMS"]["AUGMENT"]:
            self.Aug = DiffAug({"colour": True, "translation": True, "cutout": True})
        else:
            self.Aug = None

        # Initialise generator and discriminators
        self.generator_init(config)
        self.discriminator_init(config)

        # Spatial transformer if necessary
        if config["HYPERPARAMS"]["STN_LAYERS"] > 0:
            self.STN = SpatialTransformer(config=config)
        else:
            self.STN = None

    def generator_init(self, config):
        # Check generator output dims match input
        G_input_size = [1] + config["DATA"]["IMG_DIMS"] + [1]
        self.Generator = Generator(self.initialiser, config, name="generator")
        assert self.Generator.build_model(tf.zeros(G_input_size)) == G_input_size

    def discriminator_init(self, config):
        # Get discriminator patch size
        D_input_size = [1] + config["DATA"]["IMG_DIMS"] + [self.d_in_ch]
        self.Discriminator = Discriminator(self.initialiser, config, name="discriminator")
        self.patch_size = self.Discriminator.build_model(tf.zeros(D_input_size))

    def compile(self, g_optimiser, d_optimiser, loss):
        self.g_optimiser = g_optimiser
        self.d_optimiser = d_optimiser
        self.loss_type = loss

        # Losses
        if loss == "minmax":
            self.d_loss = minimax_D
            self.g_loss = minimax_G
        elif loss == "wasserstein-GP":
            self.d_loss = wasserstein_D
            self.g_loss = wasserstein_G

        # Set up metrics
        self.d_metric = tf.keras.metrics.Mean(name="d_metric")
        self.g_metric = tf.keras.metrics.Mean(name="g_metric")
        self.train_L1_metric = tf.keras.metrics.Mean(name="train_L1")
        self.val_L1_metric = tf.keras.metrics.Mean(name="val_L1")
    
    def summary(self):
        source = tf.keras.Input(shape=self.img_dims + [1])
        outputs = self.Generator.call(source)
        print("===========================================================")
        print("Generator")
        print("===========================================================")
        tf.keras.Model(inputs=source, outputs=outputs).summary()
        source = tf.keras.Input(shape=self.img_dims + [1])
        if self.STN: target = self.STN.call(source, source)
        outputs = self.Discriminator.call(tf.concat([source, source], axis=4))
        print("===========================================================")
        print("Discriminator")
        print("===========================================================")
        tf.keras.Model(inputs=source, outputs=outputs).summary()
    
    @tf.function
    def train_step(self, source, target, seg=None):
        """ Expects data in order 'source, target' or 'source, target, segmentations'"""

        # Determine size of mb for each critic training run
        # (size of real_images = minibatch size * number of critic runs)
        g_mb = source.shape[0] // (1 + self.n_critic)
        if g_mb < 1: g_mb = source.shape[0]
        d_mb = g_mb * self.n_critic # TODO: TEST WITH N_CRITIC > 1

        # Select minibatch of images and masks for generator training
        g_source = source[0:g_mb, :, :, :, :]
        g_real_target = target[0:g_mb, :, :, :, :]
        if seg: g_seg = seg[0:g_mb, :, :, :, :]

        # Critic training loop
        for idx in range(self.n_critic):
            # Select minibatch of real images and generate fake images for critic run
            # If not enough different images for both generator and discriminator, share minibatch
            if g_mb == source.shape[0]:
                d_source = source[0:d_mb, :, :, :, :]
                d_fake_target = self.Generator(d_source)
                d_real_target = target[0:d_mb, :, :, :, :]
                if seg: d_seg = seg[0:d_mb, :, :, :, :]

            else:
                d_source = source[g_mb + idx * d_mb:g_mb + (idx + 1) * d_mb, :, :, :, :]
                d_real_target = target[g_mb + idx * d_mb:g_mb + (idx + 1) * d_mb, :, :, :, :]
                d_fake_target = self.Generator(d_source)
                if seg: d_seg = seg[g_mb + idx * d_mb:g_mb + (idx + 1) * d_mb, :, :, :, :]
            
            # Spatial transformer if required TODO: BN TRAINING
            if self.STN:
                if seg:
                    d_real_target, d_seg = self.STN(source=d_source, target=d_real_target, seg=d_seg, training=True)

                else:
                    d_real_target = self.STN(source=d_source, target=d_real_target, training=True)

            # DiffAug if required
            if self.Aug:
                imgs, _ = self.Aug.augment(imgs=[d_source, d_real_target, d_fake_target])
                d_source, d_real_target, d_fake_target = imgs

            d_fake_in = tf.concat([d_source, d_fake_target], axis=4, name="d_fake_concat")
            d_real_in = tf.concat([d_source, d_real_target], axis=4, name="d_real_concat")

            # Get gradients from critic predictions and update weights
            with tf.GradientTape() as d_tape:
                d_pred_fake = self.Discriminator(d_fake_in)
                d_pred_real = self.Discriminator(d_real_in)
                d_loss = self.d_loss(d_pred_real, d_pred_fake)

                # Gradient penalty if indicated
                if self.loss_type == "wasserstein-GP":
                    grad_penalty = gradient_penalty(d_real_target, d_fake_target, self.Discriminator)
                    d_loss += 10 * grad_penalty
            
            d_grads = d_tape.gradient(d_loss, self.Discriminator.trainable_variables)
            self.d_optimiser.apply_gradients(zip(d_grads, self.Discriminator.trainable_variables))

            # Update metrics
            self.d_metric.update_state(d_loss)

        # Generator training       
        # Get gradients from discriminator predictions of generated fake images and update weights
        with tf.GradientTape() as g_tape:
            if self.STN:
                g_real_target = self.STN(source=g_source, target=g_real_target, training=True)

            g_fake_target = self.Generator(g_source)

            # Calculate L1 before augmentation
            g_L1 = L1(g_real_target, g_fake_target)
            self.train_L1_metric.update_state(g_L1)
            
            if self.Aug:
                imgs, g_mask = self.Aug.augment(imgs=[g_source, g_fake_target])
                g_source, g_fake_target = imgs

            g_fake_in = tf.concat([g_source, g_fake_target], axis=4, name="g_fake_concat")

            g_pred_fake = self.Discriminator(g_fake_in)
            g_loss = self.g_loss(g_pred_fake)
            g_total_loss = g_loss + self.lambda_ * g_L1

        if self.STN:
            g_grads = g_tape.gradient(g_total_loss, self.Generator.trainable_variables + self.STN.trainable_variables)
            self.g_optimiser.apply_gradients(zip(g_grads, self.Generator.trainable_variables + self.STN.trainable_variables))
        
        else:
            g_grads = g_tape.gradient(g_total_loss, self.Generator.trainable_variables)
            self.g_optimiser.apply_gradients(zip(g_grads, self.Generator.trainable_variables))

        # Update metric
        self.g_metric.update_state(g_loss)

    @tf.function
    def val_step(self, source, target):
        target = self.STN(source=source, target=target, training=False)
        g_fake = self.Generator(source)
        g_L1 = L1(target, g_fake)
        self.val_L1_metric.update_state(g_L1)


#-------------------------------------------------------------------------
""" GAN acting on ROIs cropped around ROIs (Version 01) - inherits from base GAN """

class CropGAN_v01(GAN):

    def __init__(self, config, GAN_type="original", name="GAN"):
        super().__init__(config, GAN_type="original", name="GAN")

    def crop_ROI(self, source, target, mask, coords):

        """ Crops images to ROI centred around coords """

        # TODO: why is retracing occuring?
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

    def compile():
        self.g_optimiser = g_optimiser
        self.d_optimiser = d_optimiser
        self.loss_type = loss

        # Losses
        if loss == "minmax":
            self.d_loss = minimax_D
            self.g_loss = minimax_G
        elif loss == "wasserstein-GP":
            self.d_loss = wasserstein_D
            self.g_loss = wasserstein_G

        # Set up metrics
        self.discriminator_metric = keras.metrics.Mean(name="d_metric")
        self.g_metric = keras.metrics.Mean(name="g_metric")
        self.train_L1_metric = keras.metrics.Mean(name="train_L1")
        self.val_L1_metric = keras.metrics.Mean(name="val_L1") 
    
    @tf.function
    def train_step(self, data):
        source, target, mask, _ = data
        # Determine size of mb for each critic training run
        # (size of real_images = minibatch size * number of critic runs)

        g_mb = source.shape[0] // (1 + self.n_critic)
        if g_mb < 1: g_mb = source.shape[0]
        d_mb = g_mb * self.n_critic # TODO: TEST WITH N_CRITIC > 1

        # TODO: ADD NOISE TO LABELS AND/OR IMAGES

        # Select minibatch of images and masks for generator training
        g_source = source[0:g_mb, :, :, :, :]
        g_real_target = target[0:g_mb, :, :, :, :]
        g_mask = mask[0:g_mb, :, :, :, :]

        # Critic training loop
        for idx in range(self.n_critic):
            # Select minibatch of real images and generate fake images for critic run
            # If not enough different images for both generator and discriminator, share minibatch
            if g_mb == source.shape[0]:
                d_source = source[0:d_mb, :, :, :, :]
                d_fake_target = self.Generator(d_source)
                d_real_target = target[0:d_mb, :, :, :, :]
                d_mask = mask[0:d_mb, :, :, :, :]
            else:
                d_source = source[g_mb + idx * d_mb:g_mb + (idx + 1) * d_mb, :, :, :, :]
                d_real_target = target[g_mb + idx * d_mb:g_mb + (idx + 1) * d_mb, :, :, :, :]
                d_fake_target = self.Generator(d_source)
                d_mask = mask[g_mb + idx * d_mb:g_mb + (idx + 1) * d_mb, :, :, :, :]

            # DiffAug if required
            if self.Aug:
                imgs, d_mask = self.Aug.augment(imgs=[d_source, d_real_target, d_fake_target], seg=d_mask)
                d_source, d_real_target, d_fake_target = imgs

            if self.d_in_ch == 3:
                d_fake_in = tf.concat([d_source, d_fake_target, d_mask], axis=4, name="f_concat")
                d_real_in = tf.concat([d_source, d_real_target, d_mask], axis=4, name="r_concat")
            else:
                d_fake_in = tf.concat([d_source, d_fake_target], axis=4, name="f_concat")
                d_real_in = tf.concat([d_source, d_real_target], axis=4, name="r_concat")

            # Get gradients from critic predictions and update weights
            with tf.GradientTape() as d_tape:
                d_pred_fake = self.Discriminator(d_fake_in)
                d_pred_real = self.Discriminator(d_real_in)
                d_predictions = tf.concat([d_pred_fake, d_pred_real], axis=0)
                d_loss_1 = self.loss(d_labels[0:d_mb, ...], d_predictions[0:d_mb, ...]) # Fake
                d_loss_2 = self.loss(d_labels[d_mb:, ...], d_predictions[d_mb:, ...]) # Real
                d_loss = d_loss_1 + d_loss_2

                # Gradient penalty if indicated
                if self.loss_type == "wasserstein-GP":
                    grad_penalty = gradient_penalty(d_real_batch, d_fake_images, self.Discriminator)
                    d_loss += 10 * grad_penalty
            
            d_grads = d_tape.gradient(d_loss, self.Discriminator.trainable_variables)
            self.d_optimiser.apply_gradients(zip(d_grads, self.Discriminator.trainable_variables))

            # Update metrics
            self.discriminator_metric.update_state(d_loss)

        # Generator training       
        # TODO: ADD NOISE TO LABELS AND/OR IMAGES
        # Get gradients from discriminator predictions of generated fake images and update weights
        with tf.GradientTape() as g_tape:
            g_fake_target = self.Generator(g_source)

            # Calculate L1 before augmentation
            g_L1 = self.L1(g_real_target, g_fake_target, g_mask)
            self.train_L1_metric.update_state(g_real_target, g_fake_target, g_mask)
            
            if self.Aug:
                imgs, g_mask = self.Aug.augment(imgs=[g_source, g_fake_target], seg=g_mask)
                g_source, g_fake_target = imgs

            if self.d_in_ch == 3:
                g_fake_in = tf.concat([g_source, g_fake_target, g_mask], axis=4, name="f_concat")
            else:
                g_fake_in = tf.concat([g_source, g_fake_target], axis=4, name="f_concat")

            g_pred_fake = self.Discriminator(g_fake_in)
            g_loss = self.loss(g_labels, g_pred_fake)
            g_total_loss = g_loss + self.lambda_ * g_L1

        g_grads = g_tape.gradient(g_total_loss, self.Generator.trainable_variables)
        self.g_optimiser.apply_gradients(zip(g_grads, self.Generator.trainable_variables))

        # Update metric
        self.generator_metric.update_state(g_loss)

    @tf.function
    def val_step(self, data):
        source, target, mask, _ = data
        g_fake = self.Generator(source)
        self.val_L1_metric.update_state(target, g_fake, mask)

    @tf.function
    def train_step(self, data):
        source, target, mask, coords = data
        
        for i in range(coords.shape[1]):
            # Crop ROI
            source, target, mask = self.crop_ROI(source, target, mask, coords[:, i, :])
            super().train_step((source, target, mask, coords))

    @tf.function
    def val_step(self, data):
        source, target, mask, coords = data
        
        for i in range(coords.shape[1]):
            # Crop ROI
            source, target, mask = self.crop_ROI(source, target, mask, coords[:, i, :])
            super().val_step((source, target, mask, coords))
