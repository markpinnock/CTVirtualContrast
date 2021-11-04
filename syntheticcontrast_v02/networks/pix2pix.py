import numpy as np
import tensorflow as tf

from .models import Discriminator, Generator, HyperGenerator
from syntheticcontrast_v02.utils.augmentation import DiffAug, StdAug
from syntheticcontrast_v02.utils.losses import minimax_D, minimax_G, L1, FocalLoss, FocalMetric


#-------------------------------------------------------------------------
""" Wrapper for standard Pix2pix GAN """

class Pix2Pix(tf.keras.Model):

    def __init__(self, config, name="Pix2Pix"):
        super().__init__(name=name)
        self.initialiser = tf.keras.initializers.HeNormal()
        self.config = config
        self.lambda_ = config["hyperparameters"]["lambda"]
        self.mb_size = config["expt"]["mb_size"]
        self.d_in_ch = 2
        self.g_in_ch = 1
        self.img_dims = config["hyperparameters"]["img_dims"]

        if config["data"]["times"] is None:
            self.input_times = False
            assert config["hyperparameters"]["g_time_layers"] is None
            assert config["hyperparameters"]["d_time_layers"] is None
        else:
            self.input_times = True
            assert config["hyperparameters"]["g_time_layers"] is not None
            assert config["hyperparameters"]["d_time_layers"] is not None

        # Set up augmentation
        if config["augmentation"]["type"] == "standard":
            self.Aug = StdAug(config=config)
        elif config["augmentation"]["type"] == "differentiable":
            self.Aug = DiffAug(config=config["augmentation"])
        else:
            self.Aug = None

        # Initialise generator and discriminators
        self.generator_init(config["hyperparameters"])
        self.discriminator_init(config["hyperparameters"])

    def generator_init(self, config):
        # Check generator output dims match input
        G_input_size = [1] + self.img_dims + [1]
        G_output_size = [1] + self.img_dims + [1]
        self.Generator = Generator(self.initialiser, config, name="generator")

        if self.input_times:
            assert self.Generator.build_model(tf.zeros(G_input_size), tf.zeros(1)) == G_output_size, f"{self.Generator.build_model(tf.zeros(G_input_size), tf.zeros(1))} vs {G_input_size}"
        else:
            assert self.Generator.build_model(tf.zeros(G_input_size)) == G_output_size, f"{self.Generator.build_model(tf.zeros(G_input_size))} vs {G_input_size}"

    def discriminator_init(self, config):
        # Get discriminator patch size
        D_input_size = [2] + self.img_dims + [2]
        self.Discriminator = Discriminator(self.initialiser, config, name="discriminator")

        if self.input_times:
            self.patch_size = self.Discriminator.build_model(tf.zeros(D_input_size), tf.zeros(1))
        else:
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

    def summary(self):
        source = tf.keras.Input(shape=self.img_dims + [1])

        if self.input_times:
            outputs = self.Generator.call(source, tf.zeros(1))
        else:
            outputs = self.Generator.call(source)

        print("===========================================================")
        print("Generator")
        print("===========================================================")
        tf.keras.Model(inputs=source, outputs=outputs).summary()
        source = tf.keras.Input(shape=self.img_dims + [2])
        
        if self.input_times:
            outputs = self.Discriminator.call(source, tf.zeros(1))
        else:
            outputs = self.Discriminator.call(source)

        print("===========================================================")
        print("Discriminator")
        print("===========================================================")
        tf.keras.Model(inputs=source, outputs=outputs).summary()

    @tf.function
    def train_step(self, real_source, real_target, seg=None, source_times=None, target_times=None):

        """ Expects data in order 'source, target' or 'source, target, segmentations'"""

        with tf.GradientTape(persistent=True) as tape:

            # Generate fake target
            if self.input_times:
                fake_target = self.Generator(real_source, target_times)
            else:
                fake_target = self.Generator(real_source)
            
            # Calculate generator L1 before augmentation
            if seg is not None:
                g_L1 = self.L1_loss(real_target, fake_target, seg)
                self.train_L1_metric.update_state(real_target, fake_target, seg)

            else:
                g_L1 = self.L1_loss(real_target, fake_target)
                self.train_L1_metric.update_state(g_L1)

            # Augmentation if required
            if self.Aug:
                imgs, seg = self.Aug(imgs=[real_source, real_target, fake_target], seg=seg)
                real_source, real_target, fake_target = imgs

            # Concatenate targets and sources
            fake_in = tf.concat([fake_target, real_source], axis=4, name="d_fake_concat")
            real_in = tf.concat([real_target, real_source], axis=4, name="d_real_concat")
            d_in = tf.concat([real_in, fake_in], axis=0, name="d_real_fake_concat")

            # Generate predictions and calculate losses
            d_pred = self.Discriminator(d_in, target_times)
            mb_size = d_pred.shape[0] // 2
            d_loss = self.d_loss(d_pred[0:mb_size, ...], d_pred[mb_size:, ...])
            g_loss = self.g_loss(d_pred[mb_size:, ...])
            g_total_loss = g_loss + self.lambda_ * g_L1

        # Get gradients and update weights
        d_grads = tape.gradient(d_loss, self.Discriminator.trainable_variables)
        self.d_optimiser.apply_gradients(zip(d_grads, self.Discriminator.trainable_variables))
        g_grads = tape.gradient(g_total_loss, self.Generator.trainable_variables)
        self.g_optimiser.apply_gradients(zip(g_grads, self.Generator.trainable_variables))

        # Update metrics
        self.d_metric.update_state(d_loss)
        self.g_metric.update_state(g_loss)

    @tf.function
    def test_step(self, real_source, real_target, seg=None, source_times=None, target_times=None):

        # Generate fake target
        if self.input_times:
            fake_target = self.Generator(real_source, target_times)
        else:
            fake_target = self.Generator(real_source)

        g_L1 = L1(real_target, fake_target)

        if seg is not None:
            self.val_L1_metric.update_state(real_target, fake_target, seg)
        else:
            self.val_L1_metric.update_state(g_L1)
    
    def reset_train_metrics(self):
        self.d_metric.reset_states()
        self.g_metric.reset_states()
        self.train_L1_metric.reset_states()


#-------------------------------------------------------------------------
""" Wrapper for Pix2pix GAN with HyperNetwork """

class HyperPix2Pix(Pix2Pix):

    """ GAN class using HyperNetwork for generator """

    def __init__(self, config: dict, name: str = "HyperGAN"):
        super().__init__(config, name=name)

    def generator_init(self, config):
        # Check generator output dims match input
        G_input_size = [1] + self.img_dims + [1]
        G_output_size = [1] + self.img_dims + [1]

        self.Generator = HyperGenerator(self.initialiser, config, name="generator")

        if self.input_times:
            assert self.Generator.build_model(tf.zeros(G_input_size), tf.zeros(1)) == G_output_size, f"{self.Generator.build_model(tf.zeros(G_input_size), tf.zeros(1))} vs {G_input_size}"
        else:
            assert self.Generator.build_model(tf.zeros(G_input_size)) == G_output_size, f"{self.Generator.build_model(tf.zeros(G_input_size))} vs {G_input_size}"

    def summary(self):
        source = tf.keras.Input(shape=self.img_dims + [1])

        if self.input_times:
            outputs = self.Generator.call(source, tf.zeros(1))
        else:
            outputs = self.Generator.call(source)

        print("===========================================================")
        print(f"Generator: {np.sum([np.prod(v.shape) for v in self.Generator.trainable_variables])}")
        print("===========================================================")
        tf.keras.Model(inputs=source, outputs=outputs).summary()
        source = tf.keras.Input(shape=self.img_dims + [2])

        if self.input_times:
            outputs = self.Discriminator.call(source, tf.zeros(1))
        else:
            outputs = self.Discriminator.call(source)    

        print("===========================================================")
        print(f"Discriminator: {np.sum([np.prod(v.shape) for v in self.Discriminator.trainable_variables])}")
        print("===========================================================")
        tf.keras.Model(inputs=source, outputs=outputs).summary()
