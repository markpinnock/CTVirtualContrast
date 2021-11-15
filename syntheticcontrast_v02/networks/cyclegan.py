import numpy as np
import tensorflow as tf

from .models import Discriminator, Generator
from syntheticcontrast_v02.utils.augmentation import DiffAug, StdAug
from syntheticcontrast_v02.utils.losses import (
    minimax_D, minimax_G, least_squares_D, least_squares_G,
    L1, gradient_difference, mutual_information)


#-------------------------------------------------------------------------
""" Wrapper for CycleGAN """

class CycleGAN(tf.keras.Model):

    def __init__(self, config, name="CycleGAN"):
        super().__init__(name=name)
        self.initialiser = tf.keras.initializers.HeNormal()
        self.config = config
        self.lambda_ = config["hyperparameters"]["lambda"]
        self.mb_size = config["expt"]["mb_size"]
        self.d_in_ch = 2
        self.img_dims = config["hyperparameters"]["img_dims"]

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

        # Initialise image buffers
        self.buffer_forward = ImageBuffer()
        self.buffer_backward = ImageBuffer()

        # Initialise kernel for gradient difference loss
        self.gradient_kernel = get_gradient_filters()

    def generator_init(self, config):
        # Check generator output dims match input
        G_input_size = [1] + self.img_dims + [1]
        self.G_forward = Generator(self.initialiser, config, name="forward_generator")
        self.G_backward = Generator(self.initialiser, config, name="backward_generator")
        assert self.G_forward.build_model(tf.zeros(G_input_size)) == G_input_size
        assert self.G_backward.build_model(tf.zeros(G_input_size)) == G_input_size

    def discriminator_init(self, config):
        # Get discriminator patch size
        D_input_size = [1] + self.img_dims + [self.d_in_ch]
        self.D_forward = Discriminator(self.initialiser, config, name="forward_discriminator")
        self.D_backward = Discriminator(self.initialiser, config, name="backward_discriminator")
        self.patch_size = self.D_forward.build_model(tf.zeros(D_input_size))
        assert self.patch_size == self.D_backward.build_model(tf.zeros(D_input_size))

    def compile(self, g_forward_opt, g_backward_opt, d_forward_opt, d_backward_opt):
        self.g_forward_opt = g_forward_opt
        self.g_backward_opt = g_backward_opt
        self.d_forward_opt = d_forward_opt
        self.d_backward_opt = d_backward_opt

        """ Using minmax loss rather than least squares as in 
            ResCycleGAN paper """

        self.d_loss = least_squares_D
        self.g_loss = least_squares_G
        self.L1_loss = L1

        # Set up metrics
        self.d_forward_metric = tf.keras.metrics.Mean(name="d_forward_metric")
        self.g_forward_metric = tf.keras.metrics.Mean(name="g_forward_metric")
        self.d_backward_metric = tf.keras.metrics.Mean(name="d_backward_metric")
        self.g_backward_metric = tf.keras.metrics.Mean(name="g_backward_metric")
        self.train_L1_metric = tf.keras.metrics.Mean(name="train_L1")
        self.val_L1_metric = tf.keras.metrics.Mean(name="val_L1")
    
    def summary(self):
        source = tf.keras.Input(shape=self.img_dims + [1])
        outputs = self.G_forward.call(source)
        print("===========================================================")
        print("Generator")
        print("===========================================================")
        tf.keras.Model(inputs=source, outputs=outputs).summary()
        source = tf.keras.Input(shape=self.img_dims + [2])
        outputs = self.D_forward.call(source)
        print("===========================================================")
        print("Discriminator")
        print("===========================================================")
        tf.keras.Model(inputs=source, outputs=outputs).summary()
    
    def generator_step(self, real_source, real_target):

        """ Generator training """

        # Get gradients from discriminator predictions of generated fake images and update weights
        with tf.GradientTape(persistent=True) as g_tape:
            fake_target = self.G_forward(real_source)
            cycled_source = self.G_backward(fake_target)
            fake_source = self.G_backward(real_target)
            cycled_target = self.G_forward(fake_source)
            same_target = self.G_forward(real_target)
            same_source = self.G_backward(real_source)

            """ ResCycleGAN paper uses cycle consistency, gradient norm and MI """

            # Calculate L1 cycle consistency loss before augmentation
            cycle_loss = self.L1_loss(real_source, cycled_source) + self.L1_loss(real_target, cycled_target)
            # gradient_loss = gradient_difference(real_source, cycled_source, self.gradient_kernel) + gradient_difference(real_target, cycled_target, self.gradient_kernel)
            # mutual_information_loss = mutual_information(real_source, cycled_source) + mutual_information(real_target, cycled_target)
            identity_loss = self.L1_loss(real_source, same_source) + self.L1_loss(real_target, same_target)
            self.train_L1_metric.update_state(cycle_loss)

            if self.Aug:
                imgs, _ = self.Aug(imgs=[fake_source, fake_target], seg=None)
                fake_source, fake_target = imgs

            fake_target_pred = self.D_forward(tf.concat([fake_target, real_source], axis=4, name="d_forward_concat"))
            fake_source_pred = self.D_backward(tf.concat([fake_source, real_target], axis=4, name="d_backward_concat"))

            """ Not using identity loss as in ResCycleGAN paper """

            g_forward_loss = self.g_loss(fake_target_pred)
            g_backward_loss = self.g_loss(fake_source_pred)
            g_forward_total_loss = g_forward_loss + self.lambda_ * cycle_loss + self.lambda_ / 2 * identity_loss
            g_backward_total_loss = g_backward_loss + self.lambda_ * cycle_loss + self.lambda_ / 2 * identity_loss

        g_forward_grads = g_tape.gradient(g_forward_total_loss, self.G_forward.trainable_variables)
        g_backward_grads = g_tape.gradient(g_backward_total_loss, self.G_backward.trainable_variables)
        self.g_forward_opt.apply_gradients(zip(g_forward_grads, self.G_forward.trainable_variables))
        self.g_backward_opt.apply_gradients(zip(g_backward_grads, self.G_backward.trainable_variables))

        # Update metric
        self.g_forward_metric.update_state(g_forward_loss)
        self.g_backward_metric.update_state(g_backward_loss)

    def discriminator_step(self, real_source, real_target, fake_source, fake_target):

        """ Discriminator training """

        # DiffAug if required
        if self.Aug:
            imgs, _ = self.Aug(imgs=[real_source, fake_source], seg=None)
            real_source, fake_source = imgs
            imgs, _ = self.Aug(imgs=[real_target, fake_target], seg=None)
            real_target, fake_target = imgs

        D_forward_in = tf.concat(
            [
                tf.concat([real_target, real_source], axis=4, name="real_forward_concat"),
                tf.concat([fake_target, real_source], axis=4, name="fake_forward_concat")
            ], axis=0, name="d_forward_concat"
        )
        
        D_backward_in = tf.concat(
            [
                tf.concat([real_source, real_target], axis=4, name="real_backward_concat"),
                tf.concat([fake_source, real_target], axis=4, name="fake_backward_concat")
            ], axis=0, name="d_backward_concat"
        )

        # Get gradients from discriminator predictions and update weights
        with tf.GradientTape(persistent=True) as d_tape:
            d_forward_pred = self.D_forward(D_forward_in)
            d_backward_pred = self.D_backward(D_backward_in)
            d_forward_loss = self.d_loss(d_forward_pred[0, ...], d_forward_pred[1, ...])
            d_backward_loss = self.d_loss(d_backward_pred[0, ...], d_backward_pred[1, ...])
        
        d_forward_grads = d_tape.gradient(d_forward_loss, self.D_forward.trainable_variables)
        d_backward_grads = d_tape.gradient(d_backward_loss, self.D_backward.trainable_variables)
        self.d_forward_opt.apply_gradients(zip(d_forward_grads, self.D_forward.trainable_variables))
        self.d_backward_opt.apply_gradients(zip(d_backward_grads, self.D_backward.trainable_variables))

        # Update metrics
        self.d_forward_metric.update_state(d_forward_loss)
        self.d_backward_metric.update_state(d_backward_loss)
    
    @tf.function
    def train_step(self, real_source, real_target, seg=None, source_time=None, seg_time=None):

        """ Expects data in order 'real_source, real_target'"""

        fake_target = self.G_forward(real_source)
        fake_source = self.G_backward(real_target)
        fake_target = self.buffer_forward.query(fake_target)
        fake_source = self.buffer_backward.query(fake_source)
        
        self.discriminator_step(real_source, real_target, fake_source, fake_target)
        self.generator_step(real_source, real_target)

    @tf.function
    def test_step(self, real_source, real_target, seg=None):
        fake_target = self.G_forward(real_source)
        cycled_source = self.G_backward(fake_target)
        fake_source = self.G_backward(real_target)
        cycled_target = self.G_forward(fake_source)

        cycle_loss = self.L1_loss(real_source, cycled_source) + self.L1_loss(real_target, cycled_target)
        self.val_L1_metric.update_state(cycle_loss)

    def reset_train_metrics(self):
        self.g_forward_metric.reset_states()
        self.g_backward_metric.reset_states()
        self.d_forward_metric.reset_states()
        self.d_backward_metric.reset_states()
        self.train_L1_metric.reset_states()


#-------------------------------------------------------------------------
""" https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/util/image_pool.py """

class ImageBuffer:

    def __init__(self):
        self.buffer_size = 50
        self.num_imgs = 0
    
        self.pool = []
    
    def query(self, img):
        if self.num_imgs < self.buffer_size:
            self.pool.append(img)
            self.num_imgs += 1

            return img

        else:
            if tf.random.uniform(shape=1) > 0.5:
                idx = tf.random.uniform(shape=1, minval=0, maxval=self.buffer_size, dtype="int32")
                tmp = self.pool[idx]
                self.pool[idx] = img

                return tmp

            else:
                return img


#-------------------------------------------------------------------------

def get_gradient_filters():
    g = tf.constant([[[1.0, -1.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]])
    k = tf.stack([g, tf.transpose(g, [0, 2, 1]), tf.transpose(g, [2, 0, 1])], axis=3)[:, :, :, tf.newaxis, :]

    return k
