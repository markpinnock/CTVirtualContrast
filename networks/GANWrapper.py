import sys
import tensorflow as tf
import tensorflow.keras as keras

from networks.Pix2Pix import Discriminator, Generator
# from utils.TrainFuncs import least_square_loss, wasserstein_loss, gradient_penalty
from utils.Losses import FocalLoss, FocalMetric


class GAN(keras.Model):

    """ GAN class
        - latent_dims: size of generator latent distribution
        - g_nc: number of channels in generator first layer
        - d_nc: number of channels in discriminator first layer
        - g_optimiser: generator optimiser e.g. keras.optimizers.Adam()
        - d_optimiser: discriminator optimiser e.g. keras.optimizers.Adam()
        - GAN_type: 'original', 'least_square', 'wasserstein' or 'wasserstein-GP'
        - n_critic: number of discriminator/critic training runs (5 in WGAN, 1 otherwise)
        - lambda_: L1 hyperparameter """

    def __init__(self, config, g_optimiser, d_optimiser, lambda_, GAN_type="original", n_critic=1):
        super(GAN, self).__init__()
        self.initialiser = keras.initializers.RandomNormal(0, 0.02)

        # Choose appropriate loss and initialise metrics
        # self.loss_dict = {
        #     "original": keras.losses.BinaryCrossentropy(from_logits=True),
        #     "least_square": least_square_loss,
        #     "wasserstein": wasserstein_loss,
        #     "wasserstein-GP": wasserstein_loss,
        #     "progressive": wasserstein_loss
        #     }
        self.loss_dict = {"original": keras.losses.BinaryCrossentropy(from_logits=True)}
        # TODO: import losses
        self.metric_dict = {
            "g_metric": keras.metrics.Mean(),
            "d_metric_1": keras.metrics.Mean(),
            "d_metric_2": keras.metrics.Mean()
        }

        # Set up real/fake labels
        if GAN_type == "wasserstein":
            self.d_real_label = -1.0
            self.d_fake_label = 1.0
            self.g_label = -1.0
            cons = True
        elif GAN_type == "wasserstein-GP":
            self.d_real_label = -1.0
            self.d_fake_label = 1.0
            self.g_label = -1.0
            cons = False
        elif GAN_type == "progressive":
            self.d_real_label = -1.0
            self.d_fake_label = 1.0
            self.g_label = -1.0
            cons = "maxnorm"
        else:
            self.d_real_label = 0.0
            self.d_fake_label = 1.0
            self.g_label = 0.0
            cons = False
        # TODO: IMPLEMENT CONSTRAINT TYPE
        self.loss = self.loss_dict[GAN_type]
        self.GAN_type = GAN_type
        self.L1 = FocalLoss(mu=0.5, loss_fn="mae")
        self.lambda_ = lambda_
        self.L1metric = FocalMetric(loss_fn="mae")
        self.Generator = Generator(self.initialiser)
        self.Discriminator = Discriminator(self.initialiser)
        self.patch_size = self.Discriminator(
            tf.zeros((1, 512 // config["DOWN_SAMP"], 512 // config["DOWN_SAMP"], 12, 1)),
            tf.zeros((1, 512 // config["DOWN_SAMP"], 512 // config["DOWN_SAMP"], 12, 1)),
            tf.zeros((1, 512 // config["DOWN_SAMP"], 512 // config["DOWN_SAMP"], 12, 1)),
            training=True).shape
        self.g_optimiser = g_optimiser
        self.d_optimiser = d_optimiser
        self.n_critic = n_critic
    
    def compile(self, g_optimiser, d_optimiser, loss_key):
        # Not currently used
        raise NotImplementedError
        super(GAN, self).compile()
        self.g_optimiser = g_optimiser
        self.d_optimiser = d_optimiser
        self.loss = self.loss_dict[loss_key]
    
    @tf.function
    def train_step(self, source, target, mask):
        # Determine labels and size of mb for each critic training run
        # (size of real_images = minibatch size * number of critic runs)
        mb_size = source.shape[0] // self.n_critic
        # TODO: ensure different G and D minibatch
        d_labels = tf.concat(
            [tf.ones([source.shape[0]] + self.patch_size[1:]) * self.d_fake_label,
             tf.ones([source.shape[0]] + self.patch_size[1:]) * self.d_real_label
             ], axis=0)
            
        g_labels = tf.ones([source.shape[0]] + self.patch_size[1:]) * self.g_label

        # TODO: ADD NOISE TO LABELS AND/OR IMAGES

        # Critic training loop
        for idx in range(self.n_critic):
            # Select minibatch of real images and generate fake images
            d_source_batch = source[idx * mb_size:(idx + 1) * mb_size, :, :, :]
            d_target_batch = target[idx * mb_size:(idx + 1) * mb_size, :, :, :]
            d_fake_target = self.Generator(d_source_batch, training=True)

            # Get gradients from critic predictions and update weights
            with tf.GradientTape() as d_tape:
                d_pred_fake = self.Discriminator(d_source_batch, d_fake_target, mask)
                d_pred_real = self.Discriminator(d_source_batch, d_target_batch, mask)
                d_predictions = tf.concat([d_pred_fake, d_pred_real], axis=0)
                d_loss_1 = self.loss(d_labels[0:mb_size, ...], d_predictions[0:mb_size, ...]) # Fake
                d_loss_2 = self.loss(d_labels[mb_size:, ...], d_predictions[mb_size:, ...]) # Real
                d_loss = 0.5 * d_loss_1 + 0.5 * d_loss_2
            
                # Gradient penalty if indicated
                # TODO: tidy up loss selection
                # if self.GAN_type == "wasserstein-GP" or "progressive":
                #     grad_penalty = gradient_penalty(d_real_batch, d_fake_images, self.Discriminator)
                    # d_loss += 10 * grad_penalty
            
            d_grads = d_tape.gradient(d_loss, self.Discriminator.trainable_variables)
            self.d_optimiser.apply_gradients(zip(d_grads, self.Discriminator.trainable_variables))

            # Update metrics
            self.metric_dict["d_metric_1"].update_state(d_loss_1)
            self.metric_dict["d_metric_2"].update_state(d_loss_2)

        # Generator training       
        # TODO: ADD NOISE TO LABELS AND/OR IMAGES
        # Get gradients from critic predictions of generated fake images and update weights
        with tf.GradientTape() as g_tape:
            g_fake_target = self.Generator(d_source_batch)
            g_predictions = self.Discriminator(d_source_batch, g_fake_target, mask)
            g_loss = self.loss(g_labels, g_predictions)
            g_L1 = self.L1(d_target_batch, g_fake_target, mask)
            g_total_loss = g_loss + self.lambda_ * g_L1

        g_grads = g_tape.gradient(g_total_loss, self.Generator.trainable_variables)
        self.g_optimiser.apply_gradients(zip(g_grads, self.Generator.trainable_variables))

        # Update metric
        self.metric_dict["g_metric"].update_state(g_loss)
        self.L1metric.update_state(d_target_batch, g_fake_target, mask)

    @tf.function
    def val_step(self, source, target, mask):
        g_fake = self.Generator(source)
        self.L1metric.update_state(target, g_fake, mask)