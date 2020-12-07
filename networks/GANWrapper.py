import sys
import tensorflow as tf
import tensorflow.keras as keras

from networks.Pix2Pix import Discriminator, Generator
# from utils.TrainFuncs import least_square_loss, wasserstein_loss, gradient_penalty
from utils.DataLoader import DiffAug
from utils.Losses import FocalLoss, FocalMetric


class GAN(keras.Model):

    """ GAN class
        - config: configuration json
        - g_optimiser: generator optimiser e.g. keras.optimizers.Adam()
        - d_optimiser: discriminator optimiser e.g. keras.optimizers.Adam()
        - GAN_type: 'original', 'least_square', 'wasserstein' or 'wasserstein-GP' """

    def __init__(self, config, GAN_type="original", name="GAN"):
        super(GAN, self).__init__(name=name)
        self.initialiser = keras.initializers.RandomNormal(0, 0.02)

        # Choose appropriate loss and initialise metrics
        # self.loss_dict = {
        #     "original": keras.losses.BinaryCrossentropy(from_logits=True),
        #     "least_square": least_square_loss,
        #     "wasserstein": wasserstein_loss,
        #     "wasserstein-GP": wasserstein_loss,
        #     "progressive": wasserstein_loss
        #     }
        self.loss_dict = {"original": keras.losses.BinaryCrossentropy(from_logits=True, name="bce_loss")}
        # TODO: import losses

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
        self.L1 = FocalLoss(mu=config["HYPERPARAMS"]["MU"], loss_fn="mae", name="focal_loss")
        self.lambda_ = config["HYPERPARAMS"]["LAMBDA"]
        self.d_in_ch = config["HYPERPARAMS"]["D_IN_CH"]
        
        # Initialise generator and discriminators
        G_input_size = [1] + config["EXPT"]["IMG_DIMS"] + [1]
        D_input_size = [1] + config["EXPT"]["IMG_DIMS"] + [self.d_in_ch]
        
        self.Generator = Generator(self.initialiser, config, name="generator")
        
        if config["HYPERPARAMS"]["D_LAYERS_F"]:
            Discriminator_G = Discriminator(self.initialiser, config, d_focal=False, name="discriminator_G")
            Discriminator_F = Discriminator(self.initialiser, config, d_focal=True, name="discriminator_F")
            self.Discriminators = {"Discriminator_G": Discriminator_G, "Discriminator_F": Discriminator_F}
        else:
            self.Discriminators = {"Discriminator": Discriminator(self.initialiser, config, d_focal=False, name="discriminator")}

        self.patch_sizes = {name: D.build_model(tf.zeros(D_input_size)) for name, D in self.Discriminators.items()}
        assert self.Generator.build_model(tf.zeros(G_input_size)) == G_input_size

        self.g_optimiser = keras.optimizers.Adam(2e-4, 0.5, 0.999, name="g_opt")
        self.d_optimisers = {
            name: keras.optimizers.Adam(2e-4, 0.5, 0.999, name=f"{name}_opt")\
                for name in self.Discriminators.keys()}

        self.generator_metrics = {
            name:
            {
                "g_metric": keras.metrics.Mean(name=f"{name}_g_metric"),
                "g_L1": FocalMetric(loss_fn="mae", name=f"{name}_L1")
            } 
            for name in self.Discriminators.keys()
        }

        self.generator_val_metric = FocalMetric(loss_fn="mae", name="val_L1")

        self.discriminator_metrics = {
            name:
            {
                "d_metric_1": keras.metrics.Mean(name=f"{name}_d_metric_1"),
                "d_metric_2": keras.metrics.Mean(name=f"{name}_d_metric_2")
            }
            for name in self.Discriminators.keys()
        }

        self.n_critic = config["HYPERPARAMS"]["N_CRITIC"]
        self.mb_size = config["HYPERPARAMS"]["MB_SIZE"]

        if config["HYPERPARAMS"]["AUGMENT"]:
            self.Aug = DiffAug({"colour": True, "translation": True, "cutout": True})
        else:
            self.Aug = None
    
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
        g_mb = self.mb_size
        d_mb = (source.shape[0] - self.mb_size) // self.n_critic
        
        d_labels = {name: tf.concat(
            [tf.ones([d_mb] + patch_size[1:]) * self.d_fake_label,
             tf.ones([d_mb] + patch_size[1:]) * self.d_real_label
             ], axis=0) for name, patch_size in self.patch_sizes.items()}
            
        g_labels = {name: tf.ones([g_mb] + patch_size[1:]) * self.g_label for name, patch_size in self.patch_sizes.items()}

        # TODO: ADD NOISE TO LABELS AND/OR IMAGES

        # Select minibatch of images and masks for generator training
        g_source = source[0:g_mb, :, :, :, :]
        g_real_target = target[0:g_mb, :, :, :, :]
        g_mask = mask[0:g_mb, :, :, :, :]

        # Critic training loop
        for idx in range(self.n_critic):
            # Select minibatch of real images and generate fake images for critic run
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
            for d_name, Discriminator in self.Discriminators.items():
                if "F" in d_name:
                    tile_mask = tf.tile(d_mask, [1, 1, 1, 1, self.d_in_ch])
                    d_fake_in = d_fake_in * tile_mask
                    d_real_in = d_real_in * tile_mask

                with tf.GradientTape() as d_tape:
                    d_pred_fake = Discriminator(d_fake_in)
                    d_pred_real = Discriminator(d_real_in)
                    d_predictions = tf.concat([d_pred_fake, d_pred_real], axis=0)
                    d_loss_1 = self.loss(d_labels[d_name][0:d_mb, ...], d_predictions[0:d_mb, ...]) # Fake
                    d_loss_2 = self.loss(d_labels[d_name][d_mb:, ...], d_predictions[d_mb:, ...]) # Real
                    d_loss = d_loss_1 + d_loss_2

                    # Gradient penalty if indicated
                    # TODO: tidy up loss selection
                    # if self.GAN_type == "wasserstein-GP" or "progressive":
                    #     grad_penalty = gradient_penalty(d_real_batch, d_fake_images, self.Discriminator)
                        # d_loss += 10 * grad_penalty
                
                d_grads = d_tape.gradient(d_loss, Discriminator.trainable_variables)
                self.d_optimisers[d_name].apply_gradients(zip(d_grads, Discriminator.trainable_variables))

                # Update metrics
                self.discriminator_metrics[d_name]["d_metric_1"].update_state(d_loss_1)
                self.discriminator_metrics[d_name]["d_metric_2"].update_state(d_loss_2)

        # Generator training       
        # TODO: ADD NOISE TO LABELS AND/OR IMAGES
        # Get gradients from discriminator predictions of generated fake images and update weights
        for d_name, Discriminator in self.Discriminators.items():
            with tf.GradientTape() as g_tape:
                g_fake_target = self.Generator(g_source)

                # Calculate L1 before augmentation
                g_L1 = self.L1(g_real_target, g_fake_target, g_mask)
                self.generator_metrics[d_name]["g_L1"].update_state(g_real_target, g_fake_target, g_mask)
                
                if self.Aug:
                    imgs, g_mask = self.Aug.augment(imgs=[g_source, g_fake_target], seg=g_mask)
                    g_source, g_fake_target = imgs

                if self.d_in_ch == 3:
                    g_fake_in = tf.concat([g_source, g_fake_target, g_mask], axis=4, name="f_concat")
                else:
                    g_fake_in = tf.concat([g_source, g_fake_target], axis=4, name="f_concat")

                if "F" in d_name:
                    tile_mask = tf.tile(g_mask, [1, 1, 1, 1, self.d_in_ch])
                    g_fake_in = g_fake_in * tile_mask

                g_pred_fake = Discriminator(g_fake_in)
                g_loss = self.loss(g_labels[d_name], g_pred_fake)
                g_total_loss = g_loss + self.lambda_ * g_L1

            g_grads = g_tape.gradient(g_total_loss, self.Generator.trainable_variables)
            self.g_optimiser.apply_gradients(zip(g_grads, self.Generator.trainable_variables))

            # Update metric
            self.generator_metrics[d_name]["g_metric"].update_state(g_loss)

    @tf.function
    def val_step(self, source, target, mask):
        g_fake = self.Generator(source)
        self.generator_val_metric.update_state(target, g_fake, mask)
