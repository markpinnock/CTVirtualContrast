import matplotlib.pyplot as plt
import sys
import tensorflow as tf
import tensorflow.keras as keras

from networks.Pix2Pix import Discriminator, Generator
# from utils.TrainFuncs import least_square_loss, wasserstein_loss, gradient_penalty
from utils.DataLoader import DiffAug
from utils.Losses import FocalLoss, FocalMetric


#-------------------------------------------------------------------------
""" Wrapper for standard Pix2pix GAN """

class BaseGAN(keras.Model):

    """ GAN class
        - config: configuration json
        - GAN_type: 'original', 'least_square', 'wasserstein' or 'wasserstein-GP' """

    def __init__(self, config, GAN_type="original", name="GAN"):
        super().__init__(name=name)
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

        if config["HYPERPARAMS"]["MU"] > 0.0:
            self.L1 = FocalLoss(mu=config["HYPERPARAMS"]["MU"], loss_fn="mae", name="focal_loss")
        elif config["HYPERPARAMS"]["MU"] == 0.0:
            self.L1 = keras.losses.MeanAbsoluteError()
        else:
            raise ValueError

        self.lambda_ = config["HYPERPARAMS"]["LAMBDA"]
        self.d_in_ch = config["HYPERPARAMS"]["D_IN_CH"]
        self.img_dims = config["EXPT"]["IMG_DIMS"]
        self.n_critic = config["HYPERPARAMS"]["N_CRITIC"]
        self.mb_size = config["HYPERPARAMS"]["MB_SIZE"]

        if config["HYPERPARAMS"]["AUGMENT"]:
            self.Aug = DiffAug({"colour": True, "translation": True, "cutout": True})
        else:
            self.Aug = None
        # Initialise generator and discriminators
        self.generator_init(config)
        self.discriminator_init(config)

    def generator_init(self, config):

        self.g_optimiser = keras.optimizers.Adam(2e-4, 0.5, 0.999, name="g_opt")

        # Check generator output dims match input
        G_input_size = [1] + config["EXPT"]["IMG_DIMS"] + [1]
        self.Generator = Generator(self.initialiser, config, name="generator")
        assert self.Generator.build_model(tf.zeros(G_input_size)) == G_input_size

        # Set up metrics
        self.generator_metric = keras.metrics.Mean(name="g_metric")
        self.train_L1_metric = FocalMetric(loss_fn="mae", name="train_L1")
        self.val_L1_metric = FocalMetric(loss_fn="mae", name="val_L1")
        
    def discriminator_init(self, config):

        self.d_optimiser = keras.optimizers.Adam(2e-4, 0.5, 0.999, name="d_opt")

        # Get discriminator patch size
        D_input_size = [1] + config["EXPT"]["IMG_DIMS"] + [self.d_in_ch]
        self.Discriminator = Discriminator(self.initialiser, config, name="discriminator")
        self.patch_size = self.Discriminator.build_model(tf.zeros(D_input_size))

        # Set up metrics
        self.discriminator_metric = keras.metrics.Mean(name="d_metric")
    
    def compile(self, g_optimiser, d_optimiser, loss_key):
        # Not currently used
        raise NotImplementedError
        super(GAN, self).compile()
        self.g_optimiser = g_optimiser
        self.d_optimiser = d_optimiser
        self.loss = self.loss_dict[loss_key]      
    
    @tf.function
    def train_step(self, data):
        source, target, mask, _ = data
        # Determine size of mb for each critic training run
        # (size of real_images = minibatch size * number of critic runs)

        g_mb = source.shape[0] // (1 + self.n_critic)
        if g_mb < 1: g_mb = source.shape[0]
        d_mb = g_mb * self.n_critic # TODO: TEST WITH N_CRITIC > 1

        # Labels
        d_labels = tf.concat(
            [tf.ones([d_mb] + self.patch_size[1:]) * self.d_fake_label,
             tf.ones([d_mb] + self.patch_size[1:]) * self.d_real_label
             ], axis=0)
            
        g_labels = tf.ones([g_mb] + self.patch_size[1:]) * self.g_label

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
                # TODO: tidy up loss selection
                # if self.GAN_type == "wasserstein-GP" or "progressive":
                #     grad_penalty = gradient_penalty(d_real_batch, d_fake_images, self.Discriminator)
                    # d_loss += 10 * grad_penalty
            
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

#-------------------------------------------------------------------------
""" Pix2pix GAN with focal and global discriminator - inherits from BaseGAN """

class DoubleGAN(BaseGAN):

    def __init__(self, config, GAN_type="original", name="GAN"):
        
        assert config["HYPERPARAMS"]["D_LAYERS_F"]
        super().__init__(config, GAN_type="original", name="GAN")
    
    def discriminator_init(self):
        raise NotImplementedError
        Discriminator_G = Discriminator(self.initialiser, config, d_focal=False, name="discriminator_G")
        Discriminator_F = Discriminator(self.initialiser, config, d_focal=True, name="discriminator_F")
        self.Discriminators = {"Discriminator_G": Discriminator_G, "Discriminator_F": Discriminator_F}

        self.patch_sizes = {name: D.build_model(tf.zeros(D_input_size)) for name, D in self.Discriminators.items()}
        assert self.Generator.build_model(tf.zeros(G_input_size)) == G_input_size

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
    
    @tf.function
    def train_step(self, source, target, mask, coords):
        raise NotImplementedError
        # Determine size of mb for each critic training run
        # (size of real_images = minibatch size * number of critic runs)
        g_mb = source.shape[0] // (1 + self.n_critic)
        if g_mb < 1: g_mb = source.shape[0]
        d_mb = g_mb * self.n_critic # TODO: TEST WITH N_CRITIC > 1

        # Labels
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

#-------------------------------------------------------------------------
""" GAN acting on cropped ROIs - inherits from BaseGAN """

class CropGAN(BaseGAN):

    def __init__(self, config, GAN_type="original", name="GAN"):
        super().__init__(config, GAN_type="original", name="GAN")

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
        
        for i in range(coords.shape[1]):
            # Crop ROI
            source, target, mask = self.crop_ROI(source, target, mask, coords[:, i, :])
            super().train_step((source, target, mask, None))

    @tf.function
    def val_step(self, data):
        source, target, mask, coords = data
        
        for i in range(coords.shape[1]):
            # Crop ROI
            source, target, mask = self.crop_ROI(source, target, mask, coords[:, i, :])
            super().val_step((source, target, mask, None))
