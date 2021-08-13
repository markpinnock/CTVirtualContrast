import tensorflow as tf

from utils.affinetransformation import AffineTransform2D


""" Modified spatial transformer network:
    Jaderberg et al. Spatial transformer networks. NeurIPS 28 (2015)
    https://arxiv.org/abs/1506.02025 """


class SpatialTransformer(tf.keras.layers.Layer):
    def __init__(self, config, name="spatial_transformer"):
        super().__init__(name=name)
        self.conv = []
        self.batch_norm = []
        self.dense = []
        nc = 8
        zero_init = tf.keras.initializers.RandomNormal(0, 0.001)
        self.identity = tf.constant([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])

        for i in range(1, config["hyperparameters"]["stn_layers"] + 1):
            self.conv.append(tf.keras.layers.Conv2D(filters=nc * i, kernel_size=(2, 2), strides=(2, 2), activation="relu", kernel_initializer=zero_init))
            self.batch_norm.append(tf.keras.layers.BatchNormalization())

        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(units=config["hyperparameters"]["stn_output"], activation="linear", kernel_initializer=zero_init)

        # If segmentations available, these can be stacked on the target for transforming
        if len(config["data"]["segs"]) > 0:
            self.transform = AffineTransform2D(config["hyperparameters"]["img_dims"] + [2])
        else:
            self.transform = AffineTransform2D(config["hyperparameters"]["img_dims"] + [1])
    
    def call(self, source, target, seg=None, training=False, print_matrix=False):
        mb_size = source.shape[0]
        if mb_size == None: mb_size = 1
        x = tf.concat([source[:, :, :, 0, :], target[:, :, :, 0, :]], axis=3)

        for conv, bn in zip(self.conv, self.batch_norm):
            x = bn(conv(x), training)
        
        x = self.dense(self.flatten(x))
        x = self.identity - x

        # For debugging
        if print_matrix:
            print(tf.reshape(x[0, ...], [2, 3]))

        if seg is not None:
            target_seg = tf.concat([target, seg], axis=4)
            target_seg = self.transform(im=target_seg, mb_size=mb_size, thetas=x)
            
            return target_seg[:, :, :, :, 0][:, :, :, :, tf.newaxis], target_seg[:, :, :, :, 1][:, :, :, :, tf.newaxis]
        
        else:
            target = self.transform(im=target, mb_size=mb_size, thetas=x)

            return target, None


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from ..utils.dataloader import PairedLoader

    """ Routine for visually testing STN output """

    FILE_PATH = "D:/ProjectImages/SyntheticContrast"
    segs = ["AC"]
    TestLoader = PairedLoader({"data_path": FILE_PATH, "target": ["AC"], "source": ["HQ"], "segs": segs, "times": None, "down_sample": 4, "num_examples": 4, "cv_folds": 3, "fold": 2}, dataset_type="training")
    TestLoader.set_normalisation(norm_type="std", param_1=-288, param_2=254)
    STN = SpatialTransformer({"data": {"segs": segs}, "hyperparameters": {"img_dims": [128, 128, 12], "stn_layers" : 2, "stn_output": 6}})

    # Specify output types
    output_types = ["float32", "float32"]

    if len(segs) > 0:
        output_types += ["float8"]

    train_ds = tf.data.Dataset.from_generator(TestLoader.data_generator, output_types=output_types)

    for data in train_ds.batch(4).take(2):
        if len(segs) > 0:
            source, target, seg = data
            target, seg = STN(source, target, seg=seg)
        else:
            source, target = data
            target, _ = STN(source, target, seg=None)

        plt.subplot(3, 2, 1)
        plt.imshow(source[0, :, :, 0, 0], cmap="gray")
        plt.axis("off")
        plt.subplot(3, 2, 2)
        plt.imshow(source[1, :, :, 0, 0], cmap="gray")
        plt.axis("off")

        plt.subplot(3, 2, 3)
        plt.imshow(target[0, :, :, 0, 0], cmap="gray")
        plt.axis("off")
        plt.subplot(3, 2, 4)
        plt.imshow(target[1, :, :, 0, 0], cmap="gray")
        plt.axis("off")

        if len(segs) > 0:
            plt.subplot(3, 2, 5)
            plt.imshow(seg[0, :, :, 0, 0])
            plt.axis("off")
            plt.subplot(3, 2, 6)
            plt.imshow(seg[1, :, :, 0, 0])
            plt.axis("off")

        plt.show()

    if len(segs) > 0:
        source, target, seg = TestLoader.example_images()
        target, seg = STN(source, target, seg=seg)

    else:
        source, target = TestLoader.example_images()
        target, _ = STN(source, target, seg=None)
    
    fig, axs = plt.subplots(target.shape[0], 3)

    for i in range(target.shape[0]):
        axs[i, 0].imshow(source[i, :, :, 11, 0], cmap="gray")
        axs[i, 0].axis("off")
        axs[i, 1].imshow(target[i, :, :, 11, 0], cmap="gray")
        axs[i, 1].axis("off")

        if len(segs) > 0:
            axs[i, 2].imshow(seg[i, :, :, 11, 0])
            axs[i, 2].axis("off")
    
    plt.show()