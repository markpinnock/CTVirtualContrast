import tensorflow as tf
from tensorflow.python.ops.gen_array_ops import identity

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

        for i in range(1, config["HYPERPARAMS"]["STN_LAYERS"] + 1):
            self.conv.append(tf.keras.layers.Conv2D(filters=nc * i, kernel_size=(2, 2), strides=(2, 2), activation="relu", kernel_initializer=zero_init))
            self.batch_norm.append(tf.keras.layers.BatchNormalization())

        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(units=config["HYPERPARAMS"]["STN_OUT"], activation="linear", kernel_initializer=zero_init)

        # If segmentations available, these can be stacked on the target for transforming
        if len(config["DATA"]["SEGS"]) > 0:
            self.transform = AffineTransform2D(config["DATA"]["IMG_DIMS"] + [2])
        else:
            self.transform = AffineTransform2D(config["DATA"]["IMG_DIMS"] + [1])
    
    def call(self, source, target, seg=None, training=False, print_matrix=False):
        mb_size = source.shape[0]
        if mb_size == None: mb_size = 1
        x = tf.concat([source[:, :, :, 0, :], target[:, :, :, 0, :]], axis=3)

        for conv, bn in zip(self.conv, self.batch_norm):
            x = bn(conv(x), training)
        
        x = self.dense(self.flatten(x))
        x = self.identity - x
        if print_matrix: print(tf.reshape(x[0, ...], [2, 3]))
        if seg != None:
            target_seg = tf.concat([source, seg], axis=4)
            target_seg = self.transform(im=target_seg, mb_size=mb_size, thetas=x)
            
            return target_seg[:, :, :, :, 0], target_seg[:, :, :, :, 1]
        
        else:
            target = self.transform(im=target, mb_size=mb_size, thetas=x)

            return target
