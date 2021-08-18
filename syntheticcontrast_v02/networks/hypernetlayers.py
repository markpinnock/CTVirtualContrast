import tensorflow as tf


#-------------------------------------------------------------------------
""" Down-sampling convolutional block for generator with HyperNet """

class HyperGANDownBlock(tf.keras.layers.Layer):

    """ Input:
        - nc: number of feature maps
        - strides: tuple of strides e.g. (2, 2, 1)
        - initialiser: e.g. keras.initializers.RandomNormal
        - batch_norm: True/False """

    def __init__(self, strides, batch_norm=True, name=None):
        super().__init__(name=name)
        self.batch_norm = batch_norm
        self.strides = [1] + list(strides) + [1]
        
        if batch_norm:
            self.bn = tf.keras.layers.BatchNormalization(name="batchnorm")

    def call(self, x, w, training):
        x = tf.nn.conv3d(x, w, self.strides, padding="SAME", name="conv")

        if self.batch_norm:
            x = self.bn(x, training)

        return tf.nn.leaky_relu(x, alpha=0.2, name="l_relu")


#-------------------------------------------------------------------------
""" Up-sampling convolutional block for generator with HyperNet """

class HyperGANUpBlock(tf.keras.layers.Layer):

    """ Input:
        - nc: number of feature maps
        - strides: tuple of strides e.g. (2, 2, 1)
        - initialiser: e.g. keras.initializers.RandomNormal
        - batch_norm: True/False
        - dropout: True/False """

    def __init__(self, strides, batch_norm=True, dropout=False, name=None):
        super().__init__(name=name)
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.strides = [1] + list(strides) + [1]

        if batch_norm:
            self.bn = tf.keras.layers.BatchNormalization(name="batchnorm")
        if dropout:
            self.dropout = tf.keras.layers.Dropout(0.5, name="dropout")
        
        self.upsample = tf.keras.layers.UpSampling3D(strides, name="upsample")
        self.concat = tf.keras.layers.Concatenate(name="concat")
    
    def call(self, x, w, skip, training):
        x = self.upsample(x)
        x = tf.nn.conv3d(x, w, (1, 1, 1, 1, 1), padding="SAME", name="conv")

        if self.batch_norm:
            x = self.bn(x, training=training)
        
        if self.dropout:
            x = self.dropout(x, training=training)
    
        x = self.concat([x, skip])

        return tf.nn.relu(x)
