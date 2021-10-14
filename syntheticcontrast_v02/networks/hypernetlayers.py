import tensorflow as tf


#-------------------------------------------------------------------------
""" Down-sampling convolutional block for generator with HyperNet """

class HyperGANDownBlock(tf.keras.layers.Layer):

    """ Input:
        - nc: number of feature maps
        - strides: tuple of strides e.g. (2, 2, 1)
        - batch_norm: True/False """

    def __init__(self, strides, batch_norm=True, name=None):
        super().__init__(name=name)
        self.batch_norm = batch_norm
        self.strides = [1] + list(strides) + [1]
        
        if batch_norm:
            self.bn = tf.keras.layers.BatchNormalization(name="batchnorm")

    def call(self, x, w, training):
        if w.shape[0] == 1:
            x = tf.nn.conv3d(x, w[0, ...], self.strides, padding="SAME", name="conv")
        
        else:
            num_kernels = w.shape[0]

            for i in range(num_kernels):
                x[i, ...] = tf.nn.conv3d(x[i, ...], w[i, ...], self.strides, padding="SAME", name="conv")

        # As we are using batchnorm, there is no bias term
        if self.batch_norm:
            x = self.bn(x, training)

        return tf.nn.leaky_relu(x, alpha=0.2, name="l_relu")


#-------------------------------------------------------------------------
""" Up-sampling convolutional block for generator with HyperNet """

class HyperGANUpBlock(tf.keras.layers.Layer):

    """ Input:
        - nc: number of feature maps
        - strides: tuple of strides e.g. (2, 2, 1)
        - batch_norm: True/False
        - dropout: True/False """

    def __init__(self, strides, batch_norm=True, dropout=False, name=None):
        super().__init__(name=name)
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.strides = [1] + list(strides) + [1]

        if batch_norm:
            self.bn1 = tf.keras.layers.BatchNormalization(name="batchnorm1")
            self.bn2 = tf.keras.layers.BatchNormalization(name="batchnorm2")

        if dropout:
            self.dropout = tf.keras.layers.Dropout(0.5, name="dropout")
        
        self.upsample = tf.keras.layers.UpSampling3D(strides, name="upsample")
        self.concat = tf.keras.layers.Concatenate(name="concat")
    
    def call(self, x, w1, w2, skip, training):
        x = self.upsample(x)

        if w1.shape[0] == 1:
            x = tf.nn.conv3d(x, w1[0, ...], (1, 1, 1, 1, 1), padding="SAME", name="tconv")

        else:
            num_kernels = w1.shape[0]

            for i in range(num_kernels):
                x[i, ...] = tf.nn.conv3d(x[i, ...], w1[i, ...], self.strides, padding="SAME", name="tconv")

        # As we are using batchnorm, there is no bias term
        if self.batch_norm:
            x = self.bn1(x, training=training)
        
        if self.dropout:
            x = self.dropout(x, training=training)

        x = tf.nn.relu(x)
        x = self.concat([x, skip])

        if w2.shape[0] == 1:
            x = tf.nn.conv3d(x, w2[0, ...], (1, 1, 1, 1, 1), padding="SAME", name="conv")

        else:
            num_kernels = w2.shape[0]

            for i in range(num_kernels):
                x[i, ...] = tf.nn.conv3d(x[i, ...], w1[i, ...], self.strides, padding="SAME", name="conv")

        if self.batch_norm:
            x = self.bn2(x, training=training)

        if self.dropout:
            x = self.dropout(x, training=training)

        return tf.nn.relu(x)
