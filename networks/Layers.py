import tensorflow as tf
import tensorflow.keras as keras


class DownBlock(keras.layers.Layer):

    """ Implements encoding block for U-Net
        - nc: number of channels in block
        - pool_strides: number of pooling strides e.g. (2, 2, 1) """

    def __init__(self, nc, pool_strides):
        super(DownBlock, self).__init__(self)

        self.conv1 = keras.layers.Conv3D(nc, (3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu')
        self.conv2 = keras.layers.Conv3D(nc, (3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu')
        self.pool = keras.layers.MaxPool3D((2, 2, 2), strides=pool_strides, padding='same')
        # Consider group normalisation
        # Consider pool -> conv3
    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return self.pool(x), x


class UpBlock(keras.layers.Layer):

    """ Implements encoding block for U-Net
        - nc: number of channels in block
        - tconv_strides: number of transpose conv  strides e.g. (2x2x1) """
    
    def __init__(self, nc, tconv_strides):
        super(UpBlock, self).__init__(self)

        self.tconv = keras.layers.Conv3DTranspose(nc, (2, 2, 2), strides=tconv_strides, padding='same', activation='relu')
        self.conv1 = keras.layers.Conv3D(nc, (3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu')
        self.conv2 = keras.layers.Conv3D(nc, (3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu')
    
    def call(self, x, skip):
        x = self.tconv(x)
        x = keras.layers.concatenate([x, skip], axis=4)
        x = self.conv1(x)
        return self.conv2(x)


class GANDownBlock(keras.layers.Layer):

    """ Implements down-sampling block for both
        discriminator and generator
        Input:
            - nc: number of feature maps
            - strides: tuple of strides e.g. (2, 2, 1)
            - initialiser: e.g. keras.initializers.RandomNormal
            - batch_norm: True/False """

    def __init__(self, nc, weights, strides, initialiser, batch_norm=True):
        super(GANDownBlock, self).__init__(self)
        self.batch_norm = batch_norm

        self.conv = keras.layers.Conv3D(nc, weights, strides=strides, padding='same', kernel_initializer=initialiser)
        
        if batch_norm:
            self.bn = keras.layers.BatchNormalization()

    def call(self, x, training):

        x = self.conv(x)

        if self.batch_norm:
            x = self.bn(x, training)

        return tf.nn.leaky_relu(x, 0.2)


class GANUpBlock(keras.layers.Layer):

    """ Implements up-sampling block for generator
        Input:
            - nc: number of feature maps
            - strides: tuple of strides e.g. (2, 2, 1)
            - initialiser: e.g. keras.initializers.RandomNormal
            - batch_norm: True/False
            - dropout: True/False """

    def __init__(self, nc, strides, initialiser, batch_norm=True, dropout=False):
        super(GANUpBlock, self).__init__(self)
        self.batch_norm = batch_norm
        self.dropout = dropout

        self.tconv = keras.layers.Conv3DTranspose(nc, (4, 4, 2), strides=strides, padding='same')

        if batch_norm:
            self.bn = keras.layers.BatchNormalization()
        if dropout:
            self.dropout = keras.layers.Dropout(0.5)
        
        self.concat = keras.layers.Concatenate()
    
    def call(self, x, skip, training):
        x = self.tconv(x)

        if self.batch_norm:
            x = self.bn(x, training=training)
        
        if self.dropout:
            x = self.dropout(x, training=training)
    
        x = self.concat([x, skip])

        return tf.nn.relu(x)