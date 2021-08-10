import tensorflow as tf


#-------------------------------------------------------------------------

class Linear(tf.keras.layers.Layer):
    def __init__(self, Nz: int, f: int, in_dims: int, out_dims: int, name: str = None):

        """ Inputs:
            - Nz: z vector dims
            - f: kernel height/width
            - in_dims: kernel in channels
            - out_dims: kernel out channels """

        super().__init__(self, name=name)
        # Hidden dims assumed to be Nz
        self.Nz = Nz
        self.f = f
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.init = tf.keras.initializers.TruncatedNormal(0, 1) # Different to modulo 2 technique used in github
    
    def build(self, input_shape):
        # Instead of outputting i matrices using W and b for each i as in paper, combine into one op
        self.Wi = self.add_weight(name='Wi', shape=[self.Nz, self.Nz * self.in_dims], initializer=self.init, trainable=True)
        self.bi = self.add_weight(name='bi', shape=[self.Nz * self.in_dims], initializer="zeros", trainable=True)
        self.Wo = self.add_weight(name='Wo', shape=[self.Nz, self.out_dims * self.f * self.f], initializer=self.init, trainable=True)
        self.bo = self.add_weight(name='bo', shape=[self.out_dims * self.f * self.f], initializer="zeros", trainable=True)

    def call(self, z):
        a = tf.add(tf.multiply(z, self.Wi), self.bi)
        a = tf.reshape(a, [self.in_dims, self.Nz])
        k = tf.add(tf.multiply(a, self.Wo), self.bo)
        k = tf.reshape(k, [self.f, self.f, self.in_dims, self.out_dims])

        return k


#-------------------------------------------------------------------------

class HyperNet(tf.keras.layers.Layer):
    def __init__(self, d: int, f: int, in_dims: int, out_dims: int, name: str = None):
        super().__init__(self, name=name)
        self.d = d                      # z vector dimension
        self.f = f                      # filter height/width
    

    
    def call(self, x):
        return x