import tensorflow as tf


""" https://github.com/g1910/HyperNetworks """

#-------------------------------------------------------------------------

class HyperNet(tf.keras.layers.Layer):

    def __init__(self, Nz: int, f: int, d: int, in_dims: int, out_dims: int, name: str = None):

        """ Inputs:
            - Nz: z vector dims
            - f: kernel height/width
            - d: kernel depth
            - in_dims: kernel in channels (in first layer)
            - out_dims: kernel out channels (in first layer) """

        super().__init__(name=name)

        # Hidden dims assumed to be Nz
        self.Nz = Nz
        self.f = f
        self.d = d
        self.in_dims = in_dims
        self.out_dims = out_dims

    def build(self, input_shape=None):
        # Instead of outputting i matrices using W and b for each i as in paper, combine into one op
        self.Wi = self.add_weight(name='Wi', shape=[self.Nz, self.Nz * self.in_dims], initializer="glorot_normal", trainable=True)
        self.bi = self.add_weight(name='bi', shape=[self.Nz * self.in_dims], initializer="zeros", trainable=True)
        self.Wo = self.add_weight(name='Wo', shape=[self.Nz, self.out_dims * self.d * self.f * self.f], initializer="glorot_normal", trainable=True)
        self.bo = self.add_weight(name='bo', shape=[self.out_dims * self.d * self.f * self.f], initializer="zeros", trainable=True)

    def call(self, z, num_kernels: int = 1):

        """ Takes layer embedding z, and returns kernel for that layer """

        a = tf.add(tf.matmul(z, self.Wi), self.bi)
        a = tf.reshape(a, [num_kernels, self.in_dims, self.Nz])
        k = tf.add(tf.matmul(a, self.Wo), self.bo)
        k = tf.reshape(k, [num_kernels, self.f, self.f, self.d, self.in_dims, self.out_dims])

        return k


#-------------------------------------------------------------------------

class LayerEmbedding(tf.keras.layers.Layer):

    def __init__(self, Nz: int, depth_kernels: int, in_kernels: int, out_kernels: int, name: str = None):

        """ Inputs:
            - Nz: z vector dims
            - depth_kernels: number of kernels to combine to extend in depth direction
            - in_kernels: number of kernels to combine to make in_dims
            - out_kernels: number of kernels to combine to make out_dims """

        super().__init__(name=name)

        self.Nz = Nz
        self.depth_kernels = depth_kernels
        self.in_kernels = in_kernels
        self.out_kernels = out_kernels
        self.concat = tf.keras.layers.Concatenate()

    def build(self, input_shape):
        self.z = []

        for i in range(self.depth_kernels):
            t = []

            for j in range(self.in_kernels):
                tt = []

                for k in range(self.out_kernels):
                    tt.append(self.add_weight(name=f"z{self.out_kernels * i + j + k}", shape=[1, self.Nz], initializer="glorot_normal", trainable=True))
            
                t.append(tt)

            self.z.append(t)

    def call(self, h, t=None):

        """ Takes HyperNetwork as input and converts embedding to kernel """
    
        if t is None:
            ks = tf.concat([tf.concat([tf.concat([h(self.z[i][j][k], num_kernels=1) for k in range(self.out_kernels)], axis=5) for j in range(self.in_kernels)], axis=4) for i in range(self.depth_kernels)], axis=3)

        else:
            ks = tf.concat([tf.concat([tf.concat([h(tf.matmul(t[:, tf.newaxis], self.z[i][j][k]), num_kernels=t.shape[0]) for k in range(self.out_kernels)], axis=5) for j in range(self.in_kernels)], axis=4) for i in range(self.depth_kernels)], axis=3)

        return ks
