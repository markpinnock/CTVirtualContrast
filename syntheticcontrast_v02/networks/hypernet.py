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

        super().__init__(self, name=name)

        # Hidden dims assumed to be Nz
        self.Nz = Nz
        self.f = f
        self.d = d
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.time_dims = 2
        # self.init = tf.keras.initializers.TruncatedNormal(0, 1) # Different to modulo 2 technique used in github repo
        self.init = tf.keras.initializers.GlorotNormal()
        self.time_concat = tf.keras.layers.Concatenate(name="z_concatenate")

    def build(self, input_shape=None):
        # Instead of outputting i matrices using W and b for each i as in paper, combine into one op
        self.Wi = self.add_weight(name='Wi', shape=[self.Nz + self.time_dims, self.Nz * self.in_dims], initializer=self.init, trainable=True)
        self.bi = self.add_weight(name='bi', shape=[self.Nz * self.in_dims], initializer="zeros", trainable=True)
        self.Wo = self.add_weight(name='Wo', shape=[self.Nz, self.out_dims * self.d * self.f * self.f], initializer=self.init, trainable=True)
        self.bo = self.add_weight(name='bo', shape=[self.out_dims * self.d * self.f * self.f], initializer="zeros", trainable=True)

    def call(self, z, source_time, target_time):

        """ Takes layer embedding z, and returns kernel for that layer """
        z = self.time_concat([z, source_time, target_time], axis=1)

        a = tf.add(tf.matmul(z, self.Wi), self.bi)
        a = tf.reshape(a, [self.in_dims, self.Nz])
        k = tf.add(tf.matmul(a, self.Wo), self.bo)
        k = tf.reshape(k, [self.f, self.f, self.d, self.in_dims, self.out_dims])

        return k


#-------------------------------------------------------------------------

class LayerEmbedding(tf.keras.layers.Layer):

    def __init__(self, Nz: int, depth_kernels: int, in_kernels: int, out_kernels: int, name: str = None):

        """ Inputs:
            - Nz: z vector dims
            - depth_kernels: number of kernels to combine to extend in depth direction
            - in_kernels: number of kernels to combine to make in_dims
            - out_kernels: number of kernels to combine to make out_dims """

        super().__init__(self, name=name)

        self.Nz = Nz
        self.depth_kernels = depth_kernels
        self.in_kernels = in_kernels
        self.out_kernels = out_kernels
        self.init = tf.keras.initializers.TruncatedNormal(0, 0.01) # Different to modulo 2 technique used in github repo

    def build(self, input_shape):
        self.z = []
        # for k in range... depthwise
        for i in range(self.depth_kernels):
            t = []

            for j in range(self.in_kernels):
                tt = []

                for k in range(self.out_kernels):
                    tt.append(self.add_weight(name=f"z{self.out_kernels * i + j + k}", shape=[1, self.Nz], initializer=self.init, trainable=True))
            
                t.append(tt)

            self.z.append(t)

    def call(self, h, st, tt):

        """ Takes HyperNetwork as input and converts embedding to kernel """

        ks = tf.concat([tf.concat([tf.concat([h(self.z[i][j][k], st, tt) for k in range(self.out_kernels)], axis=4) for j in range(self.in_kernels)], axis=3) for i in range(self.depth_kernels)], axis=2)

        return ks
