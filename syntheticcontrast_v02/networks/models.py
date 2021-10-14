import numpy as np
import tensorflow as tf

from .layers import GANDownBlock, GANUpBlock
from .hypernet_v01 import HyperNet_v01, LayerEmbedding
from .hypernet_v02 import HyperNetwork_v02
from .hypernetlayers import HyperGANDownBlock, HyperGANUpBlock


#-------------------------------------------------------------------------
""" PatchGAN discriminator for Pix2pix """

class Discriminator(tf.keras.Model):

    """ Input:
        - initialiser: e.g. keras.initializers.RandomNormal
        - config: configuration json
        Returns:
        - keras.Model """

    def __init__(self, initialiser, config, name=None):
        super().__init__(name=name)
    
        # Check network and image dimensions
        img_dims = config["img_dims"]
        assert len(img_dims) == 3, "3D input only"
        max_num_layers = int(np.log2(np.min([img_dims[0], img_dims[1]]) / 4))
        max_z_downsample = int(np.floor(np.log2(img_dims[2])))

        ndf = config["ndf"]
        num_layers = config["d_layers"]
       
        assert num_layers <= max_num_layers and num_layers >= 0, f"Maximum numnber of discriminator layers: {max_num_layers}"
        self.conv_list = []

        # TODO: FIX
        # PixelGAN i.e. 1x1 receptive field
        if num_layers == 0:
            self.conv_list.append(
                GANDownBlock(
                    ndf, (1, 1, 1),
                    (1, 1, 1),
                    initialiser=initialiser,
                    batch_norm=False)) 
            
            self.conv_list.append(
                GANDownBlock(
                    ndf * 2,
                    (1, 1, 1),
                    (1, 1, 1),
                    initialiser=initialiser,
                    batch_norm=True))

            self.conv_list.append(tf.keras.layers.Conv3D(
                1, (1, 1, 1), (1, 1, 1),
                padding='same',
                kernel_initializer=initialiser))       

        # PatchGAN i.e. NxN receptive field
        else:
            batch_norm = False

            for i in range(0, num_layers):
                if i > 0: batch_norm = True
                channels = tf.minimum(ndf * 2 ** i, 512)

                if i > max_z_downsample:
                    strides = (2, 2, 1)
                    kernel = (4, 4, 2)
                else:
                    strides = (2, 2, 2)
                    kernel = (4, 4, 4)
                
                self.conv_list.append(
                    GANDownBlock(
                        channels,
                        kernel,
                        strides,
                        initialiser=initialiser,
                        batch_norm=batch_norm, name=f"downblock_{i}"))
            
            self.conv_list.append(tf.keras.layers.Conv3D(
                1, (4, 4, 1), (1, 1, 1),
                padding='valid',
                kernel_initializer=initialiser, name="output"))

    def build_model(self, x):

        """ Build method takes tf.zeros((input_dims)) and returns
            shape of output - all layers implicitly built and weights set to trainable """

        return self(x).shape

    def call(self, x, training=True):

        for conv in self.conv_list:
            x = conv(x, training=training)

        return x

#-------------------------------------------------------------------------
""" Generator for Pix2pix """

class Generator(tf.keras.Model):

    """ Input:
        - initialiser e.g. keras.initializers.RandomNormal
        - nc: number of channels in first layer
        - num_layers: number of layers
        - img_dims: input image size
        Returns:
        - keras.Model """

    def __init__(self, initialiser, config, name=None):
        super().__init__(name=name)

        # Check network and image dimensions
        img_dims = config["img_dims"]
        assert len(img_dims) == 3, "3D input only"
        max_num_layers = int(np.log2(np.min([img_dims[0], img_dims[1]])))
        max_z_downsample = int(np.floor(np.log2(img_dims[2])))
        ngf = config["ngf"]
        num_layers = config["g_layers"]
        
        if config["g_time_layers"] is not None:
            self.time_layers = config["g_time_layers"]
        else:
           self.time_layers = []

        assert num_layers <= max_num_layers and num_layers >= 0, f"Maximum number of generator layers: {max_num_layers}"
        self.encoder = []

        # Cache channels, strides and weights
        cache = {"channels": [], "strides": [], "kernels": []}

        for i in range(0, num_layers - 1):
            channels = np.min([ngf * 2 ** i, 512])

            if i >= max_z_downsample - 1:
                strides = (2, 2, 1)
                kernel = (4, 4, 2)
            else:
                strides = (2, 2, 2)
                kernel = (4, 4, 4)

            cache["channels"].append(channels)
            cache["strides"].append(strides)
            cache["kernels"].append(kernel)

            self.encoder.append(
                GANDownBlock(
                    channels,
                    kernel,
                    strides,
                    initialiser=initialiser,
                    batch_norm=True, name=f"down_{i}"))

        self.bottom_layer = GANDownBlock(
            channels,
            kernel,
            strides,
            initialiser=initialiser,
            batch_norm=True, name="bottom")

        cache["strides"].append(strides)
        cache["kernels"].append(kernel)

        cache["channels"].reverse()
        cache["kernels"].reverse()
        cache["strides"].reverse()

        self.decoder = []
        dropout = True

        for i in range(0, num_layers - 1):
            if i > 2: dropout = False
            channels = cache["channels"][i]
            strides = cache["strides"][i]
            kernel = cache["kernels"][i]

            self.decoder.append(
                GANUpBlock(
                    channels,
                    kernel,
                    strides,
                    initialiser=initialiser,
                    batch_norm=True,
                    dropout=dropout, name=f"up_{i}"))

        self.final_layer = tf.keras.layers.Conv3DTranspose(
            1, (4, 4, 4), (2, 2, 2),
            padding="same", activation="linear",
            kernel_initializer=initialiser, name="output")
        
        layer_names = [layer.name for layer in self.encoder] + ["bottom"] + [layer.name for layer in self.decoder]

        for time_input in self.time_layers:
            assert time_input in layer_names, (time_input, layer_names)

    def build_model(self, x, st=None, tt=None):

        """ Build method takes tf.zeros((input_dims)) and returns
            shape of output - all layers implicitly built and weights set to trainable """
        
        return self(x, st, tt).shape

    def call(self, x, st=None, tt=None):
        skip_layers = []

        for conv in self.encoder:
            if conv.name in self.time_layers:
                x = conv(x, st, tt, training=True)
            else:
                x = conv(x, training=True)
            skip_layers.append(x)

        if self.bottom_layer.name in self.time_layers:
            x = self.bottom_layer(x, st, tt, training=True)
        else:
            x = self.bottom_layer(x, training=True)

        x = tf.nn.relu(x)
        skip_layers.reverse()

        for skip, tconv in zip(skip_layers, self.decoder):
            if tconv.name in self.time_layers:
                x = tconv(x, skip, st, tt, training=True)
            else:
                x = tconv(x, skip, training=True)

        if self.final_layer.name in self.time_layers:

            x = self.final_layer(x, st, tt, training=True)
        else:
            x = self.final_layer(x, training=True)

        return x


#-------------------------------------------------------------------------
""" Generator for Pix2pix with HyperNetwork """

class HyperGenerator_v01(tf.keras.Model):

    """ Input:
        - initialiser e.g. keras.initializers.RandomNormal
        - nc: number of channels in first layer
        - num_layers: number of layers
        - img_dims: input image size
        Returns:
        - keras.Model """

    def __init__(self, initialiser, config, name=None):
        super().__init__(name=name)

        # Check network and image dimensions
        img_dims = config["img_dims"]
        assert len(img_dims) == 3, "3D input only"
        max_num_layers = int(np.log2(np.min([img_dims[0], img_dims[1]])))
        max_z_downsample = int(np.floor(np.log2(img_dims[2])))
        ngf = config["ngf"]
        num_layers = config["g_layers"]
        assert num_layers <= max_num_layers and num_layers >= 0, f"Maximum number of generator layers: {max_num_layers}"
        
        self.encoder = []
        self.encoder_embedding = []

        # Initialise HyperNet
        self.Hypernet = HyperNet_v01(Nz=config["Nz"], f=4, d=2, in_dims=ngf, out_dims=ngf, name="HyperNet")

        # Cache channels, strides and weights
        cache = {"channels": [], "strides": [], "kernels": []}

        cache["channels"].append(ngf)
        cache["strides"].append((2, 2, 2))
        cache["kernels"].append((4, 4, 4))

        # First layer is standard non-HyperNet conv layer
        self.first_layer = GANDownBlock(
            ngf, (4, 4, 4), (2, 2, 2),
            initialiser=initialiser, batch_norm=True, name="down_0")

        for i in range(1, num_layers - 1):
            channels = np.min([ngf * 2 ** i, 512])

            if i >= max_z_downsample - 1:
                strides = (2, 2, 1)
                kernel = (4, 4, 2)
            else:
                strides = (2, 2, 2)
                kernel = (4, 4, 4)

            cache["channels"].append(channels)
            cache["strides"].append(strides)
            cache["kernels"].append(kernel)

            self.encoder.append(HyperGANDownBlock(strides, batch_norm=True, name=f"down_{i}"))
            self.encoder_embedding.append(
                LayerEmbedding(
                    Nz=config["Nz"], depth_kernels=kernel[-1] // 2, in_kernels=cache["channels"][-2] // ngf, out_kernels=cache["channels"][-1] // ngf, name=f"z_down_{i}"))

        self.bottom_layer = HyperGANDownBlock(strides, batch_norm=True, name="bottom")
        self.bottom_embedding = LayerEmbedding(
            Nz=config["Nz"], depth_kernels=cache["kernels"][-1][-1] // 2, in_kernels=cache["channels"][-1] // ngf, out_kernels=cache["channels"][-1] // ngf, name="z_bottom")

        cache["strides"].append(strides)
        cache["kernels"].append(kernel)

        cache["channels"].reverse()
        cache["kernels"].reverse()
        cache["strides"].reverse()

        self.decoder = []
        self.decoder_embedding_1 = []
        self.decoder_embedding_2 = []
        dropout = True

        self.decoder.append(HyperGANUpBlock(strides, batch_norm=True, dropout=dropout, name="up_0"))
        self.decoder_embedding_1.append(
            LayerEmbedding(
                Nz=config["Nz"], depth_kernels=cache["kernels"][0][-1] // 2, in_kernels=cache["channels"][0] // ngf, out_kernels=cache["channels"][0] // ngf, name="z_up_0"))

        self.decoder_embedding_2.append(
            LayerEmbedding(
                Nz=config["Nz"], depth_kernels=cache["kernels"][0][-1] // 2, in_kernels=cache["channels"][0] // ngf * 2, out_kernels=cache["channels"][0] // ngf, name="z_up_0"))

        for i in range(1, num_layers - 1):
            if i > 2: dropout = False
            strides = cache["strides"][i]
            kernel = cache["kernels"][i]
            
            self.decoder.append(HyperGANUpBlock(strides, batch_norm=True, dropout=dropout, name=f"up_{i}"))
            self.decoder_embedding_1.append(
                LayerEmbedding(
                    Nz=config["Nz"], depth_kernels=kernel[-1] // 2, in_kernels=cache["channels"][i - 1] // ngf, out_kernels=cache["channels"][i] // ngf, name=f"z_up_{i}"))

            self.decoder_embedding_2.append(
                LayerEmbedding(
                    Nz=config["Nz"], depth_kernels=kernel[-1] // 2, in_kernels=cache["channels"][i] // ngf * 2, out_kernels=cache["channels"][i] // ngf, name=f"z_up_{i}"))

        # Last layer is standard non-HyperNet conv layer
        self.final_layer = tf.keras.layers.Conv3DTranspose(
            1, (4, 4, 4), (2, 2, 2),
            padding="same", activation="linear",
            kernel_initializer=initialiser, name="output")

    def build_model(self, x, st=None, tt=None):

        """ Build method takes tf.zeros((input_dims)) and returns
            shape of output - all layers implicitly built and weights set to trainable """
        
        return self(x, st, tt).shape

    def call(self, x, st=None, tt=None):
        skip_layers = []

        x = self.first_layer(x, training=True)
        skip_layers.append(x)

        for conv, embedding in zip(self.encoder, self.encoder_embedding):
            w = embedding(self.Hypernet, st, tt)
            x = conv(x, w, training=True)
            skip_layers.append(x)
        
        w = self.bottom_embedding(self.Hypernet, st, tt)
        x = self.bottom_layer(x, w, training=True)
        x = tf.nn.relu(x)

        skip_layers.reverse()

        for skip, conv, embedding_1, embedding_2 in zip(skip_layers, self.decoder, self.decoder_embedding_1, self.decoder_embedding_2):
            w1 = embedding_1(self.Hypernet, st, tt)
            w2 = embedding_2(self.Hypernet, st, tt)
            x = conv(x, w1, w2, skip, training=True)

        x = self.final_layer(x)

        return x


#-------------------------------------------------------------------------
""" Generator for Pix2pix with HyperNetwork """

class HyperGenerator_v02(tf.keras.Model):

    """ Input:
        - initialiser e.g. keras.initializers.RandomNormal
        - nc: number of channels in first layer
        - num_layers: number of layers
        - img_dims: input image size
        Returns:
        - keras.Model """

    def __init__(self, initialiser, config, name=None):
        super().__init__(name=name)

        # Check network and image dimensions
        img_dims = config["img_dims"]
        assert len(img_dims) == 3, "3D input only"
        max_num_layers = int(np.log2(np.min([img_dims[0], img_dims[1]])))
        max_z_downsample = int(np.floor(np.log2(img_dims[2])))
        ngf = config["ngf"]
        num_layers = config["g_layers"]
        assert num_layers <= max_num_layers and num_layers >= 0, f"Maximum number of generator layers: {max_num_layers}"
        
        self.encoder = []
        self.kernel_dims = {}

        # Cache channels, strides and weights
        cache = {"channels": [], "strides": [], "kernels": []}

        for i in range(0, num_layers - 1):
            channels = np.min([ngf * 2 ** i, 512])

            if i >= max_z_downsample - 1:
                strides = (2, 2, 1)
                kernel = (4, 4, 2)
            else:
                strides = (2, 2, 2)
                kernel = (4, 4, 4)

            cache["channels"].append(channels)
            cache["strides"].append(strides)
            cache["kernels"].append(kernel)

            name = f"down_{i}"
            self.encoder.append(HyperGANDownBlock(strides, batch_norm=True, name=name))
            
            if i == 0:
                self.kernel_dims[name] = list(kernel) + [1] + [channels]
            else:
                self.kernel_dims[name] = list(kernel) + [ngf * 2 ** (i - 1)] + [channels]

        cache["strides"].append(strides)
        cache["kernels"].append(kernel)

        self.bottom_layer = HyperGANDownBlock(strides, batch_norm=True, name="bottom")
        self.kernel_dims["bottom"] = list(kernel) + [cache["channels"][-1]] * 2

        cache["channels"].reverse()
        cache["kernels"].reverse()
        cache["strides"].reverse()

        self.decoder = []
        dropout = True

        for i in range(0, num_layers - 1):
            if i > 2: dropout = False
            channels = cache["channels"][i]
            strides = cache["strides"][i]
            kernel = cache["kernels"][i]

            name = f"up_{i}"
            self.decoder.append(HyperGANUpBlock(strides, batch_norm=True, dropout=dropout, name=name))
            self.kernel_dims[f"{name}A"] = list(kernel) + [cache["channels"][i - 1]] + [channels]
            self.kernel_dims[f"{name}B"] = list(kernel) + [channels * 2] + [channels]

        # Last layer is standard non-HyperNet conv layer
        self.final_layer = tf.keras.layers.Conv3DTranspose(
            1, (4, 4, 4), (2, 2, 2),
            padding="same", activation="linear",
            kernel_initializer=initialiser, name="output")

        self.HyperNet = HyperNetwork_v02(self.kernel_dims, name="HyperNet")

    def build_model(self, x, st=None, tt=None):

        """ Build method takes tf.zeros((input_dims)) and returns
            shape of output - all layers implicitly built and weights set to trainable """
        
        return self(x, st, tt).shape

    def call(self, x, st=None, tt=None):
        skip_layers = []

        w = self.HyperNet(st, tt)

        for conv in self.encoder:
            x = conv(x, w[conv.name], training=True)
            skip_layers.append(x)
        
        x = self.bottom_layer(x, w["bottom"], training=True)
        x = tf.nn.relu(x)

        skip_layers.reverse()

        for skip, conv in zip(skip_layers, self.decoder):
            x = conv(x, w[f"{conv.name}A"], w[f"{conv.name}B"], skip, training=True)

        x = self.final_layer(x)

        return x
