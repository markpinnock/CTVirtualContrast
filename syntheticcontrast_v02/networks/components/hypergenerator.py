import numpy as np
import tensorflow as tf

from .layer.hypernet import HyperNet, LayerEmbedding
from .layer.layers import GANDownBlock
from .layer.hypernetlayers import HyperGANDownBlock, HyperGANUpBlock


""" Generator for Pix2pix with HyperNetwork """

class HyperGenerator(tf.keras.Model):

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
        self.Hypernet = HyperNet(Nz=config["Nz"], f=4, d=2, in_dims=ngf, out_dims=ngf, name="HyperNet")

        # Cache channels, strides and weights
        cache = {"channels": [], "strides": [], "kernels": []}

        cache["channels"].append(ngf)
        cache["strides"].append((2, 2, 2))
        cache["kernels"].append((4, 4, 4))

        # First layer is standard non-HyperNet conv layer
        self.first_layer = GANDownBlock(
            ngf, (4, 4, 4), (2, 2, 2),
            initialiser=initialiser, model="generator", batch_norm=True, name="down_0")

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

    def build_model(self, x, t=None):

        """ Build method takes tf.zeros((input_dims)) and returns
            shape of output - all layers implicitly built and weights set to trainable """
        
        return self(x, t).shape

    def call(self, x, t=None):
        skip_layers = []

        x = self.first_layer(x, training=True)
        skip_layers.append(x)

        for conv, embedding in zip(self.encoder, self.encoder_embedding):
            w = embedding(self.Hypernet, t)
            x = conv(x, w, training=True)
            skip_layers.append(x)
        
        w = self.bottom_embedding(self.Hypernet, t)
        x = self.bottom_layer(x, w, training=True)
        x = tf.nn.relu(x)

        skip_layers.reverse()

        for skip, conv, embedding_1, embedding_2 in zip(skip_layers, self.decoder, self.decoder_embedding_1, self.decoder_embedding_2):
            w1 = embedding_1(self.Hypernet, t)
            w2 = embedding_2(self.Hypernet, t)
            x = conv(x, w1, w2, skip, training=True)

        x = self.final_layer(x)

        return x
