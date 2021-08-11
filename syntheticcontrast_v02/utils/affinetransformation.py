import numpy as np
import tensorflow as tf

from abc import ABC, abstractmethod


""" Based on implementation of spatial transformer networks:
    https://github.com/daviddao/spatial-transformer-tensorflow/blob/master/spatial_transformer.py """

#-------------------------------------------------------------------------
""" Base class for affine transform, subclasses Keras Layer class """

class AffineTransform(tf.keras.layers.Layer, ABC):

    def __init__(self, img_dims: list, name: str):
        super().__init__(name=name)
        self.num_dims = len(img_dims)
        self.height_i = img_dims[0]
        self.width_i = img_dims[1]
        self.height_f = tf.cast(self.height_i, "float32")
        self.width_f = tf.cast(self.width_i, "float32")

        if self.num_dims == 3:
            self.depth_i = 1
            self.n_ch = img_dims[2]
        
        elif self.num_dims == 4:
            self.depth_i = img_dims[2]
            self.n_ch = img_dims[3]
        
        else:
            raise ValueError(f"Invalid image dimensions: {img_dims}")

        self.flat_coords = None

    @abstractmethod
    def coord_gen(self):
        raise NotImplementedError
    
    @abstractmethod
    def transform_coords(self):
        raise NotImplementedError
    
    @abstractmethod
    def get_img_indices(self):
        raise NotImplementedError
    
    @abstractmethod
    def get_weights(self):
        raise NotImplementedError
    
    def interpolate(self):
        raise NotImplementedError

    def call(self, im: object, mb_size: int, thetas: object) -> object:
        self.transform_coords(mb_size, thetas)
        base, indices = self.get_img_indices()
        weights = self.get_weights(*indices)
        output = self.interpolate(im, base, weights, *indices)

        if self.num_dims == 3:
            return tf.reshape(output, [mb_size, self.height_i, self.width_i, self.n_ch])
        else:
            return tf.reshape(output, [mb_size, self.height_i, self.width_i, self.depth_i, self.n_ch])

#-------------------------------------------------------------------------
""" 2D affine transform class, acts on 2D images and also depth-wise on 3D volumes """

class AffineTransform2D(AffineTransform):

    def __init__(self, img_dims: list, name: str = "affine2D"):
        super().__init__(img_dims, name=name)
        self.mb_size = None
        self.X, self.Y = None, None
        self.coord_gen()

    def coord_gen(self) -> None:
        """ Generate flattened coordinates [3, height * width] """

        # Coords in range [-1, 1] (assuming origin in centre)
        X, Y = tf.meshgrid(tf.linspace(-1.0, 1.0, self.width_i), tf.linspace(-1.0, 1.0, self.height_i))
        flat_X = tf.reshape(X, (1, -1))
        flat_Y = tf.reshape(Y, (1, -1))

        # Rows are X, Y and row of ones (row length is height * width)
        self.flat_coords = tf.concat([flat_X, flat_Y, tf.ones((1, self.height_i * self.width_i))], axis=0)

    def transform_coords(self, mb_size: int, thetas: object) -> None:
        """ Transform flattened coordinates with transformation matrix theta
            thetas: 6 params for transform [mb, 6] """

        self.mb_size = mb_size
        new_flat_coords = tf.tile(self.flat_coords[tf.newaxis, :, :], [mb_size, 1, 1])
        thetas = tf.reshape(thetas, [-1, 2, 3])
        new_flat_coords = tf.matmul(thetas, new_flat_coords)

        # Unroll coords
        # These are 1D vectors containing consecutive X/Y coords for each img
        # E.g. X = [img1_x1...img1_xn, img2_x1...img2_xn, ... imgn_x1...imgn_xn]
        self.X = tf.reshape(new_flat_coords[:, 0, :], [-1])
        self.Y = tf.reshape(new_flat_coords[:, 1, :], [-1])
    
    def get_img_indices(self) -> tuple:
        """ Generates base indices corresponding to each image in mb
            e.g. [0   0   0
                  hw  hw  hw
                  2hw 2hw 2hw]

            where hw = height * width
            Allows selecting e.g. x, y pixel in second img in minibatch by selecting hw + x + y """

        # Convert coords to [0, width/height]
        self.X = (self.X + 1.0) / 2.0 * (self.width_f)
        self.Y = (self.Y + 1.0) / 2.0 * (self.height_f)

        # Generate integer indices bracketing transformed coordinates
        x0 = tf.cast(tf.floor(self.X), "int32")
        x1 = x0 + 1
        y0 = tf.cast(tf.floor(self.Y), "int32")
        y1 = y0 + 1

        # Ensure indices don't extend past image height/width
        x0 = tf.clip_by_value(x0, 0, self.width_i - 1)
        x1 = tf.clip_by_value(x1, 0, self.width_i - 1)
        y0 = tf.clip_by_value(y0, 0, self.height_i - 1)
        y1 = tf.clip_by_value(y1, 0, self.height_i - 1)

        # Creates a vector of base indices corresponding to each img in mb
        # Allows finding pixel in unrolled image vector
        img_indices = tf.reshape(tf.range(self.mb_size) * self.height_i * self.width_i, [-1, 1])
        img_indices = tf.matmul(img_indices, tf.ones((1, self.height_i * self.width_i), dtype="int32"))
        img_indices = tf.reshape(img_indices, [-1])

        return img_indices, (x0, x1, y0, y1)
    
    def get_weights(self, x0, x1, y0, y1) -> list:
        """ Generate weights representing how close bracketing indices are to transformed coords """

        x0_f = tf.cast(x0, "float32")
        x1_f = tf.cast(x1, "float32")
        y0_f = tf.cast(y0, "float32")
        y1_f = tf.cast(y1, "float32")

        wa = tf.expand_dims((x1_f - self.X) * (y1_f - self.Y), 1)
        wb = tf.expand_dims((x1_f - self.X) * (self.Y - y0_f), 1)
        wc = tf.expand_dims((self.X - x0_f) * (y1_f - self.Y), 1)
        wd = tf.expand_dims((self.X - x0_f) * (self.Y - y0_f), 1)

        return [wa, wb, wc, wd]

    def interpolate(self, im: object, base: object, weights: list, x0, x1, y0, y1) -> object:
        """ Perform interpolation of minibatch of images """

        # Add base image indices to the integer indices bracketing the transformed coordinates
        indices = []
        indices.append(base + y0 * self.width_i + x0)
        indices.append(base + y1 * self.width_i + x0)
        indices.append(base + y0 * self.width_i + x1)
        indices.append(base + y1 * self.width_i + x1)

        # Get images using bracketed indices and take weighted average
        im_flat = tf.reshape(im, [-1, self.depth_i * self.n_ch])
        imgs = [tf.gather(im_flat, idx) for idx in indices]

        return tf.add_n([img * weight for img, weight in zip(imgs, weights)])

#-------------------------------------------------------------------------
""" 3D affine transform class, not yet functional """

class AffineTransform3D(AffineTransform):

    def __init__(self, img_dims: list, name: str = "affine2D"):
        super().__init__(img_dims, name=name)
        assert self.num_dims == 4, f"Invalid image dimensions: {self.num_dims}"
        self.depth_f = tf.cast(self.depth_i, tf.float32)
        self.mb_size = None
        self.X, self.Y, self.Z = None, None, None
        self.coord_gen()
    
    def coord_gen(self) -> None:
        """ Generate flattened coordinates [4, height * width * depth] """

        X, Y, Z = tf.meshgrid(tf.linspace(-1.0, 1.0, self.width_i), tf.linspace(-1.0, 1.0, self.height_i), tf.linspace(-1.0, 1.0, self.depth_i))
        flat_X = tf.reshape(X, (1, -1))
        flat_Y = tf.reshape(Y, (1, -1))
        flat_Z = tf.reshape(Z, (1, -1))

        # Rows are X, Y, Z and row of ones (row length is height * width * depth)
        self.flat_coords = tf.concat([flat_X, flat_Y, flat_Z, tf.ones((1, self.height_i * self.width_i * self.depth_i))], axis=0)
    
    def transform_coords(self, mb_size: int, thetas: object) -> None:
        """ Transform flattened coordinates with transformation matrix
            thetas: 12 params for transform [mb, 12] """

        self.mb_size = mb_size
        new_flat_coords = tf.tile(self.flat_coords[tf.newaxis, :, :], [mb_size, 1, 1])
        thetas = tf.reshape(thetas, [-1, 3, 4])
        new_flat_coords = tf.matmul(thetas, new_flat_coords)

        # Unroll coords
        # These are 1D vectors containing consecutive X/Y/Z coords for each img
        # E.g. X = [img1_x1...img1_xn, img2_x1...img2_xn, ... imgn_x1...imgn_xn]
        self.X = tf.reshape(new_flat_coords[:, 0, :], [-1])
        self.Y = tf.reshape(new_flat_coords[:, 1, :], [-1])
        self.Z = tf.reshape(new_flat_coords[:, 2, :], [-1])

    def get_img_indices(self) -> tuple:
        """ Generates base indices corresponding to each image in mb
            e.g. [0    0    0
                  hwd  hwd  hwd
                  2hwd 2hwd 2hwd]

            where hw = height * width * depth
            Allows selecting e.g. x, y pixel in second img in minibatch by selecting hw + x + y + z"""

        # Convert coords to [0, width/height/depth]
        self.X = (self.X + 1.0) / 2.0 * (self.width_f)
        self.Y = (self.Y + 1.0) / 2.0 * (self.height_f)
        self.Z = (self.Z + 1.0) / 2.0 * (self.depth_f)

        # Generate integer indices bracketing transformed coordinates
        x0 = tf.cast(tf.floor(self.X), "int32")
        x1 = x0 + 1
        y0 = tf.cast(tf.floor(self.Y), "int32")
        y1 = y0 + 1
        z0 = tf.cast(tf.floor(self.Z), "int32")
        z1 = z0 + 1

        # Ensure indices don't extend past image height/width
        x0 = tf.clip_by_value(x0, 0, self.width_i - 1)
        x1 = tf.clip_by_value(x1, 0, self.width_i - 1)
        y0 = tf.clip_by_value(y0, 0, self.height_i - 1)
        y1 = tf.clip_by_value(y1, 0, self.height_i - 1)
        z0 = tf.clip_by_value(z0, 0, self.depth_i - 1)
        z1 = tf.clip_by_value(z1, 0, self.depth_i - 1)

        # Creates a vector of base indices corresponding to each img in mb
        # Allows finding pixel in unrolled image vector
        img_indices = tf.reshape(tf.range(self.mb_size) * self.height_i * self.width_i * self.depth_i, [-1, 1])
        img_indices = tf.matmul(img_indices, tf.ones((1, self.height_i * self.width_i * self.depth_i), dtype="int32"))
        img_indices = tf.reshape(img_indices, [-1])

        return img_indices, (x0, x1, y0, y1, z0, z1)
    
    def get_weights(self):
        """ Generate weights representing how close bracketing indices are to transformed coords """
        return super().get_weights()
    
    def interpolate(self, im, base, weights, x0, x1, y0, y1, z0, z1):
        return super().interpolate()


def interpolate(input_vol, X, Y, Z, mb_size):
    """ Performs interpolation input_vol, using deformation fields X, Y, Z """

    # Generate 8 vectors of indices corresponding to x0, y0, x1, y1 around base indices
    base_y0 = base + y0 * depth * width
    base_y1 = base + y1 * depth * width
    base_y0_x0 = base_y0 + x0 * height
    base_y1_x0 = base_y1 + x0 * height
    base_y0_x1 = base_y0 + x1 * height
    base_y1_x1 = base_y1 + x1 * height
    idx_a = base_y0_x0 + z0
    idx_b = base_y1_x0 + z0
    idx_c = base_y0_x1 + z0
    idx_d = base_y1_x1 + z0
    idx_e = base_y0_x0 + z1
    idx_f = base_y1_x0 + z1
    idx_g = base_y0_x1 + z1
    idx_h = base_y1_x1 + z1

    # Flatten image vector and look up with indices
    # Gather pixel values from flattened img based on 4 index vectors
    input_vol_flat = tf.reshape(input_vol, [-1, nc])
    input_vol_flat = tf.cast(input_vol_flat, tf.float32)
    ImgA = tf.gather(input_vol_flat, idx_a)
    ImgB = tf.gather(input_vol_flat, idx_b)
    ImgC = tf.gather(input_vol_flat, idx_c)
    ImgD = tf.gather(input_vol_flat, idx_d)
    ImgE = tf.gather(input_vol_flat, idx_e)
    ImgF = tf.gather(input_vol_flat, idx_f)
    ImgG = tf.gather(input_vol_flat, idx_g)
    ImgH = tf.gather(input_vol_flat, idx_h)

    # Generate vectors of the fractional difference between original indices and rounded indices i.e. weights
    x0_f = tf.cast(x0, 'float32')
    x1_f = tf.cast(x1, 'float32')
    y0_f = tf.cast(y0, 'float32')
    y1_f = tf.cast(y1, 'float32')
    z0_f = tf.cast(z0, 'float32')
    z1_f = tf.cast(z1, 'float32')
    wa = tf.expand_dims(((x1_f - X) * (y1_f - Y) * (z1_f - Z)), 1)
    wb = tf.expand_dims(((x1_f - X) * (Y - y0_f) * (z1_f - Z)), 1)
    wc = tf.expand_dims(((X - x0_f) * (y1_f - Y) * (z1_f - Z)), 1)
    wd = tf.expand_dims(((X - x0_f) * (Y - y0_f) * (z1_f - Z)), 1)
    we = tf.expand_dims(((x1_f - X) * (y1_f - Y) * (Z - z0_f)), 1)
    wf = tf.expand_dims(((x1_f - X) * (Y - y0_f) * (Z - z0_f)), 1)
    wg = tf.expand_dims(((X - x0_f) * (y1_f - Y) * (Z - z0_f)), 1)
    wh = tf.expand_dims(((X - x0_f) * (Y - y0_f) * (Z - z0_f)), 1)

    # Add weighted imgs from each of four indices and return img_vol
    output_vol = tf.add_n(
        [wa * ImgA, wb * ImgB, wc * ImgC, wd * ImgD, we * ImgE, wf * ImgF, wg * ImgG, wh * ImgH])

    return output_vol
