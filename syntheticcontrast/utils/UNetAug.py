import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import time


""" Based on work found at:
    https://github.com/daviddao/spatial-transformer-tensorflow/blob/master/spatial_transformer.py """


def affine_transformation(input_vol, thetas):

    """ - Implements affine 2D transformation on 3D image volumes
        - input_vol: 3D img volume (mb, height, width, depth, nc)
        - thetas: 2x2 matrix for transform (mb, 2, 2) 
        
        Returns transformed image volume """

    mb_size = input_vol.shape[0]
    height = input_vol.shape[1]
    width = input_vol.shape[2]
    depth = input_vol.shape[3]

    # Generate flattened coordinates and transform
    flat_coords = coord_gen(mb_size, height, width, depth)
    trans_mat = tf.concat([thetas, tf.zeros([mb_size, 2, 1])], axis=2)
    trans_mat = tf.concat([trans_mat, tf.tile(tf.constant([[[0.0, 0.0, 1.0]]]), [mb_size, 1, 1])], axis=1)
    new_coords = tf.matmul(trans_mat, flat_coords)

    # Unroll entire coords
    # These are 1D vectors containing consecutive subsections for each img
    # E.g. X_new = [img1_y1...img1_yn, img2_y1...img2_yn, ... imgn_y1...imgn_yn]
    X_new = tf.reshape(new_coords[:, 0, :], [-1])
    Y_new = tf.reshape(new_coords[:, 1, :], [-1])

    # Perform interpolation on input_vol
    output_vol = interpolate(input_vol, X_new, Y_new, mb_size)

    return output_vol


def coord_gen(mb_size, height, width, depth):

    """ Generates coordinates (mb_size, 3, height * width)
        3rd dim consists of height * width rows for X, Y and ones
        
        Returns flat pixel coordinates """
    
    height_f = tf.cast(height, tf.float32)
    width_f = tf.cast(width, tf.float32)
    X, Y = tf.meshgrid(tf.linspace(-1.0, 1.0, width), tf.linspace(-1.0, 1.0, height))
    flat_X = tf.reshape(X, (1, -1))
    flat_Y = tf.reshape(Y, (1, -1))

    # Rows are X, Y, Z and row of ones (row length is height * width * depth)
    # Replicate for each minibatch
    flat_coords = tf.concat([flat_X, flat_Y, tf.ones((1, height_f * width_f))], axis=0)
    flat_coords = tf.tile(flat_coords[tf.newaxis, :, :], [mb_size, 1, 1])

    return flat_coords


def interpolate(input_vol, X, Y, mb_size):

    """ Implements interpolation of input image volume,
        using deformation fields X, Y 
        - input_vol: input image volume,
        - X: generated X def field
        - Y: generated Y def field
        - mb_size: minibatch size
    
        Returns interpolated image volume of dimension 5 """

    height = input_vol.shape[1]
    width = input_vol.shape[2]
    depth = input_vol.shape[3]
    nc = input_vol.shape[4]
    height_f = tf.cast(height, tf.float32)
    width_f = tf.cast(width, tf.float32)

    # Convert to image coords
    X = (X + 1.0) * width_f / 2.0
    Y = (Y + 1.0) * height_f / 2.0

    # Generate integer coord indices either side of actual value
    x0 = tf.cast(tf.floor(X), tf.int32)
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(Y), tf.int32)
    y1 = y0 + 1

    # Ensure indices don't extend past image height/width
    x0 = tf.clip_by_value(x0, 0, width - 1)
    x1 = tf.clip_by_value(x1, 0, width - 1)
    y0 = tf.clip_by_value(y0, 0, height - 1)
    y1 = tf.clip_by_value(y1, 0, height - 1)

    # Creates a vector of base indices corresponding to each img in mb
    # Allows finding index in unrolled image vector
    base = tf.matmul(tf.reshape(tf.range(mb_size) * height * width, [-1, 1]), tf.cast(tf.ones((1, height * width)), tf.int32))
    base = tf.reshape(base, [-1])

    # Generate 4 vectors of indices corresponding to x0, y0, x1, y1 around base indices
    base_y0 = base + y0 * width
    base_y1 = base + y1 * width
    idx_a = base_y0 + x0
    idx_b = base_y1 + x0
    idx_c = base_y0 + x1
    idx_d = base_y1 + x1

    # Flatten image vector and look up with indices (image vector arranged z1, z2, z3, z1, z2, z3, etc...)
    # Gather pixel values from flattened img based on 4 index vectors
    input_vol_flat = tf.reshape(input_vol, [-1, nc])
    input_vol_flat = tf.cast(input_vol_flat, tf.float32)

    # TODO: FIX THIS
    ImgA1 = tf.gather(input_vol_flat[0::12], idx_a)
    ImgA2 = tf.gather(input_vol_flat[1::12], idx_a)
    ImgA3 = tf.gather(input_vol_flat[2::12], idx_a)
    ImgA4 = tf.gather(input_vol_flat[3::12], idx_a)
    ImgA5 = tf.gather(input_vol_flat[4::12], idx_a)
    ImgA6 = tf.gather(input_vol_flat[5::12], idx_a)
    ImgA7 = tf.gather(input_vol_flat[6::12], idx_a)
    ImgA8 = tf.gather(input_vol_flat[7::12], idx_a)
    ImgA9 = tf.gather(input_vol_flat[8::12], idx_a)
    ImgA10 = tf.gather(input_vol_flat[9::12], idx_a)
    ImgA11 = tf.gather(input_vol_flat[10::12], idx_a)
    ImgA12 = tf.gather(input_vol_flat[11::12], idx_a)

    ImgB1 = tf.gather(input_vol_flat[0::12], idx_b)
    ImgB2 = tf.gather(input_vol_flat[1::12], idx_b)
    ImgB3 = tf.gather(input_vol_flat[2::12], idx_b)
    ImgB4 = tf.gather(input_vol_flat[3::12], idx_b)
    ImgB5 = tf.gather(input_vol_flat[4::12], idx_b)
    ImgB6 = tf.gather(input_vol_flat[5::12], idx_b)
    ImgB7 = tf.gather(input_vol_flat[6::12], idx_b)
    ImgB8 = tf.gather(input_vol_flat[7::12], idx_b)
    ImgB9 = tf.gather(input_vol_flat[8::12], idx_b)
    ImgB10 = tf.gather(input_vol_flat[9::12], idx_b)
    ImgB11 = tf.gather(input_vol_flat[10::12], idx_b)
    ImgB12 = tf.gather(input_vol_flat[11::12], idx_b)

    ImgC1 = tf.gather(input_vol_flat[0::12], idx_c)
    ImgC2 = tf.gather(input_vol_flat[1::12], idx_c)
    ImgC3 = tf.gather(input_vol_flat[2::12], idx_c)
    ImgC4 = tf.gather(input_vol_flat[3::12], idx_c)
    ImgC5 = tf.gather(input_vol_flat[4::12], idx_c)
    ImgC6 = tf.gather(input_vol_flat[5::12], idx_c)
    ImgC7 = tf.gather(input_vol_flat[6::12], idx_c)
    ImgC8 = tf.gather(input_vol_flat[7::12], idx_c)
    ImgC9 = tf.gather(input_vol_flat[8::12], idx_c)
    ImgC10 = tf.gather(input_vol_flat[9::12], idx_c)
    ImgC11 = tf.gather(input_vol_flat[10::12], idx_c)
    ImgC12 = tf.gather(input_vol_flat[11::12], idx_c)

    ImgD1 = tf.gather(input_vol_flat[0::12], idx_d)
    ImgD2 = tf.gather(input_vol_flat[1::12], idx_d)
    ImgD3 = tf.gather(input_vol_flat[2::12], idx_d)
    ImgD4 = tf.gather(input_vol_flat[3::12], idx_d)
    ImgD5 = tf.gather(input_vol_flat[4::12], idx_d)
    ImgD6 = tf.gather(input_vol_flat[5::12], idx_d)
    ImgD7 = tf.gather(input_vol_flat[6::12], idx_d)
    ImgD8 = tf.gather(input_vol_flat[7::12], idx_d)
    ImgD9 = tf.gather(input_vol_flat[8::12], idx_d)
    ImgD10 = tf.gather(input_vol_flat[9::12], idx_d)
    ImgD11 = tf.gather(input_vol_flat[10::12], idx_d)
    ImgD12 = tf.gather(input_vol_flat[11::12], idx_d)

    # Generate vectors of the fractional difference between original indices and rounded indices i.e. weights
    x0_f = tf.cast(x0, 'float32')
    x1_f = tf.cast(x1, 'float32')
    y0_f = tf.cast(y0, 'float32')
    y1_f = tf.cast(y1, 'float32')

    wa = ((x1_f - X) * (y1_f - Y))[:, tf.newaxis]
    wb = ((x1_f - X) * (Y - y0_f))[:, tf.newaxis]
    wc = ((X - x0_f) * (y1_f - Y))[:, tf.newaxis]
    wd = ((X - x0_f) * (Y - y0_f))[:, tf.newaxis]

    # Add weighted imgs from each of four indices and return img_vol
    output_vol_1 = tf.reshape(tf.add_n([wa * ImgA1, wb * ImgB1, wc * ImgC1, wd * ImgD1]), [mb_size, height, width])
    output_vol_2 = tf.reshape(tf.add_n([wa * ImgA2, wb * ImgB2, wc * ImgC2, wd * ImgD2]), [mb_size, height, width])
    output_vol_3 = tf.reshape(tf.add_n([wa * ImgA3, wb * ImgB3, wc * ImgC3, wd * ImgD3]), [mb_size, height, width])
    
    output_vol_4 = tf.reshape(tf.add_n([wa * ImgA4, wb * ImgB4, wc * ImgC4, wd * ImgD4]), [mb_size, height, width])
    output_vol_5 = tf.reshape(tf.add_n([wa * ImgA5, wb * ImgB5, wc * ImgC5, wd * ImgD5]), [mb_size, height, width])
    output_vol_6 = tf.reshape(tf.add_n([wa * ImgA6, wb * ImgB6, wc * ImgC6, wd * ImgD6]), [mb_size, height, width])
    
    output_vol_7 = tf.reshape(tf.add_n([wa * ImgA7, wb * ImgB7, wc * ImgC7, wd * ImgD7]), [mb_size, height, width])
    output_vol_8 = tf.reshape(tf.add_n([wa * ImgA8, wb * ImgB8, wc * ImgC8, wd * ImgD8]), [mb_size, height, width])
    output_vol_9 = tf.reshape(tf.add_n([wa * ImgA9, wb * ImgB9, wc * ImgC9, wd * ImgD9]), [mb_size, height, width])
    
    output_vol_10 = tf.reshape(tf.add_n([wa * ImgA10, wb * ImgB10, wc * ImgC10, wd * ImgD10]), [mb_size, height, width])
    output_vol_11 = tf.reshape(tf.add_n([wa * ImgA11, wb * ImgB11, wc * ImgC11, wd * ImgD11]), [mb_size, height, width])
    output_vol_12 = tf.reshape(tf.add_n([wa * ImgA12, wb * ImgB12, wc * ImgC12, wd * ImgD12]), [mb_size, height, width])

    return tf.stack(
        [
            output_vol_1,
            output_vol_2,
            output_vol_3,
            output_vol_4,
            output_vol_5,
            output_vol_6,
            output_vol_7,
            output_vol_8,
            output_vol_9,
            output_vol_10,
            output_vol_11,
            output_vol_12,
        ], axis=3
    )[:, :, :, :, tf.newaxis]


class TransMatGen:
    """ Implements affine transformation matrix generator """
    # TODO: reimplement being able to pass parameters
    #   - this is simplified version

    def flipMat(self, mb_size):
        # Generates random flip matrix
        flip_mat = tf.round(tf.random.uniform([mb_size, 2, 2], 0, 1))
        flip_mat = (flip_mat * 2) - 1
        flip_mat = flip_mat * tf.tile(tf.eye(2)[tf.newaxis, :, :], [mb_size, 1, 1])

        return flip_mat


    def rotMat(self, mb_size):
        # Generates random rotation matrix
        thetas = tf.random.uniform([mb_size], -90, 90)
        thetas = thetas / 180 * 3.14159265359

        rot_00 = tf.math.cos(thetas)
        rot_01 = -tf.math.sin(thetas)
        rot_10 = tf.math.sin(thetas)
        rot_11 = tf.math.cos(thetas)
        rot_mat = tf.stack([[rot_00, rot_01], [rot_10, rot_11]])
        rot_mat = tf.transpose(rot_mat, [2, 0, 1])

        return rot_mat


    def scaleMat(self, mb_size):
        # Generates random scaling matrix
        z = tf.random.uniform([mb_size], 0.75, 1.25)
        scale_mat = tf.tile(tf.eye(2)[tf.newaxis, :, :], [mb_size, 1, 1])
        scale_mat = (scale_mat * z[:, tf.newaxis, tf.newaxis])

        return scale_mat


    def shearMat(self, phi):
        # Generates random shear matrix - not currently implemented
        # TODO: IMPLEMENT

        phi = phi / 180 * np.pi
        phi = np.random.uniform(-phi, phi)

        p = False
        p = bool(np.random.binomial(1, 0.5))

        shear_mat = np.copy(self._ident_mat)

        if p:
            shear_mat[0, 1] = phi
        else:
            shear_mat[1, 0] = phi

        return shear_mat


    def transMatGen(self, mb_size):
        # TODO: REIMPLEMENT

        # trans_mat = self._ident_mat

        # if flip == None:
        #     pass
        # elif flip < 0 or flip > 1:
        #     raise ValueError("Flip probability out must be between 0 and 1")
        # else:
        #     trans_mat = np.matmul(trans_mat, self.flipMat(flip))
        #
        # if rot != None:
        #     trans_mat = np.matmul(trans_mat, self.rotMat(rot))
        #
        # if scale != None:
        #     trans_mat = np.matmul(trans_mat, self.scaleMat(scale))
        #
        # if shear != None:
        #     trans_mat = np.matmul(trans_mat, self.shearMat(shear))
        trans_mat = tf.matmul(self.flipMat(mb_size), tf.matmul(self.rotMat(mb_size), self.scaleMat(mb_size)))
        # trans_mat = tf.concat([trans_mat, tf.zeros([mb_size, 2, 2])], axis=2)
        # trans_mat = tf.concat([trans_mat, tf.tile(tf.constant([[[0.0, 0.0, 1.0, 0.0]]]), [mb_size, 1, 1])], axis=1)
        
        # flat_mat = tf.zeros((mb_size, 12))
        # flat_mat[:, 0] = trans_mat[:, 0, 0]
        # flat_mat[:, 1] = trans_mat[:, 0, 1]
        # flat_mat[:, 4] = trans_mat[:, 1, 0]
        # flat_mat[:, 5] = trans_mat[:, 1, 1]
        # flat_mat[:, 10] = 1

        return trans_mat


if __name__ == "__main__":
    # """ Tests matrix generator using toy example """
    # TestMatGen = TransMatGen()
    # print(TestMatGen.flipMat())
    # print(TestMatGen.rotMat())
    # print(TestMatGen.scaleMat())
    # print(TestMatGen.transMatGen())

    """ Testing of above functions on toy example """

    # from TransGen import TransMatGen

    start_t = time.time()
    img_vol = np.zeros((4, 128, 128, 12, 1))
    img_vol[:, 40:88, 40:88, :, :] = 1

    TestMatGen = TransMatGen()
    new_vol = affine_transformation(img_vol, TestMatGen.transMatGen(4))
    print(time.time() - start_t)

    for i in range(0, 3):
        fig, axs = plt.subplots(2, 4)
        axs[0, 0].imshow(img_vol[0, :, :, i, 0])
        axs[0, 1].imshow(img_vol[1, :, :, i, 0])
        axs[0, 2].imshow(img_vol[2, :, :, i, 0])
        axs[0, 3].imshow(img_vol[3, :, :, i, 0])
        axs[1, 0].imshow(new_vol[0, :, :, i, 0])
        axs[1, 1].imshow(new_vol[1, :, :, i, 0])
        axs[1, 2].imshow(new_vol[2, :, :, i, 0])
        axs[1, 3].imshow(new_vol[3, :, :, i, 0])
        plt.show()
    
    img1 = np.load("test1.npy")
    img2 = np.load("test2.npy")
    img3 = np.load("test3.npy")
    img4 = np.load("test4.npy")
    imgs = np.stack([img1, img2, img3, img4], axis=0)[:, :, :, :, np.newaxis]

    TestMatGen2 = TransMatGen()
    new_imgs = affine_transformation(imgs, TestMatGen.transMatGen(4))
    print(time.time() - start_t)

    for i in range(0, 3):
        fig, axs = plt.subplots(2, 4)
        axs[0, 0].imshow(np.fliplr(imgs[0, :, :, i, 0].T), origin='lower', cmap='gray', vmin=0.12, vmax=0.18)
        axs[0, 1].imshow(np.fliplr(imgs[1, :, :, i, 0].T), origin='lower', cmap='gray', vmin=0.12, vmax=0.18)
        axs[0, 2].imshow(np.fliplr(imgs[2, :, :, i, 0].T), origin='lower', cmap='gray', vmin=0.12, vmax=0.18)
        axs[0, 3].imshow(np.fliplr(imgs[3, :, :, i, 0].T), origin='lower', cmap='gray', vmin=0.12, vmax=0.18)
        axs[1, 0].imshow(np.fliplr(new_imgs[0, :, :, i, 0].numpy().T), origin='lower', cmap='gray', vmin=0.12, vmax=0.18)
        axs[1, 1].imshow(np.fliplr(new_imgs[1, :, :, i, 0].numpy().T), origin='lower', cmap='gray', vmin=0.12, vmax=0.18)
        axs[1, 2].imshow(np.fliplr(new_imgs[2, :, :, i, 0].numpy().T), origin='lower', cmap='gray', vmin=0.12, vmax=0.18)
        axs[1, 3].imshow(np.fliplr(new_imgs[3, :, :, i, 0].numpy().T), origin='lower', cmap='gray', vmin=0.12, vmax=0.18)
        plt.show()