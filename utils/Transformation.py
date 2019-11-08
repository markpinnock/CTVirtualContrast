import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import time


""" Based on work by Tensorflow authors, found at:
    https://github.com/daviddao/spatial-transformer-tensorflow/blob/master/spatial_transformer.py """


def affineTransformation(input_vol, thetas):
    """ input_vol: 3D img volume (mb, height, width, depth, nc)
        thetas: 6 params for transform (mb, 6) """

    mb_size = input_vol.shape[0]
    height = input_vol.shape[1]
    width = input_vol.shape[2]

    # Generate flattened coordinates and transform
    flat_coords = coordGen(mb_size, height, width)
    trans_mat = tf.reshape(thetas, (-1, 2, 3))
    new_coords = tf.matmul(trans_mat, flat_coords)

    # Unroll entire coords
    # These are 1D vectors containing consecutive subsections for each img
    # E.g. X_new = [img1_y1...img1_yn, img2_y1...img2_yn, ... imgn_y1...imgn_yn]
    X_new = tf.reshape(new_coords[:, 0, :], [-1]) # CAUTION HERE
    Y_new = tf.reshape(new_coords[:, 1, :], [-1])

    # Perform interpolation on input_vol
    output_vol = interpolate(input_vol, X_new, Y_new)
    return tf.reshape(output_vol, input_vol.shape)


def coordGen(mb_size, height, width):
    """ Generate coordinates (mb, 3, height * width)
        3rd dim consists of height * width rows for X, Y and ones """
    
    height_f = tf.cast(height, tf.float32)
    width_f = tf.cast(width, tf.float32)
    X, Y = tf.meshgrid(tf.linspace(-1.0,  1.0, width), tf.linspace(-1.0, 1.0, height))
    flat_X = tf.reshape(X, (1, -1))
    flat_Y = tf.reshape(Y, (1, -1))
    # Y, Z, X = tf.meshgrid(tf.linspace(0.0, 3.0, 4), tf.linspace(0.0, 2.0, 3), tf.linspace(0.0, 5.0, 6))
    # Rows are X, Y and row of ones (length is height * width)
    # Replicate for each minibatch
    flat_coords = tf.concat([flat_X, flat_Y, tf.ones((1, height_f * width_f))], axis=0)
    flat_coords = tf.tile(flat_coords[tf.newaxis, :, :], [mb_size, 1, 1])

    return flat_coords


def interpolate(input_vol, X, Y):
    mb_size = input_vol.shape[0]
    height = input_vol.shape[1]
    width = input_vol.shape[2]
    depth = input_vol.shape[3]
    # nc = input_vol.shape[4]
    height_f = tf.cast(height, tf.float32)
    width_f = tf.cast(width, tf.float32)
    depth_f = tf.cast(depth, tf.float32)

    # Convert to image coords
    X = (X + 1.0)*(width_f) / 2.0
    Y = (Y + 1.0)*(height_f) / 2.0

    # Generate integer coord indices either side of actual value
    x0 = tf.cast(tf.floor(X), tf.int32)
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(Y), tf.int32)
    y1 = y0 + 1

    # Ensure indices don't extend past image height/width
    x0 = tf.clip_by_value(x0, 0, width - 1)
    x1 = tf.clip_by_value(x1, 0, height - 1)
    y0 = tf.clip_by_value(y0, 0, width - 1)
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

    # Flatten image vector and look up with indices
    # Gather pixel values from flattened img based on 4 index vectors
    input_vol_flat = tf.reshape(input_vol, [-1, 1])
    input_vol_flat = tf.cast(input_vol_flat, tf.float32)
    ImgA = tf.gather(input_vol_flat, idx_a)
    ImgB = tf.gather(input_vol_flat, idx_b)
    ImgC = tf.gather(input_vol_flat, idx_c)
    ImgD = tf.gather(input_vol_flat, idx_d)

    # Generate vectors of the fractional difference between original indices and rounded indices i.e. weights
    x0_f = tf.cast(x0, 'float32')
    x1_f = tf.cast(x1, 'float32')
    y0_f = tf.cast(y0, 'float32')
    y1_f = tf.cast(y1, 'float32')
    wa = tf.expand_dims(((x1_f - X) * (y1_f - Y)), 1)
    wb = tf.expand_dims(((x1_f - X) * (Y - y0_f)), 1)
    wc = tf.expand_dims(((X - x0_f) * (y1_f - Y)), 1)
    wd = tf.expand_dims(((X - x0_f) * (Y - y0_f)), 1)

    # Add weighted imgs from each of four indices and return img_vol
    output_vol = tf.add_n([wa * ImgA, wb * ImgB, wc * ImgC, wd * ImgD])
    return output_vol


if __name__ == "__main__":
    start_t = time.time()
    img_vol = np.zeros((4, 128, 128, 1)) # Alter nc bit
    img_vol[:, 54:74, 54:74, :] = 1

    theta0 = np.array([0.5, 0, 0, 0, 0.5, 0], dtype=np.float32)
    theta1 = np.array([2, 0, 0, 0, 2, 0], dtype=np.float32)
    theta2 = np.array([1, 0, -0.5, 0, 1, 0.25], dtype=np.float32)
    theta3 = np.array([0.707, -0.707, 0, 0.707, 0.707, 0], dtype=np.float32)

    theta = tf.convert_to_tensor(np.stack([theta0, theta1, theta2, theta3], axis=0))

    new_vol = affineTransformation(img_vol, theta)

    print(time.time() - start_t)

    fig, axs = plt.subplots(2, 4)
    axs[0, 0].imshow(img_vol[0, :, :, 0])
    axs[0, 1].imshow(img_vol[1, :, :, 0])
    axs[0, 2].imshow(img_vol[2, :, :, 0])
    axs[0, 3].imshow(img_vol[3, :, :, 0])
    axs[1, 0].imshow(new_vol[0, :, :, 0])
    axs[1, 1].imshow(new_vol[1, :, :, 0])
    axs[1, 2].imshow(new_vol[2, :, :, 0])
    axs[1, 3].imshow(new_vol[3, :, :, 0])
    plt.show()