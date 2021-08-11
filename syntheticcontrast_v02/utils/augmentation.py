import matplotlib.pyplot as plt
import tensorflow as tf

from utils.dataloader import PairedLoader


""" DiffAug class for differentiable augmentation
    Paper: https://arxiv.org/abs/2006.10738
    Adapted from: https://github.com/mit-han-lab/data-efficient-gans """


class DiffAug(tf.keras.layers.Layer):
    
    def __init__(self, aug_config, name="augmentation"):
        super().__init__(name=name)
        self.aug_config = aug_config

    def brightness(self, x):
        """ Random brightness in range [-0.5, 0.5] """

        factor = tf.random.uniform((x.shape[0], 1, 1, 1, 1)) - 0.5
        return x + factor
    
    def saturation(self, x):
        """ Random saturation in range [0, 2] """

        factor = tf.random.uniform((x.shape[0], 1, 1, 1, 1)) * 2
        x_mean = tf.reduce_mean(x, axis=(1, 2, 3), keepdims=True)
        return (x - x_mean) * factor + x_mean

    def contrast(self, x):
        """ Random contrast in range [0.5, 1.5] """

        factor = tf.random.uniform((x.shape[0], 1, 1, 1, 1)) + 0.5
        x_mean = tf.reduce_mean(x, axis=-1, keepdims=True)
        return (x - x_mean) * factor + x_mean
    
    def translation(self, imgs, seg=None, ratio=0.125):
        """ Random translation by ratio 0.125 """

        # NB: This assumes NHWDC format and does not (yet) act in z direction
        num_imgs = len(imgs)
        batch_size = tf.shape(imgs[0])[0]
        image_size = tf.shape(imgs[0])[1:3]
        image_depth = tf.shape(imgs[0])[3]

        if seg != None:
            x = tf.concat(imgs + [seg], axis=3)
        
        else:
            x = tf.concat(imgs, axis=3)

        shift = tf.cast(tf.cast(image_size, tf.float32) * ratio + 0.5, tf.int32)
        translation_x = tf.random.uniform([batch_size, 1], -shift[0], shift[0] + 1, dtype=tf.int32)
        translation_y = tf.random.uniform([batch_size, 1], -shift[1], shift[1] + 1, dtype=tf.int32)
        grid_x = tf.clip_by_value(tf.expand_dims(tf.range(image_size[0], dtype=tf.int32), 0) + translation_x + 1, 0, image_size[0] + 1)
        grid_y = tf.clip_by_value(tf.expand_dims(tf.range(image_size[1], dtype=tf.int32), 0) + translation_y + 1, 0, image_size[1] + 1)
        x = tf.gather_nd(tf.pad(x, [[0, 0], [1, 1], [0, 0], [0, 0], [0, 0]]), tf.expand_dims(grid_x, -1), batch_dims=1)
        x = tf.transpose(tf.gather_nd(tf.pad(tf.transpose(x, [0, 2, 1, 3, 4]), [[0, 0], [1, 1], [0, 0], [0, 0], [0, 0]]), tf.expand_dims(grid_y, -1), batch_dims=1), [0, 2, 1, 3, 4])
        
        imgs = [x[:, :, :, i * 12:(i + 1) * 12, :] for i in range(num_imgs)]

        if seg != None:
            seg = x[:, :, :, -12:, :]

        return imgs, seg

    def cutout(self, imgs, seg=None, ratio=0.5):
        """ Random cutout by ratio 0.5 """
        # NB: This assumes NHWDC format and does not (yet) act in z direction

        num_imgs = len(imgs)
        batch_size = tf.shape(imgs[0])[0]
        image_size = tf.shape(imgs[0])[1:3]
        image_depth = tf.shape(imgs[0])[3]

        if seg != None:
            x = tf.concat(imgs + [seg], axis=3)
        
        else:
            x = tf.concat(imgs, axis=3)

        cutout_size = tf.cast(tf.cast(image_size, tf.float32) * ratio + 0.5, tf.int32)
        offset_x = tf.random.uniform([tf.shape(x)[0], 1, 1], maxval=image_size[0] + (1 - cutout_size[0] % 2), dtype=tf.int32)
        offset_y = tf.random.uniform([tf.shape(x)[0], 1, 1], maxval=image_size[1] + (1 - cutout_size[1] % 2), dtype=tf.int32)

        grid_batch, grid_x, grid_y = tf.meshgrid(tf.range(batch_size, dtype=tf.int32), tf.range(cutout_size[0], dtype=tf.int32), tf.range(cutout_size[1], dtype=tf.int32), indexing='ij')
        cutout_grid = tf.stack([grid_batch, grid_x + offset_x - cutout_size[0] // 2, grid_y + offset_y - cutout_size[1] // 2], axis=-1)

        mask_shape = tf.stack([batch_size, image_size[0], image_size[1]])
        cutout_grid = tf.maximum(cutout_grid, 0)

        cutout_grid = tf.minimum(cutout_grid, tf.reshape(mask_shape - 1, [1, 1, 1, 3]))

        mask = tf.maximum(1 - tf.scatter_nd(cutout_grid, tf.ones([batch_size, cutout_size[0], cutout_size[1]], dtype=tf.float32), mask_shape), 0)
        x = x * tf.expand_dims(tf.expand_dims(mask, axis=3), axis=4)
       
        imgs = [x[:, :, :, i * 12:(i + 1) * 12, :] for i in range(num_imgs)]

        if seg != None:
            seg = x[:, :, :, -12:, :]

        return imgs, seg

    def call(self, imgs, seg=None):
        if self.aug_config["colour"]: imgs = [self.contrast(self.saturation(self.brightness(img))) for img in imgs]
        if self.aug_config["translation"]: imgs, seg = self.translation(imgs, seg)
        if self.aug_config["cutout"]: imgs, seg = self.cutout(imgs, seg)

        return imgs, seg


# To try:
""" https://github.com/NVlabs/stylegan2-ada/blob/main/training/augment.py """



if __name__ == "__main__":
    from dataloader import PairedLoader

    FILE_PATH = "D:/ProjectImages/SyntheticContrast"
    TestLoader = PairedLoader({"DATA": {"DATA_PATH": FILE_PATH, "TARGET": ["AC"], "SOURCE": ["HQ"], "SEGS": ["AC"], "JSON": "", "DOWN_SAMP": 4, "NUM_EXAMPLES": 4}, "EXPT": {"CV_FOLDS": 3, "FOLD": 2}}, dataset_type="training")
    TestLoader.set_normalisation(norm_type="std", param_1=-288, param_2=253)
    TestAug = DiffAug({"colour": True, "translation": True, "cutout": True})

    train_ds = tf.data.Dataset.from_generator(
        TestLoader.data_generator, output_types=(tf.float32, tf.float32, tf.float32))

    for data in train_ds.batch(4):
        imgs, segs = TestAug(imgs=[data[0], data[1]], seg=data[2])
        source, target = imgs

        plt.subplot(2, 3, 1)
        plt.imshow(source[0, :, :, 0, 0])
        plt.subplot(2, 3, 4)
        plt.imshow(source[1, :, :, 0, 0])
        
        plt.subplot(2, 3, 2)
        plt.imshow(target[0, :, :, 0, 0])
        plt.subplot(2, 3, 5)
        plt.imshow(target[1, :, :, 0, 0])

        plt.subplot(2, 3, 3)
        plt.imshow(segs[0, :, :, 0, 0])
        plt.subplot(2, 3, 6)
        plt.imshow(segs[1, :, :, 0, 0])

        plt.show()