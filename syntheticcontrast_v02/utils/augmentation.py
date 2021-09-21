import tensorflow as tf

from syntheticcontrast_v02.utils.affinetransformation import AffineTransform2D


#-------------------------------------------------------------------------
""" DiffAug class for differentiable augmentation
    Paper: https://arxiv.org/abs/2006.10738
    Adapted from: https://github.com/mit-han-lab/data-efficient-gans """

class DiffAug(tf.keras.layers.Layer):
    
    def __init__(self, config, name="diff_aug"):
        super().__init__(name=name)
        self.aug_config = config
        self.depth = config["depth"]

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
        
        imgs = [x[:, :, :, i * self.depth:(i + 1) * self.depth, :] for i in range(num_imgs)]

        if seg != None:
            seg = x[:, :, :, -self.depth:, :]

        return imgs, seg

    def cutout(self, imgs, seg=None, ratio=0.5):
        """ Random cutout by ratio 0.5 """
        # NB: This assumes NHWDC format and does not (yet) act in z direction

        num_imgs = len(imgs)
        batch_size = tf.shape(imgs[0])[0]
        image_size = tf.shape(imgs[0])[1:3]
        # image_depth = tf.shape(imgs[0])[3]

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
       
        imgs = [x[:, :, :, i * self.depth:(i + 1) * self.depth, :] for i in range(num_imgs)]

        if seg != None:
            seg = x[:, :, :, -self.depth:, :]

        return imgs, seg

    def call(self, imgs, seg=None):
        num_imgs = len(imgs)

        imgs = tf.concat(imgs, axis=4)
        if self.aug_config["colour"]: imgs = self.contrast(self.saturation(self.brightness(imgs)))
        imgs = [tf.expand_dims(imgs[:, :, :, :, i], 4) for i in range(num_imgs)]

        if self.aug_config["translation"]: imgs, seg = self.translation(imgs, seg)
        if self.aug_config["cutout"]: imgs, seg = self.cutout(imgs, seg)

        return imgs, seg


#-------------------------------------------------------------------------
# To try:
""" https://github.com/NVlabs/stylegan2-ada/blob/main/training/augment.py """


#-------------------------------------------------------------------------

class StdAug(tf.keras.layers.Layer):

    def __init__(self, config, name="std_aug"):
        super().__init__(name=name)

        # If segmentations available, these can be stacked on the target for transforming
        if len(config["data"]["segs"]) > 0:
            self.transform = AffineTransform2D(config["hyperparameters"]["img_dims"] + [2])
        else:
            self.transform = AffineTransform2D(config["hyperparameters"]["img_dims"] + [1])

        self.flip_probs = tf.math.log([[config["augmentation"]["flip_prob"], 1 - config["augmentation"]["flip_prob"]]])
        self.rot_angle = config["augmentation"]["rotation"] / 180 * 3.14159265359
        self.scale_factor = config["augmentation"]["scale"]
        self.shear_angle = config["augmentation"]["shear"] / 180 * 3.14159265359
        self.x_shift = [-config["augmentation"]["translate"][0], config["augmentation"]["translate"][0]]
        self.y_shift = [-config["augmentation"]["translate"][1], config["augmentation"]["translate"][1]]

    def flip_matrix(self, mb_size: int):
        updates = tf.reshape(tf.cast(tf.random.categorical(logits=self.flip_probs, num_samples=mb_size * 2), "float32"), [mb_size * 2])
        updates = 2.0 * updates - 1.0
        indices = tf.concat([tf.repeat(tf.range(0, mb_size), 2)[:, tf.newaxis], tf.tile(tf.constant([[0, 0], [1, 1]]), [mb_size, 1])], axis=1)
        flip_mat = tf.scatter_nd(indices, updates, [mb_size, 2, 2])

        return flip_mat

    def rotation_matrix(self, mb_size: int):
        thetas = tf.random.uniform([mb_size], -self.rot_angle, self.rot_angle)
        rot_mat = tf.stack(
            [
                [tf.math.cos(thetas), -tf.math.sin(thetas)],
                [tf.math.sin(thetas), tf.math.cos(thetas)]
            ]
        )

        rot_mat = tf.transpose(rot_mat, [2, 0, 1])

        return rot_mat

    def scale_matrix(self, mb_size: int):
        updates = tf.repeat(tf.random.uniform([mb_size], * self.scale_factor), 2)
        indices = tf.concat([tf.repeat(tf.range(0, mb_size), 2)[:, tf.newaxis], tf.tile(tf.constant([[0, 0], [1, 1]]), [mb_size, 1])], axis=1)
        scale_mat = tf.scatter_nd(indices, updates, [mb_size, 2, 2])

        return scale_mat

    def shear_matrix(self, mb_size: int):
        mask = tf.cast(tf.random.categorical(logits=[[0.5, 0.5]], num_samples=mb_size), "float32")
        mask = tf.reshape(tf.transpose(tf.concat([mask, 1 - mask], axis=0), [1, 0]), [1, -1])
        updates = tf.repeat(tf.random.uniform([mb_size], -self.shear_angle, self.shear_angle), 2)
        updates = tf.reshape(updates * mask, [-1])
        indices = tf.concat([tf.repeat(tf.range(0, mb_size), 2)[:, tf.newaxis], tf.tile(tf.constant([[0, 1], [1, 0]]), [mb_size, 1])], axis=1)
        shear_mat = tf.scatter_nd(indices, updates, [mb_size, 2, 2])
        shear_mat += tf.tile(tf.eye(2)[tf.newaxis, :, :], [mb_size, 1, 1])

        return shear_mat

    def translation_matrix(self, m, mb_size: int):
        xs = tf.random.uniform([mb_size], *self.x_shift)
        ys = tf.random.uniform([mb_size], *self.y_shift)
        xys = tf.stack([xs, ys], axis=0)
        xys = tf.transpose(xys, [1, 0])[:, :, tf.newaxis]
        m = tf.concat([m, xys], axis=2)

        return m

    def transformation(self, mb_size: int):
        trans_mat = tf.matmul(
            self.shear_matrix(mb_size), tf.matmul(
                self.flip_matrix(mb_size), tf.matmul(
                    self.rotation_matrix(mb_size), self.scale_matrix(mb_size))))

        trans_mat = self.translation_matrix(trans_mat, mb_size)

        return trans_mat
    
    def call(self, imgs, seg=None):
        l = len(imgs)
        imgs = tf.concat(imgs, axis=4)
        mb_size = imgs.shape[0]
        thetas = tf.reshape(self.transformation(mb_size), [mb_size, -1])

        if seg is not None:
            img_seg = tf.concat([imgs, seg], axis=4)
            img_seg = self.transform(im=img_seg, mb_size=mb_size, thetas=thetas)
            imgs = [img_seg[:, :, :, :, i][:, :, :, :, tf.newaxis] for i in range(l)]
            seg = img_seg[:, :, :, :, -1][:, :, :, :, tf.newaxis]

            return tuple(imgs), seg
        
        else:
            imgs = self.transform(im=imgs, mb_size=mb_size, thetas=thetas)
            imgs = [imgs[:, :, :, :, i][:, :, :, :, tf.newaxis] for i in range(l)]
            return tuple(imgs), None


#-------------------------------------------------------------------------
""" Short routine for visually testing augmentations """

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import yaml
    from syntheticcontrast_v02.utils.dataloader import PairedLoader

    test_config = yaml.load(open("syntheticcontrast_v02/utils/test_config.yml", 'r'), Loader=yaml.FullLoader)

    FILE_PATH = "D:/ProjectImages/SyntheticContrast"
    TestLoader = PairedLoader(test_config["data"], dataset_type="training")
    TestLoader.set_normalisation()

    if test_config["augmentation"]["type"] == "differentiable":
        TestAug = DiffAug(test_config)
    else:
        TestAug = StdAug(test_config)

    output_types = ["float32", "float32"]

    if len(test_config["data"]["segs"]) > 0:
        output_types += ["float32"]

    train_ds = tf.data.Dataset.from_generator(TestLoader.data_generator, output_types=tuple(output_types))

    for data in train_ds.batch(4):
        if len(test_config["data"]["segs"]) > 0:
            source, target, seg = data
            target, seg = TestAug([source, target], seg=seg)
        else:
            source, target = data
            target, _ = TestAug([source, target], seg=None)

        imgs, segs = TestAug(imgs=[data[0], data[1]], seg=data[2])

        source, target = imgs
        # source = TestLoader.un_normalise(source)
        # target = TestLoader.un_normalise(target)

        plt.subplot(2, 4, 1)
        plt.imshow(data[0][0, :, :, 0, 0], cmap="gray")#, vmin=-150, vmax=250)
        plt.axis("off")
        plt.subplot(2, 4, 5)
        plt.imshow(data[0][1, :, :, 0, 0], cmap="gray")#, vmin=-150, vmax=250)
        plt.axis("off")

        plt.subplot(2, 4, 2)
        plt.imshow(source[0, :, :, 0, 0], cmap="gray")#, vmin=-150, vmax=250)
        plt.axis("off")
        plt.subplot(2, 4, 6)
        plt.imshow(source[1, :, :, 0, 0], cmap="gray")#, vmin=-150, vmax=250)
        plt.axis("off")
        
        plt.subplot(2, 4, 3)
        plt.imshow(target[0, :, :, 0, 0], cmap="gray")#, vmin=-150, vmax=250)
        plt.axis("off")
        plt.subplot(2, 4, 7)
        plt.imshow(target[1, :, :, 0, 0], cmap="gray")#, vmin=-150, vmax=250)
        plt.axis("off")

        if len(test_config["data"]["segs"]) > 0:
            plt.subplot(2, 4, 4)
            plt.imshow(segs[0, :, :, 0, 0])
            plt.axis("off")
            plt.subplot(2, 4, 8)
            plt.imshow(segs[1, :, :, 0, 0])
            plt.axis("off")

        plt.show()
