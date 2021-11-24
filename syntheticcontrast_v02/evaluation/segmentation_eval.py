import argparse
import tensorflow as tf
import yaml

from .unet import UNet
from .util import load_data, Augmentation, dice_loss

#-------------------------------------------------------------------------

def train(config):

    train_ds = tf.data.Dataset.from_tensor_slices(
        load_data("train", config["model_config"]["img_dims"][0] + 12, **config["data_config"])).shuffle(256).batch(config["model_config"]["mb_size"]).map(Augmentation(config["model_config"]["img_dims"]))
    test_ds = tf.data.Dataset.from_tensor_slices(
        load_data("test", config["model_config"]["img_dims"][0], **config["data_config"])).batch(config["model_config"]["mb_size"])

    Model = UNet(config["model_config"])

    inputs = tf.keras.Input(shape=config["model_config"]["img_dims"] + [1])
    outputs = Model.call(inputs, training=True)
    tf.keras.Model(inputs=inputs, outputs=outputs).summary()

    Model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss=dice_loss)
    Model.fit(train_ds, epochs=config["model_config"]["epochs"], validation_data=test_ds)


#-------------------------------------------------------------------------

if __name__ == "__main__":

    """ Segmentation-based evaluation training routine """

    # Handle arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", "-p", help="Expt path", type=str)
    arguments = parser.parse_args()

    EXPT_PATH = arguments.path

    # Parse config json
    with open(f"{EXPT_PATH}/config.yml", 'r') as infile:
        CONFIG = yaml.load(infile, yaml.FullLoader)

    train(CONFIG)
