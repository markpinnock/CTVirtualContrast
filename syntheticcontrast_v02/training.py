import argparse
import datetime
import os
import tensorflow as tf
import yaml

from syntheticcontrast_v02.trainingtuningclasses.trainingclasses import TrainingPix2Pix, TrainingCycleGAN
from syntheticcontrast_v02.networks.pix2pix import Pix2Pix, HyperPix2Pix
from syntheticcontrast_v02.networks.cyclegan import CycleGAN
from syntheticcontrast_v02.utils.dataloader import PairedLoader, UnpairedLoader


#-------------------------------------------------------------------------

def train(CONFIG):

    if CONFIG["data"]["data_type"] == "paired":
        Loader = PairedLoader

    elif CONFIG["data"]["data_type"] == "unpaired":
        Loader = UnpairedLoader

    else:
        raise ValueError("Select paired or unpaired dataloader")

    # TODO: move into dataloader
    # Initialise datasets and set normalisation parameters
    TrainGenerator = Loader(config=CONFIG["data"], dataset_type="training")
    param_1, param_2 = TrainGenerator.set_normalisation()
    ValGenerator = Loader(config=CONFIG["data"], dataset_type="validation")
    _, _ = ValGenerator.set_normalisation(param_1, param_2)

    CONFIG["data"]["norm_param_1"] = param_1
    CONFIG["data"]["norm_param_2"] = param_2

    # Specify output types
    output_types = ["float32", "float32"]

    if len(CONFIG["data"]["segs"]) > 0:
        output_types += ["float32"]

    # Create dataloader (one mb for generator, one for discriminator)
    train_ds = tf.data.Dataset.from_generator(
        generator=TrainGenerator.data_generator,
        output_types=tuple(output_types)
        ).batch(CONFIG["expt"]["mb_size"] * 2)

    val_ds = tf.data.Dataset.from_generator(
        generator=ValGenerator.data_generator,
        output_types=tuple(output_types)
        ).batch(CONFIG["expt"]["mb_size"] * 2)

    # Compile model
    if CONFIG["expt"]["model"] == "Pix2Pix":
        Model = Pix2Pix(config=CONFIG)
        Model.compile(
            g_optimiser=tf.keras.optimizers.Adam(*CONFIG["hyperparameters"]["g_opt"], name="g_opt"),
            d_optimiser=tf.keras.optimizers.Adam(*CONFIG["hyperparameters"]["g_opt"], name="g_opt")
            )

    elif CONFIG["expt"]["model"] == "HyperPix2Pix":
        Model = HyperPix2Pix(config=CONFIG)
        Model.compile(
            g_optimiser=tf.keras.optimizers.Adam(*CONFIG["hyperparameters"]["g_opt"], name="g_opt"),
            d_optimiser=tf.keras.optimizers.Adam(*CONFIG["hyperparameters"]["g_opt"], name="g_opt")
            )

    elif CONFIG["expt"]["model"] == "CycleGAN":
        Model = CycleGAN(config=CONFIG)
        Model.compile(
            g_forward_opt=tf.keras.optimizers.Adam(*CONFIG["hyperparameters"]["g_opt"], name="g_foward_opt"),
            g_backward_opt=tf.keras.optimizers.Adam(*CONFIG["hyperparameters"]["g_opt"], name="g_backward_opt"),
            d_forward_opt=tf.keras.optimizers.Adam(*CONFIG["hyperparameters"]["d_opt"], name="d_forward_opt"),
            d_backward_opt=tf.keras.optimizers.Adam(*CONFIG["hyperparameters"]["d_opt"], name="d_backward_opt")
            )

    else:
        raise ValueError

    if CONFIG["expt"]["verbose"]:
        Model.summary()

    # Write graph for visualising in Tensorboard
    if CONFIG["expt"]["graph"]:
        curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = f"{CONFIG['paths']['expt_path']}/logs/{curr_time}"
        writer = tf.summary.create_file_writer(log_dir)

        @tf.function
        def trace(x):
            return Model.Generator(x)

        tf.summary.trace_on(graph=True)
        trace(tf.zeros([1] + CONFIG["hyperparameters"]["img_dims"] + [1]))

        with writer.as_default():
            tf.summary.trace_export("graph", step=0)

    if CONFIG["expt"]["model"] == "Pix2Pix" or CONFIG["expt"]["model"] == "HyperPix2Pix":
        TrainingLoop = TrainingPix2Pix(
            Model=Model,
            dataset=(train_ds, val_ds),
            train_generator=TrainGenerator,
            val_generator=ValGenerator,
            config=CONFIG
            )

    else:
        TrainingLoop = TrainingCycleGAN(
            Model=Model,
            dataset=(train_ds, val_ds),
            train_generator=TrainGenerator,
            val_generator=ValGenerator,
            config=CONFIG
            )

    # Run training loop
    TrainingLoop.train()
    TrainingLoop.save_results()


#-------------------------------------------------------------------------

if __name__ == "__main__":

    """ Training routine """

    # Handle arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", "-p", help="Expt path", type=str)
    parser.add_argument("--gpu", "-g", help="GPU number", type=int)
    arguments = parser.parse_args()

    EXPT_PATH = arguments.path

    if not os.path.exists(f"{EXPT_PATH}/images"):
        os.makedirs(f"{EXPT_PATH}/images")

    if not os.path.exists(f"{EXPT_PATH}/logs"):
        os.makedirs(f"{EXPT_PATH}/logs")

    if not os.path.exists(f"{EXPT_PATH}/models"):
        os.makedirs(f"{EXPT_PATH}/models")

    # Parse config json
    with open(f"{EXPT_PATH}/config.yml", 'r') as infile:
        CONFIG = yaml.load(infile, yaml.FullLoader)
    
    CONFIG["paths"]["expt_path"] = arguments.path

    # Set GPU
    if arguments.gpu is not None:
        gpu_number = arguments.gpu
        os.environ["LD_LIBRARY_PATH"] = CONFIG["paths"]["cuda_path"]
        gpus = tf.config.experimental.list_physical_devices("GPU")
        tf.config.set_visible_devices(gpus[gpu_number], "GPU")
        tf.config.experimental.set_memory_growth(gpus[gpu_number], True)
    
    train(CONFIG)
