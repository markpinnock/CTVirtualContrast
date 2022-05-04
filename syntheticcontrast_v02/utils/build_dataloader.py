import tensorflow as tf

from .dataloader import PairedLoader, UnpairedLoader


def get_dataloader(config: dict,
                   dataset: str,
                   mb_size: int = None,
                   stride_length: int = None):

    if config["data"]["data_type"] == "paired":
        Loader = PairedLoader

    elif config["data"]["data_type"] == "unpaired":
        Loader = UnpairedLoader

    else:
        raise ValueError("Select paired or unpaired dataloader")

    # Specify output types
    output_types = ["real_source", "real_target"]

    if len(config["data"]["segs"]) > 0:
        output_types += ["seg"]
    
    if config["data"]["times"] is not None:
        output_types += ["source_times", "target_times"]


    if dataset == "train_val":
        # Initialise datasets and set normalisation parameters
        TrainGenerator = Loader(config=config["data"], dataset_type="training")
        param_1, param_2 = TrainGenerator.set_normalisation()

        ValGenerator = Loader(config=config["data"], dataset_type="validation")
        _, _ = ValGenerator.set_normalisation(param_1, param_2)

        config["data"]["norm_param_1"] = param_1
        config["data"]["norm_param_2"] = param_2

        # Create dataloader
        train_ds = tf.data.Dataset.from_generator(
            generator=TrainGenerator.data_generator,
            output_types={k: "float32" for k in output_types}
            ).batch(config["expt"]["mb_size"])

        val_ds = tf.data.Dataset.from_generator(
            generator=ValGenerator.data_generator,
            output_types={k: "float32" for k in output_types}
            ).batch(config["expt"]["mb_size"])

        return train_ds, val_ds, TrainGenerator, ValGenerator

    elif dataset == "test":
        assert mb_size is not None, "Set minibatch size"
        assert stride_length is not None, "Set stride length"

        # Inference-specific config settings
        config["data"]["cv_folds"] = 1
        config["data"]["fold"] = 0
        config["data"]["segs"] = []
        config["data"]["times"] = None # TODO: WHY
        config["data"]["xy_patch"] = True
        config["data"]["stride_length"] = stride_length
        config["expt"]["mb_size"] = mb_size

        # Initialise datasets and set normalisation parameters
        TestGenerator = Loader(config=config["data"], dataset_type="validation")
        _, _ = TestGenerator.set_normalisation()

        # Specify output types
        output_types = {"real_source": "float32", "subject_ID": tf.string, "x": "int32", "y": "int32", "z": "int32"}

        # Create dataloader
        test_ds = tf.data.Dataset.from_generator(
            generator=TestGenerator.inference_generator,
            output_types=output_types).batch(config["expt"]["mb_size"])

        return test_ds, TestGenerator

    else:
        raise ValueError(f"Select 'train_val' or 'test', not {dataset}")
