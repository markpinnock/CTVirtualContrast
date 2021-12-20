import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import yaml

from .unet import UNet
from .util import load_data, dice_loss, bootstrap

np.set_printoptions(4)

#-------------------------------------------------------------------------

def test(expt_path):

    with open(f"{expt_path}/config.yml", 'r') as infile:
        config = yaml.load(infile, yaml.FullLoader)

    test_ds = tf.data.Dataset.from_tensor_slices(
        load_data("test", config["model_config"]["img_dims"][0], **config["data_config"]))

    segs = next(iter(test_ds.batch(256)))[1]

    test_ds = test_ds.batch(config["model_config"]["mb_size"])

    Model = UNet(config["model_config"])
    Model.load_weights(f"{expt_path}/models/model.ckpt").expect_partial()
    prediction = Model.predict(test_ds)

    dice = 1 - dice_loss(prediction, segs, axis=[1, 2, 3]).numpy()

    return dice


#-------------------------------------------------------------------------

def bootstrap_and_display(expt1, expt2):
    diff = np.median(results[expt1]) - np.median(results[expt2])
    boot_results = bootstrap(results[expt1], results[expt2], N=100000)

    h = plt.hist(boot_results, bins=20)
    plt.axvline(diff, c='k', ls='--')
    plt.errorbar(x=diff, y=(0.75 * np.max(h[0])), xerr=(1.96 * np.std(boot_results)))
    plt.title(f"{expt1} - {expt2}")
    plt.show()

    # Pivot method
    percentiles = np.quantile(boot_results, [0.975, 0.025]) # NB: these are switched

    return expt1, expt2, diff, 2 * diff - percentiles, f"Bias {np.mean(boot_results) - diff}, std err {np.std(boot_results)}"


#-------------------------------------------------------------------------

if __name__ == "__main__":

    """ Segmentation-based evaluation testing routine """

    expts_path = "syntheticcontrast_v02/evaluation/expts"
    expts = [
        "HQ", "AC", "VC", "UNetT_save1000_AP", "UNetT_save1000_VP",
        "CycleGAN_save880_AP", "2_save230_AP", "2_save230_VP",
        "2_save170_patch_AP", "2_save170_patch_VP",
        "H2_save280_AP", "H2_save280_VP",
        "H2_save300_patch_AP", "H2_save300_patch_VP"
        ]

    results = {}

    for expt in expts:
        results[expt] = test(f"{expts_path}/{expt}")

    box_results = []

    for k, v in results.items():
        print(k, np.median(v), np.quantile(v, [0.05, 0.95]))
        box_results.append(v)

    # plt.boxplot(box_results)
    # plt.xticks(list(range(1, len(expts) + 1)), expts)
    # plt.ylabel("Dice")
    # plt.show()

    print(bootstrap_and_display("AC", "VC"))
    print(bootstrap_and_display("AC", "HQ"))
    print(bootstrap_and_display("VC", "HQ"))
    print(bootstrap_and_display("AC", "UNetT_save1000_AP"))
    print(bootstrap_and_display("VC", "UNetT_save1000_VP"))
    print(bootstrap_and_display("AC", "CycleGAN_save880_AP"))
    print(bootstrap_and_display("AC", "2_save230_AP"))
    print(bootstrap_and_display("VC", "2_save230_VP"))
    print(bootstrap_and_display("AC", "2_save170_patch_AP"))
    print(bootstrap_and_display("VC", "2_save170_patch_VP"))
    print(bootstrap_and_display("AC", "H2_save280_AP"))
    print(bootstrap_and_display("VC", "H2_save280_VP"))
    print(bootstrap_and_display("AC", "H2_save300_patch_AP"))
    print(bootstrap_and_display("VC", "H2_save300_patch_VP"))
