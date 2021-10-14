import json
import matplotlib.pyplot as plt
import numpy as np
import os

from .util import load_images, resample, display_imgs, get_HUs, aggregate_HUs


#-------------------------------------------------------------------------

def display_subjects(subject_list: list, subject_ignore: list = [], image_ignore: list = [], depth_idx: int = None):
    for subject in subject_list:
        if subject in subject_ignore or 'F' in subject:
            continue

        print(subject)
        imgs, segs = load_images(subject, IMG_PATH, SEG_PATH, TRANS_PATH, ignore=image_ignore)
        imgs, segs = resample(imgs, segs)

        AC = [n for n in imgs.keys() if 'AC' in n]
        VC = [n for n in imgs.keys() if 'VC' in n]
        HQ = [n for n in imgs.keys() if 'HQ' in n]
        keys = sorted(AC + VC + HQ, key=lambda x: int(x[-3:]))

        with open(f"{TIME_PATH}/{subject}/time.json", 'r') as fp:
            times = json.load(fp)

        t = [times[f"{k}.nrrd"] for k in keys]

        overlay = np.copy(segs[AC[0]])

        display_imgs(imgs, segs, keys, overlay=overlay, depth_idx=depth_idx)
        display_HUs(imgs, segs[AC[0]], keys, t)


#-------------------------------------------------------------------------

def display_HUs(imgs: dict, seg: object, keys: list, t: list = None):
    Ao, RK, LK, Tu = get_HUs(imgs, seg, keys)

    plt.figure(figsize=(18, 10))

    if t is None:
        plt.plot(Ao, label="Ao")
        plt.plot(RK, label="RK")
        plt.plot(LK, label="LK")
        plt.plot(Tu, label="Tu")

    else:
        plt.plot(t, Ao, label="Ao")
        plt.plot(t, RK, label="RK")
        plt.plot(t, LK, label="LK")
        plt.plot(t, Tu, label="Tu")

    plt.xlabel("Series")
    plt.ylabel("HU")
    plt.title(keys[0][0:6])
    plt.legend()
    plt.show()


#-------------------------------------------------------------------------

def display_aggregate_HUs(HUs):
    t = HUs.pop("times")

    if None in t:
        N = HUs['Ao'].shape[1] # TODO: FIX
        t = np.linspace(0, N - 1, N)

    min_HU = np.min([a for a in HUs.values()])
    max_HU = np.max([a for a in HUs.values()])

    fig, axs = plt.subplots(2, 2)

    for i in range(HUs['Ao'].shape[0]):
        axs[0, 0].semilogx(t[i, :], HUs['Ao'][i, :], marker='+')#, c='k')
        axs[0, 1].semilogx(t[i, :], HUs['RK'][i, :], marker='+')#, c='k')
        axs[1, 0].semilogx(t[i, :], HUs['LK'][i, :], marker='+')#, c='k')
        axs[1, 1].semilogx(t[i, :], HUs['Tu'][i, :], marker='+')#, c='k')
    
    # axs[0, 0].plot(t, HUs['Ao'].mean(axis=0), c='r')
    # axs[0, 1].plot(t, HUs['RK'].mean(axis=0), c='r')
    # axs[1, 0].plot(t, HUs['LK'].mean(axis=0), c='r')
    # axs[1, 1].plot(t, HUs['Tu'].mean(axis=0), c='r')

    for ax in axs.ravel():
        ax.set_xlabel("Time")
        ax.set_ylabel("HU")
        ax.set_xlabel("Time")
        ax.set_ylim([min_HU, max_HU])

    axs[0, 0].set_title("Aorta")
    axs[0, 1].set_title("RK")
    axs[1, 0].set_title("LK")
    axs[1, 1].set_title("Tumour")

    plt.show()


#-------------------------------------------------------------------------

if __name__ == "__main__":

    with open("syntheticcontrast_v02/preproc/ignore.json", 'r') as fp:
        ignore = json.load(fp)

    subject_ignore = list(ignore["subject_ignore"].keys())
    image_ignore = ignore["image_ignore"]

    IMG_PATH = "Z:/Clean_CT_Data/Toshiba/Images"
    SEG_PATH = "Z:/Clean_CT_Data/Toshiba/Segmentations"
    TRANS_PATH = "Z:/Clean_CT_Data/Toshiba/Transforms"
    TIME_PATH = "Z:/Clean_CT_Data/Toshiba/Times"
    subjects = os.listdir(IMG_PATH)[15:]
    subject_ignore += ["T005A0", "T016A0", "T021A0", "T024A0", "T030A0", "T031A0"]#, "T038A0"] # misregistered

    display_subjects(subjects, subject_ignore=subject_ignore, image_ignore=image_ignore, depth_idx=None)
    HUs = aggregate_HUs(subjects, subject_ignore=subject_ignore, image_ignore=image_ignore, img_path=IMG_PATH, seg_path=SEG_PATH)
    display_aggregate_HUs(HUs)

# Not enough segs? T016A0
# Misregistered: T005A0HQ004, T016A0, T021A0HQ002, T024A0HQ056, T030A0, T031A0HQ007, T038A0
