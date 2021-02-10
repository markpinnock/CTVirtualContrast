import matplotlib.pyplot as plt
import numpy as np
import os
import SimpleITK as itk


FILE_PATH = "C:/ProjectImages/Imgs/"
SAVE_PATH = "C:/Users/roybo/OneDrive - University College London/PhD/PhD_Prog/007_CNN_Virtual_Contrast/DataExploration/"
subjects = os.listdir(FILE_PATH)
NC_IDX = 1


def plot_image(im1, im2=None, x=None, y=None, z=None):
    if z:
        if im2 is None:
            plt.imshow(np.flipud(im1[:, :, z].T), cmap="gray", origin="lower")
            plt.axis("off")
        else:
            plt.imshow(np.flipud(im1[:, :, z].T - im2[:, :, z].T), cmap="gray", origin="lower")
            plt.axis("off")
    elif y:
        if im2 is None:
            plt.imshow(np.flipud(im1[:, y, :].T), cmap="gray", origin="lower")
            plt.axis("off")
        else:
            plt.imshow(np.flipud(im1[:, y, :].T - im2[:, y, :].T), cmap="gray", origin="lower")
            plt.axis("off")
    elif x:
        if im2 is None:
            plt.imshow(np.flipud(im1[x, :, :].T), cmap="gray", origin="lower")
            plt.axis("off")
        else:
            plt.imshow(np.flipud(im1[x, :, :].T - im2[x, :, :].T), cmap="gray", origin="lower")
            plt.axis("off")
    else:
        raise ValueError


for subject in subjects:

    imgs = os.listdir(f"{FILE_PATH}{subject}")
    ACs = [img for img in imgs if 'AC' in img]
    VCs = [img for img in imgs if 'VC' in img]
    NCs = [img for img in imgs if 'HQ' in img]

    if len(ACs) > 0 and len(VCs) > 0:

        source = itk.ReadImage(f"{FILE_PATH}{subject}/{ACs[0]}", itk.sitkInt32)
        target = itk.ReadImage(f"{FILE_PATH}{subject}/{VCs[NC_IDX - 1]}", itk.sitkInt32)
        resamp = itk.Resample(source, target)

        mid_depth = target.GetSize()[2] // 2

        s_dir = source.GetDirection()
        t_dir = target.GetDirection()
        assert np.isclose(np.array(s_dir), np.array(t_dir)).all()

        s = np.transpose(itk.GetArrayViewFromImage(source), [2, 1, 0])
        t = np.transpose(itk.GetArrayViewFromImage(target), [2, 1, 0])
        r = np.transpose(itk.GetArrayViewFromImage(resamp), [2, 1, 0])

        if s_dir[0] < 0 and s_dir[4] < 0:
            s = np.flipud(np.fliplr(s))
            t = np.flipud(np.fliplr(t))
            r = np.flipud(np.fliplr(r))

        s = np.clip(s, -350, 450)
        t = np.clip(t, -350, 450)
        r = np.clip(r, -350, 450)

        plt.figure(figsize=(8, 12))
        plt.subplot(4, 3, 1)
        try:
            plot_image(s, z=mid_depth)
            plt.title("Non-aligned AC")
        except IndexError:
            plot_image(s, z=mid_depth // 2)
            plt.title("Non-aligned AC")
            plt.subplot(4, 3, 2)
            plot_image(t, z=mid_depth // 2)
            plt.title("NC")
            plt.subplot(4, 3, 3)
            plot_image(s, t, z=mid_depth // 2)
            plt.title("Diff")
        else:
            plt.subplot(4, 3, 2)
            plot_image(t, z=mid_depth)
            plt.title("NC")
            plt.subplot(4, 3, 3)
            plot_image(s, t, z=mid_depth)
            plt.title("Diff")

        plt.subplot(4, 3, 4)
        plot_image(r, z=mid_depth)
        plt.title("Aligned AC")
        plt.subplot(4, 3, 5)
        plot_image(t, z=mid_depth)
        plt.title("NC")
        plt.subplot(4, 3, 6)
        plot_image(r, t, z=mid_depth)
        plt.title("Diff")
        plt.subplot(4, 3, 7)
        plot_image(r, y=256)
        plt.title("Aligned AC")
        plt.subplot(4, 3, 8)
        plot_image(t, y=256)
        plt.title("NC")
        plt.subplot(4, 3, 9)
        plot_image(r, t, y=256)
        plt.title("Diff")
        plt.subplot(4, 3, 10)
        plot_image(r, x=256)
        plt.title("Aligned AC")
        plt.subplot(4, 3, 11)
        plot_image(t, x=256)
        plt.title("NC")
        plt.subplot(4, 3, 12)
        plot_image(r, t, x=256)
        plt.title("Diff")
        # plt.savefig(f"{SAVE_PATH}NC{NC_IDX}/{subject}.png")
        # plt.close()
        plt.show()

    else:
        continue
