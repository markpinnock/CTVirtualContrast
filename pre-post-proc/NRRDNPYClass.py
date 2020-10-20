import numpy as np
import os
import SimpleITK as itk


class ImgConv:

    """ Implements converter - images and segmentations
        - file_path: data directory
        - output_dims: output size (w, h, d)

        Returns: dict of .npy images and segmentations 
            or saves to disc """

    def __init__(self, file_path, output_dims):
        self.img_path = f"{file_path}/Imgs/"
        self.seg_path = f"{file_path}/Segs/"
        self.output_dims = output_dims
        self.img_files = [name for name in os.listdir(self.img_path) if "F" not in name]
        self.seg_files = [name for name in os.listdir(self.seg_path) if "F" not in name]

    def list_images(self):
        # Build list of image names
        self.AC_names = {}
        self.VC_names = {}
        self.seg_names = {}
        self.NC_names = None # TODO

        for name in self.img_files:
            img_path = f"{self.img_path}{name}/"
            files = os.listdir(img_path)
            self.AC_names[name] = [f"{img_path}{im}" for im in files if "AC" in im]
            self.VC_names[name] = [f"{img_path}{im}" for im in files if "VC" in im]
            
            if len(self.AC_names[name]) == 0:
                continue

            assert len(self.AC_names[name]) == 1
            seg_stem = f"{self.seg_path}{name}S/{self.AC_names[name][0][-16:-5]}M.seg.nrrd"
            self.seg_names[name] = [seg_stem]

        return self

    def load_images(self):

        self.ACs = {}
        self.VCs = {}
        self.segs = {}
        self.NCs = None # TODO

        for key, val in self.AC_names.items():
            if len(val) == 0 or len(self.VC_names[key]) == 0:
                continue

            self.ACs[key] = []
            self.VCs[key] = []
            self.segs[key] = []

            AC = itk.ReadImage(val)
            VC = itk.ReadImage(self.VC_names[key])
            seg = itk.ReadImage(self.seg_names[key])
            tempVC = itk.ReadImage(self.VC_names[key])

            # Resample images and segmentations to match coordinates
            AC, VC = self._resize(AC, VC)
            seg, _ = self._resize(seg, tempVC)
            del tempVC

            assert AC.GetSize() == VC.GetSize(), (key, AC.GetSize(), VC.GetSize())
            assert AC.GetSize() == seg.GetSize(), (key, AC.GetSize(), seg.GetSize())
            assert np.isclose(AC.GetOrigin()[2], VC.GetOrigin()[2], 1.0), (key, AC.GetOrigin(), VC.GetOrigin())
            assert np.isclose(AC.GetOrigin()[2], seg.GetOrigin()[2], 1.0), (key, AC.GetOrigin(), seg.GetOrigin())
            vol_thick = AC.GetSize()[2]

            # Partition in sub-volumes
            for i in range(0, vol_thick, self.output_dims[2]):
                if i + self.output_dims[2] > vol_thick:
                    break

                self.ACs[key].append(itk.GetArrayFromImage(AC[:, :, i:i + self.output_dims[2]]).squeeze().transpose((1, 2, 0)))
                self.VCs[key].append(itk.GetArrayFromImage(VC[:, :, i:i + self.output_dims[2]]).squeeze().transpose((1, 2, 0)))
                sub_seg = itk.GetArrayFromImage(seg[:, :, i:i + self.output_dims[2]])
                self.segs[key].append(np.bitwise_or.reduce(sub_seg[:, :, :, :, :-1], axis=4).squeeze().transpose((1, 2, 0)))

        return self
    
    def return_data(self, save_path=None, normalise=True):
        if save_path is not None:
            count = 0

            if not os.path.exists(save_path):
                os.mkdir(f"{save_path}/")
            if not os.path.exists(f"{save_path}/AC/"):
                os.mkdir(f"{save_path}/AC/")
            if not os.path.exists(f"{save_path}/VC/"):
                os.mkdir(f"{save_path}/VC/")
            if not os.path.exists(f"{save_path}/Segs/"):
                os.mkdir(f"{save_path}/Segs/")

            for key in self.ACs.keys():
                AC_list = self.ACs[key]
                VC_list = self.VCs[key]
                seg_list = self.segs[key]

                for idx, (AC, VC, seg) in enumerate(zip(AC_list, VC_list, seg_list)):
                    AC_stem = self.AC_names[key][0][-16:-5]
                    VC_stem = self.VC_names[key][0][-16:-5]
                    seg_stem = f"{AC_stem}M"

                    AC = (AC - AC.min()) / (AC.max() - AC.min())
                    VC = (VC - VC.min()) / (VC.max() - VC.min())
                    np.save(f"{save_path}/AC/{AC_stem}_{idx:02d}.npy", AC)
                    np.save(f"{save_path}/VC/{VC_stem}_{idx:02d}.npy", VC)
                    np.save(f"{save_path}/Segs/{seg_stem}_{idx:02d}.npy", seg)
                    count += 1
            
            assert len(os.listdir(f"{save_path}/AC")) == len(os.listdir(f"{save_path}/VC"))
            return count

        else:
            return self.ACs, self.VCs, self.segs

    def _resize(self, a, b):
        a_orig = a.GetOrigin()[2]
        b_orig = b.GetOrigin()[2]
        a_thick = a.GetSize()[2]
        b_thick = b.GetSize()[2]
        orig_diff = int(b_orig - a_orig)
        thick_diff = b_thick - a_thick

        # Bottom of b >= bottom of a and top of b < top of a
        if orig_diff >= 0 and orig_diff + b_thick < a_thick:
            a = a[:, :, orig_diff:orig_diff + b_thick]

        # Bottom of b >= bottom of a and b top of b >= top of a
        elif orig_diff >= 0 and orig_diff + b_thick >= a_thick:
            a = a[:, :, orig_diff:]
            b = b[:, :, 0:a_thick - orig_diff]

        # Bottom of b < bottom of a and top of a >= top of b
        elif orig_diff < 0 and abs(orig_diff) + a_thick >= b_thick:
            a = a[: ,:, 0:b_thick - abs(orig_diff)]
            b = b[:, :, abs(orig_diff):]

        # Bottom of b < bottom of a and top of a < top of b
        elif orig_diff < 0 and abs(orig_diff) + a_thick < b_thick:
            b = b[:, :, abs(orig_diff):abs(orig_diff) + a_thick]

        else:
            raise ValueError

        return a, b


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    """ Preview data """

    FILE_PATH = "C:/ProjectImages/"
    Normal = ImgConv(FILE_PATH, (512, 512, 12))
    Normal.list_images().load_images()
    # AC_dict, VC_dict, seg_dict = Normal.return_data()
    num = Normal.return_data("C:/ProjectImages/VirtualContrast/")

    # for ACs, VCs, segs in zip(list(AC_dict.values()), list(VC_dict.values()), list(seg_dict.values())):
    #     for AC, VC, seg in zip(ACs, VCs, segs):
    #         plt.figure(figsize=(18, 9))
    #         plt.subplot(1, 3, 1)
    #         plt.imshow(np.flipud(AC[:, :, 6]), cmap="gray", origin="lower")
    #         plt.subplot(1, 3, 2)
    #         plt.imshow(np.flipud(VC[:, :, 6]), cmap="gray", origin="lower")
    #         plt.subplot(1, 3, 3)
    #         plt.imshow(np.flipud(seg[:, :, 6]), cmap="gray", origin="lower")
    #         plt.pause(3)
    #         plt.close()

    print(num)
