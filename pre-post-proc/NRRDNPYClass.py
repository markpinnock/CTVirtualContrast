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
        self.file_path = file_path
        self.output_dims = output_dims
        self.files = [name for name in os.listdir(file_path) if "F" not in name]

    def list_images(self):
        # Build list of image names
        self.AC_names = {}
        self.VC_names = {}
        self.NC_names = None # TODO

        for name in self.files:
            path = f"{self.file_path}{name}/"
            files = os.listdir(path)
            self.AC_names[name] = [f"{path}{im}" for im in files if "AC" in im]
            self.VC_names[name] = [f"{path}{im}" for im in files if "VC" in im]

        return self

    def load_images(self):

        self.ACs = {}
        self.VCs = {}
        self.NCs = None # TODO

        for key, val in self.AC_names.items():
            if len(val) == 0 or len(self.VC_names[key]) == 0:
                continue

            self.ACs[key] = []
            self.VCs[key] = []

            AC = itk.ReadImage(val)
            VC = itk.ReadImage(self.VC_names[key])

            # Resample images to match coordinates
            AC, VC = self._resize(AC, VC)
            assert AC.GetSize() == VC.GetSize(), (key, AC.GetSize(), VC.GetSize())
            assert np.isclose(AC.GetOrigin()[2], VC.GetOrigin()[2], 1.0), (key, AC.GetOrigin(), VC.GetOrigin())
            vol_thick = AC.GetSize()[2]

            # Partition in sub-volumes
            for i in range(0, vol_thick, self.output_dims[2]):
                if i + self.output_dims[2] > vol_thick:
                    break

                self.ACs[key].append(itk.GetArrayFromImage(AC[:, :, i:i + self.output_dims[2]]).squeeze().transpose((1, 2, 0)))
                self.VCs[key].append(itk.GetArrayFromImage(VC[:, :, i:i + self.output_dims[2]]).squeeze().transpose((1, 2, 0)))

        return self
    
    def return_data(self, save_path=None):
        if save_path is not None:
            count = 0

            if not os.path.exists(save_path):
                os.mkdir(f"{save_path}/")
                os.mkdir(f"{save_path}/AC/")
                os.mkdir(f"{save_path}/VC/")

            for key in self.ACs.keys():
                AC_list = self.ACs[key]
                VC_list = self.VCs[key]
                
                for idx, (AC, VC) in enumerate(zip(AC_list, VC_list)):
                    AC_stem = self.AC_names[key][0][-16:-5]
                    VC_stem = self.VC_names[key][0][-16:-5]
                    np.save(f"{save_path}/AC/{AC_stem}_{idx:02d}.npy", AC)
                    np.save(f"{save_path}/VC/{VC_stem}_{idx:02d}.npy", VC)
                    count += 1
            
            assert len(os.listdir(f"{save_path}/AC")) == len(os.listdir(f"{save_path}/VC"))
            return count
        else:
            return self.ACs, self.VCs

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

    """ Testing for ImgConv class """

    FILE_PATH = "C:/ProjectImages/Imgs/"
    Normal = ImgConv(FILE_PATH, (512, 512, 12))
    AC_dict, VC_dict = Normal.list_images().load_images().return_data()
    num = Normal.return_data("C:/ProjectImages/VirtualContrast/")

    # for ACs, VCs in zip(list(AC_dict.values()), list(VC_dict.values())):
        # for AC, VC in zip(ACs, VCs):
            # plt.figure(figsize=(18, 9))
            # plt.subplot(1, 3, 1)
            # plt.imshow(np.flipud(AC[:, :, 6]), cmap="gray", origin="lower")
            # plt.subplot(1, 3, 2)
            # plt.imshow(np.flipud(VC[:, :, 6]), cmap="gray", origin="lower")
            # plt.subplot(1, 3, 3)
            # plt.imshow(np.flipud(AC[:, :, 6] - VC[:, :, 6]), cmap="gray", origin="lower")
            # plt.pause(3)
            # plt.close()

    print(num)
