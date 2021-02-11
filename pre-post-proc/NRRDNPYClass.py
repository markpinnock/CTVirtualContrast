import json
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
        self.seg_X, self.seg_Y = np.meshgrid(np.linspace(0, output_dims[0] - 1, output_dims[0]), np.linspace(0, output_dims[1] - 1, output_dims[1]))
        self.HU_min = -350
        self.HU_max = 450

    def list_images(self):
        # Build list of image names
        self.AC_names = {}
        self.VC_names = {}
        self.seg_names = {}
        self.NC_names = {}

        for name in self.img_files:
            img_path = f"{self.img_path}{name}/"
            files = os.listdir(img_path)
            self.AC_names[name] = [f"{img_path}{im}" for im in files if "AC" in im]
            self.VC_names[name] = [f"{img_path}{im}" for im in files if "VC" in im]
            self.NC_names[name] = [f"{img_path}{im}" for im in files if "HQ" in im]
            
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
        self.NCs = {}

        for key, val in self.AC_names.items():
            if len(val) == 0:# or len(self.VC_names[key]) == 0:
                continue

            self.ACs[key] = []
            self.VCs[key] = []
            self.segs[key] = []
            self.NCs = {}
            NC_idx = 0

            # NB: THIS ASSUMES SEGMENTATIONS BASED ON AC!!!!!
            AC = itk.ReadImage(val, itk.sitkInt32)[:, :, :, 0]
            VC = itk.ReadImage(self.VC_names[key], itk.sitkInt32)[:, :, :, 0]
            seg = itk.ReadImage(self.seg_names[key])[:, :, :, 0]

            # NC = itk.ReadImage(self.NC_names[key][NC_idx], itk.sitkInt32)
            AC = itk.Resample(AC, VC, defaultPixelValue=1e6)
            seg = itk.Resample(seg, VC, defaultPixelValue=1e6)

            assert np.isclose(AC.GetOrigin()[2], VC.GetOrigin()[2], 1.0), (key, AC.GetOrigin(), VC.GetOrigin())
            assert np.isclose(AC.GetOrigin()[2], seg.GetOrigin()[2], 1.0), (key, AC.GetOrigin(), seg.GetOrigin())
            AC_dir = AC.GetDirection()
            VC_dir = VC.GetDirection()
            seg_dir = seg.GetDirection()
            assert np.isclose(np.array(AC_dir), np.array(VC_dir)).all()
            assert np.isclose(np.array(AC_dir), np.array(seg_dir)).all()

            # # Resample images and segmentations to match coordinates
            # seg, _ = self._resize(seg, VC)
            # AC, VC = self._resize(AC, VC)

            AC = np.transpose(itk.GetArrayFromImage(AC), [2, 1, 0])
            VC = np.transpose(itk.GetArrayFromImage(VC), [2, 1, 0])
            seg = np.transpose(itk.GetArrayFromImage(seg), [2, 1, 0, 3])
            AC_idx = np.argwhere(~np.all(AC == 1e6, axis=(0, 1)) == True)

            AC = AC[:, :, AC_idx[0][0]:AC_idx[-1][0]]
            VC = VC[:, :, AC_idx[0][0]:AC_idx[-1][0]]
            seg = seg[:, :, AC_idx[0][0]:AC_idx[-1][0], :]
            AC = np.clip(AC, self.HU_min, self.HU_max)
            VC = np.clip(VC, self.HU_min, self.HU_max)

            assert (AC == 1e6).sum() == 0, f"1e6 in AC {key}"
            assert (seg == 1e6).sum() == 0, f"1e6 in seg {key}"
            assert np.isclose(AC.shape, VC.shape).all(), (AC.shape, VC.shape)
            assert np.isclose(seg.shape[:-1], VC.shape).all(), (seg.shape, VC.shape)

            if AC_dir[0] < 0 and AC_dir[4] < 0:
                AC = np.flipud(np.fliplr(AC))
                VC = np.flipud(np.fliplr(VC))
                seg = np.flipud(np.fliplr(seg))

            vol_thick = AC.shape[2]

            # Partition in sub-volumes
            for i in range(0, vol_thick, self.output_dims[2]):
                if i + self.output_dims[2] > vol_thick:
                    break

                self.ACs[key].append(AC[:, :, i:i + self.output_dims[2]])
                self.VCs[key].append(VC[:, :, i:i + self.output_dims[2]])
                self.segs[key].append(seg[:, :, i:i + self.output_dims[2], :])

        return self
    
    def segmentation_com(self, seg):

        if not seg[:, :, 5].sum():
            return [seg.shape[0] // 2, seg.shape[0] // 2]

        x_coord = self.seg_X[seg[:, :, 5] == 1].sum() / ((seg[:, :, 5] == 1).sum() + 1e-8)
        y_coord = self.seg_Y[seg[:, :, 5] == 1].sum() / ((seg[:, :, 5] == 1).sum() + 1e-8)

        return [x_coord, y_coord]
    
    def return_data(self, save_path=None, normalise=True):
        coords = {}

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

                    # Calculate centroids of segmentations
                    if seg.shape[3] == 5:
                        arteries = self.segmentation_com(seg[:, :, :, 0])
                        r_kidney = self.segmentation_com(seg[:, :, :, 1])
                        l_kidney = self.segmentation_com(seg[:, :, :, 2])
                    elif seg.shape[3] == 4:
                        arteries = [seg.shape[1] // 2, seg.shape[2] // 2]
                        r_kidney = self.segmentation_com(seg[:, :, :, 1])
                        l_kidney = self.segmentation_com(seg[:, :, :, 2])
                    else:
                        raise ValueError("Incorrect seg dims")

                    seg = np.bitwise_or.reduce(seg[:, :, :, :-1], axis=3).squeeze()

                    if normalise:
                        AC = (AC - self.HU_min) / (self.HU_max - self.HU_min)
                        VC = (VC - self.HU_min) / (self.HU_max - self.HU_min)

                    np.save(f"{save_path}/AC/{AC_stem}_{idx:02d}.npy", AC)
                    np.save(f"{save_path}/VC/{VC_stem}_{idx:02d}.npy", VC)
                    np.save(f"{save_path}/Segs/{seg_stem}_{idx:02d}.npy", seg)
                    coords[f"{AC_stem}_{idx:02d}"] = [arteries, r_kidney, l_kidney]
                    count += 1
            
            assert len(os.listdir(f"{save_path}/AC")) == len(os.listdir(f"{save_path}/VC"))
            json.dump(coords, open(f"{save_path}coords.json", 'w'), indent=4)

            return count

        else:
            return self.ACs, self.VCs, self.segs, coords

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
    # AC_dict, VC_dict, seg_dict, _ = Normal.return_data()
    num = Normal.return_data("C:/ProjectImages/VirtualContrast/")
    print(num)

    # plt.figure(figsize=(18, 9))

    # for ACs, VCs, segs in zip(list(AC_dict.values()), list(VC_dict.values()), list(seg_dict.values())):
    #     for AC, VC, seg in zip(ACs, VCs, segs):
    #         plt.subplot(1, 3, 1)
    #         plt.imshow(np.flipud(AC[:, :, 6].T), cmap="gray", origin="lower")
    #         plt.subplot(1, 3, 2)
    #         plt.imshow(np.flipud(VC[:, :, 6].T), cmap="gray", origin="lower")
    #         plt.subplot(1, 3, 3)
    #         plt.imshow(np.flipud(seg[:, :, 6, 0].T), cmap="gray", origin="lower")
    #         plt.pause(3)
    #         plt.gca().clear()
