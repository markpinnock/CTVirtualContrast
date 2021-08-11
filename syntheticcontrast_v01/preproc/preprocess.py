import json
import matplotlib.pyplot as plt
import numpy as np
import os
import SimpleITK as itk


class ImgConv:

    def __init__(self, file_path: str, save_path: str, output_dims: tuple, NCC_tol: float, source: str, target: str):

        """ file_path: location of image data
                (expects sub-folders of AC, VC and HQ)
            save_path: location of data to be saved 
            output_dims: tuple of output image dimensions
            target: one of AC, VC, HQ
            source: one of AC, VC, HQ """

        self.img_path = f"{file_path}/Imgs/"
        self.seg_path = f"{file_path}/Segs/"
        self.save_path = save_path
        self.source = source
        self.target = target
        assert target in ["AC", "VC", "HQ"]
        assert source in ["AC", "VC", "HQ"]
        if not os.path.exists(f"{save_path}r{source}/"): os.makedirs(f"{save_path}r{source}/")
        if not os.path.exists(f"{save_path}r{target}/"): os.makedirs(f"{save_path}r{target}/")
        if not os.path.exists(f"{save_path}rSegs/"): os.makedirs(f"{save_path}rSegs/")
        self.output_dims = output_dims
        self.subjects = [name for name in os.listdir(self.img_path) if "F" not in name]#[-3:]
        self.subjects.sort()
        self.seg_X, self.seg_Y = np.meshgrid(np.linspace(0, output_dims[0] - 1, output_dims[0]), np.linspace(0, output_dims[1] - 1, output_dims[1]))
        self.abdo_window_min = -150
        self.abdo_window_max = 250
        self.HU_min = -2048
        self.NCC_tol = NCC_tol

    def list_images(self, num_sources: int = 1, num_targets: int = 1):
        assert num_targets == 1, "Only handles one target img currently"
        assert num_sources == 1, "Only handles one source img currently"
        self.num_sources = num_sources
        self.num_targets = num_targets
        self.source_names = {}
        self.target_names = {}
        self.source_seg_names = {}
        self.target_seg_names = {}

        for name in self.subjects:
            img_path = f"{self.img_path}{name}/"
            seg_path = f"{self.seg_path}{name}S/"
            imgs = os.listdir(img_path)
            imgs.sort()

            self.source_names[name] = [f"{img_path}{im}" for im in imgs if f"{self.source}" in im]
            self.target_names[name] = [f"{img_path}{im}" for im in imgs if f"{self.target}" in im]

            if len(self.source_names[name]) < 1 or len(self.target_names[name]) < 1:
                continue

            """ NB num_source !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"""
            self.source_names[name] = self.source_names[name]#[0:num_sources]
            self.target_names[name] = self.target_names[name][0:num_targets]

            try:
                segs = os.listdir(seg_path)
            except FileNotFoundError:
                segs = []
            else:
                segs.sort()
                self.source_seg_names[name] = [f"{seg_path}{img[-16:-5]}M.seg.nrrd" for img in self.source_names[name] if f"{img[-16:-5]}M.seg.nrrd" in segs]
                self.target_seg_names[name] = [f"{seg_path}{img[-16:-5]}M.seg.nrrd" for img in self.target_names[name] if f"{img[-16:-5]}M.seg.nrrd" in segs]

        return self

    def _load_images(self, subject_name: str) -> np.array:
        source_names = self.source_names[subject_name]
        target_names = self.target_names[subject_name]
        source_seg_names = self.source_seg_names[subject_name]
        target_seg_names = self.target_seg_names[subject_name]
        assert len(target_names) == 1
        assert len(target_names) == len(target_seg_names)
        sources = []

        for i in range(len(target_names)):
            target = itk.ReadImage(target_names[i], itk.sitkInt32)
            seg = itk.ReadImage(target_seg_names[i])
            target_spacing = target.GetSpacing()[2]
            target_dir = target.GetDirection()
            seg_spacing = seg.GetSpacing()[2]
            assert target_spacing == 1.0
            assert seg_spacing == 1.0
            target_dim_z = target.GetSize()[2]
            target_origin_z = target.GetOrigin()[2]
            tightest_bounds_diff = [-1000, 1000]

            # Get start/end coords data for sources
            for j in range(len(source_names)):
                source = itk.ReadImage(source_names[j], itk.sitkInt32)
                source_stem = f"{subject_name}_{self.source}_{i:03d}"
                source_spacing = source.GetSpacing()[2]
                assert source_spacing == 1.0
                source_dim_z = source.GetSize()[2]
                source_origin_z = source.GetOrigin()[2]
                source_bounds = np.around([source_origin_z, source_origin_z + source_dim_z - 1]).astype(np.int32)
                target_bounds = np.around([target_origin_z, target_origin_z + target_dim_z - 1]).astype(np.int32)
                bounds_diff = source_bounds - target_bounds

                if source_bounds[0] - target_bounds[1] + source_bounds[1] - target_bounds[0] > 140:
                    print("skipped", i)
                    continue

                # Save tightest start/end coords, add/subtract 1 for rounding errors
                if bounds_diff[0] > tightest_bounds_diff[0]:
                    tightest_bounds_diff[0] = bounds_diff[0] + 1
                if bounds_diff[1] < tightest_bounds_diff[1]:
                    tightest_bounds_diff[1] = bounds_diff[1] - 1

                # Resample source to target coords
                source = itk.Resample(source, target, defaultPixelValue=self.HU_min)
                sources.append(source)

            # Crop target and seg to tightest start/end coords
            target = np.transpose(itk.GetArrayFromImage(target), [2, 1, 0])
            seg = np.transpose(itk.GetArrayFromImage(seg), [2, 1, 0, 3]).astype(np.uint8)

            if (np.array(tightest_bounds_diff) > 0).all():
                target = target[:, :, tightest_bounds_diff[0]:]
                seg = seg[:, :, tightest_bounds_diff[0]:]

            elif (np.array(tightest_bounds_diff) < 0).all():
                target = target[:, :, :tightest_bounds_diff[1]]
                seg = seg[:, :, :tightest_bounds_diff[1]]

            elif tightest_bounds_diff[0] > 0 and tightest_bounds_diff[1] < 0:
                target = target[:, :, tightest_bounds_diff[0]:tightest_bounds_diff[1]]
                seg = seg[:, :, tightest_bounds_diff[0]:tightest_bounds_diff[1]]

            elif tightest_bounds_diff[0] < 0 and tightest_bounds_diff[1] > 0:
                print("!!!!!!!!!!!!!!!!!!!!!!!!")
                pass
            
            else:
                return None, None, None

            # Repeat for all source images
            for j in range(len(sources)):
                source = sources[j]

                # Ensure both source and target are same way up
                source_dir = source.GetDirection()
                assert np.isclose(np.array(source_dir), np.array(target_dir)).all()

                source = np.transpose(itk.GetArrayFromImage(source), [2, 1, 0])

                if (np.array(tightest_bounds_diff) > 0).all():
                    source = source[:, :, tightest_bounds_diff[0]:]

                elif (np.array(tightest_bounds_diff) < 0).all():
                    source = source[:, :, :tightest_bounds_diff[1]]

                elif tightest_bounds_diff[0] > 0 and tightest_bounds_diff[1] < 0:
                    source = source[:, :, tightest_bounds_diff[0]:tightest_bounds_diff[1]]

                elif tightest_bounds_diff[0] < 0 and tightest_bounds_diff[1] > 0:
                    print("!!!!!!!!!!!!!!!!!!!!!!!!")
                    pass

                else:
                    return None, None, None

                assert np.isclose(source.shape, target.shape).all(), (source.shape, target.shape)

                # Clip source to window
                source = np.clip(source, self.abdo_window_min, self.abdo_window_max)

                # If coords are reversed, flip images
                if target_dir[0] < 0 and target_dir[4] < 0:
                    source = np.flipud(np.fliplr(source))
                
                sources[j] = source

            # Clip target to window
            target = np.clip(target, self.abdo_window_min, self.abdo_window_max)

            # If coords are reversed, flip images
            if target_dir[0] < 0 and target_dir[4] < 0:
                target = np.flipud(np.fliplr(target))
                seg = np.flipud(np.fliplr(seg))

        return sources, target, seg
    
    def normalise_image(self, im):
        return (im - self.abdo_window_min) / (self.abdo_window_max - self.abdo_window_min)


    def calc_NCC(self, a: object, b: object) -> int:
        assert len(a.shape) == 3
        a = self.normalise_image(a)
        b = self.normalise_image(b)
        N = np.prod(a.shape)

        mu_a = np.mean(a)
        mu_b = np.mean(b)
        sig_a = np.std(a)
        sig_b = np.std(b)

        return np.sum((a - mu_a) * (b - mu_b) / (N * sig_a * sig_b))
    
    def calc_RBF(self, a: object, b: object, gamma: int) -> int:
        a = self.normalise_image(a)
        b = self.normalise_image(b)

        return np.exp(-gamma * np.sum(np.power(a - b, 2)))
   
    def segmentation_com(self, seg):

        if not seg[:, :, 5].sum():
            return [seg.shape[0] // 2, seg.shape[0] // 2]

        x_coord = self.seg_X[seg[:, :, 5] == 1].sum() / ((seg[:, :, 5] == 1).sum() + 1e-8)
        y_coord = self.seg_Y[seg[:, :, 5] == 1].sum() / ((seg[:, :, 5] == 1).sum() + 1e-8)

        return [x_coord, y_coord]
    
    def save_data(self, normalise: bool=True) -> int:
        count = 1
        total = len(self.source_names.keys())
        coords = {}

        for key in self.source_names.keys():
            if key not in self.target_seg_names:
                print(f"{key} failed, no seg mask")
                continue

            print(f"Loading {key}: {count} of {total}")
            sources, target, seg = self._load_images(key)

            if sources is None or target is None or seg is None:
                print(f"{key} failed, no source/target/seg")
                continue

            target_stem = self.target_names[key][0][-16:-5]
            seg_stem = f"{target_stem}M"

            # Partition source into sub-volumes
            vol_thick = target.shape[2]
            idx = 0

            # Normalise
            if normalise:
                target = self.normalise_image(target)           

            # Partition into sub-volumes
            for i in range(0, vol_thick, self.output_dims[2]):
                if i + self.output_dims[2] > vol_thick:
                    break

                target_sub_vol = target[:, :, i:i + self.output_dims[2]]
                seg_sub_vol = seg[:, :, i:i + self.output_dims[2], :]

                # Calculate centroids of segmentations
                assert seg_sub_vol.shape[3] == 5, f"Incorrect seg dims {seg_sub_vol}"
                arteries = self.segmentation_com(seg_sub_vol[:, :, :, 0])
                r_kidney = self.segmentation_com(seg_sub_vol[:, :, :, 1])
                l_kidney = self.segmentation_com(seg_sub_vol[:, :, :, 2])

                seg_sub_vol = np.bitwise_or.reduce(seg_sub_vol[:, :, :, :-1], axis=3).squeeze()
                coords[f"{seg_stem}_{idx:03d}"] = [arteries, r_kidney, l_kidney]
                np.save(f"{self.save_path}r{self.target}/{target_stem}_{idx:03d}.npy", target_sub_vol)
                np.save(f"{self.save_path}/rSegs/{seg_stem}_{idx:03d}.npy", seg_sub_vol)
                idx += 1

            for j in range(len(sources)):
                source = sources[j]
                source_stem = self.source_names[key][j][-16:-5]
                
                # Discard if alignment inadequate
                NCC = self.calc_NCC(source, target)
                if NCC < self.NCC_tol:
                    print(f"{key} number {i + 1} discarded: NCC {NCC}")
                    continue

                # Normalise
                if normalise:
                    source = self.normalise_image(source)

                # Partition into sub-volumes
                idx = 0

                for i in range(0, vol_thick, self.output_dims[2]):
                    if i + self.output_dims[2] > vol_thick:
                        break

                    source_sub_vol = source[:, :, i:i + self.output_dims[2]]
                    np.save(f"{self.save_path}r{self.source}/{source_stem}_{idx:03d}.npy", source_sub_vol)
                    idx += 1

            count += 1

        # assert len(os.listdir(f"{self.save_path}{self.source}")) == len(os.listdir(f"{self.save_path}{self.target}"))
        assert len(os.listdir(f"{self.save_path}/rSegs/")) == len(os.listdir(f"{self.save_path}r{self.target}"))
        json.dump(coords, open(f"{self.save_path}rcoords.json", 'w'), indent=4)

    def view_data(self):
        count = 1
        total = len(self.source_names.keys())
        nccs, rbfs = [], []
        fig, axs = plt.subplots(self.num_targets, 6, figsize=(18, 8))

        for key in self.source_names.keys():
            if key not in self.target_seg_names:
                print(f"{key} failed, no seg mask")
                continue

            print(f"Loading {key}: {count} of {total}")
            sources, target, seg = self._load_images(key)

            if sources is None or target is None or seg is None:
                print(f"{key} failed, no source/target/seg")
                continue

            for i in range(self.num_targets):
                for j in range(len(sources)):
                    source = sources[j]
                
                    axs[0].imshow(source[:, :, 10].T, cmap="gray")
                    axs[0].axis("off")
                    axs[0].set_title(f"{key} {j}")
                    axs[1].imshow(target[:, :, 10].T, cmap="gray")
                    axs[1].axis("off")
                    axs[2].imshow(source[:, :, 10].T - target[:, :, 10].T, cmap="gray")
                    axs[2].axis("off")
                    axs[2].set_title(f"NCC: {self.calc_NCC(source, target)}")
                    axs[3].imshow(np.flipud(source[:, 256, :].T), cmap="gray")
                    axs[3].axis("off")
                    axs[4].imshow(np.flipud(target[:, 256, :].T), cmap="gray")
                    axs[4].axis("off")
                    axs[5].imshow(seg[:, :, 10, 0].T, cmap="gray")
                    axs[5].axis("off")
                    axs[5].set_title(f"RBF: {self.calc_RBF(source, target, 1e-8)}")

                    plt.pause(0.5)
                    plt.cla()

    def check_saved(self):
        subjects = []
        sources, targets, segs, coord_list = [], [], [], []
        coord_json = json.load(open(f"{self.save_path}rcoords.json", 'r'))

        for img in os.listdir(f"{self.save_path}r{self.source}"):
            if img[:6] not in subjects: subjects.append(img[:6])
        
        for img_name in subjects:
            imgs = [im for im in os.listdir(f"{self.save_path}r{self.source}") if img_name in im]
            sources.append(np.load(f"{self.save_path}r{self.source}/{imgs[-1]}"))
            imgs = [im for im in os.listdir(f"{self.save_path}r{self.target}") if img_name in im]
            targets.append(np.load(f"{self.save_path}r{self.target}/{imgs[-1]}"))
            imgs = [im for im in os.listdir(f"{self.save_path}rSegs/") if img_name in im]
            segs.append(np.load(f"{self.save_path}rSegs/{imgs[-1]}"))
            cx = np.array([int(c[0]) for c in coord_json[imgs[-1][:-4]]])
            cy = np.array([int(c[1]) for c in coord_json[imgs[-1][:-4]]])
            coord_list.append((cx, cy))

        plt.figure(figsize=(18, 8))

        for source, target, seg, coord in zip(sources, targets, segs, coord_list):
            coord_x, coord_y = coord

            plt.subplot(2, 2, 1)
            plt.imshow(source[:, :, 11].T, cmap="gray")
            plt.plot(coord_y, coord_x, 'r+', markersize=15)
            plt.axis("off")
            plt.subplot(2, 2, 2)
            plt.imshow(target[:, :, 11].T, cmap="gray")
            plt.axis("off")
            plt.subplot(2, 2, 3)
            plt.imshow(target[:, :, 11].T - source[:, :, 11].T, cmap="gray")
            plt.axis("off")
            plt.title(f"{self.calc_NCC(target, source):.3f} {self.calc_RBF(target, source, 1):.3f}")
            plt.subplot(2, 2, 4)
            plt.imshow(seg[:, :, 11].T)
            plt.axis("off")

            plt.pause(5)
            plt.clf()


if __name__ == "__main__":

    FILE_PATH = "C:/ProjectImages/"
    SAVE_PATH = "C:/ProjectImages/VirtualContrastTest/"

    Normal = ImgConv(
        file_path=FILE_PATH,
        save_path=SAVE_PATH,
        output_dims=(512, 512, 12),
        NCC_tol=0.0,
        source="HQ",
        target="AC"
        )

    Normal.list_images()
    Normal.view_data()
    # Normal.save_data(normalise=True)
    # Normal.check_saved()
