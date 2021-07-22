from abc import ABC, abstractmethod
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import SimpleITK as itk


"""
>>> def load(directory):
...     ACs = [f for f in os.listdir(directory) if 'AC' in f]
...     assert len(ACs) == 1
...     HQs = [f for f in os.listdir(directory) if 'HQ' in f]
...     AC = sitk.ReadImage(f"{directory}/{ACs[0]}")
...     HQ = {HQs[i]: sitk.ReadImage(f"{directory}/{HQs[i]}") for i in range(len(HQs))}
...     return AC, HQ

>>> def print_data(directory):
...     AC, HQ = load(directory)
...     print("==================")
...     print(AC.GetDepth(), AC.GetDirection(), AC.GetMetaData("NRRD_space"), AC.GetOrigin(), AC.GetSpacing())
...     print("\n")
...     for n, f in HQ.items():
...         print(n)
...         print(f.GetDepth(), f.GetDirection(), f.GetMetaData("NRRD_space"), f.GetOrigin(), f.GetSpacing())

"""

class ImgConvBase(ABC):

    def __init__(self, image_path, segmentation_path, save_path, output_dims, ignore, NCC_tol):
        self.img_path = image_path
        self.seg_path = segmentation_path
        self.save_path = save_path

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        self.output_dims = output_dims
        self.NCC_tol = NCC_tol
        self.HU_min = -2048
        self.abdo_window_min = -150
        self.abdo_window_max = 250
        self.HU_min_all = 2048
        self.HU_max_all = -2048

        self.subjects = [name for name in os.listdir(self.img_path) if 'F' not in name and name not in ignore]
        self.subjects.sort()

    def list_images(self, ignore: list = None, num_HQ: int = 1, num_AC: int = 1, num_VC: int = 1):
        self.num_HQ = num_HQ
        self.num_AC = num_AC
        self.num_VC = num_VC
        self.HQ_names = {}
        self.AC_names = {}
        self.VC_names = {}
        self.seg_names = {}

        for name in self.subjects:
            img_path = f"{self.img_path}/{name}/"
            imgs = os.listdir(img_path)
            imgs.sort()

            self.HQ_names[name] = []
            self.AC_names[name] = []
            self.VC_names[name] = []

            for im in imgs:
                if im not in ignore:
                    if 'HQ' in im:
                        self.HQ_names[name].append(f"{img_path}{im}")
                    elif 'AC' in im:
                        self.AC_names[name].append(f"{img_path}{im}")
                    elif 'VC' in im:
                        self.VC_names[name].append(f"{img_path}{im}")
                    else:
                        continue

            self.HQ_names[name] = self.HQ_names[name][0:num_HQ]
            self.AC_names[name] = self.AC_names[name][0:num_AC]
            self.VC_names[name] = self.VC_names[name][0:num_VC]

            assert len(self.AC_names[name]) == 1
            # assert len(self.VC_names[name]) == 1                                      # TODO: uncomment when VCs used

            if len(self.HQ_names[name]) < num_HQ:
                print(f"{name} has only {len(self.HQ_names[name])} HQs: {self.HQ_names[name]}")
                continue

            # try:
            #     segs = os.listdir(self.seg_names)

            # except FileNotFoundError:
            #     segs = []

            # else:
            #     segs.sort()
            #     self.source_seg_names[name] = [f"{self.seg_names}{img[-16:-5]}M.seg.nrrd" for img in self.source_names[name] if f"{img[-16:-5]}M.seg.nrrd" in segs]
            #     self.target_seg_names[name] = [f"{self.seg_names}{img[-16:-5]}M.seg.nrrd" for img in self.target_names[name] if f"{img[-16:-5]}M.seg.nrrd" in segs]

        return self
    
    @abstractmethod
    def load_subject(self, subject_ID: str) -> list:
        raise NotImplementedError
    
    def normalise_image(self, im):
        return (im - self.HU_min_all) / (self.HU_max_all - self.HU_min_all)

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
    
    # TODO: enable ability to select reference image
    def display(self, subject_ID: list = None, display=True, HU_min=None, HU_max=None):
        if subject_ID:
            subjects = subject_ID
        else:
            subjects = self.subjects
        
        for subject in subjects:
            proc = self.load_subject(subject, HU_min=HU_min, HU_max=HU_max)

            if isinstance(proc, tuple):
                imgs = proc[0]
                segs = proc[1]

            elif proc == None:
                print(f"{subject} skipped")
                continue
            
            elif isinstance(proc, dict):
                imgs = proc
                segs = None
            
            else:
                raise ValueError(type(proc))

            img_arrays = [itk.GetArrayFromImage(i).transpose([1, 2, 0]) for i in imgs.values()]
            img_names = list(imgs.keys())
            NCCs = [self.calc_NCC(img_arrays[i], img_arrays[0]) for i in range(len(img_arrays))]
            print(subject, img_arrays[0].shape)

            if segs is not None:
                seg_arrays = [itk.GetArrayFromImage(s).transpose([1, 2, 0, 3]) for s in segs.values()]
                seg_names = list(segs.keys())
            else:
                seg_arrays = []

            if display:
                mid_image = img_arrays[0].shape[2] // 2

                if segs is not None:
                    fig, axs = plt.subplots(4, len(img_arrays))
                else:
                    fig, axs = plt.subplots(3, len(img_arrays))

                for i in range(len(img_arrays)):
                    axs[0, i].imshow(img_arrays[i][:, :, mid_image], cmap="gray")
                    axs[0, i].set_title(img_names[i][-16:])
                    axs[0, i].axis("off")
                    axs[1, i].hist(img_arrays[i].ravel(), bins=100)
                    axs[2, i].imshow(img_arrays[0][:, :, mid_image] - img_arrays[i][:, :, mid_image], cmap="gray")
                    axs[2, i].axis("off")
                    axs[2, i].set_title(f"{NCCs[i]:.4f}")
                
                    if segs is not None and len(seg_arrays) > i:
                        axs[3, i].imshow(seg_arrays[i][:, :, mid_image, 0:3] * 255, cmap="gray")
                        axs[3, i].axis("off")

                plt.show()

    @abstractmethod
    def save_data(self, subject_ID: list = None, HU_min=None, HU_max=None) -> int:
        raise NotImplementedError

    @staticmethod
    def check_seg_dims(img_path: str, seg_path: str, phase: str = 'AC'):
        for f in os.listdir(img_path):
            if f"{f}S" not in os.listdir(seg_path):
                print(f"No segmentations matching folder {f}")
                continue

            ims = [i for i in os.listdir(f"{img_path}/{f}") if phase in i]
            
            if len(ims) == 0:
                print(f"No images matching phase {phase} in folder {f}")
                continue
            
            for im in ims:
                img = itk.ReadImage(f"{img_path}/{f}/{im}")

                try:
                    seg = itk.ReadImage(f"{seg_path}/{f}S/{im[:-5]}.seg.nrrd")
                except FileNotFoundError:
                    print(f"No segmentation matching {im}")
                else:
                    print(im, img.GetSize(), seg.GetSize())
                    if (np.array(img.GetSize()) != np.array(seg.GetSize())).all(): print(f"Mismatched dims for {im}")
    
    @staticmethod
    def check_processed_imgs(file_path: str, phase: str = 'AC'):
        imgs = os.listdir(f"{file_path}/Images/{phase}")
        segs = os.listdir(f"{file_path}/Segmentations/{phase}")
        print(len(imgs), len(segs))

        if len(imgs) >= len(segs):
            for im in imgs:
                if im not in segs:
                    print(f"No segmentation matching {im}")

        else:
            for seg in segs:
                if seg not in imgs:
                    print(f"No image matching {seg}")


class ImgConv02(ImgConvBase):

    def __init__(self, image_path=None, segmentation_path=None, save_path=None, output_dims=None, ignore=[], NCC_tol=None):
        super().__init__(image_path, segmentation_path, save_path, output_dims, ignore, NCC_tol)
    
    def load_subject(self, subject_ID: str, HU_min: int = None, HU_max: int = None) -> list:
        image_names = self.AC_names[subject_ID] + self.VC_names[subject_ID] + self.HQ_names[subject_ID]
        assert len(image_names) > 1
        images = []
        seg_names = []
        segs = []

        for i in range(len(image_names)):
            img = itk.ReadImage(image_names[i], itk.sitkInt32)
            image_dir = np.around(img.GetDirection())
            assert img.GetSpacing()[2] == 1.0, f"{image_names[i]}: {img.GetSpacing()}"

            if self.seg_path is not None:
                seg_name = f"{self.seg_path}/{subject_ID}S/{image_names[i][-16:-5]}.seg.nrrd"

                try:
                    seg = itk.ReadImage(seg_name)

                except RuntimeError:
                    print(f"Segmentation not found for {image_names[i][-16:-5]}")
                    seg = None
                
                else:

                    seg_names.append(seg_name)
                    assert np.isclose(img.GetSpacing(), seg.GetSpacing()).all() and np.isclose(img.GetDirection(), seg.GetDirection()).all(), f"{image_names[i]}: {img.GetSpacing()}, {seg.GetSpacing()}, {img.GetDirection()}, {seg.GetDirection()}"
                    
                    if 'AC' in image_names[i]:
                        assert np.isclose(img.GetOrigin(), seg.GetOrigin()).all(), f"{image_names[i]}: {img.GetOrigin()}, {seg.GetOrigin()}"
                    else:
                        print(f"{image_names[i]}, {seg_name} origin issues: {img.GetOrigin()} {seg.GetOrigin()}")

            # Check image is orientated correctly and flip/rotate if necessary
            if image_dir[0] == 0.0 or image_dir[4] == 0.0:
                img = itk.PermuteAxes(img, [1, 0, 2])
                image_dir = np.around(img.GetDirection())
                img = img[::int(image_dir[0]), ::int(image_dir[4]), :]

                if seg is not None:
                    seg = itk.PermuteAxes(seg, [1, 0, 2])
                    seg = seg[::int(image_dir[0]), ::int(image_dir[4]), :]
                    segs.append(seg)

            else:
                img = img[::int(image_dir[0]), ::int(image_dir[4]), :]

                if seg is not None:
                    seg = seg[::int(image_dir[0]), ::int(image_dir[4]), :]
                    segs.append(seg)

            images.append(img)

        image_bounds = []

        # Get start/end coords data for images
        for img in images:
            image_dim_z = img.GetSize()[2]
            image_origin_z = img.GetOrigin()[2]

            # TODO: Will this work without rounding?
            image_bounds.append(np.around([image_origin_z, image_origin_z + image_dim_z - 1]).astype(np.int32))

        image_bounds = np.vstack(image_bounds)
        tightest_bounds = [image_bounds[:, 0].max(), image_bounds[:, 1].min()]

        for i in range(len(images)):
            start = tightest_bounds[0] - image_bounds[i, 0]
            end = tightest_bounds[1] - image_bounds[i, 1]

            if end == 0:
                images[i] = images[i][:, :, start:]
            else:
                images[i] = images[i][:, :, start:end]

            if images[i].GetSize()[2] == 0:
                print(f"{subject_ID} img: size {images[i].GetSize()}")
                return None

        for i in range(len(segs)):
            # TODO: Will this work without rounding?
            seg_bounds = np.around([segs[i].GetOrigin()[2], segs[i].GetOrigin()[2] + segs[i].GetSize()[2] - 1]).astype(np.int32)
            start = tightest_bounds[0] - seg_bounds[0]
            end = tightest_bounds[1] - seg_bounds[1]

            if end == 0:
                segs[i] = segs[i][:, :, start:]
            else:
                segs[i] = segs[i][:, :, start:end]

            if segs[i].GetSize()[2] == 0:
                print(f"{subject_ID} seg: size {segs[i].GetSize()}")
                return None

        if HU_min is not None and HU_max is not None:
            self.HU_min_all = HU_min
            self.HU_max_all = HU_max
            filter = itk.ClampImageFilter()
            filter.SetLowerBound(HU_min)
            filter.SetUpperBound(HU_max)

        # TODO: allow choosing which image to resample to
        # Resample source to target coords
        for i in range(1, len(images)):
            images[i] = itk.Resample(images[i], images[0], defaultPixelValue=self.HU_min)
            assert images[i].GetSize() == images[0].GetSize()

            # Clip target to window if needed
            if HU_min is not None and HU_max is not None:
                images[i] = filter.Execute(images[i])
            else:
                self.HU_min_all = np.min([self.HU_min_all, itk.GetArrayFromImage(images[i]).min()])
                self.HU_max_all = np.max([self.HU_max_all, itk.GetArrayFromImage(images[i]).max()])

        if HU_min is not None and HU_max is not None:
            images[0] = filter.Execute(images[0])

        # TODO: allow choosing which image to resample to
        for i in range(len(segs)):
            segs[i] = itk.Resample(segs[i], images[0], defaultPixelValue=self.HU_min)
            assert segs[i].GetSize() == images[0].GetSize()

        assert len(image_names) == len(images)

        if len(segs) > 0:
            return {name: img for name, img in zip(image_names, images)}, {name: seg for name, seg in zip(seg_names, segs)}
        else:
            return {name: img for name, img in zip(image_names, images)}
    
    def save_data(self, subject_ID: list = None, HU_min=None, HU_max=None) -> int:
        if subject_ID:
            subjects = subject_ID
        else:
            subjects = self.subjects
        
        count = 1
        total = len(subjects)
        
        for subject in subjects:
            proc = self.load_subject(subject, HU_min=HU_min, HU_max=HU_max)

            if isinstance(proc, tuple):
                imgs = proc[0]
                segs = proc[1]

            elif proc == None:
                print(f"{subject} skipped")
                continue
            
            elif isinstance(proc, dict):
                imgs = proc
                segs = None
            
            else:
                raise ValueError(type(proc))

            img_arrays = [itk.GetArrayFromImage(i).transpose([1, 2, 0]).astype("float32") for i in imgs.values()]
            img_names = list(imgs.keys())

            if segs is not None:
                seg_arrays = [itk.GetArrayFromImage(s).transpose([1, 2, 0, 3]).astype("bool") for s in segs.values()]
                seg_names = list(segs.keys())

            vol_thick = img_arrays[0].shape[2]      

            # Partition into sub-volumes
            for name, img in zip(img_names, img_arrays):
                idx = 0  
                stem = name[-16:-5]

                if not os.path.exists(f"{self.save_path}/Images/{stem[6:8]}"):
                    os.makedirs(f"{self.save_path}/Images/{stem[6:8]}")

                for i in range(0, vol_thick, self.output_dims[2]):
                    if i + self.output_dims[2] > vol_thick:
                        break

                    sub_vol = img[:, :, i:i + self.output_dims[2]]
                    np.save(f"{self.save_path}/Images/{stem[6:8]}/{stem}_{idx:03d}.npy", sub_vol)
                    idx += 1
                    count += 1

            for name, seg in zip(seg_names, seg_arrays):
                idx = 0  
                stem = name[-20:-9]

                if not os.path.exists(f"{self.save_path}/Segmentations/{stem[6:8]}"):
                    os.makedirs(f"{self.save_path}/Segmentations/{stem[6:8]}")

                for i in range(0, vol_thick, self.output_dims[2]):
                    if i + self.output_dims[2] > vol_thick:
                        break

                    sub_vol = seg[:, :, i:i + self.output_dims[2]]
                    np.save(f"{self.save_path}/Segmentations/{stem[6:8]}/{stem}_{idx:03d}.npy", sub_vol)
                    idx += 1
                    count += 1

        return count


class ImgConv01(ImgConvBase):

    def __init__(self, image_path, segmentation_path, save_path, output_dims, NCC_tol, source, target):      
        super().__init__(image_path, segmentation_path, save_path, output_dims, NCC_tol)
        self.source = source
        self.target = target
        assert target in ["AC", "VC", "HQ"]
        assert source in ["AC", "VC", "HQ"]

        if not os.path.exists(f"{save_path}r{source}/"): os.makedirs(f"{save_path}r{source}/")
        if not os.path.exists(f"{save_path}r{target}/"): os.makedirs(f"{save_path}r{target}/")
        if not os.path.exists(f"{save_path}rSegs/"): os.makedirs(f"{save_path}rSegs/")

        """
        self.seg_X, self.seg_Y = np.meshgrid(np.linspace(0, output_dims[0] - 1, output_dims[0]), np.linspace(0, output_dims[1] - 1, output_dims[1]))
        """
        """

        """

    def load_subject(self, subject_name: str, HU_min: int = None, HU_max: int = None) -> list:
        super().load_subject(subject_name)
        image_bounds = []

        target = []
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

    FILE_PATH = "D:/ProjectImages"
    SAVE_PATH = "D:/ProjectImages/SyntheticContrast"
    subject_ignore = ["T017A0", "T020A0", "T025A1", "T033A0", "T037A0"]
    image_ignore = [
        "T005A0HQ002.nrrd", "T006A1HQ002.nrrd", "T009A0HQ002.nrrd", "T009A0HQ003.nrrd", "T024A0HQ002.nrrd",
        "T026A0HQ002.nrrd", "T026A0HQ003.nrrd", "T027A0HQ002.nrrd", "T029A0HQ002.nrrd", "T029A0HQ004.nrrd",
        "T031A0HQ002.nrrd", "T031A0HQ003.nrrd", "T031A0HQ005.nrrd", "T031A0HQ006.nrrd"]


    Test = ImgConv02(image_path=FILE_PATH + "/Imgs", segmentation_path=FILE_PATH + "/Segs", save_path=SAVE_PATH, output_dims=(512, 512, 12), ignore=subject_ignore, NCC_tol=0.0)
    # Test.list_images(ignore=image_ignore, num_AC=1, num_VC=1, num_HQ=2).display(display=True, HU_min=-500, HU_max=2500)
    print(Test.list_images(ignore=image_ignore, num_AC=1, num_VC=0, num_HQ=1).save_data(HU_min=-500, HU_max=2500))
    ImgConv02.check_processed_imgs(SAVE_PATH)
