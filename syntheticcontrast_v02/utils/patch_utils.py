import numpy as np


def extract_patches(img, stride_length, patch_size, downsample):
    H, W, D = img.shape
    patches = []
    indices = []

    if patch_size == 0:
        for k in range(0, D - stride_length, stride_length):
            patches.append(img[::downsample, ::downsample, k:(k + patch_size[2]), np.newaxis])
            indices.append([0, 0, k]])

            if (k + patch_size[2]) > D:  
                patches.append(img[::downsample, ::downsample, -patch_size[2]:, np.newaxis])
                indices.append([0, 0, -patch_size[2]])
                return patches, indices

            return patches, indices

    else:
        for i in range(0, H - stride_length, stride_length):
            for j in range(0, W - stride_length, stride_length):
                for k in range(0, D - stride_length, stride_length):
                    patches.append(img[i:(i + patch_size[0]):downsample, j:(j + patch_size[1]):downsample, k:(k + patch_size[2]), np.newaxis])
                    indices.append([i, j, k])

                    if (k + patch_size[2]) > D:  
                        for i in range(0, H - stride_length, stride_length):
                            for j in range(0, W - stride_length, stride_length):
                                patches.append(img[::downsample, ::downsample, -patch_size[2]:, np.newaxis])
                                indices.append([i, j, -patch_size[2]])

                        return patches, indices

                    return patches, indices
