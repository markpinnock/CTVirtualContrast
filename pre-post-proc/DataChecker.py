import json
import matplotlib.pyplot as plt
import numpy as np
import os


FILE_PATH = "C:/ProjectImages/VirtualContrast/"
coords = json.load(open(f"{FILE_PATH}coords.json", 'r'))
ACs = os.listdir(f"{FILE_PATH}AC/")
VCs = os.listdir(f"{FILE_PATH}VC/")
segs = os.listdir(f"{FILE_PATH}Segs/")
ACs.sort()
VCs.sort()
segs.sort()

plt.figure(figsize=(18, 18))

for AC, VC, seg in zip(ACs, VCs, segs):
    ac = np.load(f"{FILE_PATH}AC/{AC}")
    vc = np.load(f"{FILE_PATH}VC/{VC}")
    s = np.load(f"{FILE_PATH}Segs/{seg}")
    cx = np.array([int(c[0]) for c in coords[AC[:-4]]])
    cy = np.array([int(c[1]) for c in coords[AC[:-4]]])
    coord_idx = 2

    plt.subplot(2, 2, 1)
    plt.imshow(ac[cy[coord_idx]-64:cy[coord_idx]+64, cx[coord_idx]-128:cx[coord_idx]+128, 5], cmap="gray")
    # plt.imshow(np.flipud(ac[:, :, 5].T), cmap="gray", origin="lower")
    # plt.plot(cy, 512 - cx, 'r+', markersize=15)
    plt.axis("off")
    plt.subplot(2, 2, 2)
    plt.imshow(np.flipud(vc[:, :, 5].T), cmap="gray", origin="lower")
    plt.axis("off")
    plt.subplot(2, 2, 3)
    plt.imshow(np.flipud(ac[:, :, 5].T - vc[:, :, 5].T), cmap="gray", origin="lower")
    plt.axis("off")
    plt.subplot(2, 2, 4)
    plt.imshow(np.flipud(s[:, :, 5].T), cmap="gray", origin="lower")
    plt.axis("off")
    plt.title(f"{s.min(), s.max()}")
    # plt.show()
    plt.pause(0.1)
    plt.clf()
