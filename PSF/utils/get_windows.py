import os
import json
from pathlib import Path

import numpy as np
import nd2
from skimage.transform import downscale_local_mean
from skimage.filters import gaussian
from skimage.feature import peak_local_max
from scipy.ndimage import center_of_mass, map_coordinates
from tifffile import imwrite

def find_peaks(img, threshold_rel=0.2, min_distance=3):
    sm = gaussian(img, sigma=1)
    peaks = peak_local_max(sm, threshold_rel=threshold_rel, min_distance=min_distance)
    return peaks

def extract_cuboid_bead(img, peak, crop_shape=(6,10,10), normalize=True):
    zc, yc, xc = peak
    dz, dy, dx = crop_shape
    z0, z1 = max(zc - dz, 0), min(zc + dz + 1, img.shape[0])
    y0, y1 = max(yc - dy, 0), min(yc + dy + 1, img.shape[1])
    x0, x1 = max(xc - dx, 0), min(xc + dx + 1, img.shape[2])
    crop = img[z0:z1, y0:y1, x0:x1]
    if crop.size == 0 or crop.max() == 0:
        return None
    return crop / crop.max() if normalize else crop
