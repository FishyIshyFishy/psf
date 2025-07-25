import csv
import numpy as np
import matplotlib.pyplot as plt
import napari
import nd2
import time

from scipy.ndimage import map_coordinates, center_of_mass
from skimage.filters import gaussian
from skimage.feature import peak_local_max
from skimage.segmentation import flood
from skimage.measure import label, regionprops
from skimage.transform import downscale_local_mean

import utils.get_metrics
import utils.get_windows
import utils.load

# User-editable parameters
IMAGE_PATH = r"Z:\BioMIID_Nonsync\BioMIID_Users_Nonsync\singhi7_BioMIID_Nonsync\20250618_Fluosphere-small-PSF\split\multipoint_psf_xy1.nd2"
OUTPUT_CSV = r'C:\Users\singhi7\Documents\psf\results.csv'
DOWNSAMPLE = (1, 1, 1)  # (z, y, x)
THRESH_REL = 0.2
MIN_DISTANCE = 3
CROP_SHAPE = (80, 20, 20)  # (z, y, x) half-sizes
NORMALIZE = True


def main():
    print(f'Loading image...')
    img_raw, vox = utils.load.load_image(IMAGE_PATH)

    print(f'Downsampling image...')
    img_ds = utils.load.downsample_image(img_raw, DOWNSAMPLE)
    vox_ds = tuple(v * d for v, d in zip(vox, DOWNSAMPLE))

    print(f'Finding peaks...')
    peaks = utils.get_windows.find_peaks(img_ds, threshold_rel=THRESH_REL, min_distance=MIN_DISTANCE)

    fieldnames = [
        'bead_index', 'peak_z', 'peak_y', 'peak_x',
        'centroid_z_um', 'centroid_y_um', 'centroid_x_um',
        'fwhm_z_um', 'fwhm_y_um', 'fwhm_x_um',
        'fwhm_pca1_um', 'fwhm_pca2_um', 'fwhm_pca3_um',
        'pca_axis1', 'pca_axis2', 'pca_axis3'
    ]
    with open(OUTPUT_CSV, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        print(f'Processing peaks...')
        for idx, pk in enumerate(peaks):
            bead = utils.get_windows.extract_cuboid_bead(img_ds, tuple(pk), crop_shape=CROP_SHAPE, normalize=NORMALIZE)
            if bead is None or bead.sum() == 0 or np.count_nonzero(bead) < 10:
                continue
            
            m = utils.get_metrics.compute_psf_metrics(bead, vox_ds)
            rec = {
                'bead_index': idx,
                'peak_z': int(pk[0]),
                'peak_y': int(pk[1]),
                'peak_x': int(pk[2]),
                **m
            }
            writer.writerow(rec)
            csvfile.flush()
            print(f'Processed peak {idx}')

if __name__ == '__main__':
    main()
