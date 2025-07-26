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

from PSF.utils import get_metrics
from PSF.utils import get_windows
from PSF.utils import load

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
    img_raw, vox = load.load_image(IMAGE_PATH)

    print(f'Downsampling image...')
    img_ds = load.downsample_image(img_raw, DOWNSAMPLE)
    vox_ds = tuple(v * d for v, d in zip(vox, DOWNSAMPLE))

    print(f'Finding peaks using fast method...')
    peaks = get_windows.find_peaks_fast(
        img_full=img_raw,
        img_ds=img_ds,
        downsample_factors=DOWNSAMPLE,
        voxel_size_ds=vox_ds,
        min_sep_um=1.0,
        threshold_rel=THRESH_REL,
        min_distance=MIN_DISTANCE,
        exclude_border_vox=CROP_SHAPE
    )

    fieldnames = [
        'bead_index', 'peak_z', 'peak_y', 'peak_x',
        'centroid_z_um', 'centroid_y_um', 'centroid_x_um',
        'fwhm_z_um', 'fwhm_y_um', 'fwhm_x_um',
        'fwhm_pca1_um', 'fwhm_pca2_um', 'fwhm_pca3_um',
        'pca_axis1', 'pca_axis2', 'pca_axis3',
        'tilt_angle_y_deg', 'tilt_angle_x_deg',
        'skew_pc1', 'kurt_pc1',
        'snr', 'vol_ratio_05_01',
        'pc1_z_angle_deg', 'astig_um'
    ]
    with open(OUTPUT_CSV, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        print(f'Processing peaks...')
        for idx, pk in enumerate(peaks):
            # Extract bead from full resolution image using peak location
            bead = get_windows.extract_cuboid_bead_from_full_resolution(
                img_full=img_raw,
                peak_full=pk,  # pk is already in full resolution coordinates
                downsample_factors=DOWNSAMPLE,
                crop_shape=CROP_SHAPE,
                normalize=NORMALIZE
            )
            
            if bead is None or bead.sum() == 0 or np.count_nonzero(bead) < 10:
                continue
            
            m = get_metrics.compute_psf_metrics(bead, vox)  # Use full resolution voxel size
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
