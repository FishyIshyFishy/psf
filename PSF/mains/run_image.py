import csv
import numpy as np
import matplotlib.pyplot as plt
import napari
import nd2
import time
import pandas as pd

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
DOWNSAMPLE = (2,2,2)  # (z, y, x)
THRESH_REL = 0.2
MIN_DISTANCE = 3
CROP_SHAPE = (80, 20, 20)  # (z, y, x) half-sizes
NORMALIZE = False

# QC filtering parameters
FILTER_BEADS = False  # Set to True to enable QC filtering
QC_PARAMS = {
    'min_snr': 8.0,
    'max_bg_cv': 0.6,
    'max_secondary_peak_ratio': 0.65
}


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
    
    # Add QC fields if filtering is enabled
    if FILTER_BEADS:
        fieldnames.extend(['qc_passed', 'qc_failure_reason'])
    
    with open(OUTPUT_CSV, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        print(f'Processing peaks...')
        passed_beads = 0
        total_beads = 0
        
        for idx, pk in enumerate(peaks):
            tic = time.time()
            
            bead_raw, new_pk = get_windows.extract_bead_adaptive(
                img=img_raw,
                peak=tuple(pk),
                crop_shape=CROP_SHAPE,
                normalize=False
            )
            
            if bead_raw is None or np.count_nonzero(bead_raw) < 10:
                continue
            
            total_beads += 1
            
            # Compute metrics on raw bead
            m = get_metrics.compute_psf_metrics(bead_raw, vox)
            
            # Apply QC filtering if enabled
            if FILTER_BEADS:
                qc_passed, failure_reason = get_windows.apply_qc_filtering(
                    bead_raw=bead_raw,
                    m=m,
                    vox=vox,
                    qc_params=QC_PARAMS
                )
                
                if not qc_passed:
                    print(f'Peak {idx} failed QC: {failure_reason}')
                    continue
                
                passed_beads += 1
            else:
                qc_passed = True
                failure_reason = ""
            
            # Record results
            rec = {
                'bead_index': idx,
                'peak_z': int(new_pk[0]),
                'peak_y': int(new_pk[1]),
                'peak_x': int(new_pk[2]),
                **m
            }
            
            # Add QC information if filtering is enabled
            if FILTER_BEADS:
                rec['qc_passed'] = qc_passed
                rec['qc_failure_reason'] = failure_reason
            
            writer.writerow(rec)
            csvfile.flush()
            print(f'Processed peak {idx} in time {(time.time() - tic):.2f}')
    
    print(f'QC Summary: {passed_beads}/{total_beads} beads passed filtering')


if __name__ == '__main__':
    main()
