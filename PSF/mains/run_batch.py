import os
import csv
import json
import numpy as np
from tifffile import imwrite
from PSF.utils import load, get_windows, get_metrics
from tqdm import tqdm
import time

IMAGE_PATHS = [
    r"Z:\BioMIID_Nonsync\BioMIID_Users_Nonsync\singhi7_BioMIID_Nonsync\20250320_Controlled-misalignment-HD-MDJ\psf-tilt-angle-sweep\Psf-mp-mo3-4900-offset-0000-slit-2250-z-258617_XY001_T001__Channel_GFP-MPSOPi.nd2",
    r"Z:\BioMIID_Nonsync\BioMIID_Users_Nonsync\singhi7_BioMIID_Nonsync\20250320_Controlled-misalignment-HD-MDJ\psf-tilt-angle-sweep\Psf-mp-mo3-4900-offset-0000-slit-2250-z-258617_XY002_T001__Channel_GFP-MPSOPi.nd2",
    r"Z:\BioMIID_Nonsync\BioMIID_Users_Nonsync\singhi7_BioMIID_Nonsync\20250320_Controlled-misalignment-HD-MDJ\psf-tilt-angle-sweep\Psf-mp-mo3-4900-offset-0000-slit-2250-z-258617_XY003_T001__Channel_GFP-MPSOPi.nd2",
    r"Z:\BioMIID_Nonsync\BioMIID_Users_Nonsync\singhi7_BioMIID_Nonsync\20250320_Controlled-misalignment-HD-MDJ\psf-tilt-angle-sweep\Psf-mp-mo3-4900-offset-0000-slit-2250-z-258617_XY004_T001__Channel_GFP-MPSOPi.nd2",
    r"Z:\BioMIID_Nonsync\BioMIID_Users_Nonsync\singhi7_BioMIID_Nonsync\20250320_Controlled-misalignment-HD-MDJ\psf-tilt-angle-sweep\Psf-mp-mo3-4900-offset-0500-slit-2250-z-258617_XY001_T001__Channel_GFP-MPSOPi.nd2",
    r"Z:\BioMIID_Nonsync\BioMIID_Users_Nonsync\singhi7_BioMIID_Nonsync\20250320_Controlled-misalignment-HD-MDJ\psf-tilt-angle-sweep\Psf-mp-mo3-4900-offset-0500-slit-2250-z-258617_XY002_T001__Channel_GFP-MPSOPi.nd2",
    r"Z:\BioMIID_Nonsync\BioMIID_Users_Nonsync\singhi7_BioMIID_Nonsync\20250320_Controlled-misalignment-HD-MDJ\psf-tilt-angle-sweep\Psf-mp-mo3-4900-offset-0500-slit-2250-z-258617_XY003_T001__Channel_GFP-MPSOPi.nd2",
    r"Z:\BioMIID_Nonsync\BioMIID_Users_Nonsync\singhi7_BioMIID_Nonsync\20250320_Controlled-misalignment-HD-MDJ\psf-tilt-angle-sweep\Psf-mp-mo3-4900-offset-0500-slit-2250-z-258617_XY004_T001__Channel_GFP-MPSOPi.nd2",
    r"Z:\BioMIID_Nonsync\BioMIID_Users_Nonsync\singhi7_BioMIID_Nonsync\20250320_Controlled-misalignment-HD-MDJ\psf-tilt-angle-sweep\Psf-mp-mo3-4900-offset-1000-slit-2250-z-258617_XY001_T001__Channel_GFP-MPSOPi.nd2",
    r"Z:\BioMIID_Nonsync\BioMIID_Users_Nonsync\singhi7_BioMIID_Nonsync\20250320_Controlled-misalignment-HD-MDJ\psf-tilt-angle-sweep\Psf-mp-mo3-4900-offset-1000-slit-2250-z-258617_XY002_T001__Channel_GFP-MPSOPi.nd2",
    r"Z:\BioMIID_Nonsync\BioMIID_Users_Nonsync\singhi7_BioMIID_Nonsync\20250320_Controlled-misalignment-HD-MDJ\psf-tilt-angle-sweep\Psf-mp-mo3-4900-offset-1000-slit-2250-z-258617_XY003_T001__Channel_GFP-MPSOPi.nd2",
    r"Z:\BioMIID_Nonsync\BioMIID_Users_Nonsync\singhi7_BioMIID_Nonsync\20250320_Controlled-misalignment-HD-MDJ\psf-tilt-angle-sweep\Psf-mp-mo3-4900-offset-1000-slit-2250-z-258617_XY004_T001__Channel_GFP-MPSOPi.nd2",
    r"Z:\BioMIID_Nonsync\BioMIID_Users_Nonsync\singhi7_BioMIID_Nonsync\20250320_Controlled-misalignment-HD-MDJ\psf-tilt-angle-sweep\Psf-mp-mo3-4900-offset-neg0060-slit-2250-z-258617_XY001_T001__Channel_GFP-MPSOPi.nd2",
    r"Z:\BioMIID_Nonsync\BioMIID_Users_Nonsync\singhi7_BioMIID_Nonsync\20250320_Controlled-misalignment-HD-MDJ\psf-tilt-angle-sweep\Psf-mp-mo3-4900-offset-neg0060-slit-2250-z-258617_XY002_T001__Channel_GFP-MPSOPi.nd2",
    r"Z:\BioMIID_Nonsync\BioMIID_Users_Nonsync\singhi7_BioMIID_Nonsync\20250320_Controlled-misalignment-HD-MDJ\psf-tilt-angle-sweep\Psf-mp-mo3-4900-offset-neg0060-slit-2250-z-258617_XY003_T001__Channel_GFP-MPSOPi.nd2",
    r"Z:\BioMIID_Nonsync\BioMIID_Users_Nonsync\singhi7_BioMIID_Nonsync\20250320_Controlled-misalignment-HD-MDJ\psf-tilt-angle-sweep\Psf-mp-mo3-4900-offset-neg0060-slit-2250-z-258617_XY004_T001__Channel_GFP-MPSOPi.nd2"
]
OUTPUT_ROOT = r"Z:\BioMIID_Nonsync\BioMIID_Users_Nonsync\singhi7_BioMIID_Nonsync\20250320_Controlled-misalignment-HD-MDJ\psf-tilt-angle-sweep\250725_RESULTS"  # All results will go here
DOWNSAMPLE = (1, 1, 1)
THRESH_REL = 0.2
MIN_DISTANCE = 3
CROP_SHAPE = (80, 20, 20)
NORMALIZE = True

FIELDNAMES = [
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

def process_image(image_path, output_dir):
    tic = time.time()
    print(f'Processing {image_path}')
    try:
        img_raw, vox = load.load_image(image_path)
    except Exception as e:
        print(f'  Could not load {image_path}: {e}')
        return
    if img_raw is None or img_raw.size == 0:
        print(f'  Image {image_path} is empty, skipping.')
        return

    img_ds = load.downsample_image(img_raw, DOWNSAMPLE)
    vox_ds = tuple(v * d for v, d in zip(vox, DOWNSAMPLE))
    peaks = get_windows.find_peaks(img_ds, threshold_rel=THRESH_REL, min_distance=MIN_DISTANCE)

    os.makedirs(output_dir, exist_ok=True)
    metadata = {'voxel_size_um': {'z': vox_ds[0], 'y': vox_ds[1], 'x': vox_ds[2]}}
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    csv_path = os.path.join(output_dir, 'results.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES)
        writer.writeheader()

        for idx, pk in enumerate(tqdm(peaks, desc='  Extracting beads')):
            bead = get_windows.extract_cuboid_bead(img_ds, tuple(pk), crop_shape=CROP_SHAPE, normalize=NORMALIZE)
            if bead is None or bead.sum() == 0 or np.count_nonzero(bead) < 10:
                continue

            bead_path = os.path.join(output_dir, f'bead_{idx:04d}.tiff')
            imwrite(bead_path, bead.astype(np.float32))
            
            m = get_metrics.compute_psf_metrics(bead, vox_ds)
            rec = {
                'bead_index': idx,
                'peak_z': int(pk[0]),
                'peak_y': int(pk[1]),
                'peak_x': int(pk[2]),
                **m
            }
            writer.writerow(rec)
            csvfile.flush()

    print(f'  Done: {len(peaks)} peaks processed, results in {output_dir}')
    toc = time.time()
    print(f'  Time taken: {toc - tic} seconds')

def main():
    for image_path in IMAGE_PATHS:
        base = os.path.splitext(os.path.basename(image_path))[0]
        out_dir = os.path.join(OUTPUT_ROOT, base)
        process_image(image_path, out_dir)

if __name__ == '__main__':
    main() 