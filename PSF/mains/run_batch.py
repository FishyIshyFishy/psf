import os
import csv
import json
import numpy as np
from tifffile import imwrite
from PSF.utils import load, get_windows, get_metrics
from tqdm import tqdm
import time
import napari
import pandas as pd

DEBUG = True

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
DOWNSAMPLE = (2, 2, 2)  # Updated to match run_image.py
THRESH_REL = 0.2
MIN_DISTANCE = 3
CROP_SHAPE = (40,40,40)
NORMALIZE = True

# QC filtering parameters
FILTER_BEADS = True  # Set to True to enable QC filtering
QC_PARAMS = {
    'min_snr': 8.0,
    'max_bg_cv': 0.6,
    'max_secondary_peak_ratio': 1.,
    # Shape-based QC parameters
    'min_linearity': 0.7,           # Linearity (elongation) threshold
    'max_sphericity': 0.15,         # Maximum sphericity (avoid blobs)
    'max_planarity': 0.3,           # Maximum planarity (avoid sheets)
    'min_length_width_ratio': 4.0,  # Minimum length/width ratio
    'min_width_um': 0.3,            # Minimum width in µm
    'max_width_um': 1.2,            # Maximum width in µm
    'min_thickness_um': 0.3,        # Minimum thickness in µm
    'max_thickness_um': 1.2,        # Maximum thickness in µm
    'max_tortuosity': 1.15,         # Maximum tortuosity
    'max_width_CV': 0.35,           # Maximum width coefficient of variation
    'max_border_fraction': 0.1,     # Maximum fraction touching borders
    'max_branches': 0               # Maximum number of skeleton branches
}

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

# Add QC fields if filtering is enabled
if FILTER_BEADS:
    FIELDNAMES.extend(['qc_passed', 'qc_failure_reason'])

def visualize_beads_in_napari(beads, image_name, start_idx=0, batch_size=20):
    viewer = napari.Viewer(title=f"Beads from {image_name} (batch {start_idx//batch_size + 1})")
    
    for i, bead in enumerate(beads):
        if bead is not None and bead.size > 0:
            viewer.add_image(
                bead, 
                name=f"Bead {start_idx + i:04d}",
                colormap='viridis',
                blending='additive'
            )
    
    print(f"  Showing beads {start_idx} to {start_idx + len(beads) - 1} in napari")
    napari.run()

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

    os.makedirs(output_dir, exist_ok=True)
    metadata = {'voxel_size_um': {'z': vox[0], 'y': vox[1], 'x': vox[2]}}  # Use full resolution voxel size
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    csv_path = os.path.join(output_dir, 'results.csv')
    
    # For debug mode, collect all beads for visualization
    if DEBUG:
        all_beads = []
        valid_bead_indices = []
    
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES)
        writer.writeheader()

        passed_beads = 0
        total_beads = 0
        
        for idx, pk in enumerate(tqdm(peaks, desc='  Extracting beads')):
            
            bead_raw, refined_pk = get_windows.extract_bead_adaptive(
                img=img_raw,
                peak=tuple(pk),
                crop_shape=CROP_SHAPE,
                normalize=NORMALIZE,
                threshold_rel_conn=0.2
            )
            
            if DEBUG:
                all_beads.append(bead_raw)
            
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
                    if DEBUG:
                        print(f'  Peak {idx} failed QC: {failure_reason}')
                    continue
                
                passed_beads += 1
            else:
                qc_passed = True
                failure_reason = ""
            
            if DEBUG:
                if qc_passed:
                    valid_bead_indices.append(idx)

            bead_path = os.path.join(output_dir, f'bead_{idx:04d}.tiff')
            imwrite(bead_path, bead_raw.astype(np.float32))
            
            # Record results
            rec = {
                'bead_index': idx,
                'peak_z': int(refined_pk[0]),
                'peak_y': int(refined_pk[1]),
                'peak_x': int(refined_pk[2]),
                **m
            }
            
            # Add QC information if filtering is enabled
            if FILTER_BEADS:
                rec['qc_passed'] = qc_passed
                rec['qc_failure_reason'] = failure_reason
            
            writer.writerow(rec)
            csvfile.flush()

    print(f'  Done: {len(peaks)} peaks processed, results in {output_dir}')
    print(f'  QC Summary: {passed_beads}/{total_beads} beads passed filtering')
    toc = time.time()
    print(f'  Time taken: {toc - tic} seconds')
    
    # Debug visualization
    if DEBUG and all_beads:
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        print(f'  DEBUG: Found {len(valid_bead_indices)} valid beads out of {len(peaks)} peaks')
        
        # Show beads in batches of 20
        batch_size = 20
        for batch_start in range(0, len(all_beads), batch_size):
            batch_end = min(batch_start + batch_size, len(all_beads))
            batch_beads = all_beads[batch_start:batch_end]
            
            # Filter out None beads for visualization
            valid_batch_beads = [b for b in batch_beads if b is not None and b.size > 0]
            
            if valid_batch_beads:
                visualize_beads_in_napari(valid_batch_beads, image_name, batch_start, batch_size)
            else:
                print(f"  No valid beads in batch {batch_start//batch_size + 1}")

def main():
    for image_path in IMAGE_PATHS:
        base = os.path.splitext(os.path.basename(image_path))[0]
        out_dir = os.path.join(OUTPUT_ROOT, base)
        process_image(image_path, out_dir)
        
        # In debug mode, ask if user wants to continue to next image
        if DEBUG:
            response = input(f"\nContinue to next image? (y/n): ")
            if response.lower() != 'y':
                print("Stopping batch processing.")
                break

if __name__ == '__main__':
    main() 