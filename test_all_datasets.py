#!/usr/bin/env python3
"""
Test script to run peak finding on all specified datasets.
This helps verify that the fast peak finding method works consistently across all datasets.
"""

import csv
import numpy as np
import time
import os
from pathlib import Path

from PSF.utils import get_metrics
from PSF.utils import get_windows
from PSF.utils import load

# Dataset paths
DATASETS = [
    r"\\BioMIID_Central\BioMIID_Central\BioMIID_Nonsync\BioMIID_Users_Nonsync\singhi7_BioMIID_Nonsync\20250320_Controlled-misalignment-HD-MDJ\psf-tilt-angle-sweep\Psf-mp-mo3-4900-offset-0000-slit-2250-z-258617_XY001_T001__Channel_GFP-MPSOPi.nd2",
    r"\\BioMIID_Central\BioMIID_Central\BioMIID_Nonsync\BioMIID_Users_Nonsync\singhi7_BioMIID_Nonsync\20250320_Controlled-misalignment-HD-MDJ\psf-tilt-angle-sweep\Psf-mp-mo3-4900-offset-0000-slit-2250-z-258617_XY002_T001__Channel_GFP-MPSOPi.nd2",
    r"\\BioMIID_Central\BioMIID_Central\BioMIID_Nonsync\BioMIID_Users_Nonsync\singhi7_BioMIID_Nonsync\20250320_Controlled-misalignment-HD-MDJ\psf-tilt-angle-sweep\Psf-mp-mo3-4900-offset-0000-slit-2250-z-258617_XY003_T001__Channel_GFP-MPSOPi.nd2",
    r"\\BioMIID_Central\BioMIID_Central\BioMIID_Nonsync\BioMIID_Users_Nonsync\singhi7_BioMIID_Nonsync\20250320_Controlled-misalignment-HD-MDJ\psf-tilt-angle-sweep\Psf-mp-mo3-4900-offset-0000-slit-2250-z-258617_XY004_T001__Channel_GFP-MPSOPi.nd2",
    r"\\BioMIID_Central\BioMIID_Central\BioMIID_Nonsync\BioMIID_Users_Nonsync\singhi7_BioMIID_Nonsync\20250320_Controlled-misalignment-HD-MDJ\psf-tilt-angle-sweep\Psf-mp-mo3-4900-offset-0500-slit-2250-z-258617_XY001_T001__Channel_GFP-MPSOPi.nd2",
    r"\\BioMIID_Central\BioMIID_Central\BioMIID_Nonsync\BioMIID_Users_Nonsync\singhi7_BioMIID_Nonsync\20250320_Controlled-misalignment-HD-MDJ\psf-tilt-angle-sweep\Psf-mp-mo3-4900-offset-0500-slit-2250-z-258617_XY002_T001__Channel_GFP-MPSOPi.nd2",
    r"\\BioMIID_Central\BioMIID_Central\BioMIID_Nonsync\BioMIID_Users_Nonsync\singhi7_BioMIID_Nonsync\20250320_Controlled-misalignment-HD-MDJ\psf-tilt-angle-sweep\Psf-mp-mo3-4900-offset-0500-slit-2250-z-258617_XY003_T001__Channel_GFP-MPSOPi.nd2",
    r"\\BioMIID_Central\BioMIID_Central\BioMIID_Nonsync\BioMIID_Users_Nonsync\singhi7_BioMIID_Nonsync\20250320_Controlled-misalignment-HD-MDJ\psf-tilt-angle-sweep\Psf-mp-mo3-4900-offset-0500-slit-2250-z-258617_XY004_T001__Channel_GFP-MPSOPi.nd2",
    r"\\BioMIID_Central\BioMIID_Central\BioMIID_Nonsync\BioMIID_Users_Nonsync\singhi7_BioMIID_Nonsync\20250320_Controlled-misalignment-HD-MDJ\psf-tilt-angle-sweep\Psf-mp-mo3-4900-offset-1000-slit-2250-z-258617_XY001_T001__Channel_GFP-MPSOPi.nd2",
    r"\\BioMIID_Central\BioMIID_Central\BioMIID_Nonsync\BioMIID_Users_Nonsync\singhi7_BioMIID_Nonsync\20250320_Controlled-misalignment-HD-MDJ\psf-tilt-angle-sweep\Psf-mp-mo3-4900-offset-1000-slit-2250-z-258617_XY002_T001__Channel_GFP-MPSOPi.nd2",
    r"\\BioMIID_Central\BioMIID_Central\BioMIID_Nonsync\BioMIID_Users_Nonsync\singhi7_BioMIID_Nonsync\20250320_Controlled-misalignment-HD-MDJ\psf-tilt-angle-sweep\Psf-mp-mo3-4900-offset-1000-slit-2250-z-258617_XY003_T001__Channel_GFP-MPSOPi.nd2",
    r"\\BioMIID_Central\BioMIID_Central\BioMIID_Nonsync\BioMIID_Users_Nonsync\singhi7_BioMIID_Nonsync\20250320_Controlled-misalignment-HD-MDJ\psf-tilt-angle-sweep\Psf-mp-mo3-4900-offset-1000-slit-2250-z-258617_XY004_T001__Channel_GFP-MPSOPi.nd2",
    r"\\BioMIID_Central\BioMIID_Central\BioMIID_Nonsync\BioMIID_Users_Nonsync\singhi7_BioMIID_Nonsync\20250320_Controlled-misalignment-HD-MDJ\psf-tilt-angle-sweep\Psf-mp-mo3-4900-offset-neg0060-slit-2250-z-258617_XY001_T001__Channel_GFP-MPSOPi.nd2",
    r"\\BioMIID_Central\BioMIID_Central\BioMIID_Nonsync\BioMIID_Users_Nonsync\singhi7_BioMIID_Nonsync\20250320_Controlled-misalignment-HD-MDJ\psf-tilt-angle-sweep\Psf-mp-mo3-4900-offset-neg0060-slit-2250-z-258617_XY002_T001__Channel_GFP-MPSOPi.nd2",
    r"\\BioMIID_Central\BioMIID_Central\BioMIID_Nonsync\BioMIID_Users_Nonsync\singhi7_BioMIID_Nonsync\20250320_Controlled-misalignment-HD-MDJ\psf-tilt-angle-sweep\Psf-mp-mo3-4900-offset-neg0060-slit-2250-z-258617_XY003_T001__Channel_GFP-MPSOPi.nd2",
    r"\\BioMIID_Central\BioMIID_Central\BioMIID_Nonsync\BioMIID_Users_Nonsync\singhi7_BioMIID_Nonsync\20250320_Controlled-misalignment-HD-MDJ\psf-tilt-angle-sweep\Psf-mp-mo3-4900-offset-neg0060-slit-2250-z-258617_XY004_T001__Channel_GFP-MPSOPi.nd2"
]

# Parameters
DOWNSAMPLE = (2, 2, 2)  # (z, y, x)
THRESH_REL = 0.2
MIN_DISTANCE = 3
CROP_SHAPE = (80, 20, 20)  # (z, y, x) half-sizes
NORMALIZE = True

# Output file
OUTPUT_CSV = 'test_all_datasets_results.csv'

def extract_dataset_info(filepath):
    """Extract offset and XY position from filename."""
    filename = Path(filepath).name
    parts = filename.split('_')
    
    # Extract offset
    offset_part = [p for p in parts if 'offset' in p][0]
    offset = offset_part.replace('offset-', '').replace('neg', '-')
    
    # Extract XY position
    xy_part = [p for p in parts if p.startswith('XY')][0]
    xy_pos = xy_part.replace('XY', '')
    
    return offset, xy_pos

def process_dataset(filepath, dataset_idx):
    """Process a single dataset and return summary statistics."""
    print(f"\n{'='*80}")
    print(f"Processing dataset {dataset_idx + 1}/{len(DATASETS)}")
    print(f"File: {Path(filepath).name}")
    
    try:
        # Check if file exists
        if not os.path.exists(filepath):
            print(f"ERROR: File not found: {filepath}")
            return None
        
        # Extract dataset info
        offset, xy_pos = extract_dataset_info(filepath)
        print(f"Offset: {offset}, XY Position: {xy_pos}")
        
        # Load image
        print(f'Loading image...')
        start_time = time.time()
        img_raw, vox = load.load_image(filepath)
        load_time = time.time() - start_time
        print(f'Image loaded in {load_time:.2f}s. Shape: {img_raw.shape}, Voxel size: {vox}')

        # Downsample image
        print(f'Downsampling image...')
        start_time = time.time()
        img_ds = load.downsample_image(img_raw, DOWNSAMPLE)
        vox_ds = tuple(v * d for v, d in zip(vox, DOWNSAMPLE))
        downsample_time = time.time() - start_time
        print(f'Downsampled in {downsample_time:.2f}s. Shape: {img_ds.shape}')

        # Find peaks
        print(f'Finding peaks using fast method...')
        start_time = time.time()
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
        peak_finding_time = time.time() - start_time
        print(f'Found {len(peaks)} peaks in {peak_finding_time:.2f}s')

        # Process peaks
        print(f'Processing peaks...')
        start_time = time.time()
        valid_beads = 0
        total_processing_time = 0
        
        for idx, pk in enumerate(peaks):
            bead_start = time.time()
            bead, new_pk = get_windows.extract_bead_adaptive(
                img=img_raw,
                peak=tuple(pk),
                crop_shape=CROP_SHAPE,
                normalize=NORMALIZE
            )
            
            if bead is not None and np.count_nonzero(bead) >= 10:
                valid_beads += 1
                total_processing_time += time.time() - bead_start
                
                if valid_beads <= 3:  # Only print first few for debugging
                    print(f'  Processed peak {idx}: valid bead #{valid_beads}')
        
        processing_time = time.time() - start_time
        avg_bead_time = total_processing_time / valid_beads if valid_beads > 0 else 0
        
        print(f'Processing complete: {valid_beads} valid beads in {processing_time:.2f}s')
        print(f'Average time per bead: {avg_bead_time:.3f}s')
        
        # Return summary
        return {
            'dataset_idx': dataset_idx,
            'filename': Path(filepath).name,
            'offset': offset,
            'xy_pos': xy_pos,
            'image_shape': img_raw.shape,
            'voxel_size': vox,
            'downsampled_shape': img_ds.shape,
            'total_peaks_found': len(peaks),
            'valid_beads': valid_beads,
            'load_time': load_time,
            'downsample_time': downsample_time,
            'peak_finding_time': peak_finding_time,
            'processing_time': processing_time,
            'avg_bead_time': avg_bead_time,
            'total_time': load_time + downsample_time + peak_finding_time + processing_time
        }
        
    except Exception as e:
        print(f"ERROR processing {filepath}: {str(e)}")
        return None

def main():
    """Process all datasets and generate summary report."""
    print("Starting batch processing of all datasets...")
    print(f"Total datasets: {len(DATASETS)}")
    print(f"Parameters: DOWNSAMPLE={DOWNSAMPLE}, THRESH_REL={THRESH_REL}, MIN_DISTANCE={MIN_DISTANCE}")
    
    # Process all datasets
    results = []
    start_time = time.time()
    
    for idx, filepath in enumerate(DATASETS):
        result = process_dataset(filepath, idx)
        if result is not None:
            results.append(result)
    
    total_time = time.time() - start_time
    
    # Generate summary report
    print(f"\n{'='*80}")
    print("SUMMARY REPORT")
    print(f"{'='*80}")
    
    if not results:
        print("No datasets were processed successfully!")
        return
    
    # Calculate statistics
    total_peaks = sum(r['total_peaks_found'] for r in results)
    total_beads = sum(r['valid_beads'] for r in results)
    avg_peaks_per_dataset = total_peaks / len(results)
    avg_beads_per_dataset = total_beads / len(results)
    
    print(f"Successfully processed: {len(results)}/{len(DATASETS)} datasets")
    print(f"Total peaks found: {total_peaks}")
    print(f"Total valid beads: {total_beads}")
    print(f"Average peaks per dataset: {avg_peaks_per_dataset:.1f}")
    print(f"Average beads per dataset: {avg_beads_per_dataset:.1f}")
    print(f"Total processing time: {total_time:.2f}s")
    print(f"Average time per dataset: {total_time/len(results):.2f}s")
    
    # Group by offset
    print(f"\nResults by offset:")
    offsets = {}
    for r in results:
        offset = r['offset']
        if offset not in offsets:
            offsets[offset] = []
        offsets[offset].append(r)
    
    for offset in sorted(offsets.keys()):
        offset_results = offsets[offset]
        avg_beads = sum(r['valid_beads'] for r in offset_results) / len(offset_results)
        print(f"  Offset {offset}: {len(offset_results)} datasets, avg {avg_beads:.1f} beads/dataset")
    
    # Save detailed results to CSV
    print(f"\nSaving detailed results to {OUTPUT_CSV}...")
    fieldnames = [
        'dataset_idx', 'filename', 'offset', 'xy_pos', 'image_shape', 'voxel_size',
        'downsampled_shape', 'total_peaks_found', 'valid_beads',
        'load_time', 'downsample_time', 'peak_finding_time', 'processing_time',
        'avg_bead_time', 'total_time'
    ]
    
    with open(OUTPUT_CSV, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result)
    
    print("Processing complete!")

if __name__ == '__main__':
    main() 