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
DEBUG_MODE = True  # Set to True to enable debug visualizations

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
            m = get_metrics.compute_psf_metrics(bead_raw, vox, debug_mode=DEBUG_MODE)
            
            # Debug visualization if enabled
            if DEBUG_MODE:
                print(f'Debug: Peak {idx} metrics:')
                print(f'  FWHM (Z,Y,X): {m["fwhm_z_um"]:.2f}, {m["fwhm_y_um"]:.2f}, {m["fwhm_x_um"]:.2f} µm')
                print(f'  PCA FWHM: {m["fwhm_pca1_um"]:.2f}, {m["fwhm_pca2_um"]:.2f}, {m["fwhm_pca3_um"]:.2f} µm')
                print(f'  Tilt angles: Y={m["tilt_angle_y_deg"]:.1f}°, X={m["tilt_angle_x_deg"]:.1f}°')
                print(f'  SNR: {m["snr"]:.1f}')
                print(f'  Astigmatism: {m["astig_um"]:.2f} µm')
                
                # Create debug visualization in Napari
                if idx == 0:  # Only show first bead for debug
                    viewer = napari.Viewer()
                    
                    # Add the bead volume
                    viewer.add_image(bead_raw, name=f'Bead {idx}', rendering='mip', colormap='gray')
                    
                    # Add PCA arrows if available
                    if 'pca_axis1' in m and 'pca_axis2' in m and 'pca_axis3' in m:
                        # Get centroid
                        centroid = np.array(bead_raw.shape) / 2
                        
                        # Create arrow vectors for PCA axes
                        arrow_length = min(bead_raw.shape) * 0.4
                        
                        # Convert PCA axes from lists to numpy arrays
                        pca1 = np.array(m['pca_axis1'])
                        pca2 = np.array(m['pca_axis2'])
                        pca3 = np.array(m['pca_axis3'])
                        
                        # Normalize the PCA vectors
                        pca1 = pca1 / np.linalg.norm(pca1)
                        pca2 = pca2 / np.linalg.norm(pca2)
                        pca3 = pca3 / np.linalg.norm(pca3)
                        
                        # Create vectors for arrows (origin, direction)
                        # PCA axis 1 (primary direction) - Red
                        pca1_end = centroid + pca1 * arrow_length
                        pca1_vector = np.array([centroid, pca1_end])
                        viewer.add_vectors(pca1_vector, name='PCA Axis 1', 
                                        edge_color='red', length=0)
                        
                        # PCA axis 2 (secondary direction) - Green  
                        pca2_end = centroid + pca2 * arrow_length
                        pca2_vector = np.array([centroid, pca2_end])
                        viewer.add_vectors(pca2_vector, name='PCA Axis 2', 
                                        edge_color='green', length=0)
                        
                        # PCA axis 3 (tertiary direction) - Blue
                        pca3_end = centroid + pca3 * arrow_length
                        pca3_vector = np.array([centroid, pca3_end])
                        viewer.add_vectors(pca3_vector, name='PCA Axis 3', 
                                        edge_color='blue', length=0)
                        
                        print(f'  PCA arrows added: lengths = {np.linalg.norm(pca1):.2f}, {np.linalg.norm(pca2):.2f}, {np.linalg.norm(pca3):.2f}')
     
                    napari.run()
            
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
    
    # Debug summary plot if enabled
    if DEBUG_MODE:
        print("\n=== DEBUG SUMMARY ===")
        print(f"Processed {total_beads} beads total")
        print(f"Debug mode enabled - check console output for detailed metrics")
        print("First bead visualization opened in Napari with PCA arrows")
        
        # Create a simple summary plot showing metric distributions
        if total_beads > 0:
            # Read the CSV to get metrics for plotting
            df = pd.read_csv(OUTPUT_CSV)
            
            fig, axs = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle('PSF Metrics Summary (Debug Mode)', fontsize=16)
            
            # FWHM metrics
            axs[0, 0].hist([df['fwhm_z_um'], df['fwhm_y_um'], df['fwhm_x_um']], 
                           label=['Z', 'Y', 'X'], alpha=0.7, bins=20)
            axs[0, 0].set_title('FWHM Distribution')
            axs[0, 0].set_xlabel('FWHM (µm)')
            axs[0, 0].legend()
            
            # PCA FWHM metrics
            axs[0, 1].hist([df['fwhm_pca1_um'], df['fwhm_pca2_um'], df['fwhm_pca3_um']], 
                           label=['PCA1', 'PCA2', 'PCA3'], alpha=0.7, bins=20)
            axs[0, 1].set_title('PCA FWHM Distribution')
            axs[0, 1].set_xlabel('FWHM (µm)')
            axs[0, 1].legend()
            
            # Tilt angles
            axs[0, 2].scatter(df['tilt_angle_y_deg'], df['tilt_angle_x_deg'], alpha=0.6)
            axs[0, 2].set_title('Tilt Angles')
            axs[0, 2].set_xlabel('Y Tilt (deg)')
            axs[0, 2].set_ylabel('X Tilt (deg)')
            
            # SNR distribution
            axs[1, 0].hist(df['snr'], bins=20, alpha=0.7)
            axs[1, 0].set_title('SNR Distribution')
            axs[1, 0].set_xlabel('SNR')
            
            # Astigmatism
            axs[1, 1].hist(df['astig_um'], bins=20, alpha=0.7)
            axs[1, 1].set_title('Astigmatism Distribution')
            axs[1, 1].set_xlabel('Astigmatism (µm)')
            
            # PC1-Z angle
            axs[1, 2].hist(df['pc1_z_angle_deg'], bins=20, alpha=0.7)
            axs[1, 2].set_title('PC1-Z Angle Distribution')
            axs[1, 2].set_xlabel('Angle (deg)')
            
            plt.tight_layout()
            plt.show()
            
            print("Summary plot displayed showing metric distributions across all beads")


if __name__ == '__main__':
    main()
