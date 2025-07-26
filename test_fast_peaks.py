import numpy as np
import time
from PSF.utils import get_windows, load

def test_fast_peaks():
    img = np.zeros((100, 200, 200))

    peaks_true = [
        (25, 50, 50),
        (45, 100, 100),
        (65, 150, 150),
        (35, 75, 125),
        (55, 125, 75)
    ]
    
    for z, y, x in peaks_true:
        z_range = slice(max(0, z-10), min(img.shape[0], z+11))
        y_range = slice(max(0, y-10), min(img.shape[1], y+11))
        x_range = slice(max(0, x-10), min(img.shape[2], x+11))
        
        for dz in range(-10, 11):
            for dy in range(-10, 11):
                for dx in range(-10, 11):
                    z_idx, y_idx, x_idx = z + dz, y + dy, x + dx
                    if (0 <= z_idx < img.shape[0] and 
                        0 <= y_idx < img.shape[1] and 
                        0 <= x_idx < img.shape[2]):

                        dist = np.sqrt(dz**2 + dy**2 + dx**2)
                        img[z_idx, y_idx, x_idx] += 100 * np.exp(-dist**2 / 20)
    
    img += np.random.normal(0, 5, img.shape)
    
    downsample_factors = (2, 4, 4)  # (z, y, x)
    voxel_size = (0.2, 0.1, 0.1)  # (z, y, x) in µm

    img_ds = load.downsample_image(img, downsample_factors)
    vox_ds = tuple(v * d for v, d in zip(voxel_size, downsample_factors))
    
    start_time = time.time()
    peaks_fast = get_windows.find_peaks_fast(
        img_full=img,
        img_ds=img_ds,
        downsample_factors=downsample_factors,
        voxel_size_ds=vox_ds,
        threshold_rel=0.3,
        min_distance=3,
        exclude_border_vox=(10, 10, 10)
    )
    fast_time = time.time() - start_time
    
    start_time = time.time()
    peaks_original = get_windows.find_peaks_adv(
        img_ds,
        voxel_size=vox_ds,
        min_sep_um=1.0,
        k_sigma=6.0,
        smooth_sigma_um=(0.4, 0.2, 0.2),
        exclude_border_vox=(5, 5, 5)
    )
    original_time = time.time() - start_time
    

    peaks_original_full = peaks_original * np.array(downsample_factors)
    
    print(f"\nSpeedup: {original_time/fast_time:.1f}x faster")
    print(f"Peak count ratio: {len(peaks_fast)}/{len(peaks_original)} = {len(peaks_fast)/len(peaks_original):.2f}")
    
    # Test window extraction
    print("\nTesting window extraction...")
    if len(peaks_fast) > 0:
        bead = get_windows.extract_cuboid_bead_from_full_resolution(
            img_full=img,
            peak_full=peaks_fast[0],
            downsample_factors=downsample_factors,
            crop_shape=(10, 10, 10),
            normalize=True
        )
        if bead is not None:
            print(f"Successfully extracted bead window with shape: {bead.shape}")
            print(f"Bead max value: {bead.max():.2f}")
        else:
            print("Failed to extract bead window")
    
    return peaks_fast, peaks_original_full

if __name__ == "__main__":
    peaks_fast, peaks_original = test_fast_peaks() 