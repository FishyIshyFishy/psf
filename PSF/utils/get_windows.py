import time
import numpy as np
from skimage.filters import gaussian
from skimage.feature import peak_local_max
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.feature import peak_local_max
from skimage.measure import label

def find_peaks(img, threshold_rel=0.2, min_distance=3):
    sm = gaussian(img, sigma=1)
    peaks = peak_local_max(sm, threshold_rel=threshold_rel, min_distance=min_distance)
    return peaks


def find_peaks_fast(
    img_full,
    img_ds,
    downsample_factors,
    voxel_size_ds,
    min_sep_um=1.0,
    threshold_rel=0.3,
    min_distance=2,
    exclude_border_vox=(6, 10, 10)
):
    """
    Two-stage peak finding: find rough peaks on downsampled image, then scale back to full resolution.
    
    Args:
        img_full: Full resolution image
        img_ds: Downsampled image
        downsample_factors: (z, y, x) downsampling factors
        voxel_size_ds: Voxel size of downsampled image
        min_sep_um: Minimum separation between peaks in microns
        threshold_rel: Relative threshold for peak detection
        min_distance: Minimum distance between peaks in downsampled voxels
        exclude_border_vox: Border exclusion in full resolution voxels
    
    Returns:
        List of peak coordinates in full resolution image
    """
    print(f'Finding peaks on downsampled image...')
    
    # Simple smoothing on downsampled image (much faster)
    sm_ds = gaussian(img_ds, sigma=0.5)
    
    # Find peaks on downsampled image
    peaks_ds = peak_local_max(
        sm_ds, 
        threshold_rel=threshold_rel, 
        min_distance=min_distance,
        exclude_border=False
    )
    
    print(f'Found {len(peaks_ds)} peaks on downsampled image')
    
    # Scale peaks back to full resolution
    dz, dy, dx = downsample_factors
    peaks_full = peaks_ds * np.array([dz, dy, dx])
    peaks_full = peaks_full.astype(int)
    
    # Border guard: remove peaks too close to borders for your crop
    zpad, ypad, xpad = exclude_border_vox
    Z, Y, X = img_full.shape
    keep = (
        (peaks_full[:, 0] >= zpad) & (peaks_full[:, 0] < Z - zpad) &
        (peaks_full[:, 1] >= ypad) & (peaks_full[:, 1] < Y - ypad) &
        (peaks_full[:, 2] >= xpad) & (peaks_full[:, 2] < X - xpad)
    )
    peaks_full = peaks_full[keep]
    
    print(f'Kept {len(peaks_full)} peaks after border filtering')
    
    return peaks_full


def find_peaks_adv(
    img,
    voxel_size=(0.2, 0.1, 0.1),        # (z, y, x) in µm
    min_sep_um=1.0,                    # enforce physical separation
    k_sigma=6.0,                       # abs threshold = k * noise_sigma
    smooth_sigma_um=(0.4, 0.2, 0.2),   # anisotropic smoothing in µm
    exclude_border_vox=(6, 10, 10),
):
    vz, vy, vx = voxel_size

    # 1) Background subtraction (very large Gaussian)
    tic = time.time()
    bkg = gaussian_filter(img, sigma=(10/vz, 20/vy, 20/vx))
    work = img - bkg
    work[work < 0] = 0
    print(f'background subtracting done in {time.time() - tic} s')

    # 2) Smooth with anisotropic Gaussian set in µm
    tic = time.time()
    sigma_vox = (smooth_sigma_um[0]/vz, smooth_sigma_um[1]/vy, smooth_sigma_um[2]/vx)
    sm = gaussian_filter(work, sigma=sigma_vox)
    print(f'smoothing done in {time.time() - tic} s')

    # 3) Estimate noise using MAD on the lowest quartile
    tic = time.time()
    bg_vals = sm[sm < np.percentile(sm, 25)]
    sigma_n = 1.4826 * np.median(np.abs(bg_vals - np.median(bg_vals))) if bg_vals.size else 1.0
    thr_abs = k_sigma * sigma_n

    # 5) Label a hard mask with abs threshold; one peak per CC
    tic = time.time()
    hard_mask = sm > thr_abs
    labels = label(hard_mask, connectivity=1)
    print(f'mask labeling done in {time.time() - tic} s')

    # 6) Enforce min separation (converted to voxels)
    # peak_local_max can take labels to get 1 per component, and min_distance
    tic = time.time()
    min_sep_vox = int(round(min_sep_um / min(vz, vy, vx)))  # conservative choice
    coords = peak_local_max(
        sm,
        labels=labels,
        min_distance=min_sep_vox,
        threshold_abs=thr_abs,
        exclude_border=False,  # we'll do custom border pruning
    )
    print(f'peal local max done in {time.time() - tic} s')

    # 7) Border guard: remove peaks too close to borders for your crop
    zpad, ypad, xpad = exclude_border_vox
    Z, Y, X = img.shape
    keep = (
        (coords[:, 0] >= zpad) & (coords[:, 0] < Z - zpad) &
        (coords[:, 1] >= ypad) & (coords[:, 1] < Y - ypad) &
        (coords[:, 2] >= xpad) & (coords[:, 2] < X - xpad)
    )
    coords = coords[keep]

    return coords


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


def extract_cuboid_bead_from_full_resolution(img_full, peak_full, downsample_factors, crop_shape=(6,10,10), normalize=True):
    """
    Extract bead window from full resolution image using peak location.
    
    Args:
        img_full: Full resolution image
        peak_full: Peak coordinates in full resolution image
        downsample_factors: (z, y, x) downsampling factors (for reference)
        crop_shape: Crop shape in full resolution voxels
        normalize: Whether to normalize the extracted window
    
    Returns:
        Extracted bead window or None if invalid
    """
    # Extract window from full resolution image
    return extract_cuboid_bead(img_full, peak_full, crop_shape, normalize)

