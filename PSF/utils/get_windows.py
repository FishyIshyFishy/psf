import time
import numpy as np
from skimage.filters import gaussian
from skimage.feature import peak_local_max
from scipy.ndimage import gaussian_filter, center_of_mass
from skimage.measure import label
from skimage.segmentation import flood


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
    sm_ds = gaussian(img_ds, sigma=0.5)
    peaks_ds = peak_local_max(
        sm_ds, 
        threshold_rel=threshold_rel, 
        min_distance=min_distance,
        exclude_border=False
    )
    
    print(f'Found {len(peaks_ds)} peaks on downsampled image')

    dz, dy, dx = downsample_factors
    peaks_full = peaks_ds * np.array([dz, dy, dx])
    peaks_full = peaks_full.astype(int)

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

def extract_bead_adaptive(
    img: np.ndarray,
    peak: tuple[int,int,int],
    crop_shape: tuple[int,int,int] = (6,10,10),
    normalize: bool = True,
    threshold_rel_conn: float = 0.2
) -> tuple[np.ndarray, tuple[int,int,int]] | tuple[None, tuple[int,int,int]]:
    """
    1) Coarse crop around `peak`
    2) Refine center by center_of_mass
    3) Crop again around the refined center
    4) Flood‑fill from the refined centroid to keep only its connected component
    5) (Optional) Normalize
    Returns (crop, refined_peak). On failure, returns (None, original_peak).
    """
    zc, yc, xc = peak
    dz, dy, dx = crop_shape
    Z, Y, X = img.shape

    # 1) Coarse crop
    z0, z1 = max(zc-dz, 0), min(zc+dz+1, Z)
    y0, y1 = max(yc-dy, 0), min(yc+dy+1, Y)
    x0, x1 = max(xc-dx, 0), min(xc+dx+1, X)
    crop_coarse = img[z0:z1, y0:y1, x0:x1]
    if crop_coarse.size == 0 or crop_coarse.max() == 0:
        return None, peak

    # 2) Compute local COM in coarse crop
    com_z, com_y, com_x = center_of_mass(crop_coarse)
    if not np.isfinite(com_z+com_y+com_x):
        return None, peak

    # 3) Map COM → full-image coords and re-crop
    z_ref = int(round(z0 + com_z))
    y_ref = int(round(y0 + com_y))
    x_ref = int(round(x0 + com_x))

    z0r, z1r = max(z_ref-dz, 0), min(z_ref+dz+1, Z)
    y0r, y1r = max(y_ref-dy, 0), min(y_ref+dy+1, Y)
    x0r, x1r = max(x_ref-dx, 0), min(x_ref+dx+1, X)
    crop_ref = img[z0r:z1r, y0r:y1r, x0r:x1r]

    if crop_ref.size == 0 or crop_ref.max() == 0:
        final_crop = crop_coarse
        refined_peak = peak
        # relative origin for flood = (dz, dy, dx) in coarse crop
        origin = (int(round(com_z)), int(round(com_y)), int(round(com_x)))
    else:
        final_crop = crop_ref
        refined_peak = (z_ref, y_ref, x_ref)
        origin = (
            z_ref - z0r,
            y_ref - y0r,
            x_ref - x0r
        )

    # 4) Flood-fill connectivity mask
    seed_val = final_crop[origin]
    tol = float(seed_val) * threshold_rel_conn
    mask = flood(final_crop, origin, tolerance=tol, connectivity=2)

    # zero out everything outside the bead's CC
    final_crop = final_crop * mask
    if final_crop.max() == 0:
        return None, peak

    # 5) Normalize if requested
    if normalize:
        final_crop = final_crop / final_crop.max()

    return final_crop, refined_peak


def qc_hard_gates(bead_raw, m, vox,
                  min_snr=8.0,
                  max_bg_cv=0.6,
                  max_secondary_peak_ratio=0.65,
                  # Shape-based QC parameters
                  min_linearity=0.7,
                  max_sphericity=0.15,
                  max_planarity=0.3,
                  min_length_width_ratio=4.0,
                  min_width_um=0.3,
                  max_width_um=1.2,
                  min_thickness_um=0.3,
                  max_thickness_um=1.2,
                  max_tortuosity=1.15,
                  max_width_CV=0.35,
                  max_border_fraction=0.1,
                  max_branches=0
                 ):
    """
    Comprehensive QC with shape-based filtering for narrow+elongated PSFs.
    """
    # 1) SNR gate (needs raw intensities)
    snr = float(m.get('snr', 0.0))
    if not np.isfinite(snr) or snr < min_snr:
        return False, f"Low SNR: {snr:.1f} < {min_snr}"

    # 2) Shape-based QC gates (metrics already computed in m)
    
    # Linearity (elongation) test
    L = m.get('L', 0)
    if L < min_linearity:
        return False, f"Low linearity: {L:.2f} < {min_linearity}"
    
    # Sphericity test
    S = m.get('S', 1)
    if S > max_sphericity:
        return False, f"High sphericity: {S:.2f} > {max_sphericity}"
    
    # Planarity test (avoid sheet-like blobs)
    P = m.get('P', 0)
    if P > max_planarity:
        return False, f"High planarity: {P:.2f} > {max_planarity}"
    
    # Aspect ratio test
    length_width_ratio = m.get('length_width_ratio', 1)
    if length_width_ratio < min_length_width_ratio:
        return False, f"Low length/width ratio: {length_width_ratio:.1f} < {min_length_width_ratio}"
    
    # Width bounds
    width_um = m.get('width_um', 0)
    if width_um < min_width_um or width_um > max_width_um:
        return False, f"Width out of bounds: {width_um:.2f} µm not in [{min_width_um}, {max_width_um}]"
    
    # Thickness bounds
    thickness_um = m.get('thickness_um', 0)
    if thickness_um < min_thickness_um or thickness_um > max_thickness_um:
        return False, f"Thickness out of bounds: {thickness_um:.2f} µm not in [{min_thickness_um}, {max_thickness_um}]"
    
    # Tortuosity test
    tortuosity = m.get('tortuosity', 1)
    if tortuosity > max_tortuosity:
        return False, f"High tortuosity: {tortuosity:.2f} > {max_tortuosity}"
    
    # Cross-section uniformity
    width_CV = m.get('width_CV', 1)
    if width_CV > max_width_CV:
        return False, f"High width CV: {width_CV:.2f} > {max_width_CV}"
    
    # Border fraction test
    border_fraction = m.get('border_fraction', 1)
    if border_fraction > max_border_fraction:
        return False, f"High border fraction: {border_fraction:.2f} > {max_border_fraction}"
    
    # Skeleton topology test
    branches = m.get('branches', 0)
    if branches > max_branches:
        return False, f"Too many branches: {branches} > {max_branches}"
    
    # 3) Secondary peak dominance (already computed in shape metrics)
    secondary_peak_ratio = m.get('secondary_peak_ratio', 0)
    if secondary_peak_ratio > max_secondary_peak_ratio:
        return False, f"Multiple peaks: {secondary_peak_ratio:.2f} > {max_secondary_peak_ratio}"

    return True, ""


def robust_outlier_mask(df, cols, z=3.5):
    X = df[cols].values.astype(float)
    med = np.nanmedian(X, axis=0)
    mad = np.nanmedian(np.abs(X - med), axis=0) + 1e-12
    zscores = 0.6745 * (X - med) / mad
    keep = np.all(np.abs(zscores) <= z, axis=1)
    return keep


def apply_qc_filtering(bead_raw, m, vox, qc_params=None):
    if qc_params is None:
        qc_params = {}

    qc_passed, failure_reason = qc_hard_gates(
        bead_raw=bead_raw, 
        m=m, 
        vox=vox,
        min_snr=qc_params.get('min_snr', 8.0),
        max_bg_cv=qc_params.get('max_bg_cv', 0.6),
        max_secondary_peak_ratio=qc_params.get('max_secondary_peak_ratio', 0.65),
        # Shape-based parameters
        min_linearity=qc_params.get('min_linearity', 0.7),
        max_sphericity=qc_params.get('max_sphericity', 0.15),
        max_planarity=qc_params.get('max_planarity', 0.3),
        min_length_width_ratio=qc_params.get('min_length_width_ratio', 4.0),
        min_width_um=qc_params.get('min_width_um', 0.3),
        max_width_um=qc_params.get('max_width_um', 1.2),
        min_thickness_um=qc_params.get('min_thickness_um', 0.3),
        max_thickness_um=qc_params.get('max_thickness_um', 1.2),
        max_tortuosity=qc_params.get('max_tortuosity', 1.15),
        max_width_CV=qc_params.get('max_width_CV', 0.35),
        max_border_fraction=qc_params.get('max_border_fraction', 0.1),
        max_branches=qc_params.get('max_branches', 0)
    )
    
    if not qc_passed:
        return False, failure_reason
    
    return True, ""