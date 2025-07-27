import time
import numpy as np
from skimage.filters import gaussian
from skimage.feature import peak_local_max
import numpy as np
from scipy.ndimage import gaussian_filter, center_of_mass
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
    normalize: bool = True
) -> tuple[np.ndarray, tuple[int,int,int]] | tuple[None, tuple[int,int,int]]:
    zc, yc, xc = peak
    dz, dy, dx = crop_shape
    Z, Y, X = img.shape

    # 1) Coarse crop
    z0, z1 = max(zc - dz, 0), min(zc + dz + 1, Z)
    y0, y1 = max(yc - dy, 0), min(yc + dy + 1, Y)
    x0, x1 = max(xc - dx, 0), min(xc + dx + 1, X)
    crop = img[z0:z1, y0:y1, x0:x1]
    if crop.size == 0 or crop.max() == 0:
        return None, peak

    # 2) Compute local COM (in crop‐coordinates)
    com_z, com_y, com_x = center_of_mass(crop)
    # If COM is nan (e.g. all zeros), bail out
    if np.isnan(com_z + com_y + com_x):
        return None, peak

    # 3) Map COM → full‐image coordinates
    z_ref = int(round(z0 + com_z))
    y_ref = int(round(y0 + com_y))
    x_ref = int(round(x0 + com_x))

    # 4) Refined crop around (z_ref, y_ref, x_ref)
    z0r, z1r = max(z_ref - dz, 0), min(z_ref + dz + 1, Z)
    y0r, y1r = max(y_ref - dy, 0), min(y_ref + dy + 1, Y)
    x0r, x1r = max(x_ref - dx, 0), min(x_ref + dx + 1, X)
    crop_ref = img[z0r:z1r, y0r:y1r, x0r:x1r]
    if crop_ref.size == 0 or crop_ref.max() == 0:
        # fallback to the coarse crop if refinement went out of bounds
        final_crop = crop
        refined_peak = peak
    else:
        final_crop = crop_ref
        refined_peak = (z_ref, y_ref, x_ref)

    # 5) Normalize if requested
    if normalize and final_crop.max() > 0:
        final_crop = final_crop / final_crop.max()

    return final_crop, refined_peak


def qc_hard_gates(bead_raw, m, vox,
                  min_snr=8.0,
                  max_bg_cv=0.6,
                  max_secondary_peak_ratio=0.65,
                 ):
    # 1) SNR gate (needs raw intensities)
    snr = float(m.get('snr', 0.0))
    if not np.isfinite(snr) or snr < min_snr:
        return False, f"Low SNR: {snr:.1f} < {min_snr}"

    # 2) Background homogeneity (outer shell stats)
    zc, yc, xc = [s//2 for s in bead_raw.shape]
    # define a central ellipsoid ~0.6 µm laterally, 1.5 µm axially (tune if needed)
    rad_um = (1.5, 0.6, 0.6)  # (z,y,x) in µm
    rz, ry, rx = [max(1, int(round(r/v))) for r, v in zip(rad_um, vox)]
    zz, yy, xx = np.ogrid[:bead_raw.shape[0], :bead_raw.shape[1], :bead_raw.shape[2]]
    core = ((zz - zc)**2 / (rz**2) + (yy - yc)**2 / (ry**2) + (xx - xc)**2 / (rx**2)) <= 1.0
    shell = ~core
    bg_vals = bead_raw[shell]
    if bg_vals.size > 100:
        bg_mean = np.mean(bg_vals)
        bg_std  = np.std(bg_vals)
        bg_cv   = bg_std / (bg_mean + 1e-9)
        if bg_cv > max_bg_cv:
            return False, f"High background CV: {bg_cv:.2f} > {max_bg_cv}"

    # 3) Single-peak dominance inside the crop (avoid "double beads" / strong side lobes)
    sm = bead_raw  # already reasonably smooth after optics; add Gaussian if needed
    coords = peak_local_max(sm, threshold_rel=0.3, min_distance=3, exclude_border=False)
    if coords.shape[0] >= 2:
        vals = sm[tuple(coords.T)]
        v1, v2 = np.sort(vals)[-2:] if len(vals) >= 2 else (vals.max(), 0.0)
        if v2 / (v1 + 1e-9) > max_secondary_peak_ratio:
            return False, f"Multiple peaks: {v2/v1:.2f} > {max_secondary_peak_ratio}"

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
    )
    
    if not qc_passed:
        return None, False, failure_reason
    
    return True, ""