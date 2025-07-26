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

