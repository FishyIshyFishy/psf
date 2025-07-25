import numpy as np
from scipy.ndimage import center_of_mass, map_coordinates

def centroid_um(bead, vox_ds):
    cz, cy, cx = center_of_mass(bead)
    return cz * vox_ds[0], cy * vox_ds[1], cx * vox_ds[2]

def fwhm_z_um(bead, vox_ds):
    pz = bead.sum(axis=(1,2))
    return estimate_fwhm_1d(pz, vox_ds[0])

def fwhm_y_um(bead, vox_ds):
    py = bead.sum(axis=(0,2))
    return estimate_fwhm_1d(py, vox_ds[1])

def fwhm_x_um(bead, vox_ds):
    px = bead.sum(axis=(0,1))
    return estimate_fwhm_1d(px, vox_ds[2])

def pca_axes(bead, vox_ds):
    thresh = np.percentile(bead[bead > 0], 95)
    mask = bead >= thresh
    coords = np.column_stack(np.nonzero(mask))
    weights = bead[mask]
    coords_phys = coords * np.array(vox_ds)
    cov = np.cov(coords_phys.T, aweights=weights)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    axes = vecs[:,order]
    return axes[:,0].tolist(), axes[:,1].tolist(), axes[:,2].tolist()

def estimate_fwhm_1d(profile, step, label=None):
    baseline = np.min(profile)
    prof = profile - baseline
    prof = np.clip(prof, 0, None)
    half = prof.max() / 2
    idx = np.where(prof >= half)[0]
    fwhm = (idx[-1] - idx[0]) * step if idx.size >= 2 else np.nan
    return fwhm

def sample_profile(bead, center_phys, axis, vox_ds, half_length=5.0, num_pts=200, label=None):
    ax = np.array(axis)/np.linalg.norm(axis)
    dists = np.linspace(-half_length, half_length, num_pts)
    line_phys = center_phys + np.outer(dists, ax)
    line_vox = (line_phys/vox_ds).T
    prof = map_coordinates(bead, line_vox, order=1, mode='nearest')
    step = dists[1]-dists[0]
    return estimate_fwhm_1d(prof, step, label=label)

def fwhm_along_axis(bead, center_phys, axis, vox_ds, half_length=5.0, num_pts=200, label=None):
    return sample_profile(bead, center_phys, axis, vox_ds, half_length=half_length, num_pts=num_pts, label=label)

import numpy as np
from scipy.ndimage import center_of_mass, map_coordinates
from scipy.stats import skew, kurtosis  # for weighted moments, if you prefer scipy routines

def compute_tilt_angles(bead, vox_ds):
    """Fit 2D centroid drift per Z-slice → tilt angles in degrees."""
    zs = np.arange(bead.shape[0])
    com2d = np.array([center_of_mass(bead[z]) for z in zs])  # (Z, 2) = (y,z), (x,z)
    ay, _ = np.polyfit(zs, com2d[:,0], 1)
    ax, _ = np.polyfit(zs, com2d[:,1], 1)
    # Convert voxel‐per‐slice slope → physical slope → angle
    tilt_y = np.degrees(np.arctan((ay * vox_ds[0]) / vox_ds[1]))
    tilt_x = np.degrees(np.arctan((ax * vox_ds[0]) / vox_ds[2]))
    return tilt_y, tilt_x

def compute_weighted_moments(bead, vox_ds, axis):
    """Compute weighted skewness & excess kurtosis of the intensity distribution along `axis`."""
    # flatten coords & intensities
    coords = np.indices(bead.shape).reshape(3, -1).T * np.array(vox_ds)
    vals   = bead.flatten().astype(float)
    u      = coords @ np.array(axis)   # projection onto axis
    wsum   = vals.sum()
    mean_u = (wsum > 0) and (u @ vals) / wsum
    var_u  = (u - mean_u)**2 @ vals / wsum
    std_u  = np.sqrt(var_u)
    # weighted central moments
    m3 = ((u - mean_u)**3 @ vals) / wsum
    m4 = ((u - mean_u)**4 @ vals) / wsum
    skew_u = m3 / (std_u**3) if std_u else 0.0
    kurt_u = m4 / (std_u**4) - 3 if std_u else 0.0
    return skew_u, kurt_u

def compute_snr(bead, border=2):
    """Estimate global SNR using a border‐region background."""
    bz, by, bx = bead.shape
    # border mask
    bg_mask = np.zeros_like(bead, bool)
    bg_mask[:border,:,:] = True; bg_mask[-border:,:,:] = True
    bg_mask[:,:, :border] = True; bg_mask[:,:, -border:] = True
    bg_vals = bead[bg_mask]
    mu, sigma = bg_vals.mean(), bg_vals.std()
    return (bead.max() - mu) / sigma if sigma > 0 else np.nan

def compute_volume_ratio(bead, low=0.1, high=0.5):
    """Volume(≥high·Imax) / Volume(≥low·Imax)."""
    imax = bead.max()
    if imax == 0:
        return np.nan
    v_low  = np.count_nonzero(bead >= low * imax)
    v_high = np.count_nonzero(bead >= high * imax)
    return (v_high / v_low) if v_low else np.nan

def compute_pc1_z_angle(axis1):
    """Angle (deg) between first PCA axis and the optical z‑axis."""
    z_axis = np.array([1,0,0])  # if your indexing is (z,y,x)
    cosθ    = abs(np.dot(axis1, z_axis)) / (np.linalg.norm(axis1) or 1)
    return np.degrees(np.arccos(np.clip(cosθ, -1, 1)))

def compute_astig(fx, fy):
    """Astigmatism metric = |FWHM_x − FWHM_y|."""
    return abs(fx - fy)

def compute_psf_metrics(bead, vox_ds):
    # --- basic metrics (unchanged) ---
    cz, cy, cx = centroid_um(bead, vox_ds)
    fz, fy, fx = fwhm_z_um(bead, vox_ds), fwhm_y_um(bead, vox_ds), fwhm_x_um(bead, vox_ds)
    pca1, pca2, pca3 = pca_axes(bead, vox_ds)
    center_phys = np.array(center_of_mass(bead)) * vox_ds
    f1 = fwhm_along_axis(bead, center_phys, pca1, vox_ds)
    f2 = fwhm_along_axis(bead, center_phys, pca2, vox_ds)
    f3 = fwhm_along_axis(bead, center_phys, pca3, vox_ds)

    # --- new metrics ---
    tilt_y_deg, tilt_x_deg = compute_tilt_angles(bead, vox_ds)
    skew1, kurt1          = compute_weighted_moments(bead, vox_ds, pca1)
    snr_val               = compute_snr(bead)
    vol_ratio_05_01       = compute_volume_ratio(bead, low=0.1, high=0.5)
    pc1_z_angle_deg       = compute_pc1_z_angle(pca1)
    astig_um              = compute_astig(fx, fy)

    return {
        # existing outputs
        'centroid_z_um': cz,    'centroid_y_um': cy,    'centroid_x_um': cx,
        'fwhm_z_um':     fz,    'fwhm_y_um':     fy,    'fwhm_x_um':     fx,
        'fwhm_pca1_um':  f1,    'fwhm_pca2_um':  f2,    'fwhm_pca3_um':  f3,
        'pca_axis1':     pca1,  'pca_axis2':     pca2,  'pca_axis3':     pca3,
        # new outputs
        'tilt_angle_y_deg': tilt_y_deg,
        'tilt_angle_x_deg': tilt_x_deg,
        'skew_pc1':         skew1,
        'kurt_pc1':         kurt1,
        'snr':              snr_val,
        'vol_ratio_05_01':  vol_ratio_05_01,
        'pc1_z_angle_deg':  pc1_z_angle_deg,
        'astig_um':         astig_um,
    }
