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

def compute_psf_metrics(bead, vox_ds):
    centroid = centroid_um(bead, vox_ds)
    fz = fwhm_z_um(bead, vox_ds)
    fy = fwhm_y_um(bead, vox_ds)
    fx = fwhm_x_um(bead, vox_ds)
    pca1, pca2, pca3 = pca_axes(bead, vox_ds)
    center_phys = np.array(center_of_mass(bead)) * vox_ds
    fwhm_pca1 = fwhm_along_axis(bead, center_phys, pca1, vox_ds, label='PCA1')
    fwhm_pca2 = fwhm_along_axis(bead, center_phys, pca2, vox_ds, label='PCA2')
    fwhm_pca3 = fwhm_along_axis(bead, center_phys, pca3, vox_ds, label='PCA3')
    return {
        'centroid_z_um': centroid[0],
        'centroid_y_um': centroid[1],
        'centroid_x_um': centroid[2],
        'fwhm_z_um': fz,
        'fwhm_y_um': fy,
        'fwhm_x_um': fx,
        'fwhm_pca1_um': fwhm_pca1,
        'fwhm_pca2_um': fwhm_pca2,
        'fwhm_pca3_um': fwhm_pca3,
        'pca_axis1': pca1,
        'pca_axis2': pca2,
        'pca_axis3': pca3,
    }
