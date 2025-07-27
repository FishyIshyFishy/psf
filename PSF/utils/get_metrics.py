import numpy as np
from scipy.ndimage import center_of_mass, map_coordinates
from scipy.stats import skew, kurtosis
from skimage.feature import peak_local_max
from skimage.morphology import skeletonize
from scipy.spatial.distance import cdist

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

def compute_volume_ratio(bead, low=0.1, high=0.5):
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

def compute_astig(f2, f3):
    """Astigmatism metric = |FWHM_x − FWHM_y|."""
    return abs(f2 - f3)

def compute_shape_metrics(bead_raw, vox):
    """
    Compute comprehensive shape metrics for QC.
    
    Args:
        bead_raw: Raw bead data (unnormalized)
        vox: Voxel size tuple (vz, vy, vx)
    
    Returns:
        Dictionary of shape metrics
    """
    # Create binary mask from non-zero regions
    mask = bead_raw > 0
    
    if mask.sum() < 10:  # Too small to analyze
        return None
    
    # Get coordinates of non-zero voxels
    coords = np.array(np.where(mask)).T
    intensities = bead_raw[mask]
    
    # 1) Moment/PCA-based elongation tests
    # Compute intensity-weighted covariance matrix
    centroid = np.average(coords, weights=intensities, axis=0)
    coords_centered = coords - centroid
    
    # Weighted covariance matrix
    cov_matrix = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            cov_matrix[i, j] = np.average(
                coords_centered[:, i] * coords_centered[:, j], 
                weights=intensities
            )
    
    # Eigenvalues (sorted in descending order)
    eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
    eigenvals = eigenvals[::-1]  # Sort descending
    eigenvecs = eigenvecs[:, ::-1]  # Sort descending
    
    # Shape metrics
    lambda1, lambda2, lambda3 = eigenvals
    
    # Linearity (elongation)
    L = (lambda1 - lambda2) / lambda1 if lambda1 > 0 else 0
    
    # Sphericity
    S = lambda3 / lambda1 if lambda1 > 0 else 1
    
    # Planarity
    P = (lambda2 - lambda3) / lambda1 if lambda1 > 0 else 0
    
    # 2) Skeleton topology
    try:
        skeleton = skeletonize(mask)
        skeleton_coords = np.array(np.where(skeleton)).T
        
        if len(skeleton_coords) > 1:
            # Find endpoints and branch points
            # Simple heuristic: count neighbors for each skeleton point
            distances = cdist(skeleton_coords, skeleton_coords)
            neighbor_counts = np.sum(distances <= 1.5, axis=1) - 1  # -1 to exclude self
            
            endpoints = np.sum(neighbor_counts == 1)
            branches = np.sum(neighbor_counts > 2)
            
            # Tortuosity: skeleton length / end-to-end distance
            if endpoints >= 2:
                # Find endpoints (points with only 1 neighbor)
                endpoint_indices = np.where(neighbor_counts == 1)[0]
                if len(endpoint_indices) >= 2:
                    # Use first two endpoints
                    end1, end2 = skeleton_coords[endpoint_indices[0]], skeleton_coords[endpoint_indices[1]]
                    end_to_end_dist = np.linalg.norm(end2 - end1)
                    skeleton_length = len(skeleton_coords)  # Approximate
                    tortuosity = skeleton_length / end_to_end_dist if end_to_end_dist > 0 else 1
                else:
                    tortuosity = 1
            else:
                tortuosity = 1
        else:
            endpoints = 0
            branches = 0
            tortuosity = 1
            
    except Exception:
        endpoints = 0
        branches = 0
        tortuosity = 1
    
    # 3) Cross-section uniformity along PC1 axis
    # Project coordinates onto PC1 axis
    pc1_proj = np.dot(coords_centered, eigenvecs[:, 0])
    
    # Bin along PC1 axis and compute width in each bin
    n_bins = min(10, len(pc1_proj) // 5)  # Adaptive binning
    if n_bins > 1:
        bins = np.linspace(pc1_proj.min(), pc1_proj.max(), n_bins + 1)
        widths = []
        
        for i in range(n_bins):
            mask_bin = (pc1_proj >= bins[i]) & (pc1_proj < bins[i + 1])
            if mask_bin.sum() > 0:
                # Compute width in this bin (projection onto PC2)
                coords_bin = coords_centered[mask_bin]
                pc2_proj = np.dot(coords_bin, eigenvecs[:, 1])
                width = np.std(pc2_proj) * 2  # 2*std as width measure
                widths.append(width)
        
        if widths:
            width_CV = np.std(widths) / np.mean(widths) if np.mean(widths) > 0 else 1
        else:
            width_CV = 1
    else:
        width_CV = 1
    
    # 4) Border fraction
    border_voxels = 0
    total_voxels = mask.sum()
    
    # Check if mask touches any border
    if (mask[0, :, :].any() or mask[-1, :, :].any() or 
        mask[:, 0, :].any() or mask[:, -1, :].any() or
        mask[:, :, 0].any() or mask[:, :, -1].any()):
        # Count border voxels
        border_mask = np.zeros_like(mask)
        border_mask[0, :, :] = mask[0, :, :]
        border_mask[-1, :, :] = mask[-1, :, :]
        border_mask[:, 0, :] = mask[:, 0, :]
        border_mask[:, -1, :] = mask[:, -1, :]
        border_mask[:, :, 0] = mask[:, :, 0]
        border_mask[:, :, -1] = mask[:, :, -1]
        border_voxels = border_mask.sum()
    
    border_fraction = border_voxels / total_voxels if total_voxels > 0 else 1
    
    # 5) Secondary peak dominance
    coords_peaks = peak_local_max(bead_raw, threshold_rel=0.3, min_distance=3, exclude_border=False)
    if coords_peaks.shape[0] >= 2:
        vals = bead_raw[tuple(coords_peaks.T)]
        v1, v2 = np.sort(vals)[-2:] if len(vals) >= 2 else (vals.max(), 0.0)
        secondary_peak_ratio = v2 / (v1 + 1e-9)
    else:
        secondary_peak_ratio = 0
    
    # 6) Physical dimensions (in µm)
    # Convert eigenvalues to physical dimensions
    length_um = np.sqrt(lambda1) * 2 * np.sqrt(2)  # 2*std approximation
    width_um = np.sqrt(lambda2) * 2 * np.sqrt(2)
    thickness_um = np.sqrt(lambda3) * 2 * np.sqrt(2)
    
    # Aspect ratios
    length_width_ratio = length_um / width_um if width_um > 0 else 1
    width_thickness_ratio = width_um / thickness_um if thickness_um > 0 else 1
    
    return {
        'L': L,  # Linearity
        'S': S,  # Sphericity
        'P': P,  # Planarity
        'endpoints': endpoints,
        'branches': branches,
        'tortuosity': tortuosity,
        'width_CV': width_CV,
        'border_fraction': border_fraction,
        'secondary_peak_ratio': secondary_peak_ratio,
        'length_um': length_um,
        'width_um': width_um,
        'thickness_um': thickness_um,
        'length_width_ratio': length_width_ratio,
        'width_thickness_ratio': width_thickness_ratio
    }

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
    vol_ratio_05_01       = compute_volume_ratio(bead, low=0.1, high=0.5)
    pc1_z_angle_deg       = compute_pc1_z_angle(pca1)
    astig_um              = compute_astig(f2, f3)

    # --- shape metrics ---
    shape_metrics = compute_shape_metrics(bead, vox_ds)

    result = {
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
        'vol_ratio_05_01':  vol_ratio_05_01,
        'pc1_z_angle_deg':  pc1_z_angle_deg,
        'astig_um':         astig_um,
    }
    
    # Add shape metrics if available
    if shape_metrics is not None:
        result.update(shape_metrics)
    
    return result
