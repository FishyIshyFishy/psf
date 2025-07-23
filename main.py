import csv
import numpy as np
import matplotlib.pyplot as plt
import napari
import nd2
import time

from scipy.ndimage import map_coordinates, center_of_mass
from skimage.filters import gaussian
from skimage.feature import peak_local_max
from skimage.segmentation import flood
from skimage.measure import label, regionprops
from skimage.transform import downscale_local_mean

# configuration
CSV_PATH   = r'C:\Users\singhi7\Documents\psf\results.csv' # path to your metrics CSV
ND2_PATH   = r"Z:\BioMIID_Nonsync\BioMIID_Users_Nonsync\singhi7_BioMIID_Nonsync\20250618_Fluosphere-small-PSF\split\multipoint_psf_xy1.nd2"
DOWNSAMPLE     = (1,1,1)     # z, y, x factors
THRESH_REL     = 0.2
FLOOD_REL      = 0.20
CROP_SHAPE = (80, 20, 20)  # z, y, x half-sizes → resulting crop will be (13×21×21)
MIN_DISTANCE   = 3
PADDING_PX     = 2
DEBUG          = False         # toggle all debug plotting

def load_image(path):
    with nd2.ND2File(path) as f:
        img = f.asarray()
        while img.ndim > 3:
            img = img[0]
        vx, vy, vz = f.voxel_size()  # x, y, z in um
        print(f'voxel sizes: {vx}, {vy}, {vz}')
    return img, (vz, vy, vx)



def downsample_image(img, factors):
    pads = [(0, (f - img.shape[i] % f) % f) for i, f in enumerate(factors)]
    img_p = np.pad(img, pads, mode='edge')
    return downscale_local_mean(img_p, factors).astype(img.dtype)

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

def extract_adaptive_bead(img, peak, flood_rel, padding):
    zc, yc, xc = peak
    dz, dy, dx = 3, 5, 5

    z0, z1 = zc-dz, zc+dz+1
    y0, y1 = yc-dy, yc+dy+1
    x0, x1 = xc-dx, xc+dx+1
    if z0<0 or y0<0 or x0<0 or z1>img.shape[0] or y1>img.shape[1] or x1>img.shape[2]:
        return None
    
    local = img[z0:z1, y0:y1, x0:x1]
    if local.max() == 0:
        return None
    
    com_z, com_y, com_x = center_of_mass(local)
    com = (int(round(z0+com_z)), int(round(y0+com_y)), int(round(x0+com_x)))

    seed = img[com]
    mask = flood(img, seed_point=com, tolerance=seed*(1 - flood_rel))
    if not mask[com]:
        return None
    
    lbl = label(mask)
    reg = regionprops(lbl)[lbl[com]-1]

    z0, y0, x0, z1, y1, x1 = (*reg.bbox[:3], *reg.bbox[3:])
    z0, y0, x0 = max(z0-padding,0), max(y0-padding,0), max(x0-padding,0)
    z1, y1, x1 = min(z1+padding,img.shape[0]), min(y1+padding,img.shape[1]), min(x1+padding,img.shape[2])
    crop = img[z0:z1, y0:y1, x0:x1]

    return crop/crop.max() if crop.size and crop.max()>0 else None



def estimate_fwhm_1d(profile, step, label=None):
    # 1) estimate and subtract baseline
    #    you can choose np.min(profile), or e.g. the 5th percentile
    baseline = np.min(profile)
    # baseline = np.percentile(profile, 5)
    
    prof = profile - baseline
    prof = np.clip(prof, 0, None)
    
    # 2) find half‑maximum on the corrected curve
    half = prof.max() / 2
    idx = np.where(prof >= half)[0]
    fwhm = (idx[-1] - idx[0]) * step if idx.size >= 2 else np.nan

    # 3) (optional) debug plot
    if DEBUG and label:
        x = np.arange(len(prof)) * step
        plt.figure()
        plt.plot(x, prof, '-o', label='baseline‑subtracted')
        plt.axhline(half, color='r', linestyle='--', label='½‑max')
        if idx.size >= 2:
            plt.axvline(x[idx[0]], color='g', linestyle=':')
            plt.axvline(x[idx[-1]], color='g', linestyle=':')
        plt.title(f'{label}')
        plt.xlabel('Distance (um)')
        plt.ylabel('Intensity')
        plt.legend()
        plt.show()

    return fwhm



def sample_profile(bead, center_phys, axis, vox_ds, half_length=5.0, num_pts=200, label=None):
    ax = np.array(axis)/np.linalg.norm(axis)
    dists = np.linspace(-half_length, half_length, num_pts)
    line_phys = center_phys + np.outer(dists, ax)
    line_vox = (line_phys/vox_ds).T
    prof = map_coordinates(bead, line_vox, order=1, mode='nearest')
    step = dists[1]-dists[0]
    return estimate_fwhm_1d(prof, step, label=label)



def compute_psf_metrics(bead, vox_ds):
    cz, cy, cx = center_of_mass(bead)
    centroid = (cz*vox_ds[0], cy*vox_ds[1], cx*vox_ds[2])

    pz = bead.sum(axis=(1,2))
    py = bead.sum(axis=(0,2))
    px = bead.sum(axis=(0,1))
    fz = estimate_fwhm_1d(pz, vox_ds[0], label='Z')
    fy = estimate_fwhm_1d(py, vox_ds[1], label='Y')
    fx = estimate_fwhm_1d(px, vox_ds[2], label='X')

    thresh = np.percentile(bead[bead > 0], 95)
    mask = bead >= thresh
    coords = np.column_stack(np.nonzero(mask))
    weights = bead[mask]

    coords_phys = coords * np.array(vox_ds)
    cov = np.cov(coords_phys.T, aweights=weights)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    axes = vecs[:,order]

    return {
        'centroid_z_um':   centroid[0],
        'centroid_y_um':   centroid[1],
        'centroid_x_um':   centroid[2],
        'fwhm_z_um':       fz,
        'fwhm_y_um':       fy,
        'fwhm_x_um':       fx,
        'pca_axis1':       axes[:,0].tolist(),
        'pca_axis2':       axes[:,1].tolist(),
        'pca_axis3':       axes[:,2].tolist(),
    }



def get_vec_dirs(center_vox, axes, scale=5.0):
    center = np.array(center_vox)
    vectors = []
    for i in range(3):
        direction = np.array(axes[:, i]) * scale
        vectors.append([center, direction])
    return np.array(vectors)  # shape: (3, 2, 3)




def plot_pca_projections(bead, vox_ds, pca_axes, center_phys):
    """Plot orthogonal projections of the bead intensity along PCA planes with arrows."""
    import matplotlib.pyplot as plt

    if not DEBUG:
        return

    # Normalize axes (assumed to be in physical units)
    pca_axes = np.array(pca_axes).T
    pca_axes = pca_axes / np.linalg.norm(pca_axes, axis=0)

    # Create coordinate grid in physical space
    z, y, x = np.indices(bead.shape)
    coords_vox = np.stack((z, y, x), axis=-1).reshape(-1, 3)
    coords_phys = coords_vox * np.array(vox_ds)

    intensities = bead.flatten()

    # Shift to center-of-mass coordinate frame
    coords_phys_centered = coords_phys - center_phys

    # Project into PCA space
    proj = coords_phys_centered @ pca_axes  # shape (N, 3)

    # Reshape projection back to 3D shape
    for i, (i1, i2) in enumerate([(0, 1), (0, 2), (1, 2)]):
        # 2D histogram projection
        H, xedges, yedges = np.histogram2d(
            proj[:, i1], proj[:, i2], bins=100, weights=intensities, density=False
        )

        X, Y = np.meshgrid(
            0.5 * (xedges[:-1] + xedges[1:]),
            0.5 * (yedges[:-1] + yedges[1:])
        )

        plt.figure(figsize=(6, 5))
        plt.pcolormesh(X, Y, H.T, shading='auto', cmap='magma')
        plt.colorbar(label='Projected Intensity')
        plt.quiver(0, 0, 1, 0, color='red', scale=5, label='PC%d' % (i1 + 1))
        plt.quiver(0, 0, 0, 1, color='green', scale=5, label='PC%d' % (i2 + 1))
        plt.title(f'Projection onto PC{i1+1} vs PC{i2+1}')
        plt.xlabel(f'PC{i1+1} (µm)')
        plt.ylabel(f'PC{i2+1} (µm)')
        plt.legend()
        plt.axis('equal')
        plt.tight_layout()
        plt.show()

def mip_image_perp_pc1(bead, vox_ds, pc1_axis, center_phys,
                       plane_half_width=5.0,  # µm from center along each in‑plane axis
                       plane_pixels=201,      # resolution of the output image
                       line_half_length=5.0,  # µm along PC1 to search max over
                       line_samples=101       # samples along each line
                       ):
    """
    Compute a 2D MIP image of `bead` onto the plane orthogonal to PC1.

    Returns:
      MIP    : 2D numpy array shape (plane_pixels, plane_pixels)
      U, V   : 1D arrays of length plane_pixels giving physical coords (µm)
      spacing: tuple (dU, dV) for display scaling
    """
    # 1) build orthonormal basis {dir1, dir2, dir3}
    dir1 = np.array(pc1_axis, float)
    dir1 /= np.linalg.norm(dir1)
    # pick arbitrary non‑colinear vector for Gram‑Schmidt
    tmp  = np.array([1,0,0]) if abs(dir1[0])<0.9 else np.array([0,1,0])
    dir2 = np.cross(dir1, tmp); dir2 /= np.linalg.norm(dir2)
    dir3 = np.cross(dir1, dir2)

    # 2) prepare the 2D grid in the plane
    U = np.linspace(-plane_half_width, plane_half_width, plane_pixels)
    V = np.linspace(-plane_half_width, plane_half_width, plane_pixels)
    dU = U[1] - U[0]; dV = V[1] - V[0]

    # 3) sample points along PC1 for max‐intensity
    T = np.linspace(-line_half_length, line_half_length, line_samples)

    # 4) allocate output
    MIP = np.zeros((plane_pixels, plane_pixels), dtype=bead.dtype)

    # 5) loop over each (u,v) and take max along PC1
    for i, u in enumerate(U):
        for j, v in enumerate(V):
            # build all sample points for this line
            pts_phys = (
                center_phys[None,:]
                + u * dir2[None,:]
                + v * dir3[None,:]
                + T[:,None] * dir1[None,:]
            )
            # convert phys→vox
            pts_vox = pts_phys / np.array(vox_ds)[None,:]
            coords  = np.stack([
                pts_vox[:,k] for k in range(3)
            ], axis=0)
            vals    = map_coordinates(bead, coords, order=1, mode='nearest')
            MIP[j,i] = vals.max()

    return MIP, U, V, (dU, dV)



def main():
    print(f'loading image')
    img_raw, vox = load_image(ND2_PATH)

    print(f'dowmsampling image')
    img_ds = downsample_image(img_raw, DOWNSAMPLE)
    vox_ds = tuple(v*d for v,d in zip(vox, DOWNSAMPLE))
    sm = gaussian(img_ds, sigma=1)

    print(f'finding peaks')
    peaks = peak_local_max(sm, threshold_rel=THRESH_REL, min_distance=MIN_DISTANCE)

    fieldnames = [
        'bead_index','peak_z','peak_y','peak_x',
        'total_intensity','max_intensity',
        'centroid_z_um','centroid_y_um','centroid_x_um',
        'fwhm_z_um','fwhm_y_um','fwhm_x_um',
        'fwhm_pca1_um','fwhm_pca2_um','fwhm_pca3_um',
        'pca_axis1','pca_axis2','pca_axis3'
    ]
    with open(CSV_PATH,'w',newline='') as csvfile:
        writer = csv.DictWriter(csvfile,fieldnames=fieldnames)
        writer.writeheader()

        print(f'entering peak loop')
        for idx, pk in enumerate(peaks):
            tic = time.time()
            # bead = extract_adaptive_bead(img_ds, tuple(pk), flood_rel=FLOOD_REL, padding=PADDING_PX)
            bead = extract_cuboid_bead(img_ds, tuple(pk), crop_shape=CROP_SHAPE)

            if bead is None or bead.sum()==0 or np.count_nonzero(bead)<10:
                continue

            

            m = compute_psf_metrics(bead, vox_ds)
            center_phys = np.array(center_of_mass(bead))*vox_ds

            for i, ax in enumerate([m['pca_axis1'],m['pca_axis2'],m['pca_axis3']],1):
                m[f'fwhm_pca{i}_um'] = sample_profile(bead,center_phys,ax,vox_ds,label=f'PCA{i}')

            if DEBUG:
                mip2d, U, V, (dU, dV) = mip_image_perp_pc1(
                    bead, vox_ds, m['pca_axis1'], center_phys,
                    plane_half_width=5.0,
                    plane_pixels=201,
                    line_half_length=5.0,
                    line_samples=101
                )

                # Display with Matplotlib
                plt.figure(figsize=(5,5))
                plt.imshow(
                    mip2d,
                    extent=[U[0], U[-1], V[0], V[-1]],
                    origin='lower',
                    cmap='magma',
                    aspect='equal'
                )
                plt.colorbar(label='Max intensity')
                plt.xlabel('Dir2 (µm)'); plt.ylabel('Dir3 (µm)')
                plt.title(f'Bead {idx}: MIP ⟂ PC1')
                plt.tight_layout()
                plt.show()


                v = napari.Viewer()
                v.add_image(bead, scale=vox_ds, blending='additive', name='Bead')

                # compute PCA vectors in voxel coordinates (origin is bead CoM)
                cz, cy, cx = center_of_mass(bead)
                center_vox = np.array([cz, cy, cx])
                # axes = np.column_stack((m['pca_axis1'], m['pca_axis2'], m['pca_axis3']))
                # vectors = get_vec_dirs(center_vox, axes, scale=5.0)

                axes_vox = np.column_stack((m['pca_axis1'], m['pca_axis2'], m['pca_axis3'])) / np.array(vox_ds)[:, None]
                axes_vox = axes_vox / np.linalg.norm(axes_vox, axis=0)  # normalize each axis
                vectors = get_vec_dirs(center_vox, axes_vox, scale=5.0)
                v.add_vectors(vectors, name='PCA Vectors', edge_color=['red', 'green', 'blue'], edge_width=2, scale=vox_ds)

                v.add_vectors(vectors, name='PCA Vectors', edge_color=['red', 'green', 'blue'], edge_width=2, scale=vox_ds)
                napari.run()

                plot_pca_projections(bead, vox_ds, [m['pca_axis1'], m['pca_axis2'], m['pca_axis3']], center_phys)

            rec = {
                'bead_index':idx,
                'peak_z':int(pk[0]),
                'peak_y':int(pk[1]),
                'peak_x':int(pk[2]),
                **m
            }
            writer.writerow(rec)
            csvfile.flush()
            toc = time.time()
            print(f'done with peak{idx} in time {toc-tic:.2f}')

if __name__=='__main__':
    main()
