import numpy as np
import pandas as pd
import ast
import napari
import nd2
from skimage.segmentation import flood
from skimage.measure import label, regionprops
from scipy.ndimage import center_of_mass

# ─── Configuration ───────────────────────────────────────────────────────────
CSV_PATH   = r"Z:\BioMIID_Nonsync\BioMIID_Users_Nonsync\singhi7_BioMIID_Nonsync\20250320_Controlled-misalignment-HD-MDJ\psf-tilt-angle-sweep\results.csv" # path to your metrics CSV
ND2_PATH   = r"Z:\BioMIID_Nonsync\BioMIID_Users_Nonsync\singhi7_BioMIID_Nonsync\20250320_Controlled-misalignment-HD-MDJ\psf-tilt-angle-sweep\Psf-mp-mo3-4900-offset-0000-slit-2250-z-258617_XY001_T001__Channel_GFP-MPSOPi.nd2"
DOWNSAMPLE = (1, 1, 1)
FLOOD_REL = 0.25
PADDING_PX = 2
MAX_PSF_COUNT = 30
VECTOR_SCALE_UM = 5

# ─── Load voxel size and image ───────────────────────────────────────────────
def get_voxel_size_ds(nd2_path, factors):
    with nd2.ND2File(nd2_path) as f:
        vx, vy, vz = f.voxel_size()
    return (vz * factors[0], vy * factors[1], vx * factors[2])

def load_volume(path):
    with nd2.ND2File(path) as f:
        arr = f.asarray()
        while arr.ndim > 3:
            arr = arr[0]
    return arr

# ─── Reuse exact PSF extraction logic ────────────────────────────────────────
def extract_adaptive_bead(img, peak, flood_rel=0.25, padding=2):
    zc, yc, xc = peak
    dz, dy, dx = 3, 5, 5
    z0, z1 = zc - dz, zc + dz + 1
    y0, y1 = yc - dy, yc + dy + 1
    x0, x1 = xc - dx, xc + dx + 1

    if z0 < 0 or y0 < 0 or x0 < 0 or z1 > img.shape[0] or y1 > img.shape[1] or x1 > img.shape[2]:
        return None

    local = img[z0:z1, y0:y1, x0:x1]
    if local.max() == 0:
        return None

    com_z, com_y, com_x = center_of_mass(local)
    com = (int(round(z0 + com_z)), int(round(y0 + com_y)), int(round(x0 + com_x)))
    if not all(0 <= c < dim for c, dim in zip(com, img.shape)):
        return None

    seed_val = img[com]
    mask = flood(img, seed_point=com, tolerance=seed_val * (1 - flood_rel))
    if not mask[com]:
        return None

    labeled = label(mask)
    region = regionprops(labeled)[labeled[com] - 1]
    z0, y0, x0, z1, y1, x1 = (*region.bbox[:3], *region.bbox[3:])
    z0, y0, x0 = max(z0 - padding, 0), max(y0 - padding, 0), max(x0 - padding, 0)
    z1, y1, x1 = min(z1 + padding, img.shape[0]), min(y1 + padding, img.shape[1]), min(x1 + padding, img.shape[2])

    crop = img[z0:z1, y0:y1, x0:x1]
    return crop / crop.max() if crop.size > 0 and crop.max() > 0 else None

# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    print("Loading metrics and volume...")
    df = pd.read_csv(CSV_PATH)
    df['pca_axis1'] = df['pca_axis1'].apply(ast.literal_eval)
    df = df.sort_values('total_intensity', ascending=False).head(MAX_PSF_COUNT).reset_index(drop=True)

    img = load_volume(ND2_PATH)
    voxel_size = get_voxel_size_ds(ND2_PATH, DOWNSAMPLE)

    # Downsample voxel size is Z, Y, X
    pca_vectors = []
    psf_volumes = []
    vector_origins = []

    for _, row in df.iterrows():
        peak = (int(row['peak_z']), int(row['peak_y']), int(row['peak_x']))
        bead = extract_adaptive_bead(img, peak, flood_rel=FLOOD_REL, padding=PADDING_PX)
        if bead is None:
            continue
        psf_volumes.append(bead)

        # Position in physical space
        origin_um = np.array(peak) * np.array(voxel_size)
        vec = np.array(row['pca_axis1'])
        vec_scaled = vec / np.linalg.norm(vec) * VECTOR_SCALE_UM

        # Napari expects [origin, vector] in (X, Y, Z)
        vector_origins.append([origin_um[::-1], vec_scaled[::-1]])

    print(f"Loaded {len(psf_volumes)} PSFs and vectors.")

    # ─── Napari Visualization ────────────────────────────────────────────────
    viewer = napari.Viewer()
    for i, bead in enumerate(psf_volumes):
        viewer.add_image(
            bead,
            name=f"bead_{i}",
            scale=voxel_size,
            colormap='gray',
            blending='additive'
        )

    if vector_origins:
        vectors = np.array(vector_origins)
        viewer.add_vectors(
            vectors,
            name="pca_axis1",
            edge_color='red',
            scale=voxel_size[::-1]
        )

    napari.run()

if __name__ == '__main__':
    main()
