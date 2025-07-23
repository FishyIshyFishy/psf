import os
import json
from pathlib import Path

import numpy as np
import nd2
from skimage.transform import downscale_local_mean
from skimage.filters import gaussian
from skimage.feature import peak_local_max
from scipy.ndimage import center_of_mass, map_coordinates
from tifffile import imwrite

ND2_PATH    = r"Z:\BioMIID_Nonsync\BioMIID_Users_Nonsync\singhi7_BioMIID_Nonsync\20250618_Fluosphere-small-PSF\split\multipoint_psf_xy1.nd2"
OUTPUT_DIR  = r"Z:\BioMIID_Nonsync\BioMIID_Users_Nonsync\singhi7_BioMIID_Nonsync\20250618_Fluosphere-small-PSF\split\xy1_windows"
METADATA_JSON = r"Z:\BioMIID_Nonsync\BioMIID_Users_Nonsync\singhi7_BioMIID_Nonsync\20250618_Fluosphere-small-PSF\split\xy1.json"
DOWNSAMPLE  = (1, 1, 1)
THRESH_REL   = 0.2
MIN_DISTANCE = 3
CROP_SHAPE   = (80, 20, 20)  # → window will be (2*dz+1,2*dy+1,2*dx+1)

# ─── Helper Functions ────────────────────────────────────────────────────────

def load_volume(path):
    with nd2.ND2File(path) as f:
        img = f.asarray()
        # If there are extra dimensions, drop them:
        while img.ndim > 3:
            img = img[0]
        vx, vy, vz = f.voxel_size()  # returns x, y, z
    # reorder to (z,y,x) and voxel tuple (z,y,x)
    return img, (vz, vy, vx)

def downsample(img, factors):
    pads = [(0, (f - img.shape[i] % f) % f) for i, f in enumerate(factors)]
    img_p = np.pad(img, pads, mode='edge')
    return downscale_local_mean(img_p, factors).astype(img.dtype)

def extract_cuboid(img, peak, half_sizes):
    zc, yc, xc = peak
    dz, dy, dx = half_sizes
    z0, z1 = max(zc-dz,0), min(zc+dz+1,img.shape[0])
    y0, y1 = max(yc-dy,0), min(yc+dy+1,img.shape[1])
    x0, x1 = max(xc-dx,0), min(xc+dx+1,img.shape[2])
    crop = img[z0:z1, y0:y1, x0:x1]
    if crop.size == 0 or crop.max() == 0:
        return None
    return crop


def main():
    # Prepare output
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    out_dir = Path(OUTPUT_DIR)

    # Load & downsample
    volume, vox = load_volume(ND2_PATH)
    volume_ds = downsample(volume, DOWNSAMPLE)
    vox_ds = tuple(v*ds for v,ds in zip(vox, DOWNSAMPLE))

    # Smooth & detect peaks
    sm = gaussian(volume_ds, sigma=1)
    peaks = peak_local_max(
        sm,
        threshold_rel=THRESH_REL,
        min_distance=MIN_DISTANCE
    )

    # Collect metadata
    metadata = {
        'config': {
            'nd2_path': str(ND2_PATH),
            'downsample': DOWNSAMPLE,
            'threshold_rel': THRESH_REL,
            'min_distance': MIN_DISTANCE,
            'crop_half_sizes': CROP_SHAPE
        },
        'voxel_size_um': {'z': vox_ds[0], 'y': vox_ds[1], 'x': vox_ds[2]},
        'beads': []
    }

    for idx, pk in enumerate(peaks):
        crop = extract_cuboid(volume_ds, tuple(pk), CROP_SHAPE)
        if crop is None or np.count_nonzero(crop) < 10:
            continue

        # Build record
        record = {
            'bead_index': idx,
            'peak_z': int(pk[0]),
            'peak_y': int(pk[1]),
            'peak_x': int(pk[2]),
        }
        # Save TIFF (uint16 or float as in original)
        fname = out_dir / f'bead_{idx:04d}.tif'
        imwrite(str(fname), crop.astype(np.float32))

        # Add filename to metadata
        record['tiff_file'] = fname.name
        metadata['beads'].append(record)

    # Write JSON
    with open(METADATA_JSON, 'w') as jfile:
        json.dump(metadata, jfile, indent=2)

    print(f'Extracted {len(metadata["beads"])} bead windows')
    print(f'TIFFs → {OUTPUT_DIR}')
    print(f'Metadata → {METADATA_JSON}')

if __name__ == '__main__':
    main()
