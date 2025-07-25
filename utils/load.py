import nd2
import numpy as np
from skimage.transform import downscale_local_mean

def load_image(path):
    with nd2.ND2File(path) as f:
        img = f.asarray()
        while img.ndim > 3:
            img = img[0]
        vx, vy, vz = f.voxel_size()  # x, y, z in um
    return img, (vz, vy, vx)

def downsample_image(img, factors):
    pads = [(0, (f - img.shape[i] % f) % f) for i, f in enumerate(factors)]
    img_p = np.pad(img, pads, mode='edge')
    return downscale_local_mean(img_p, factors).astype(img.dtype)
