import json
import numpy as np
import napari
from tifffile import imread
from pathlib import Path


DATA_DIR = Path(r"Z:\BioMIID_Nonsync\BioMIID_Users_Nonsync\singhi7_BioMIID_Nonsync\20250618_Fluosphere-small-PSF\split\xy1_windows")  # directory with TIFFs + metadata
META_PATH = r"Z:\BioMIID_Nonsync\BioMIID_Users_Nonsync\singhi7_BioMIID_Nonsync\20250618_Fluosphere-small-PSF\split\xy1.json"

with open(META_PATH, 'r') as f:
    metadata = json.load(f)

vox = metadata['voxel_size_um']
scale = (vox['z'], vox['y'], vox['x'])

viewer = napari.Viewer()

for i, bead in enumerate(metadata['beads']):
    tiff_path = DATA_DIR / bead['tiff_file']
    img = imread(str(tiff_path))

    layer_name = f"Bead {bead['bead_index']}"
    viewer.add_image(img, name=layer_name, scale=scale, colormap='magma', blending='additive')

    if i > 10: break

napari.run()
