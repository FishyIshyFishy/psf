import numpy as np
import json
from tifffile import imread
from pathlib import Path
from skimage.feature import peak_local_max
from scipy.ndimage import center_of_mass

# ─── Configuration ────────────────────────────────────────────────────────────
BEAD_DIR = Path(r"Z:\BioMIID_Nonsync\BioMIID_Users_Nonsync\singhi7_BioMIID_Nonsync\20250618_Fluosphere-small-PSF\split\xy1_windows")
META_PATH = r"Z:\BioMIID_Nonsync\BioMIID_Users_Nonsync\singhi7_BioMIID_Nonsync\20250618_Fluosphere-small-PSF\split\xy1.json"

CENTRAL_PEAK_THRESHOLD = 0.7
ELONGATION_RATIO_THRESHOLD = 30
MAX_PEAKS_ALLOWED = 1
MIN_NONZERO_PIXELS = 20

# ─── Load Voxel Size ──────────────────────────────────────────────────────────
with open(META_PATH, 'r') as f:
    metadata = json.load(f)

voxel_um = metadata['voxel_size_um']
scale = np.array([voxel_um['z'], voxel_um['y'], voxel_um['x']])

# ─── QC Function ──────────────────────────────────────────────────────────────
def is_bad_bead(bead, eigenvalues):
    reasons = []

    # 1. Elongation check
    elongation = eigenvalues[0] / eigenvalues[1]
    if elongation > ELONGATION_RATIO_THRESHOLD:
        reasons.append(f"elongated (PC1/PC2 = {elongation:.1f})")

    # 2. Central peak intensity check
    max_val = bead.max()
    central_coord = tuple(np.round(np.array(bead.shape) / 2).astype(int))
    central_val = bead[central_coord]
    if central_val / max_val < CENTRAL_PEAK_THRESHOLD:
        reasons.append(f"off-center peak (center={central_val:.3f}, max={max_val:.3f})")

    # 3. Multi-peak check
    peaks = peak_local_max(bead, min_distance=3, threshold_rel=1)
    if len(peaks) > MAX_PEAKS_ALLOWED:
        reasons.append(f"{len(peaks)} peaks")

    # 4. Sparsity
    if np.count_nonzero(bead) < MIN_NONZERO_PIXELS:
        reasons.append("too few nonzero voxels")

    return reasons

# ─── Processing ───────────────────────────────────────────────────────────────
bead_paths = sorted(BEAD_DIR.glob("bead_*.tif"))
print(f"Found {len(bead_paths)} bead files.\n")

for path in bead_paths:
    try:
        bead = imread(str(path))
        if bead.ndim != 3 or bead.sum() == 0:
            print(f"{path.name}: INVALID SHAPE OR EMPTY")
            continue

        coords = np.column_stack(np.nonzero(bead))
        weights = bead[bead > 0]
        if len(weights) < MIN_NONZERO_PIXELS:
            print(f"{path.name}: SKIPPED (low voxel count)")
            continue

        coords_phys = coords * scale
        cov = np.cov(coords_phys.T, aweights=weights)
        evals = np.linalg.eigvalsh(cov)[::-1]  # sort descending

        reasons = is_bad_bead(bead, evals)
        status = "BAD" if reasons else "GOOD"
        comment = " | ".join(reasons)

        print(f"{path.name} → PCA: {evals.round(2)} → {status} {comment if comment else ''}")

    except Exception as e:
        print(f"{path.name}: ERROR → {e}")
