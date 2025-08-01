import numpy as np
import napari
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift, fftfreq
from scipy.ndimage import gaussian_filter
from sklearn.decomposition import PCA
import nd2

# Configuration - Update these paths and crop parameters
ND2_FILE_PATH = r"Z:\BioMIID_Nonsync\BioMIID_Users_Nonsync\singhi7_BioMIID_Nonsync\20250320_Controlled-misalignment-HD-MDJ\psf-tilt-angle-sweep\Psf-mp-mo3-4900-offset-0000-slit-2250-z-258617_XY001_T001__Channel_GFP-MPSOPi.nd2"


# Crop parameters - specify the region of interest in the ND2 file
# These are in voxel coordinates (z, y, x)
CROP_START = (50, 100, 100)  # Start coordinates (z, y, x)
CROP_SIZE = (800, 800,800)     # Size of the crop (z, y, x)

def load_nd2_crop(nd2_path, crop_start, crop_size):
    """Load a volume crop from ND2 file."""
    try:
        with nd2.ND2File(nd2_path) as f:
            # Get the full image
            full_image = f.asarray()
            print(f"Full image shape: {full_image.shape}")
            
            # Calculate crop end coordinates
            crop_end = (
                crop_start[0] + crop_size[0],
                crop_start[1] + crop_size[1], 
                crop_start[2] + crop_size[2]
            )
            
            # Ensure we don't go out of bounds
            actual_end = (
                min(crop_end[0], full_image.shape[0]),
                min(crop_end[1], full_image.shape[1]),
                min(crop_end[2], full_image.shape[2])
            )
            
            # Extract the crop
            crop = full_image[
                crop_start[0]:actual_end[0],
                crop_start[1]:actual_end[1],
                crop_start[2]:actual_end[2]
            ]
            
            print(f"Crop shape: {crop.shape}")
            print(f"Crop value range: [{crop.min():.3f}, {crop.max():.3f}]")
            
            return crop
            
    except Exception as e:
        print(f"Error loading ND2 file: {e}")
        return None

def get_voxel_size(nd2_path):
    """Get voxel size from ND2 file."""
    try:
        with nd2.ND2File(nd2_path) as f:
            # Get voxel size in microns
            voxel_size = f.voxel_size()
            print(f"Voxel size from ND2: {voxel_size} µm")
            return voxel_size
    except Exception as e:
        print(f"Warning: Could not read voxel size from ND2 file: {e}")
        print("Using default voxel size: (0.2, 0.1, 0.1) µm")
        return (0.2, 0.1, 0.1)  # Default: (z, y, x) in µm

def preprocess_volume(volume):
    """Preprocess volume similar to skew angle estimator."""
    volume = volume.astype(np.float32)
    volume -= np.percentile(volume, 5)
    volume = np.clip(volume, 0, None)
    volume = gaussian_filter(volume, sigma=(0.5, 2.0, 0.5))
    return volume

def apply_window(volume):
    """Apply Hanning window to reduce edge effects."""
    z, y, x = volume.shape
    win_z = np.hanning(z)
    win_y = np.hanning(y)
    win_x = np.hanning(x)
    window = np.outer(win_z, np.outer(win_y, win_x).reshape(y, x)).reshape(z, y, x)
    return volume * window

def compute_fft_magnitude(volume):
    """Compute FFT magnitude with preprocessing."""
    # Apply preprocessing
    volume = preprocess_volume(volume)
    volume = apply_window(volume)
    
    # Compute FFT
    fft_result = np.fft.fftn(volume)
    fft_shifted = np.fft.fftshift(fft_result)
    magnitude = np.abs(fft_shifted)
    
    # Log scale for better visualization
    log_magnitude = np.log1p(magnitude)
    
    return log_magnitude

def crop_fft_center(log_mag, crop_fraction=0.5):
    """Crop the center of FFT magnitude."""
    z, y, x = log_mag.shape
    cz, cy, cx = z // 2, y // 2, x // 2
    dz, dy, dx = int(z * crop_fraction / 2), int(y * crop_fraction / 2), int(x * crop_fraction / 2)
    return log_mag[cz - dz:cz + dz, cy - dy:cy + dy, cx - dx:cx + dx]

def make_adaptive_axis_mask(shape, plane='yz'):
    """Create adaptive axis mask to exclude vertical/horizontal streaks."""
    z, y, x = shape
    cz, cy, cx = z // 2, y // 2, x // 2

    if plane == 'yz':
        # 2D mask over (z, y) to exclude vertical + horizontal streaks
        zz, yy = np.meshgrid(np.arange(z), np.arange(y), indexing='ij')
        w_y = max(2, int(min(y, z) * 0.02))  # ~2% of Y for horizontal axis mask
        w_z = max(2, int(min(y, z) * 0.02))  # ~2% of Z for vertical axis mask
        mask_yz = (np.abs(yy - cy) < w_y) | (np.abs(zz - cz) < w_z)
        return mask_yz  # shape: (z, y)

    elif plane == 'xz':
        # 2D mask over (z, x) with fixed exclusion widths
        zz, xx = np.meshgrid(np.arange(z), np.arange(x), indexing='ij')
        w_x = 3   
        w_z = 1  
        mask_xz = (np.abs(xx - cx) < w_x) | (np.abs(zz - cz) < w_z)
        return mask_xz  # shape: (z, x)

    else:
        raise ValueError("Invalid plane. Use 'yz' or 'xz'.")

def estimate_skew_plane(coords, dim1, dim2, shape, voxel_size, threshold_percent=25):
    """Estimate skew angle using PCA on masked coordinates."""
    if len(coords) < 2:
        return np.array([0, 1]), 0.0, 0.0
    
    center = np.array([shape[dim2] // 2, shape[dim1] // 2])
    p1 = (coords[:, dim1] - center[1]) * voxel_size[dim1]
    p2 = (coords[:, dim2] - center[0]) * voxel_size[dim2]
    coords_plane = np.stack([p1, p2], axis=1)

    pca = PCA(n_components=2)
    pca.fit(coords_plane)
    vec = pca.components_[0]
    strength = pca.explained_variance_ratio_[0]

    if vec[1] < 0:
        vec = -vec

    # For plotting: we need to swap the components
    # vec[0] should be Z (vertical), vec[1] should be Y (horizontal)
    # So angle should be arctan2(dz, dy) where dz = vec[0], dy = vec[1]
    angle = np.arctan2(vec[0], vec[1]) * 180 / np.pi
    return vec, angle, strength

def place_label(cx, cz, dx, dz, angle, strength, xlim, zlim, scale=1.4):
    """Place label for angle and strength."""
    ortho = np.array([-dz, dx]) / np.sqrt(dx**2 + dz**2)
    lx = cx + dx * scale + ortho[0] * 15
    lz = cz + dz * scale + ortho[1] * 15
    lx = np.clip(lx, 10, xlim - 10)
    lz = np.clip(lz, 10, zlim - 10)
    return {'pos': (lx, lz), 'text': f"{angle:.1f}°\n({strength*100:.0f}%)"}

def analyze_volume_with_skew(volume, voxel_size=(1.0, 1.0, 1.0), threshold_percent=25):
    """Analyze volume using FFT and skew angle estimation."""
    # Full FFT magnitude
    log_mag = compute_fft_magnitude(volume)

    # Separate FFT center crops per projection
    log_mag_yz = crop_fft_center(log_mag, crop_fraction=0.5)
    log_mag_xz = crop_fft_center(log_mag, crop_fraction=0.5)

    # Create energy masks
    energy_mask_yz = log_mag_yz > np.percentile(log_mag_yz, 100 - threshold_percent)
    energy_mask_xz = log_mag_xz > np.percentile(log_mag_xz, 100 - threshold_percent)

    # Create axis masks
    axis_mask_yz = make_adaptive_axis_mask(log_mag_yz.shape, plane='yz')
    axis_mask_xz = make_adaptive_axis_mask(log_mag_xz.shape, plane='xz')

    # Combine masks
    final_mask_yz = energy_mask_yz & (~axis_mask_yz[:, :, None])
    final_mask_xz = energy_mask_xz & (~axis_mask_xz[:, None, :])

    # Get coordinates for PCA - both unfiltered and filtered
    coords_yz_unfiltered = np.argwhere(energy_mask_yz)
    coords_yz_filtered = np.argwhere(final_mask_yz)
    coords_xz = np.argwhere(final_mask_xz)

    # Estimate skew angles - both unfiltered and filtered for YZ
    vec_yz_unfiltered, angle_yz_unfiltered, strength_yz_unfiltered = estimate_skew_plane(
        coords_yz_unfiltered, dim1=1, dim2=0, shape=log_mag_yz.shape, voxel_size=voxel_size)
    vec_yz_filtered, angle_yz_filtered, strength_yz_filtered = estimate_skew_plane(
        coords_yz_filtered, dim1=1, dim2=0, shape=log_mag_yz.shape, voxel_size=voxel_size)
    vec_xz, angle_xz, strength_xz = estimate_skew_plane(
        coords_xz, dim1=2, dim2=0, shape=log_mag_xz.shape, voxel_size=voxel_size)

    return (log_mag_yz, log_mag_xz, final_mask_yz, final_mask_xz, 
            vec_yz_unfiltered, angle_yz_unfiltered, strength_yz_unfiltered,
            vec_yz_filtered, angle_yz_filtered, strength_yz_filtered,
            vec_xz, angle_xz, strength_xz)

def plot_skew_analysis(log_mag_yz, log_mag_xz, final_mask_yz, final_mask_xz,
                      vec_yz_unfiltered, angle_yz_unfiltered, strength_yz_unfiltered,
                      vec_yz_filtered, angle_yz_filtered, strength_yz_filtered,
                      vec_xz, angle_xz, strength_xz,
                      crop_start, crop_size):
    """Create comprehensive plot showing skew analysis."""
    yz_proj = log_mag_yz.max(axis=2)
    masked_yz = np.ma.masked_where(~final_mask_yz.max(axis=2), yz_proj)

    z_dim, y_dim = yz_proj.shape
    center_z, center_y = z_dim // 2, y_dim // 2
    base_scale = min(z_dim, y_dim) * 0.4

    # Normalize vectors and calculate arrow components
    vec_yz_unfiltered = vec_yz_unfiltered / np.linalg.norm(vec_yz_unfiltered)
    vec_yz_filtered = vec_yz_filtered / np.linalg.norm(vec_yz_filtered)
    
    # For plotting: vec[0] is Z (vertical), vec[1] is Y (horizontal)
    dz_yz_unfiltered, dy_unfiltered = vec_yz_unfiltered[0] * strength_yz_unfiltered * base_scale, vec_yz_unfiltered[1] * strength_yz_unfiltered * base_scale
    dz_yz_filtered, dy_filtered = vec_yz_filtered[0] * strength_yz_filtered * base_scale, vec_yz_filtered[1] * strength_yz_filtered * base_scale

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'FFT Skew Analysis - YZ Projection\nCrop: {crop_start} to {tuple(np.array(crop_start) + np.array(crop_size))}', fontsize=16)
    
    for ax in axs:
        ax.set_facecolor('black')

    # YZ projection with arrow (with all data) - using unfiltered PCA
    axs[0].imshow(yz_proj, cmap='inferno', origin='lower')
    axs[0].arrow(center_y, center_z, dy_unfiltered, dz_yz_unfiltered, color='white', width=1, head_width=5, linewidth=2)
    label_yz_unfiltered = place_label(center_y, center_z, dy_unfiltered, dz_yz_unfiltered, angle_yz_unfiltered, strength_yz_unfiltered, y_dim, z_dim)
    axs[0].text(*label_yz_unfiltered['pos'], label_yz_unfiltered['text'], color='white', fontsize=14, ha='center', va='center',
               bbox=dict(facecolor='black', alpha=0.3, boxstyle='round,pad=0.4'))
    axs[0].set_title("YZ Projection (All Data)", color='white')
    axs[0].tick_params(colors='white')

    # YZ with excluded data removed and arrow - using filtered PCA
    axs[1].imshow(masked_yz, cmap='inferno', origin='lower')
    axs[1].arrow(center_y, center_z, dy_filtered, dz_yz_filtered, color='white', width=1, head_width=5, linewidth=2)
    label_yz_filtered = place_label(center_y, center_z, dy_filtered, dz_yz_filtered, angle_yz_filtered, strength_yz_filtered, y_dim, z_dim)
    axs[1].text(*label_yz_filtered['pos'], label_yz_filtered['text'], color='white', fontsize=14, ha='center', va='center',
               bbox=dict(facecolor='black', alpha=0.3, boxstyle='round,pad=0.4'))
    axs[1].set_title("YZ Projection (Excluded Data Removed)", color='white')
    axs[1].tick_params(colors='white')

    fig.tight_layout()
    fig.patch.set_facecolor('black')
    plt.show()

def main():
    """Main function to run the volume crop analysis."""
    print("=== Volume Crop FFT Skew Analysis ===")
    
    # Get voxel size from ND2 file
    print("\n=== Getting Voxel Size ===")
    voxel_size = get_voxel_size(ND2_FILE_PATH)
    
    # Load the volume crop
    print(f"\n=== Loading Volume Crop ===")
    print(f"Crop start: {CROP_START}")
    print(f"Crop size: {CROP_SIZE}")
    
    volume = load_nd2_crop(ND2_FILE_PATH, CROP_START, CROP_SIZE)
    
    if volume is None:
        print("Failed to load volume crop")
        return
    
    # Analyze volume with skew estimation
    print("\n=== Analyzing FFT and Skew Angles ===")
    results = analyze_volume_with_skew(volume, voxel_size=voxel_size)
    (log_mag_yz, log_mag_xz, final_mask_yz, final_mask_xz, 
     vec_yz_unfiltered, angle_yz_unfiltered, strength_yz_unfiltered,
     vec_yz_filtered, angle_yz_filtered, strength_yz_filtered,
     vec_xz, angle_xz, strength_xz) = results
    
    # Plot results
    print("\n=== Plotting Skew Analysis ===")
    plot_skew_analysis(log_mag_yz, log_mag_xz, final_mask_yz, final_mask_xz,
                      vec_yz_unfiltered, angle_yz_unfiltered, strength_yz_unfiltered,
                      vec_yz_filtered, angle_yz_filtered, strength_yz_filtered,
                      vec_xz, angle_xz, strength_xz,
                      CROP_START, CROP_SIZE)
    
    # Print summary
    print("\n=== Summary ===")
    print(f"Volume crop shape: {volume.shape}")
    print(f"YZ skew angle (unfiltered): {angle_yz_unfiltered:.2f}° (strength: {strength_yz_unfiltered*100:.1f}%)")
    print(f"YZ skew angle (filtered): {angle_yz_filtered:.2f}° (strength: {strength_yz_filtered*100:.1f}%)")
    print(f"XZ skew angle: {angle_xz:.2f}° (strength: {strength_xz*100:.1f}%)")
    print(f"Voxel size: {voxel_size} µm")
    print(f"Crop start: {CROP_START}")
    print(f"Crop size: {CROP_SIZE}")

if __name__ == "__main__":
    main() 