#!/usr/bin/env python3
"""
deskew_demo.py

Demonstrates deskew detection and correction using FFT + PCA:
1) Create or load a 3D volume with known skew
2) Estimate the deskew angles in YZ and XZ planes via FFT + PCA
3) Apply shear transformation to undo the skew
4) Show original vs. deskewed results
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, affine_transform
from scipy.fft import fftn, fftshift
from sklearn.decomposition import PCA
import nd2

# Configuration - Update these paths and parameters
ND2_FILE_PATH = r"Z:\BioMIID_Nonsync\BioMIID_Users_Nonsync\singhi7_BioMIID_Nonsync\20250320_Controlled-misalignment-HD-MDJ\psf-tilt-angle-sweep\Psf-mp-mo3-4900-offset-0000-slit-2250-z-258617_XY001_T001__Channel_GFP-MPSOPi.nd2"

# Crop parameters - specify the region of interest in the ND2 file
CROP_START = (50, 100, 100)  # Start coordinates (z, y, x)
CROP_SIZE = (200, 200, 200)  # Size of the crop (z, y, x)

def create_synthetic_skewed_volume(size=128, skew_yz=15, skew_xz=10):
    """Create a synthetic 3D volume with known skew angles."""
    z, y, x = np.mgrid[:size, :size, :size]
    center = size // 2
    
    # Create a 3D ellipsoid
    a, b, c = size // 8, size // 6, size // 4  # semi-axes
    
    # Apply skew transformations
    # YZ skew: y' = y + z * tan(skew_yz)
    # XZ skew: x' = x + z * tan(skew_xz)
    y_skewed = y + z * np.tan(np.radians(skew_yz))
    x_skewed = x + z * np.tan(np.radians(skew_xz))
    
    # Create ellipsoid
    obj = ((x_skewed - center) / a)**2 + ((y_skewed - center) / b)**2 + ((z - center) / c)**2 < 1
    
    # Add some noise and smoothing
    obj = gaussian_filter(obj.astype(float), sigma=1)
    obj += np.random.normal(0, 0.1, obj.shape)
    obj = np.clip(obj, 0, None)
    
    return obj

def apply_hann_window(vol):
    """Apply a separable 3D Hann window to suppress edge artifacts."""
    z, y, x = vol.shape
    wz = np.hanning(z)[:, None, None]
    wy = np.hanning(y)[None, :, None]
    wx = np.hanning(x)[None, None, :]
    return vol * (wz * wy * wx)

def detect_skew_angles(volume, percentile=75):
    """
    Return (angle_yz, angle_xz) in degrees.
    Both angles are measured as the tilt of the dominant energy ridge
    in the central FFT magnitude, via PCA on the thresholded coords.
    """
    # 1) Pre-process
    vol = volume.astype(np.float32)
    vol -= np.percentile(vol, 5)
    vol = np.clip(vol, 0, None)
    vol = apply_hann_window(vol)
    vol = gaussian_filter(vol, sigma=(0.5, 2.0, 0.5))

    # 2) 3D FFT → shift → log-magnitude
    fft_vol = fftn(vol)
    fft_vol = fftshift(fft_vol)
    log_mag = np.log1p(np.abs(fft_vol))

    # 3) Crop out the center cube (to focus on low-frequency lobes)
    def crop_center(arr, frac):
        z, y, x = arr.shape
        cz, cy, cx = z//2, y//2, x//2
        dz, dy, dx = int(z*frac/2), int(y*frac/2), int(x*frac/2)
        return arr[cz-dz:cz+dz, cy-dy:cy+dy, cx-dx:cx+dx]

    log_c = crop_center(log_mag, 0.5)

    # 4) Threshold for "high-energy" voxels
    thresh = np.percentile(log_c, percentile)
    mask = log_c > thresh

    # 5) Build PCA-based angle estimator
    def pca_angle(mask, dim1, dim2):
        # mask: 3D boolean array
        pts = np.argwhere(mask)  # each row = [z, y, x]
        if len(pts) < 2:
            return 0.0
        
        # project into the 2D plane given by (dim2, dim1):
        coords = pts[:, [dim2, dim1]].astype(np.float32)
        # center around zero
        center = np.array(mask.shape)[[dim2, dim1]] / 2
        coords -= center[None, :]
        # PCA
        p = PCA(n_components=2).fit(coords)
        v = p.components_[0]
        # ensure positive y component (for consistency)
        if v[1] < 0:
            v = -v
        # angle = arctan( dx / dy ) in degrees
        return np.degrees(np.arctan2(v[0], v[1]))

    # 6) Compute both angles
    angle_yz = pca_angle(mask, dim1=1, dim2=0)  # plane = (Z,Y)
    angle_xz = pca_angle(mask, dim1=2, dim2=0)  # plane = (Z,X)

    return angle_yz, angle_xz

def deskew_volume(volume, angle_yz, angle_xz):
    """
    Undo the skew by applying a single 3×3 shear matrix:
      output[z,y,x] ⟵ input[z, y + z*sh_y, x + z*sh_x]
    where sh_y = tan(angle_yz), sh_x = tan(angle_xz).
    """
    sy = np.tan(np.deg2rad(angle_yz))
    sx = np.tan(np.deg2rad(angle_xz))
    # matrix maps output coords → input coords
    M = np.array([
        [1,    0, 0],
        [sy,   1, 0],
        [sx,   0, 1],
    ], dtype=np.float32)

    # apply affine_transform (order=1 linear interpolation)
    deskewed = affine_transform(
        volume,
        matrix=M,
        offset=0,
        order=1,
        mode='constant',
        cval=0.0
    )
    return deskewed

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

def plot_deskew_results(original, deskewed, angle_yz, angle_xz, estimated_yz, estimated_xz, title="Deskew Results"):
    """Plot original vs deskewed volumes with angle information."""
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(title, fontsize=16)
    
    # Original volume projections
    axs[0, 0].imshow(original.max(axis=2), cmap='gray')
    axs[0, 0].set_title('Original - YZ Projection')
    axs[0, 0].axis('off')
    
    axs[0, 1].imshow(original.max(axis=1), cmap='gray')
    axs[0, 1].set_title('Original - XZ Projection')
    axs[0, 1].axis('off')
    
    axs[0, 2].imshow(original.max(axis=0), cmap='gray')
    axs[0, 2].set_title('Original - XY Projection')
    axs[0, 2].axis('off')
    
    # Deskewed volume projections
    axs[1, 0].imshow(deskewed.max(axis=2), cmap='gray')
    axs[1, 0].set_title('Deskewed - YZ Projection')
    axs[1, 0].axis('off')
    
    axs[1, 1].imshow(deskewed.max(axis=1), cmap='gray')
    axs[1, 1].set_title('Deskewed - XZ Projection')
    axs[1, 1].axis('off')
    
    axs[1, 2].imshow(deskewed.max(axis=0), cmap='gray')
    axs[1, 2].set_title('Deskewed - XY Projection')
    axs[1, 2].axis('off')
    
    # Add angle information
    fig.text(0.02, 0.02, f'YZ: Real={angle_yz:.1f}°, Est={estimated_yz:.1f}°', fontsize=12)
    fig.text(0.02, 0.05, f'XZ: Real={angle_xz:.1f}°, Est={estimated_xz:.1f}°', fontsize=12)
    
    plt.tight_layout()
    plt.show()

def demo_synthetic():
    """Demonstrate deskew with synthetic data."""
    print("=== SYNTHETIC DESKEW DEMO ===")
    
    # Create synthetic volume with known skew
    skew_yz = 15
    skew_xz = 10
    print(f"Creating synthetic volume with YZ skew: {skew_yz}°, XZ skew: {skew_xz}°")
    
    vol = create_synthetic_skewed_volume(size=128, skew_yz=skew_yz, skew_xz=skew_xz)
    print(f"Volume shape: {vol.shape}")
    
    # Detect skew angles
    est_yz, est_xz = detect_skew_angles(vol)
    print(f"Detected angles → YZ: {est_yz:.2f}°, XZ: {est_xz:.2f}°")
    print(f"Errors → YZ: {abs(est_yz - skew_yz):.2f}°, XZ: {abs(est_xz - skew_xz):.2f}°")
    
    # Deskew
    vol_deskew = deskew_volume(vol, est_yz, est_xz)
    
    # Plot results
    plot_deskew_results(vol, vol_deskew, skew_yz, skew_xz, est_yz, est_xz, 
                       "Synthetic Deskew Demo")

def demo_real_data():
    """Demonstrate deskew with real ND2 data."""
    print("\n=== REAL DATA DESKEW DEMO ===")
    
    # Load volume crop
    vol = load_nd2_crop(ND2_FILE_PATH, CROP_START, CROP_SIZE)
    if vol is None:
        print("Failed to load volume, skipping real data demo")
        return
    
    print(f"Loaded volume shape: {vol.shape}")
    
    # Detect skew angles
    est_yz, est_xz = detect_skew_angles(vol)
    print(f"Detected angles → YZ: {est_yz:.2f}°, XZ: {est_xz:.2f}°")
    
    # Deskew
    vol_deskew = deskew_volume(vol, est_yz, est_xz)
    
    # Plot results
    plot_deskew_results(vol, vol_deskew, 0, 0, est_yz, est_xz, 
                       "Real Data Deskew Demo")

def main():
    """Main function to demonstrate deskew detection and correction."""
    print("=== DESKEW DETECTION AND CORRECTION DEMO ===")
    
    # Demo with synthetic data (known ground truth)
    demo_synthetic()
    
    # Demo with real data
    demo_real_data()

if __name__ == "__main__":
    main() 