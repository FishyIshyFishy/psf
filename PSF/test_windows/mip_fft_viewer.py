import numpy as np
import napari
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift, fftfreq
import nd2

# Configuration - Update these paths and crop parameters
ND2_FILE_PATH = r"Z:\BioMIID_Nonsync\BioMIID_Users_Nonsync\singhi7_BioMIID_Nonsync\20250618_Fluosphere-small-PSF\split\multipoint_psf_xy1.nd2"

# Crop parameters - specify the region of interest in the ND2 file
# These are in voxel coordinates (z, y, x)
CROP_START = (50, 100, 100)  # Start coordinates (z, y, x)
CROP_SIZE = (40, 40, 40)     # Size of the crop (z, y, x)

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

def compute_yz_mip(volume):
    """Compute Maximum Intensity Projection along X-axis (YZ plane)."""
    # MIP along x-axis (yz plane)
    mip_yz = np.max(volume, axis=2)
    
    print(f"MIP YZ shape: {mip_yz.shape}")
    print(f"MIP YZ value range: [{mip_yz.min():.3f}, {mip_yz.max():.3f}]")
    
    return mip_yz

def compute_fft_magnitude(image_2d, voxel_size):
    """Compute the magnitude of the 2D Fourier transform with proper scaling."""
    # Apply FFT
    fft_result = fft2(image_2d)
    
    # Shift zero frequency to center
    fft_shifted = fftshift(fft_result)
    
    # Compute magnitude
    magnitude = np.abs(fft_shifted)
    
    # Log scale for better visualization
    log_magnitude = np.log10(magnitude + 1)
    
    return log_magnitude

def create_frequency_axes(image_shape, voxel_size):
    """Create frequency axes in physical units (1/µm)."""
    # Get Y and Z voxel sizes (assuming volume is in YZ plane)
    vz, vy = voxel_size[0], voxel_size[1]  # Z and Y voxel sizes
    
    # Create frequency axes
    freq_y = fftfreq(image_shape[1], d=vy)  # Y frequencies
    freq_z = fftfreq(image_shape[0], d=vz)  # Z frequencies
    
    # Shift frequencies to center
    freq_y_shifted = fftshift(freq_y)
    freq_z_shifted = fftshift(freq_z)
    
    return freq_y_shifted, freq_z_shifted

def plot_mip_and_fft(mip_yz, fft_yz, voxel_size, crop_start, crop_size):
    """Create a simple plot showing MIP and its FFT with proper scaling."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'X Projection (YZ MIP) and Fourier Transform\nCrop: {crop_start} to {tuple(np.array(crop_start) + np.array(crop_size))}', fontsize=14)
    
    # Create frequency axes
    freq_y, freq_z = create_frequency_axes(mip_yz.shape, voxel_size)
    
    # Plot MIP
    im1 = ax1.imshow(mip_yz, cmap='viridis', aspect='auto')
    ax1.set_title('X Projection (YZ MIP)')
    ax1.set_xlabel('Y')
    ax1.set_ylabel('Z')
    plt.colorbar(im1, ax=ax1, label='Intensity')
    
    # Plot FFT with proper frequency axes
    extent = [freq_y.min(), freq_y.max(), freq_z.min(), freq_z.max()]
    im2 = ax2.imshow(fft_yz, cmap='hot', aspect='auto', extent=extent)
    ax2.set_title('Fourier Transform Magnitude')
    ax2.set_xlabel('Frequency Y (1/µm)')
    ax2.set_ylabel('Frequency Z (1/µm)')
    plt.colorbar(im2, ax=ax2, label='Log Magnitude')
    
    # Add frequency scale information
    max_freq_y = np.max(np.abs(freq_y))
    max_freq_z = np.max(np.abs(freq_z))
    ax2.text(0.02, 0.98, f'Max freq Y: {max_freq_y:.2f} 1/µm\nMax freq Z: {max_freq_z:.2f} 1/µm', 
              transform=ax2.transAxes, verticalalignment='top',
              bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()

def visualize_in_napari(volume, mip_yz, fft_yz):
    """Visualize volume, MIP and FFT in napari."""
    viewer = napari.Viewer(title="Volume Crop Analysis")
    
    # Add the full volume crop
    viewer.add_image(
        volume,
        name="Volume Crop",
        colormap='viridis',
        blending='additive'
    )
    
    # Add MIP
    viewer.add_image(
        mip_yz,
        name="X Projection (YZ MIP)",
        colormap='plasma',
        blending='additive'
    )
    
    # Add FFT
    viewer.add_image(
        fft_yz,
        name="Fourier Transform",
        colormap='hot',
        blending='additive'
    )
    
    print("Visualization opened in napari")
    print("Volume Crop: viridis colormap")
    print("X Projection: plasma colormap")
    print("Fourier Transform: hot colormap")
    
    return viewer

def main():
    """Main function to run the volume crop analysis."""
    print("=== Volume Crop Analysis: X Projection and Fourier Transform ===")
    
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
    
    # Compute YZ MIP
    print("\n=== Computing X Projection ===")
    mip_yz = compute_yz_mip(volume)
    
    # Compute FFT
    print("\n=== Computing Fourier Transform ===")
    fft_yz = compute_fft_magnitude(mip_yz, voxel_size)
    
    # Plot results
    print("\n=== Plotting Results ===")
    plot_mip_and_fft(mip_yz, fft_yz, voxel_size, CROP_START, CROP_SIZE)
    
    # Visualize in napari
    print("\n=== Opening Napari Visualization ===")
    viewer = visualize_in_napari(volume, mip_yz, fft_yz)
    
    # Print summary
    print("\n=== Summary ===")
    print(f"Volume crop shape: {volume.shape}")
    print(f"X projection shape: {mip_yz.shape}")
    print(f"FFT shape: {fft_yz.shape}")
    print(f"Voxel size: {voxel_size} µm")
    print(f"Crop start: {CROP_START}")
    print(f"Crop size: {CROP_SIZE}")
    
    # Run napari
    napari.run()

if __name__ == "__main__":
    main() 