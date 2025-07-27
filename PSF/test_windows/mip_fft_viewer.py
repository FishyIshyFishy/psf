import numpy as np
import napari
import matplotlib.pyplot as plt
from tifffile import imread
from scipy.fft import fft2, fftshift, fftfreq
import nd2

# Configuration - Update these paths
BEAD_TIFF_PATH = r"Z:\BioMIID_Nonsync\BioMIID_Users_Nonsync\singhi7_BioMIID_Nonsync\20250618_Fluosphere-small-PSF\split\xy1_windows\bead_0011.tif"
ND2_FILE_PATH = r"Z:\BioMIID_Nonsync\BioMIID_Users_Nonsync\singhi7_BioMIID_Nonsync\20250618_Fluosphere-small-PSF\split\multipoint_psf_xy1.nd2"

def load_bead(tiff_path):
    """Load bead from TIFF file."""
    bead = imread(tiff_path)
    print(f"Loaded bead with shape: {bead.shape}")
    print(f"Bead value range: [{bead.min():.3f}, {bead.max():.3f}]")
    return bead

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

def compute_yz_mip(bead):
    """Compute Maximum Intensity Projection along X-axis (YZ plane)."""
    # MIP along x-axis (yz plane)
    mip_yz = np.max(bead, axis=2)
    
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
    # Get Y and Z voxel sizes (assuming bead is in YZ plane)
    vz, vy = voxel_size[0], voxel_size[1]  # Z and Y voxel sizes
    
    # Create frequency axes
    freq_y = fftfreq(image_shape[1], d=vy)  # Y frequencies
    freq_z = fftfreq(image_shape[0], d=vz)  # Z frequencies
    
    # Shift frequencies to center
    freq_y_shifted = fftshift(freq_y)
    freq_z_shifted = fftshift(freq_z)
    
    return freq_y_shifted, freq_z_shifted

def plot_mip_and_fft(mip_yz, fft_yz, voxel_size):
    """Create a simple plot showing MIP and its FFT with proper scaling."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('X Projection (YZ MIP) and Fourier Transform', fontsize=14)
    
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

def visualize_in_napari(mip_yz, fft_yz):
    """Visualize MIP and FFT in napari."""
    viewer = napari.Viewer(title="X Projection and Fourier Transform")
    
    # Add MIP
    viewer.add_image(
        mip_yz,
        name="X Projection (YZ MIP)",
        colormap='viridis',
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
    print("X Projection: viridis colormap")
    print("Fourier Transform: hot colormap")
    
    return viewer

def main():
    """Main function to run the MIP and FFT analysis."""
    print("=== X Projection and Fourier Transform Analysis ===")
    
    # Get voxel size from ND2 file
    print("\n=== Getting Voxel Size ===")
    voxel_size = get_voxel_size(ND2_FILE_PATH)
    
    # Load the bead
    try:
        bead = load_bead(BEAD_TIFF_PATH)
    except FileNotFoundError:
        print(f"Error: Could not find bead file at {BEAD_TIFF_PATH}")
        print("Please update BEAD_TIFF_PATH at the top of this script")
        return
    except Exception as e:
        print(f"Error loading bead: {e}")
        return
    
    # Compute YZ MIP
    print("\n=== Computing X Projection ===")
    mip_yz = compute_yz_mip(bead)
    
    # Compute FFT
    print("\n=== Computing Fourier Transform ===")
    fft_yz = compute_fft_magnitude(mip_yz, voxel_size)
    
    # Plot results
    print("\n=== Plotting Results ===")
    plot_mip_and_fft(mip_yz, fft_yz, voxel_size)
    
    # Visualize in napari
    print("\n=== Opening Napari Visualization ===")
    viewer = visualize_in_napari(mip_yz, fft_yz)
    
    # Print summary
    print("\n=== Summary ===")
    print(f"Original bead shape: {bead.shape}")
    print(f"X projection shape: {mip_yz.shape}")
    print(f"FFT shape: {fft_yz.shape}")
    print(f"Voxel size: {voxel_size} µm")
    
    # Run napari
    napari.run()

if __name__ == "__main__":
    main() 