import numpy as np
import napari
import matplotlib.pyplot as plt
from tifffile import imread
from scipy.fft import fft2, fftshift
import matplotlib.patches as patches

# Configuration - Update this path to your bead TIFF file
BEAD_TIFF_PATH = r"Z:\BioMIID_Nonsync\BioMIID_Users_Nonsync\singhi7_BioMIID_Nonsync\20250618_Fluosphere-small-PSF\split\xy1_windows\bead_0011.tif"

def load_bead(tiff_path):
    """Load bead from TIFF file."""
    bead = imread(tiff_path)
    print(f"Loaded bead with shape: {bead.shape}")
    print(f"Bead value range: [{bead.min():.3f}, {bead.max():.3f}]")
    return bead

def compute_mips(bead):
    """Compute Maximum Intensity Projections in all three directions."""
    # MIP along z-axis (xy plane)
    mip_xy = np.max(bead, axis=0)
    
    # MIP along y-axis (xz plane)
    mip_xz = np.max(bead, axis=1)
    
    # MIP along x-axis (yz plane)
    mip_yz = np.max(bead, axis=2)
    
    print(f"MIP shapes: XY={mip_xy.shape}, XZ={mip_xz.shape}, YZ={mip_yz.shape}")
    
    return mip_xy, mip_xz, mip_yz

def compute_fft_magnitude(image_2d):
    """Compute the magnitude of the 2D Fourier transform."""
    # Apply FFT
    fft_result = fft2(image_2d)
    
    # Shift zero frequency to center
    fft_shifted = fftshift(fft_result)
    
    # Compute magnitude
    magnitude = np.abs(fft_shifted)
    
    # Log scale for better visualization
    log_magnitude = np.log10(magnitude + 1)
    
    return log_magnitude

def plot_mips_and_fft(bead, mip_xy, mip_xz, mip_yz):
    """Create a comprehensive plot showing MIPs and their FFTs."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Maximum Intensity Projections and Fourier Transforms', fontsize=16)
    
    # Plot MIPs
    axes[0, 0].imshow(mip_xy, cmap='viridis', aspect='auto')
    axes[0, 0].set_title('MIP XY (Z projection)')
    axes[0, 0].set_xlabel('X')
    axes[0, 0].set_ylabel('Y')
    
    axes[0, 1].imshow(mip_xz, cmap='viridis', aspect='auto')
    axes[0, 1].set_title('MIP XZ (Y projection)')
    axes[0, 1].set_xlabel('X')
    axes[0, 1].set_ylabel('Z')
    
    axes[0, 2].imshow(mip_yz, cmap='viridis', aspect='auto')
    axes[0, 2].set_title('MIP YZ (X projection)')
    axes[0, 2].set_xlabel('Y')
    axes[0, 2].set_ylabel('Z')
    
    # Compute and plot FFTs
    fft_xy = compute_fft_magnitude(mip_xy)
    fft_xz = compute_fft_magnitude(mip_xz)
    fft_yz = compute_fft_magnitude(mip_yz)
    
    axes[1, 0].imshow(fft_xy, cmap='hot', aspect='auto')
    axes[1, 0].set_title('FFT Magnitude (XY)')
    axes[1, 0].set_xlabel('Frequency X')
    axes[1, 0].set_ylabel('Frequency Y')
    
    axes[1, 1].imshow(fft_xz, cmap='hot', aspect='auto')
    axes[1, 1].set_title('FFT Magnitude (XZ)')
    axes[1, 1].set_xlabel('Frequency X')
    axes[1, 1].set_ylabel('Frequency Z')
    
    axes[1, 2].imshow(fft_yz, cmap='hot', aspect='auto')
    axes[1, 2].set_title('FFT Magnitude (YZ)')
    axes[1, 2].set_xlabel('Frequency Y')
    axes[1, 2].set_ylabel('Frequency Z')
    
    plt.tight_layout()
    plt.show()
    
    return fft_xy, fft_xz, fft_yz

def visualize_in_napari(bead, mip_xy, mip_xz, mip_yz, fft_xy, fft_xz, fft_yz):
    """Visualize MIPs and FFTs in napari."""
    viewer = napari.Viewer(title="Bead MIPs and FFTs")
    
    # Add original bead
    viewer.add_image(
        bead,
        name="Original Bead",
        colormap='viridis',
        blending='additive'
    )
    
    # Add MIPs
    viewer.add_image(
        mip_xy,
        name="MIP XY",
        colormap='plasma',
        blending='additive'
    )
    
    viewer.add_image(
        mip_xz,
        name="MIP XZ",
        colormap='plasma',
        blending='additive'
    )
    
    viewer.add_image(
        mip_yz,
        name="MIP YZ",
        colormap='plasma',
        blending='additive'
    )
    
    # Add FFTs
    viewer.add_image(
        fft_xy,
        name="FFT XY",
        colormap='hot',
        blending='additive'
    )
    
    viewer.add_image(
        fft_xz,
        name="FFT XZ",
        colormap='hot',
        blending='additive'
    )
    
    viewer.add_image(
        fft_yz,
        name="FFT YZ",
        colormap='hot',
        blending='additive'
    )
    
    print("Visualization opened in napari")
    print("Original bead: viridis colormap")
    print("MIPs: plasma colormap")
    print("FFTs: hot colormap")
    
    return viewer

def analyze_fft_features(fft_xy, fft_xz, fft_yz):
    """Analyze FFT features and print statistics."""
    print("\n=== FFT Analysis ===")
    
    # Find peak frequencies
    for name, fft_data in [("XY", fft_xy), ("XZ", fft_xz), ("YZ", fft_yz)]:
        max_val = np.max(fft_data)
        mean_val = np.mean(fft_data)
        std_val = np.std(fft_data)
        
        # Find coordinates of maximum
        max_coords = np.unravel_index(np.argmax(fft_data), fft_data.shape)
        
        print(f"{name} FFT:")
        print(f"  Max value: {max_val:.3f}")
        print(f"  Mean value: {mean_val:.3f}")
        print(f"  Std value: {std_val:.3f}")
        print(f"  Max location: {max_coords}")
        print()

def main():
    """Main function to run the MIP and FFT analysis."""
    print("=== Bead MIP and FFT Analysis ===")
    
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
    
    # Compute MIPs
    print("\n=== Computing MIPs ===")
    mip_xy, mip_xz, mip_yz = compute_mips(bead)
    
    # Plot MIPs and FFTs
    print("\n=== Plotting MIPs and FFTs ===")
    fft_xy, fft_xz, fft_yz = plot_mips_and_fft(bead, mip_xy, mip_xz, mip_yz)
    
    # Analyze FFT features
    analyze_fft_features(fft_xy, fft_xz, fft_yz)
    
    # Visualize in napari
    print("\n=== Opening Napari Visualization ===")
    viewer = visualize_in_napari(bead, mip_xy, mip_xz, mip_yz, fft_xy, fft_xz, fft_yz)
    
    # Print summary
    print("\n=== Summary ===")
    print(f"Original bead shape: {bead.shape}")
    print(f"MIP XY shape: {mip_xy.shape}")
    print(f"MIP XZ shape: {mip_xz.shape}")
    print(f"MIP YZ shape: {mip_yz.shape}")
    print(f"FFT shapes: {fft_xy.shape}, {fft_xz.shape}, {fft_yz.shape}")
    
    # Run napari
    napari.run()

if __name__ == "__main__":
    main() 