import numpy as np
import napari
import matplotlib.pyplot as plt
from tifffile import imread
from scipy.fft import fft2, fftshift
from sklearn.decomposition import PCA
import matplotlib.patches as patches

# Configuration - Update this path to your bead TIFF file
BEAD_TIFF_PATH = r"Z:\BioMIID_Nonsync\BioMIID_Users_Nonsync\singhi7_BioMIID_Nonsync\20250618_Fluosphere-small-PSF\split\xy1_windows\bead_0011.tif"

def load_bead(tiff_path):
    """Load bead from TIFF file."""
    bead = imread(tiff_path)
    print(f"Loaded bead with shape: {bead.shape}")
    print(f"Bead value range: [{bead.min():.3f}, {bead.max():.3f}]")
    return bead

def compute_yz_mip(bead):
    """Compute Maximum Intensity Projection along X-axis (YZ plane)."""
    # MIP along x-axis (yz plane)
    mip_yz = np.max(bead, axis=2)
    
    print(f"MIP YZ shape: {mip_yz.shape}")
    print(f"MIP YZ value range: [{mip_yz.min():.3f}, {mip_yz.max():.3f}]")
    
    return mip_yz

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

def compute_pca_2d(image_2d, name="Image"):
    """Compute PCA on 2D image data."""
    # Get coordinates of non-zero pixels
    coords = np.array(np.where(image_2d > 0)).T
    intensities = image_2d[image_2d > 0]
    
    if len(coords) < 2:
        print(f"Warning: Not enough non-zero pixels for PCA in {name}")
        return None, None, None
    
    # Compute intensity-weighted PCA
    centroid = np.average(coords, weights=intensities, axis=0)
    coords_centered = coords - centroid
    
    # Weighted covariance matrix
    cov_matrix = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            cov_matrix[i, j] = np.average(
                coords_centered[:, i] * coords_centered[:, j], 
                weights=intensities
            )
    
    # Compute eigenvalues and eigenvectors
    eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
    
    # Sort in descending order
    order = np.argsort(eigenvals)[::-1]
    eigenvals = eigenvals[order]
    eigenvecs = eigenvecs[:, order]
    
    print(f"{name} PCA eigenvalues: {eigenvals}")
    print(f"{name} PCA eigenvectors (columns):\n{eigenvecs}")
    
    return eigenvals, eigenvecs, centroid

def plot_mip_fft_pca(mip_yz, fft_yz, pca_mip, pca_fft):
    """Create a comprehensive plot showing MIP, FFT, and their PCAs."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('YZ MIP Analysis: Real vs Fourier Domain', fontsize=16)
    
    # Plot original MIP
    axes[0, 0].imshow(mip_yz, cmap='viridis', aspect='auto')
    axes[0, 0].set_title('MIP YZ (X projection)')
    axes[0, 0].set_xlabel('Y')
    axes[0, 0].set_ylabel('Z')
    
    # Plot FFT
    axes[0, 1].imshow(fft_yz, cmap='hot', aspect='auto')
    axes[0, 1].set_title('FFT Magnitude (YZ)')
    axes[0, 1].set_xlabel('Frequency Y')
    axes[0, 1].set_ylabel('Frequency Z')
    
    # Plot PCA on MIP
    if pca_mip[0] is not None:
        eigenvals, eigenvecs, centroid = pca_mip
        
        # Create PCA visualization
        axes[0, 2].imshow(mip_yz, cmap='viridis', aspect='auto')
        axes[0, 2].set_title('PCA on MIP YZ')
        axes[0, 2].set_xlabel('Y')
        axes[0, 2].set_ylabel('Z')
        
        # Draw PCA axes
        scale = 20  # Scale factor for visualization
        pc1_vec = eigenvecs[:, 0] * np.sqrt(eigenvals[0]) * scale
        pc2_vec = eigenvecs[:, 1] * np.sqrt(eigenvals[1]) * scale
        
        # Draw PC1 (red)
        axes[0, 2].arrow(centroid[1], centroid[0], pc1_vec[1], pc1_vec[0], 
                         color='red', width=2, head_width=4, head_length=3)
        axes[0, 2].arrow(centroid[1], centroid[0], -pc1_vec[1], -pc1_vec[0], 
                         color='red', width=2, head_width=4, head_length=3)
        
        # Draw PC2 (green)
        axes[0, 2].arrow(centroid[1], centroid[0], pc2_vec[1], pc2_vec[0], 
                         color='green', width=2, head_width=4, head_length=3)
        axes[0, 2].arrow(centroid[1], centroid[0], -pc2_vec[1], -pc2_vec[0], 
                         color='green', width=2, head_width=4, head_length=3)
        
        axes[0, 2].text(0.02, 0.98, f'PC1: {eigenvals[0]:.2f}\nPC2: {eigenvals[1]:.2f}', 
                        transform=axes[0, 2].transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot PCA on FFT
    if pca_fft[0] is not None:
        eigenvals, eigenvecs, centroid = pca_fft
        
        # Create PCA visualization
        axes[1, 0].imshow(fft_yz, cmap='hot', aspect='auto')
        axes[1, 0].set_title('PCA on FFT YZ')
        axes[1, 0].set_xlabel('Frequency Y')
        axes[1, 0].set_ylabel('Frequency Z')
        
        # Draw PCA axes
        scale = 20  # Scale factor for visualization
        pc1_vec = eigenvecs[:, 0] * np.sqrt(eigenvals[0]) * scale
        pc2_vec = eigenvecs[:, 1] * np.sqrt(eigenvals[1]) * scale
        
        # Draw PC1 (red)
        axes[1, 0].arrow(centroid[1], centroid[0], pc1_vec[1], pc1_vec[0], 
                         color='red', width=2, head_width=4, head_length=3)
        axes[1, 0].arrow(centroid[1], centroid[0], -pc1_vec[1], -pc1_vec[0], 
                         color='red', width=2, head_width=4, head_length=3)
        
        # Draw PC2 (green)
        axes[1, 0].arrow(centroid[1], centroid[0], pc2_vec[1], pc2_vec[0], 
                         color='green', width=2, head_width=4, head_length=3)
        axes[1, 0].arrow(centroid[1], centroid[0], -pc2_vec[1], -pc2_vec[0], 
                         color='green', width=2, head_width=4, head_length=3)
        
        axes[1, 0].text(0.02, 0.98, f'PC1: {eigenvals[0]:.2f}\nPC2: {eigenvals[1]:.2f}', 
                        transform=axes[1, 0].transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Comparison plots
    if pca_mip[0] is not None and pca_fft[0] is not None:
        # Eigenvalue comparison
        mip_eigenvals, _, _ = pca_mip
        fft_eigenvals, _, _ = pca_fft
        
        axes[1, 1].bar(['PC1', 'PC2'], mip_eigenvals, alpha=0.7, label='MIP', color='blue')
        axes[1, 1].bar(['PC1', 'PC2'], fft_eigenvals, alpha=0.7, label='FFT', color='orange')
        axes[1, 1].set_title('Eigenvalue Comparison')
        axes[1, 1].set_ylabel('Eigenvalue')
        axes[1, 1].legend()
        
        # Aspect ratio comparison
        mip_aspect = mip_eigenvals[0] / mip_eigenvals[1] if mip_eigenvals[1] > 0 else 0
        fft_aspect = fft_eigenvals[0] / fft_eigenvals[1] if fft_eigenvals[1] > 0 else 0
        
        axes[1, 2].bar(['MIP', 'FFT'], [mip_aspect, fft_aspect], color=['blue', 'orange'])
        axes[1, 2].set_title('Aspect Ratio (PC1/PC2)')
        axes[1, 2].set_ylabel('Aspect Ratio')
    
    plt.tight_layout()
    plt.show()

def visualize_in_napari(mip_yz, fft_yz, pca_mip, pca_fft):
    """Visualize MIP, FFT, and their PCAs in napari."""
    viewer = napari.Viewer(title="YZ MIP Analysis: Real vs Fourier Domain")
    
    # Add MIP
    viewer.add_image(
        mip_yz,
        name="MIP YZ",
        colormap='viridis',
        blending='additive'
    )
    
    # Add FFT
    viewer.add_image(
        fft_yz,
        name="FFT YZ",
        colormap='hot',
        blending='additive'
    )
    
    # Add PCA vectors for MIP if available
    if pca_mip[0] is not None:
        eigenvals, eigenvecs, centroid = pca_mip
        
        # Scale eigenvectors by eigenvalues for visualization
        scaled_vectors = eigenvecs * np.sqrt(eigenvals[:, np.newaxis])
        
        # PC1 (red)
        viewer.add_vectors(
            np.array([centroid]),
            scaled_vectors[:, 0:1].T,
            name="MIP PC1",
            edge_color='red',
            length=10
        )
        
        # PC2 (green)
        viewer.add_vectors(
            np.array([centroid]),
            scaled_vectors[:, 1:2].T,
            name="MIP PC2",
            edge_color='green',
            length=10
        )
    
    # Add PCA vectors for FFT if available
    if pca_fft[0] is not None:
        eigenvals, eigenvecs, centroid = pca_fft
        
        # Scale eigenvectors by eigenvalues for visualization
        scaled_vectors = eigenvecs * np.sqrt(eigenvals[:, np.newaxis])
        
        # PC1 (orange)
        viewer.add_vectors(
            np.array([centroid]),
            scaled_vectors[:, 0:1].T,
            name="FFT PC1",
            edge_color='orange',
            length=10
        )
        
        # PC2 (yellow)
        viewer.add_vectors(
            np.array([centroid]),
            scaled_vectors[:, 1:2].T,
            name="FFT PC2",
            edge_color='yellow',
            length=10
        )
    
    print("Visualization opened in napari")
    print("MIP YZ: viridis colormap")
    print("FFT YZ: hot colormap")
    print("MIP PCA: red/green vectors")
    print("FFT PCA: orange/yellow vectors")
    
    return viewer

def analyze_comparison(pca_mip, pca_fft):
    """Analyze and compare PCA results between MIP and FFT."""
    print("\n=== PCA Comparison Analysis ===")
    
    if pca_mip[0] is not None and pca_fft[0] is not None:
        mip_eigenvals, mip_eigenvecs, _ = pca_mip
        fft_eigenvals, fft_eigenvecs, _ = pca_fft
        
        print("MIP PCA Results:")
        print(f"  Eigenvalues: {mip_eigenvals}")
        print(f"  PC1 direction: {mip_eigenvecs[:, 0]}")
        print(f"  PC2 direction: {mip_eigenvecs[:, 1]}")
        print(f"  Aspect ratio (PC1/PC2): {mip_eigenvals[0]/mip_eigenvals[1]:.3f}")
        
        print("\nFFT PCA Results:")
        print(f"  Eigenvalues: {fft_eigenvals}")
        print(f"  PC1 direction: {fft_eigenvecs[:, 0]}")
        print(f"  PC2 direction: {fft_eigenvecs[:, 1]}")
        print(f"  Aspect ratio (PC1/PC2): {fft_eigenvals[0]/fft_eigenvals[1]:.3f}")
        
        print("\nComparison:")
        print(f"  MIP total variance: {np.sum(mip_eigenvals):.3f}")
        print(f"  FFT total variance: {np.sum(fft_eigenvals):.3f}")
        print(f"  MIP vs FFT aspect ratio: {mip_eigenvals[0]/mip_eigenvals[1] / (fft_eigenvals[0]/fft_eigenvals[1]):.3f}")

def main():
    """Main function to run the MIP and FFT analysis."""
    print("=== YZ MIP Analysis: Real vs Fourier Domain ===")
    
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
    print("\n=== Computing YZ MIP ===")
    mip_yz = compute_yz_mip(bead)
    
    # Compute FFT
    print("\n=== Computing FFT ===")
    fft_yz = compute_fft_magnitude(mip_yz)
    
    # Compute PCA on MIP
    print("\n=== Computing PCA on MIP ===")
    pca_mip = compute_pca_2d(mip_yz, "MIP YZ")
    
    # Compute PCA on FFT
    print("\n=== Computing PCA on FFT ===")
    pca_fft = compute_pca_2d(fft_yz, "FFT YZ")
    
    # Plot results
    print("\n=== Plotting Results ===")
    plot_mip_fft_pca(mip_yz, fft_yz, pca_mip, pca_fft)
    
    # Analyze comparison
    analyze_comparison(pca_mip, pca_fft)
    
    # Visualize in napari
    print("\n=== Opening Napari Visualization ===")
    viewer = visualize_in_napari(mip_yz, fft_yz, pca_mip, pca_fft)
    
    # Print summary
    print("\n=== Summary ===")
    print(f"Original bead shape: {bead.shape}")
    print(f"MIP YZ shape: {mip_yz.shape}")
    print(f"FFT YZ shape: {fft_yz.shape}")
    
    # Run napari
    napari.run()

if __name__ == "__main__":
    main() 