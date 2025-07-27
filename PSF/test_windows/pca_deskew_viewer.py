import numpy as np
import napari
from skimage.transform import AffineTransform, warp
from scipy.ndimage import center_of_mass
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from tifffile import imread

# Configuration - Update this path to your bead TIFF file
BEAD_TIFF_PATH = "path/to/your/bead.tiff"  # Update this path

def load_bead(tiff_path):
    """Load bead from TIFF file."""
    bead = imread(tiff_path)
    print(f"Loaded bead with shape: {bead.shape}")
    print(f"Bead value range: [{bead.min():.3f}, {bead.max():.3f}]")
    return bead

def compute_pca_3d(bead):
    """Compute PCA on 3D bead data."""
    # Get coordinates of non-zero voxels
    coords = np.array(np.where(bead > 0)).T
    intensities = bead[bead > 0]
    
    if len(coords) < 3:
        print("Warning: Not enough non-zero voxels for PCA")
        return None, None, None
    
    # Compute intensity-weighted PCA
    centroid = np.average(coords, weights=intensities, axis=0)
    coords_centered = coords - centroid
    
    # Weighted covariance matrix
    cov_matrix = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
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
    
    print(f"PCA eigenvalues: {eigenvals}")
    print(f"PCA eigenvectors (columns):\n{eigenvecs}")
    
    return eigenvals, eigenvecs, centroid

def compute_deskew_angle(pc1_vector):
    """Compute the deskew angle from PC1 vector."""
    # PC1 is the direction of maximum variance
    # We want to rotate so that PC1 aligns with the z-axis
    z_axis = np.array([1, 0, 0])  # Assuming (z, y, x) indexing
    
    # Compute angle between PC1 and z-axis
    cos_angle = np.dot(pc1_vector, z_axis) / (np.linalg.norm(pc1_vector) * np.linalg.norm(z_axis))
    angle_rad = np.arccos(np.clip(cos_angle, -1, 1))
    angle_deg = np.degrees(angle_rad)
    
    print(f"Deskew angle: {angle_deg:.2f} degrees")
    return angle_deg, angle_rad

def create_deskew_transform(bead, pc1_vector, centroid):
    """Create affine transformation to deskew the bead."""
    # Compute rotation matrix to align PC1 with z-axis
    z_axis = np.array([1, 0, 0])
    
    # Create rotation matrix using Rodrigues' rotation formula
    # Rotate around the axis perpendicular to PC1 and z-axis
    rotation_axis = np.cross(pc1_vector, z_axis)
    if np.linalg.norm(rotation_axis) > 1e-10:
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
        
        # Compute angle between PC1 and z-axis
        cos_angle = np.dot(pc1_vector, z_axis)
        angle = np.arccos(np.clip(cos_angle, -1, 1))
        
        # Rodrigues' rotation formula
        K = np.array([[0, -rotation_axis[2], rotation_axis[1]],
                     [rotation_axis[2], 0, -rotation_axis[0]],
                     [-rotation_axis[1], rotation_axis[0], 0]])
        
        rotation_matrix = (np.eye(3) + 
                         np.sin(angle) * K + 
                         (1 - np.cos(angle)) * np.dot(K, K))
    else:
        # PC1 is already aligned with z-axis
        rotation_matrix = np.eye(3)
    
    # Create affine transformation
    # First translate to origin, then rotate, then translate back
    transform = AffineTransform(
        matrix=rotation_matrix,
        translation=-np.dot(rotation_matrix, centroid) + centroid
    )
    
    return transform

def deskew_bead(bead, transform):
    """Apply deskewing transformation to bead."""
    # Apply the affine transformation
    deskewed = warp(bead, transform, order=1, mode='constant', cval=0)
    
    print(f"Deskewed bead shape: {deskewed.shape}")
    print(f"Deskewed value range: [{deskewed.min():.3f}, {deskewed.max():.3f}]")
    
    return deskewed

def visualize_in_napari(bead_original, bead_deskewed, eigenvals, eigenvecs, centroid):
    """Visualize original and deskewed beads in napari."""
    viewer = napari.Viewer(title="Bead PCA Analysis and Deskewing")
    
    # Add original bead
    viewer.add_image(
        bead_original,
        name="Original Bead",
        colormap='viridis',
        blending='additive'
    )
    
    # Add deskewed bead
    viewer.add_image(
        bead_deskewed,
        name="Deskewed Bead",
        colormap='plasma',
        blending='additive'
    )
    
    # Add PCA axes as vectors
    if eigenvecs is not None:
        # Scale eigenvectors by eigenvalues for visualization
        scaled_vectors = eigenvecs * np.sqrt(eigenvals[:, np.newaxis])
        
        # PC1 (red)
        viewer.add_vectors(
            np.array([centroid]),
            scaled_vectors[:, 0:1].T,
            name="PC1",
            edge_color='red',
            length=10
        )
        
        # PC2 (green)
        viewer.add_vectors(
            np.array([centroid]),
            scaled_vectors[:, 1:2].T,
            name="PC2",
            edge_color='green',
            length=10
        )
        
        # PC3 (blue)
        viewer.add_vectors(
            np.array([centroid]),
            scaled_vectors[:, 2:3].T,
            name="PC3",
            edge_color='blue',
            length=10
        )
    
    print("Visualization opened in napari")
    print("Red: PC1 (maximum variance)")
    print("Green: PC2 (second maximum variance)")
    print("Blue: PC3 (minimum variance)")
    
    return viewer

def main():
    """Main function to run the PCA analysis and deskewing."""
    print("=== Bead PCA Analysis and Deskewing ===")
    
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
    
    # Compute PCA
    print("\n=== Computing PCA ===")
    eigenvals, eigenvecs, centroid = compute_pca_3d(bead)
    
    if eigenvals is None:
        print("Could not compute PCA, exiting")
        return
    
    # Get PC1 vector
    pc1_vector = eigenvecs[:, 0]
    print(f"PC1 vector: {pc1_vector}")
    
    # Compute deskew angle
    print("\n=== Computing Deskew Angle ===")
    angle_deg, angle_rad = compute_deskew_angle(pc1_vector)
    
    # Create deskewing transformation
    print("\n=== Creating Deskewing Transformation ===")
    transform = create_deskew_transform(bead, pc1_vector, centroid)
    
    # Apply deskewing
    print("\n=== Applying Deskewing ===")
    bead_deskewed = deskew_bead(bead, transform)
    
    # Visualize results
    print("\n=== Opening Visualization ===")
    viewer = visualize_in_napari(bead, bead_deskewed, eigenvals, eigenvecs, centroid)
    
    # Print summary
    print("\n=== Summary ===")
    print(f"Original bead shape: {bead.shape}")
    print(f"Deskewed bead shape: {bead_deskewed.shape}")
    print(f"Deskew angle: {angle_deg:.2f} degrees")
    print(f"PCA eigenvalues: {eigenvals}")
    print(f"PC1 direction: {pc1_vector}")
    
    # Run napari
    napari.run()

if __name__ == "__main__":
    main() 