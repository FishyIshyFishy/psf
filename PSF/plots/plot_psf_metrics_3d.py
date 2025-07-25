import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import nd2
import ast
import seaborn as sns

# ─── Configuration ───────────────────────────────────────────────────────────
CSV_PATH   = r'C:\Users\singhi7\Documents\psf\results.csv' # path to your metrics CSV
ND2_PATH   = r"Z:\BioMIID_Nonsync\BioMIID_Users_Nonsync\singhi7_BioMIID_Nonsync\20250618_Fluosphere-small-PSF\split\multipoint_psf_xy1.nd2"
DOWNSAMPLE = (1,1,1)
ANGLE_THRESHOLD_DEG = 10  # Max deviation from mean direction

# ─── Utilities ───────────────────────────────────────────────────────────────
def get_voxel_size_ds(nd2_path, factors):
    with nd2.ND2File(nd2_path) as f:
        vx, vy, vz = f.voxel_size()
    return (vz * factors[0], vy * factors[1], vx * factors[2])  # Z, Y, X

def compute_angles_to_mean(pca_vectors):
    unit_vecs = pca_vectors / np.linalg.norm(pca_vectors, axis=1, keepdims=True)
    mean_vec = np.mean(unit_vecs, axis=0)
    mean_vec /= np.linalg.norm(mean_vec)
    dots = np.clip(np.dot(unit_vecs, mean_vec), -1.0, 1.0)
    angles = np.degrees(np.arccos(dots))
    return angles, mean_vec

def plot_metric_scatter_3d(df, metrics, vox_ds):
    n = len(metrics)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig = plt.figure(figsize=(4 * cols, 4 * rows))
    fig.suptitle("volume scatters", fontsize=16, y=0.95)
    for idx, metric in enumerate(metrics, start=1):
        ax = fig.add_subplot(rows, cols, idx, projection='3d')
        sc = ax.scatter(df['x_um'], df['y_um'], df['z_um'],
                        c=df[metric], cmap='viridis', s=25, edgecolor='k', linewidth=0.2)
        ax.set_title(metric.replace('_', ' '), fontsize=10)
        ax.set_xlabel('X (um)')
        ax.set_ylabel('Y (um)')
        ax.set_zlabel('Z (um)')
        fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.show()

def plot_pca_quiver(df, vox_ds):
    # Arrow origin points
    x0 = df['x_um'].values
    y0 = df['y_um'].values
    z0 = df['z_um'].values

    # Vector components (ZYX → XYZ)
    pca1 = np.vstack(df['pca_axis1'].values)
    dz, dy, dx = pca1[:, 0], pca1[:, 1], pca1[:, 2]

    arrow_scale = 5  # microns
    dx *= arrow_scale
    dy *= arrow_scale
    dz *= arrow_scale

    fig2 = plt.figure(figsize=(10, 8))
    ax = fig2.add_subplot(111, projection='3d')
    ax.set_title("PCA Axis 1 Orientation Vectors")
    ax.quiver(x0, y0, z0, dx, dy, dz, length=1, normalize=False,
              color='crimson', linewidth=1)

    ax.set_xlabel("X (um)")
    ax.set_ylabel("Y (um)")
    ax.set_ylim([20, 70])
    ax.set_zlabel("Z (um)")
    ax.set_box_aspect([1, 1, 1])

    plt.tight_layout()
    plt.show()

def plot_metric_kde(df, metrics):
    rows = int(np.ceil(len(metrics) / cols))
    fig3, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    fig3.suptitle("Metric KDEs (Seaborn)", fontsize=16, y=0.95)

    for i, metric in enumerate(metrics):
        ax = axes.flat[i]
        data = df[metric].dropna()
        if len(data) > 1:
            sns.kdeplot(data, fill=True, bw_adjust=0.7, ax=ax, color='slateblue')
            ax.set_title(metric.replace('_', ' '), fontsize=10)
            ax.set_xlabel(metric.replace('_', ' '))
            ax.set_ylabel('Density')
        else:
            ax.text(0.5, 0.5, "Insufficient data", ha='center', va='center')
            ax.axis('off')

    # Hide any unused subplots
    for j in range(len(metrics), len(axes.flat)):
        axes.flat[j].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.show()

def plot_metric_boxplot(df, metrics):
    rows = int(np.ceil(len(metrics) / cols))
    fig3, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    fig3.suptitle("metric boxplots", fontsize=16, y=0.95)

    for i, metric in enumerate(metrics):
        ax = axes.flat[i]
        data = df[metric].dropna()

        # Calculate IQR and count outliers
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = data[(data < lower_bound) | (data > upper_bound)]
        outlier_count = len(outliers)

        # Draw boxplot
        ax.boxplot(data, vert=True, patch_artist=True,
                boxprops=dict(facecolor='slateblue', color='black'),
                medianprops=dict(color='black'))
        
        ax.set_title(f"{metric.replace('_', ' ')}\nOutliers: {outlier_count}", fontsize=10)
        ax.set_ylabel(metric.replace('_', ' '))

    # Hide any unused subplots
    for j in range(len(metrics), len(axes.flat)):
        axes.flat[j].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.show()
