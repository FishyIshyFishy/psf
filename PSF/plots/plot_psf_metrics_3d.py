import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import nd2
import ast
import seaborn as sns

# ---- Default metrics to plot ----
DEFAULT_METRICS = [
    'fwhm_z_um', 'fwhm_y_um', 'fwhm_x_um',
    'fwhm_pca1_um', 'fwhm_pca2_um', 'fwhm_pca3_um',
    'tilt_angle_y_deg', 'tilt_angle_x_deg',
    'skew_pc1', 'kurt_pc1',
    'snr', 'vol_ratio_05_01',
    'pc1_z_angle_deg', 'astig_um'
]

FWHM_METRICS = [
    'fwhm_z_um', 'fwhm_y_um', 'fwhm_x_um',
    'fwhm_pca1_um', 'fwhm_pca2_um', 'fwhm_pca3_um'
]

# ---- Helper: check if column is numeric ----
def is_numeric_col(df, col):
    return pd.api.types.is_numeric_dtype(df[col])

# ---- Utilities ----
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

# ---- Plotting functions ----
def plot_metric_scatter_3d(df, metrics=None, vox_ds=None):
    if metrics is None:
        metrics = DEFAULT_METRICS
    metrics = [m for m in metrics if m in df.columns and is_numeric_col(df, m)]
    n = len(metrics)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig = plt.figure(figsize=(4 * cols, 4 * rows))
    fig.suptitle("Metric 3D Scatter Plots", fontsize=16, y=0.95)
    for idx, metric in enumerate(metrics, start=1):
        ax = fig.add_subplot(rows, cols, idx, projection='3d')
        # Use x/y/z_um if available, else fallback to peak_x/y/z
        if all(c in df.columns for c in ['x_um', 'y_um', 'z_um']):
            x, y, z = df['x_um'], df['y_um'], df['z_um']
        else:
            x, y, z = df['peak_x'], df['peak_y'], df['peak_z']
        data = df[metric]
        mask = np.isfinite(data)
        sc = ax.scatter(x[mask], y[mask], z[mask],
                        c=data[mask], cmap='viridis', s=25, edgecolor='k', linewidth=0.2)
        ax.set_title(metric.replace('_', ' '), fontsize=10)
        ax.set_xlabel('X (um)')
        ax.set_ylabel('Y (um)')
        ax.set_zlabel('Z (um)')
        fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.show()

def plot_metric_kde(df, metrics=None):
    if metrics is None:
        metrics = DEFAULT_METRICS
    metrics = [m for m in metrics if m in df.columns and is_numeric_col(df, m)]
    n = len(metrics)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    fig.suptitle("Metric KDEs (Seaborn)", fontsize=16, y=0.95)
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
    for j in range(len(metrics), len(axes.flat)):
        axes.flat[j].axis('off')
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.show()

def plot_metric_boxplot(df, metrics=None):
    if metrics is None:
        metrics = DEFAULT_METRICS
    metrics = [m for m in metrics if m in df.columns and is_numeric_col(df, m)]
    n = len(metrics)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    fig.suptitle("Metric Boxplots", fontsize=16, y=0.95)
    for i, metric in enumerate(metrics):
        ax = axes.flat[i]
        data = df[metric].dropna()
        if len(data) > 0:
            Q1 = np.percentile(data, 25)
            Q3 = np.percentile(data, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = data[(data < lower_bound) | (data > upper_bound)]
            outlier_count = len(outliers)
            ax.boxplot(data, vert=True, patch_artist=True,
                    boxprops=dict(facecolor='slateblue', color='black'),
                    medianprops=dict(color='black'))
            ax.set_title(f"{metric.replace('_', ' ')}\nOutliers: {outlier_count}", fontsize=10)
            ax.set_ylabel(metric.replace('_', ' '))
        else:
            ax.text(0.5, 0.5, "No data", ha='center', va='center')
            ax.axis('off')
    for j in range(len(metrics), len(axes.flat)):
        axes.flat[j].axis('off')
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.show()

# ---- PCA Quiver remains unchanged ----
def plot_pca_quiver(df, vox_ds):
    x0 = df['x_um'].values if 'x_um' in df.columns else df['peak_x'].values
    y0 = df['y_um'].values if 'y_um' in df.columns else df['peak_y'].values
    z0 = df['z_um'].values if 'z_um' in df.columns else df['peak_z'].values
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
    ax.set_zlabel("Z (um)")
    ax.set_box_aspect([1, 1, 1])
    plt.tight_layout()
    plt.show()

def filter_outliers_fwhm(df, fwhm_metrics=None, iqr_factor=1.5):
    """Remove rows where any FWHM metric is an outlier (outside IQR*factor)."""
    if fwhm_metrics is None:
        fwhm_metrics = FWHM_METRICS
    mask = np.ones(len(df), dtype=bool)
    for col in fwhm_metrics:
        if col in df.columns and is_numeric_col(df, col):
            data = df[col].dropna()
            Q1 = np.percentile(data, 25)
            Q3 = np.percentile(data, 75)
            IQR = Q3 - Q1
            lower = Q1 - iqr_factor * IQR
            upper = Q3 + iqr_factor * IQR
            mask &= (df[col] >= lower) & (df[col] <= upper)
    filtered = df[mask].reset_index(drop=True)
    print(f"Filtered out {len(df) - len(filtered)} outliers based on FWHM metrics.")
    return filtered

if __name__ == '__main__':
    # User-editable path to CSV
    CSV_PATH = r'C:\Users\ishaa\OneDrive\Documents\VIBES2025\biomiid_stuff\psf\initial_results.csv'  # <-- Edit as needed
    print(f"Loading metrics from: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)

    # Parse PCA columns if present
    for col in ['pca_axis1', 'pca_axis2', 'pca_axis3']:
        if col in df.columns:
            df[col] = df[col].apply(ast.literal_eval)

    # Optionally compute x_um, y_um, z_um if not present
    if all(c in df.columns for c in ['peak_z', 'peak_y', 'peak_x']):
        if not all(c in df.columns for c in ['z_um', 'y_um', 'x_um']):
            # If you know your voxel size, set it here:
            VOXEL_SIZE = (1.0, 1.0, 1.0)  # (z, y, x) in um, edit as needed
            df['z_um'] = df['peak_z'] * VOXEL_SIZE[0]
            df['y_um'] = df['peak_y'] * VOXEL_SIZE[1]
            df['x_um'] = df['peak_x'] * VOXEL_SIZE[2]

    # Filter outliers based on FWHM metrics
    df_filt = filter_outliers_fwhm(df)

    # Plot all metrics for inlier beads
    plot_metric_scatter_3d(df_filt)
    plot_metric_kde(df_filt)
    plot_metric_boxplot(df_filt)

    # Plot PCA quiver if possible
    if all(col in df_filt.columns for col in ['pca_axis1', 'x_um', 'y_um', 'z_um']):
        plot_pca_quiver(df_filt, vox_ds=(1.0, 1.0, 1.0))  # Edit vox_ds as needed
    else:
        print("PCA quiver plot skipped (missing columns)")
