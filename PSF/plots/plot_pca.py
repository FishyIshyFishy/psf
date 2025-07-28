import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────────────────────────────
RESULTS_ROOT = Path(r"Z:\BioMIID_Nonsync\BioMIID_Users_Nonsync\singhi7_BioMIID_Nonsync\20250320_Controlled-misalignment-HD-MDJ\psf-tilt-angle-sweep\250726_RESULTS")
OUTPUT_PATH  = r'C:\Users\singhi7\Documents\psf\PCA_pc1bybc2.png'

# List of all FWHM metric columns you want to use for outlier filtering:
FWHM_METRICS = [
    'fwhm_z_um','fwhm_y_um','fwhm_x_um',
    'fwhm_pca1_um','fwhm_pca2_um','fwhm_pca3_um'
]

def extract_offset_from_path(path):
    """Extract offset value from folder path name."""
    # Look for pattern like "offset-0000", "offset-0500", "offset-neg0060"
    match = re.search(r'offset-([^/\\-]+)', str(path))
    if match:
        offset_str = match.group(1)
        # Handle negative offsets
        if offset_str.startswith('neg'):
            return -float(offset_str[3:])
        else:
            return float(offset_str)
    return None

def is_numeric_col(df, col):
    """Return True if df[col] is a numeric dtype."""
    return pd.api.types.is_numeric_dtype(df[col])

def filter_outliers_fwhm(df, fwhm_metrics=None, iqr_factor=1.5):
    """Remove rows where any FWHM metric is an outlier (outside IQR*factor)."""
    if fwhm_metrics is None:
        fwhm_metrics = FWHM_METRICS
    mask = np.ones(len(df), dtype=bool)
    for col in fwhm_metrics:
        if col in df.columns and is_numeric_col(df, col):
            data = df[col].dropna()
            if len(data) < 2:
                continue
            Q1, Q3 = np.percentile(data, [25, 75])
            IQR = Q3 - Q1
            lower = Q1 - iqr_factor * IQR
            upper = Q3 + iqr_factor * IQR
            mask &= df[col].between(lower, upper)
    filtered = df[mask].reset_index(drop=True)
    return filtered, (len(df) - len(filtered))

def gather_results(results_root):
    all_csvs = list(results_root.glob("**/results.csv"))
    if not all_csvs:
        raise FileNotFoundError("No results.csv files found under the specified folder.")

    cleaned_dfs = []
    for csv_path in all_csvs:
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue

        # Extract offset from the folder path
        offset = extract_offset_from_path(csv_path.parent)
        if offset is None:
            print(f"Warning: Could not extract offset from {csv_path.parent}")
            continue
            
        src = csv_path.parent.name
        n0 = len(df)

        # 1) Outlier filtering
        df_filt, n_out = filter_outliers_fwhm(df)
        print(f"{src} (offset {offset}): {n_out} outliers removed (from {n0} rows)")

        # 2) Drop rows with any NaNs
        n1 = len(df_filt)
        df_clean = df_filt.dropna()
        n_nan = n1 - len(df_clean)
        print(f"{src} (offset {offset}): {n_nan} rows dropped due to NaNs (remaining {len(df_clean)})")

        # Tag with offset instead of source folder name
        df_clean['offset'] = offset
        df_clean['source'] = src  # Keep original source for reference
        cleaned_dfs.append(df_clean)

    if not cleaned_dfs:
        raise RuntimeError("No data left after cleaning.")
    return pd.concat(cleaned_dfs, ignore_index=True)

def run_pca(df, n_components=2):
    drop_cols = [
        'bead_index', 'peak_z', 'peak_y', 'peak_x',
        'centroid_z_um', 'centroid_y_um', 'centroid_x_um',
        'source', 'offset', 'snr', 'pc1_z_angle_deg'
    ]
    numeric = df.select_dtypes(include=[np.number])
    numeric = numeric.drop(columns=[c for c in drop_cols if c in numeric.columns], errors='ignore')
    features = numeric.columns.tolist()

    scaler = StandardScaler()
    scaled = scaler.fit_transform(numeric)

    pca = PCA(n_components=n_components)
    pcs = pca.fit_transform(numeric.values)

    loadings = pd.DataFrame(pca.components_.T, columns=[f'PC{i+1}' for i in range(n_components)],index=features)
    return pcs, pca, loadings

def plot_pc1_vs_pc2(pcs, offsets, output_path):
    df_plot = pd.DataFrame({
        'PC1': pcs[:, 0],
        'PC2': pcs[:, 1],
        'offset': offsets
    })
    plt.figure(figsize=(10,8))

    # Sort offsets for consistent color mapping
    unique_offsets = sorted(df_plot['offset'].unique())
    
    # Create a color map based on offset values
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_offsets)))
    color_dict = dict(zip(unique_offsets, colors))

    # Plot each offset with its corresponding color
    for offset in unique_offsets:
        mask = df_plot['offset'] == offset
        plt.scatter(df_plot.loc[mask, 'PC1'], df_plot.loc[mask, 'PC2'], 
                   c=[color_dict[offset]], label=f'Offset {offset}', 
                   s=50, alpha=0.8)

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA Plot by Offset Parameter")
    plt.legend(loc='best', frameon=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.show()

# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 1) Gather, filter outliers, drop NaNs, and tag with offset
    df_all = gather_results(RESULTS_ROOT)

    # 2) Run PCA
    pcs, pca_model, loadings = run_pca(df_all)
    print("Explained variance ratio:", pca_model.explained_variance_ratio_)
    print(f'\n loadings')
    print(f'{loadings} \n')

    # 3) Plot by offset
    plot_pc1_vs_pc2(pcs, df_all['offset'], OUTPUT_PATH)
