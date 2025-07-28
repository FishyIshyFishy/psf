import matplotlib.pyplot as plt
import numpy as np

def plot_psf_summary(metrics, title=None):
    fig, ax = plt.subplots(figsize=(3, 2.2), dpi=300)
    fig.patch.set_facecolor('white')

    # Radar chart setup
    labels = ['FWHM Z', 'FWHM Y', 'FWHM X', 'PCA1', 'PCA2', 'PCA3', 'Astig.', 'PC1-Z']
    values = [
        metrics['fwhm_z_um'],
        metrics['fwhm_y_um'],
        metrics['fwhm_x_um'],
        metrics['fwhm_pca1_um'],
        metrics['fwhm_pca2_um'],
        metrics['fwhm_pca3_um'],
        metrics['astig_um'],
        metrics['pc1_z_angle_deg'] / 90  # normalize angle to [0–1]
    ]

    values = np.array(values)
    norm_vals = values / np.max(values[:6])  # normalize first 6 values to same scale
    norm_vals = np.clip(norm_vals, 0, 1)
    norm_vals[-2:] = np.clip(norm_vals[-2:], 0, 1)  # astig, angle

    # Close the radar loop
    data = np.concatenate([norm_vals, [norm_vals[0]]])
    angles = np.linspace(0, 2*np.pi, len(data), endpoint=True)

    # Radar plot
    radar_ax = fig.add_axes([0.05, 0.1, 0.48, 0.8], polar=True)
    radar_ax.plot(angles, data, color='tab:blue', linewidth=1.2)
    radar_ax.fill(angles, data, color='tab:blue', alpha=0.2)
    radar_ax.set_xticks(angles[:-1])
    radar_ax.set_xticklabels(labels, fontsize=6)
    radar_ax.set_yticks([])
    radar_ax.set_title('PSF Shape', fontsize=7, pad=8)

    # Right: Text box
    summary_ax = fig.add_axes([0.58, 0.1, 0.4, 0.8])
    summary_ax.axis('off')

    # Metrics to display
    txt = [
        f"SNR: {metrics['snr']:.1f}",
        f"Tilt Y: {metrics['tilt_angle_y_deg']:.1f}°",
        f"Tilt X: {metrics['tilt_angle_x_deg']:.1f}°",
        f"Skew: {metrics['skew_pc1']:.2f}",
        f"Kurt.: {metrics['kurt_pc1']:.2f}",
        f"VolRatio: {metrics['vol_ratio_05_01']:.2f}"
    ]

    for i, line in enumerate(txt):
        summary_ax.text(0, 1 - i * 0.15, line, fontsize=6, va='top')

    if title:
        fig.suptitle(title, fontsize=8)

    plt.tight_layout()
    return fig

