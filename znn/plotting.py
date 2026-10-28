import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors

__all__ = [
    "set_plot_style",
    "plot_stats",
    "compare_binned_stats",
    "visualize_the_data",
    "plot_ensemble_losses",
]


def set_plot_style():
    plt.rc("font", size=14)
    plt.rc("axes", labelsize=14, titlesize=14)
    plt.rc("legend", fontsize=14)
    plt.rc("xtick", labelsize=10)
    plt.rc("ytick", labelsize=10)


def plot_stats(
    stats, redshift_stats, imag_stats, y_train, y_pred, redshift_bins, imag_bins, i_mag_data, save_path=None
):
    Y = np.asarray(y_train)
    Y2 = y_pred.flatten()
    dz = (Y2 - Y) / (1 + Y)

    fig = plt.figure(figsize=(15, 5))  # ← reduced height; 15/3 = 5 per column → square
    gs = gridspec.GridSpec(2, 3, figure=fig, height_ratios=[2, 1], hspace=0.05, wspace=0.35)

    ax_scatter = fig.add_subplot(gs[:, 0])
    ax_zstats = fig.add_subplot(gs[0, 1])
    ax_zdz = fig.add_subplot(gs[1, 1], sharex=ax_zstats)
    ax_istats = fig.add_subplot(gs[0, 2])
    ax_idz = fig.add_subplot(gs[1, 2], sharex=ax_istats)

    # Only scatter panel needs to be square
    ax_scatter.set_box_aspect(1)  # ← only this one

    # ── Panel 1: scatter plot ────────────────────────────────────────────────
    mean, _mean_err, std, outlier_rate, abs_outlier_rate = stats
    mean, std, outlier_rate, abs_outlier_rate = (
        round(mean, 4),
        round(std, 4),
        round(outlier_rate, 4),
        round(abs_outlier_rate, 4),
    )

    # ax_scatter.scatter(y_train, y_pred, s=0.1, color='k')
    bin_edges = np.linspace(0, 3, 101)
    ax_scatter.hist2d(
        y_train,
        y_pred,
        bins=(bin_edges, bin_edges),
        norm=colors.LogNorm(),
        cmap="gray",
    )
    ax_scatter.plot([0, 3], [0, 3], "r-")
    ax_scatter.plot([0, 3], [0 - 3 * std, 3 - 3 * std], "r--")
    ax_scatter.plot([0, 3], [0 + 3 * std, 3 + 3 * std], "r--")
    ax_scatter.set_xlim([0, 3])
    ax_scatter.set_ylim([0, 3])
    ax_scatter.set_xlabel("truth redshift")
    ax_scatter.set_ylabel("predicted redshift")

    abs_out_thresh = 0.2
    label = (
        rf"$\Delta z = {mean}$"
        + "\n"
        + rf"$\sigma z = {std}$"
        + "\n"
        + rf"outlier rate (>3$\sigma$) = {outlier_rate}"
        + "\n"
        + f"outlier rate (>{abs_out_thresh}) = {abs_outlier_rate}"
    )
    ax_scatter.text(0.1, 1.8, label, fontsize=12, bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))

    # ── Panel 2: redshift statistics ─────────────────────────────────────────
    N_bin = len(redshift_bins) - 1
    z_mean = np.zeros(N_bin)
    z_err = np.zeros(N_bin)
    z_std = np.zeros(N_bin)
    z_out = np.zeros(N_bin)
    z_abs = np.zeros(N_bin)
    for i in range(N_bin):
        z_mean[i], z_err[i], z_std[i], z_out[i], z_abs[i] = redshift_stats[i]

    z_centres = (redshift_bins[1:] + redshift_bins[:-1]) / 2
    ax_zstats.plot(z_centres, z_mean, "-", label="bias")
    ax_zstats.plot(z_centres, z_std, "-", label=r"$\sigma_z$")
    ax_zstats.plot(z_centres, z_out, "-", label="outlier rate")
    ax_zstats.legend()
    ax_zstats.set_ylabel("statistics")
    plt.setp(ax_zstats.get_xticklabels(), visible=False)

    # ax_zdz.scatter(Y, dz, s=0.1, color='k')
    dzbin_edges = np.linspace(-0.7, 0.7, 51)
    ax_zdz.hist2d(
        Y,
        dz,
        bins=(bin_edges, dzbin_edges),
        norm=colors.LogNorm(),
        cmap="gray",
    )
    ax_zdz.set_xlabel("redshift")
    ax_zdz.set_ylabel(r"$(z_{pred} - z_{true})/(1 + z_{true})$")
    ax_zdz.set_ylim([-0.5, 0.5])  # ← y-axis limit on lower panel

    # ── Panel 3: i-magnitude statistics ──────────────────────────────────────
    N_bin = len(imag_bins) - 1
    i_mean = np.zeros(N_bin)
    i_err = np.zeros(N_bin)
    i_std = np.zeros(N_bin)
    i_out = np.zeros(N_bin)
    i_abs = np.zeros(N_bin)
    for i in range(N_bin):
        i_mean[i], i_err[i], i_std[i], i_out[i], i_abs[i] = imag_stats[i]

    i_centres = (imag_bins[1:] + imag_bins[:-1]) / 2
    ax_istats.plot(i_centres, i_mean, "-", label="bias")
    ax_istats.plot(i_centres, i_std, "-", label=r"$\sigma_z$")
    ax_istats.plot(i_centres, i_out, "-", label="outlier rate")
    ax_istats.legend()
    ax_istats.set_ylabel("statistics")
    plt.setp(ax_istats.get_xticklabels(), visible=False)

    # ax_idz.scatter(i_mag_data, dz, s=0.1, color='k')
    ibins = np.linspace(imag_bins[0], imag_bins[-1], 51)
    ax_idz.hist2d(
        i_mag_data,
        dz,
        bins=(ibins, dzbin_edges),
        norm=colors.LogNorm(),
        cmap="gray",
    )
    ax_idz.set_xlabel("i magnitude")
    ax_idz.set_ylabel(r"$(z_{pred} - z_{true})/(1 + z_{true})$")
    ax_idz.set_ylim([-0.5, 0.5])  # ← y-axis limit on lower panel

    plt.suptitle("Photo-z Statistics", fontsize=14)
    plt.tight_layout()
    plt.show()

    # Save if filename provided
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {save_path}")


def compare_binned_stats(redshift_bins, imag_bins, redshift_stats1, imag_stats1, redshift_stats2, imag_stats2):
    fig, axarr = plt.subplots(1, 2, figsize=(10, 4))

    xlabels = ["redshifts", "i magnitude"]

    for j, bins, stats1, stats2 in zip(
        [0, 1], [redshift_bins, imag_bins], [redshift_stats1, imag_stats1], [redshift_stats2, imag_stats2]
    ):
        plt.sca(axarr[j])
        N_bin = len(bins) - 1
        xx = (bins[1:] + bins[:-1]) / 2

        mean1 = np.zeros(N_bin)
        std1 = np.zeros(N_bin)
        outlier_rate1 = np.zeros(N_bin)
        for i in range(N_bin):
            mean1[i], _, std1[i], outlier_rate1[i], _ = stats1[i]

        mean2 = np.zeros(N_bin)
        std2 = np.zeros(N_bin)
        outlier_rate2 = np.zeros(N_bin)
        for i in range(N_bin):
            mean2[i], _, std2[i], outlier_rate2[i], _ = stats2[i]

        plt.plot(xx, mean1, "-", label="bias", color="C0")
        plt.plot(xx, std1, "-", label=r"$\sigma_z$", color="C1")
        plt.plot(xx, outlier_rate1, "-", label="outlier rate", color="C2")

        plt.plot(xx, mean2, "--", color="C0")
        plt.plot(xx, std2, "--", color="C1")
        plt.plot(xx, outlier_rate2, "--", color="C2")
        plt.legend()
        plt.ylabel("statistics")
        plt.xlabel(xlabels[j])
        plt.grid()
    plt.tight_layout()


def visualize_the_data(X, Y, lambda_bin_centers, title="Example data vector", n_examples=5):
    """Plot a handful of example data vectors: the amplitude-weighted, binned curve
    (channel 0) against wavelength, with each example's coverage channel (channel 2)
    overlaid to show where photometry actually backs the curve."""
    for i in range(n_examples):
        z = Y[i]
        plt.plot(lambda_bin_centers, X[i, :, 0], label=f"z={round(z,2)}")
        plt.plot(lambda_bin_centers, X[i, :, 2], color="k")
    plt.plot(lambda_bin_centers, X[i, :, 2], color="k", label="coverage")
    plt.ylabel("mag (normed)")
    plt.xlabel("wavelength")
    plt.legend()
    plt.title(title)


def plot_ensemble_losses(histories, ylim=(0, 0.1)):
    n_folds = len(histories)

    # Define colours and line styles explicitly
    style = {
        "loss": {"color": "blue", "linestyle": "-"},  # solid blue
        "val_loss": {"color": "blue", "linestyle": "--"},  # dashed blue
        "mae": {"color": "red", "linestyle": "-"},  # solid red
        "val_mae": {"color": "red", "linestyle": "--"},  # dashed red
    }

    # n_folds panels + 1 summary panel
    fig, axes = plt.subplots(1, n_folds + 1, figsize=(5 * (n_folds + 1), 5), sharey=True)

    # --- Individual fold panels ---
    for fold, (ax, history) in enumerate(zip(axes[:-1], histories)):
        for metric, s in style.items():
            if metric in history.history:
                ax.plot(history.history[metric], label=metric, color=s["color"], linestyle=s["linestyle"], linewidth=2)
        ax.set_title(f"Member {fold+1}")
        ax.set_xlabel("Epoch")
        ax.set_ylim(ylim)
        ax.grid(True)
        ax.legend()
        if fold == 0:
            ax.set_ylabel("Loss / MAE")

    # --- Summary panel ---
    ax_summary = axes[-1]

    for metric, s in style.items():
        # collect this metric across all folds that have it
        all_folds = [h.history[metric] for h in histories if metric in h.history]

        if len(all_folds) == 0:
            continue

        # pad to same length (early stopping may stop folds at different epochs)
        max_len = max(len(f) for f in all_folds)
        padded = [np.pad(f, (0, max_len - len(f)), mode="edge") for f in all_folds]

        mean = np.mean(padded, axis=0)
        std = np.std(padded, axis=0)
        epochs = np.arange(max_len)

        # plot mean line
        ax_summary.plot(epochs, mean, label=metric, color=s["color"], linestyle=s["linestyle"], linewidth=2)

        # plot shaded std band around mean
        ax_summary.fill_between(epochs, mean - std, mean + std, color=s["color"], alpha=0.15)

    ax_summary.set_title("Summary (mean ± std)")
    ax_summary.set_xlabel("Epoch")
    ax_summary.set_ylim(ylim)
    ax_summary.grid(True)
    ax_summary.legend()

    plt.suptitle("Ensemble Model — Loss per Member", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()
