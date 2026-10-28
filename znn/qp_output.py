import warnings

import numpy as np
import pandas as pd

try:
    import qp

    QP_INSTALLED = True
except Exception:  # pragma: no cover - qp missing or unimportable in this env
    QP_INSTALLED = False

__all__ = [
    "QP_INSTALLED",
    "DEFAULT_Z_BINS",
    "package_samples",
    "save_predictions",
    "predictions_mean_std",
]

# Redshift bin edges of the histogram p(z) built by package_samples
DEFAULT_Z_BINS = np.linspace(0.0, 3.0, 301)


def package_samples(samples, ids=None, bins=DEFAULT_Z_BINS):
    """
    Package per-object redshift samples (e.g. every ensemble member's prediction on every noise
    realization) into a histogram p(z).

    Each object's p(z) is the normalised histogram of its samples on ``bins``. Samples outside the bins
    are counted in the first or last bin, so every object keeps all of its probability. If ``qp`` is
    installed, returns a ``qp.Ensemble`` of histograms with ``object_id`` and ``zmode`` (the centre of
    the most populated bin) as ancillary data, following the PZ data challenge convention. If ``qp`` is
    not installed, falls back to a ``pandas.DataFrame`` with "object_id", "mean", "std" columns.

    Parameters
    ----------
    samples : array-like, shape (N, n_samples)
        Redshift samples per object.
    ids : array-like, optional
        Per-object identifiers. Defaults to the sample order index (0..N-1).
    bins : array-like
        Histogram bin edges.

    Returns
    -------
    qp.Ensemble or pandas.DataFrame
    """
    samples = np.asarray(samples, dtype=float)
    bins = np.asarray(bins, dtype=float)
    ids = np.arange(len(samples)) if ids is None else np.asarray(ids)

    if not QP_INSTALLED:
        warnings.warn(
            "qp is not installed; returning a pandas DataFrame of id/mean/std instead "
            "of a qp Ensemble. Install qp-prob to get p(z) ensembles.",
            stacklevel=2,
        )
        return pd.DataFrame({"object_id": ids, "mean": samples.mean(axis=1), "std": samples.std(axis=1)})

    n_bins = len(bins) - 1
    which = np.clip(np.searchsorted(bins, samples, side="right") - 1, 0, n_bins - 1)
    counts = np.zeros((len(samples), n_bins))
    np.add.at(counts, (np.arange(len(samples))[:, None], which), 1)
    pdfs = counts / (samples.shape[1] * np.diff(bins))

    centres = 0.5 * (bins[1:] + bins[:-1])
    ancil = {"object_id": ids, "zmode": centres[counts.argmax(axis=1)]}
    return qp.Ensemble(qp.hist, data={"bins": bins, "pdfs": pdfs}, ancil=ancil)


def predictions_mean_std(result):
    """
    Extract per-object predicted redshift mean and std from the output of
    :func:`package_samples` (a qp Ensemble or the fallback DataFrame).

    Returns
    -------
    y_pred_mean, y_pred_std : np.ndarray
        Arrays of shape (N, 1), matching the old ``ensemble_predict`` output.
    """
    if isinstance(result, pd.DataFrame):
        return result["mean"].to_numpy().reshape(-1, 1), result["std"].to_numpy().reshape(-1, 1)

    return np.asarray(result.mean()).reshape(-1, 1), np.asarray(result.std()).reshape(-1, 1)


def save_predictions(samples, output_filename, ids=None, bins=DEFAULT_Z_BINS):
    """
    Package per-object redshift samples as a histogram p(z) (see :func:`package_samples`) and write
    them to disk.

    Writes a qp Ensemble via ``write_to`` if qp is installed, otherwise writes
    the fallback id/mean/std DataFrame as CSV.
    """
    result = package_samples(samples, ids=ids, bins=bins)

    if QP_INSTALLED:
        result.write_to(output_filename)
    else:
        result.to_csv(output_filename, index=False)

    return result
