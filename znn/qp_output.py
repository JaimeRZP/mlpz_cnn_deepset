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
    "package_samples",
    "save_predictions",
    "predictions_mean_std",
]

def package_samples(samples, ids=None):
    """
    Package per-object redshift samples (e.g. every ensemble member's prediction on every noise
    realization) into a Gaussian p(z) with the samples' mean and standard deviation.

    If ``qp`` is installed, returns a ``qp.Ensemble`` of Gaussians (one per object), with ``object_id``
    and ``zmode`` (the Gaussian mode, i.e. the mean) stored as ancillary data, following the PZ data
    challenge convention. If ``qp`` is not installed, falls back to a ``pandas.DataFrame`` with
    "object_id", "mean", "std" columns.

    Parameters
    ----------
    samples : array-like, shape (N, n_samples)
        Redshift samples per object.
    ids : array-like, optional
        Per-object identifiers. Defaults to the sample order index (0..N-1).

    Returns
    -------
    qp.Ensemble or pandas.DataFrame
    """
    samples = np.asarray(samples, dtype=float)
    mean = samples.mean(axis=1)
    # floored so the Gaussian is defined when all samples coincide (a single member and no errors)
    std = np.maximum(samples.std(axis=1), 1e-6)
    ids = np.arange(len(samples)) if ids is None else np.asarray(ids)

    if not QP_INSTALLED:
        warnings.warn(
            "qp is not installed; returning a pandas DataFrame of id/mean/std instead "
            "of a qp Ensemble. Install qp-prob to get p(z) ensembles.",
            stacklevel=2,
        )
        return pd.DataFrame({"object_id": ids, "mean": mean, "std": std})

    data = {"loc": mean.reshape(-1, 1), "scale": std.reshape(-1, 1)}
    ancil = {"object_id": ids, "zmode": mean}
    return qp.Ensemble(qp.stats.norm, data=data, ancil=ancil)


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


def save_predictions(samples, output_filename, ids=None):
    """
    Package per-object redshift samples as a Gaussian p(z) (see :func:`package_samples`) and write
    them to disk.

    Writes a qp Ensemble via ``write_to`` if qp is installed, otherwise writes
    the fallback id/mean/std DataFrame as CSV.
    """
    result = package_samples(samples, ids=ids)

    if QP_INSTALLED:
        result.write_to(output_filename)
    else:
        result.to_csv(output_filename, index=False)

    return result
