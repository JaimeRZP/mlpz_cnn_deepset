import warnings

import numpy as np
import pandas as pd

try:
    import qp

    QP_INSTALLED = True
except Exception:  # pragma: no cover - qp missing or unimportable in this env
    QP_INSTALLED = False

__all__ = ["QP_INSTALLED", "package_predictions", "save_predictions", "predictions_mean_std"]


def package_predictions(y_pred_mean, y_pred_std, ids=None):
    """
    Package per-object predicted redshift mean/std into a p(z) representation.

    Each object's predicted redshift is represented as a Gaussian centred on
    ``y_pred_mean`` with width ``y_pred_std``.

    If ``qp`` is installed, returns a ``qp.Ensemble`` of Gaussians (one per
    object), with ``object_id`` and ``zmode`` (the Gaussian mode, i.e. the
    mean) stored as ancillary data, following the PZ data challenge
    convention. If ``qp`` is not installed, falls back to a
    ``pandas.DataFrame`` with "object_id", "mean", "std" columns.

    Parameters
    ----------
    y_pred_mean, y_pred_std : array-like
        Per-object predicted redshift mean and std.
    ids : array-like, optional
        Per-object identifiers. Defaults to the sample order index (0..N-1).

    Returns
    -------
    qp.Ensemble or pandas.DataFrame
    """
    y_pred_mean = np.asarray(y_pred_mean).reshape(-1)
    y_pred_std = np.asarray(y_pred_std).reshape(-1)
    ids = np.arange(len(y_pred_mean)) if ids is None else np.asarray(ids)

    if not QP_INSTALLED:
        warnings.warn(
            "qp is not installed; returning a pandas DataFrame of id/mean/std instead "
            "of a qp Ensemble. Install qp-prob to get p(z) ensembles.",
            stacklevel=2,
        )
        return pd.DataFrame({"object_id": ids, "mean": y_pred_mean, "std": y_pred_std})

    data = {"loc": y_pred_mean.reshape(-1, 1), "scale": y_pred_std.reshape(-1, 1)}
    ancil = {"object_id": ids, "zmode": y_pred_mean}
    return qp.Ensemble(qp.stats.norm, data=data, ancil=ancil)


def predictions_mean_std(result):
    """
    Extract per-object predicted redshift mean and std from the output of
    :func:`package_predictions` (a qp Ensemble or the fallback DataFrame).

    Returns
    -------
    y_pred_mean, y_pred_std : np.ndarray
        Arrays of shape (N, 1), matching the old ``ensemble_predict`` output.
    """
    if isinstance(result, pd.DataFrame):
        return result["mean"].to_numpy().reshape(-1, 1), result["std"].to_numpy().reshape(-1, 1)

    return np.asarray(result.mean()).reshape(-1, 1), np.asarray(result.std()).reshape(-1, 1)


def save_predictions(y_pred_mean, y_pred_std, output_filename, ids=None):
    """
    Package predictions (see :func:`package_predictions`) and write them to disk.

    Writes a qp Ensemble via ``write_to`` if qp is installed, otherwise writes
    the fallback id/mean/std DataFrame as CSV.
    """
    result = package_predictions(y_pred_mean, y_pred_std, ids=ids)

    if QP_INSTALLED:
        result.write_to(output_filename)
    else:
        result.to_csv(output_filename, index=False)

    return result
