import warnings

import numpy as np
import pandas as pd

try:
    import qp

    QP_INSTALLED = True
except Exception:  # pragma: no cover - qp missing or unimportable in this env
    QP_INSTALLED = False

__all__ = ["QP_INSTALLED", "package_pdfs", "predictions_mean_std"]


def package_pdfs(zgrid, pdfs, ids=None):
    """
    Package per-object p(z) evaluated on a redshift grid, pdfs of shape (N, len(zgrid)).

    If ``qp`` is installed, returns a ``qp.Ensemble`` interpolated on the grid (``qp.interp``), with
    ``object_id`` and ``zmode`` (the grid point where the p(z) peaks) as ancillary data. If ``qp`` is not
    installed, falls back to a ``pandas.DataFrame`` with "object_id", "mean", "std" columns.
    """
    zgrid = np.asarray(zgrid, dtype=float)
    pdfs = np.asarray(pdfs, dtype=float)
    ids = np.arange(len(pdfs)) if ids is None else np.asarray(ids)

    if not QP_INSTALLED:
        warnings.warn(
            "qp is not installed; returning a pandas DataFrame of id/mean/std instead "
            "of a qp Ensemble. Install qp-prob to get p(z) ensembles.",
            stacklevel=2,
        )
        weights = pdfs / pdfs.sum(axis=1, keepdims=True)
        mean = weights @ zgrid
        std = np.sqrt(np.maximum(weights @ zgrid**2 - mean**2, 0))
        return pd.DataFrame({"object_id": ids, "mean": mean, "std": std})

    ancil = {"object_id": ids, "zmode": zgrid[pdfs.argmax(axis=1)]}
    return qp.Ensemble(qp.interp, data={"xvals": zgrid, "yvals": pdfs}, ancil=ancil)


def predictions_mean_std(result):
    """
    Extract per-object predicted redshift mean and std from the output of
    :func:`package_pdfs` (a qp Ensemble or the fallback DataFrame).

    Returns
    -------
    y_pred_mean, y_pred_std : np.ndarray
        Arrays of shape (N, 1).
    """
    if isinstance(result, pd.DataFrame):
        return result["mean"].to_numpy().reshape(-1, 1), result["std"].to_numpy().reshape(-1, 1)

    return np.asarray(result.mean()).reshape(-1, 1), np.asarray(result.std()).reshape(-1, 1)
