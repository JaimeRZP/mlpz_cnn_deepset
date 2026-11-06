import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # silence TensorFlow's start-up logging

import numpy as np
import pandas as pd
import pytest

# Synthetic photometric system: 4 overlapping triangular filters spanning 3000-9000 Å
BAND_CENTRES = {"g": 4000.0, "r": 5500.0, "i": 7000.0, "z": 8500.0}
BAND_HALF_WIDTH = 1000.0


def triangle_curve(centre, half_width=BAND_HALF_WIDTH, n=101):
    lam = np.linspace(centre - half_width, centre + half_width, n)
    return np.column_stack([lam, 1.0 - np.abs(lam - centre) / half_width])


@pytest.fixture
def filter_curves():
    return {b: triangle_curve(c) for b, c in BAND_CENTRES.items()}


@pytest.fixture
def bands(tmp_path, filter_curves):
    """band -> (catalogue column, filter file), with the filter files written to tmp_path."""
    out = {}
    for b, curve in filter_curves.items():
        path = tmp_path / f"{b}.res"
        np.savetxt(path, curve)
        out[b] = (f"mag_{b}", str(path))
    return out


@pytest.fixture
def catalog():
    """20 objects with magnitudes, a redshift and one non-detection (NaN) in g."""
    rng = np.random.default_rng(0)
    n = 20
    df = pd.DataFrame({f"mag_{b}": 23.0 - 0.3 * k + rng.normal(0, 0.1, n) for k, b in enumerate(BAND_CENTRES)})
    df.loc[3, "mag_g"] = np.nan
    df["redshift"] = rng.uniform(0.05, 2.0, n)
    return df


@pytest.fixture
def feature_config():
    return {
        "nondetect_value": np.inf,
        "n_lambda": 200,
        "lambda_range": (3000.0, 9500.0),
        "n_bins": 8,
        "apply_stretch": False,
        "c": 0.8,
        "k": 20,
    }
