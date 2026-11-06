"""
Tests of the features built by znn.data.catalog_to_XY, on a toy photometric system of box filters aligned with the
wavelength bins, so the discretised SED can be written down by hand: each band fills its own bins, every other bin is
zero.

Grid: lambda_range (1000, 9000) Å in N_BINS = 8 bins of 1000 Å; bin k spans [1000 (k + 1), 1000 (k + 2)].
"""

import numpy as np
import pandas as pd
import pytest

from znn.data import catalog_to_XY

N_BINS = 8
CONFIG = {
    "nondetect_value": np.inf,
    "n_lambda": 800,  # spacing 8000/799 Å: no grid point falls exactly on an inner bin edge
    "lambda_range": (1000.0, 9000.0),
    "n_bins": N_BINS,
    "apply_stretch": False,
    "c": 0.8,
    "k": 20,
}


def bin_edges(k):
    return 1000.0 * (k + 1), 1000.0 * (k + 2)


def box(lo, hi):
    """Transmission 1 on [lo, hi], 0 outside."""
    return np.array([[lo, 1.0], [hi, 1.0]])


def write_bands(tmp_path, curves):
    bands = {}
    for b, curve in curves.items():
        path = tmp_path / f"{b}.res"
        np.savetxt(path, curve)
        bands[b] = (f"mag_{b}", str(path))
    return bands


# each band fills exactly one bin; bins 0, 2, 5 and 7 have no filter
BAND_BIN = {"u": 1, "g": 3, "r": 4, "i": 6}


@pytest.fixture
def box_bands(tmp_path):
    return write_bands(tmp_path, {b: box(*bin_edges(k)) for b, k in BAND_BIN.items()})


@pytest.fixture
def box_catalog():
    return pd.DataFrame(
        {
            "mag_u": [25.0, 24.0, 23.0, 22.0],
            "mag_g": [24.0, 24.0, np.nan, 22.0],  # object 2: g not detected
            "mag_r": [23.0, 24.0, 22.0, 22.0],
            "mag_i": [22.0, 24.0, 21.0, 22.0],  # the reference band
            "redshift": [0.1, 0.5, 1.0, 2.0],
        }
    )


def expected_sed(row, bands_bins=BAND_BIN, ref="i"):
    """By hand: in each band's bin, its magnitude minus the reference band's; zero in every other bin."""
    sed = np.zeros(N_BINS)
    for b, k in bands_bins.items():
        m = row[f"mag_{b}"]
        if np.isfinite(m):
            sed[k] = m - row[f"mag_{ref}"]
    return sed


def expected_coverage(row, bands_bins=BAND_BIN):
    cov = np.zeros(N_BINS)
    for b, k in bands_bins.items():
        cov[k] = float(np.isfinite(row[f"mag_{b}"]))
    return cov


# --- b) shape ----------------------------------------------------------------------------------------------------


def test_feature_matrix_shape(box_catalog, box_bands):
    X, Y = catalog_to_XY(box_catalog, box_bands, "i", CONFIG)
    assert X.shape == (len(box_catalog), N_BINS, 3)  # (objects, wavelength bins, channels)
    assert Y.shape == (len(box_catalog),)
    np.testing.assert_array_equal(Y, box_catalog["redshift"])
    assert np.all(np.isfinite(X))


@pytest.mark.parametrize("n_bins", [4, 8, 16, 32])
def test_shape_follows_n_bins(box_catalog, box_bands, n_bins):
    X, _ = catalog_to_XY(box_catalog, box_bands, "i", {**CONFIG, "n_bins": n_bins})
    assert X.shape == (len(box_catalog), n_bins, 3)
    np.testing.assert_allclose(X[0, :, 1], np.linspace(0, 1, n_bins))


def test_shape_without_redshift(box_catalog, box_bands):
    X, Y = catalog_to_XY(box_catalog.drop(columns="redshift"), box_bands, "i", CONFIG)
    assert X.shape == (len(box_catalog), N_BINS, 3)
    assert Y is None


# --- a) the discretised SED --------------------------------------------------------------------------------------


@pytest.mark.parametrize("obj", range(4))
def test_sed_is_band_colours_in_filter_bins_and_zero_elsewhere(box_catalog, box_bands, obj):
    X, _ = catalog_to_XY(box_catalog, box_bands, "i", CONFIG)
    row = box_catalog.iloc[obj]
    np.testing.assert_allclose(X[obj, :, 0], expected_sed(row), atol=1e-12)


def test_sed_spelled_out(box_catalog, box_bands):
    """Object 0 (u, g, r, i = 25, 24, 23, 22), relative to i: 3, 2, 1, 0 in bins 1, 3, 4, 6."""
    X, _ = catalog_to_XY(box_catalog, box_bands, "i", CONFIG)
    np.testing.assert_allclose(X[0, :, 0], [0, 3, 0, 2, 1, 0, 0, 0], atol=1e-12)
    np.testing.assert_allclose(X[0, :, 2], [0, 1, 0, 1, 1, 0, 1, 0], atol=1e-12)


def test_flat_sed_is_zero_everywhere(box_catalog, box_bands):
    """Object 1 has the same magnitude in every band: nothing left after subtracting the reference band."""
    X, _ = catalog_to_XY(box_catalog, box_bands, "i", CONFIG)
    np.testing.assert_allclose(X[1, :, 0], 0.0, atol=1e-12)


@pytest.mark.parametrize("obj", range(4))
def test_coverage_is_one_in_observed_filter_bins_and_zero_elsewhere(box_catalog, box_bands, obj):
    X, _ = catalog_to_XY(box_catalog, box_bands, "i", CONFIG)
    np.testing.assert_allclose(X[obj, :, 2], expected_coverage(box_catalog.iloc[obj]), atol=1e-12)


def test_non_detection_empties_its_bin(box_catalog, box_bands):
    """Object 2 has no g: bin 3 is zero in both the SED and the coverage, the other bands are unaffected."""
    X, _ = catalog_to_XY(box_catalog, box_bands, "i", CONFIG)
    assert X[2, BAND_BIN["g"], 0] == pytest.approx(0, abs=1e-12)
    assert X[2, BAND_BIN["g"], 2] == pytest.approx(0, abs=1e-12)
    for b in ("u", "r", "i"):
        assert X[2, BAND_BIN[b], 2] == pytest.approx(1, abs=1e-12)


def test_wavelength_label_channel(box_catalog, box_bands):
    X, _ = catalog_to_XY(box_catalog, box_bands, "i", CONFIG)
    np.testing.assert_allclose(X[:, :, 1], np.tile(np.linspace(0, 1, N_BINS), (len(box_catalog), 1)))


def test_band_missing_from_catalog_is_unobserved(box_catalog, box_bands):
    X, _ = catalog_to_XY(box_catalog.drop(columns="mag_u"), box_bands, "i", CONFIG)
    np.testing.assert_allclose(X[:, BAND_BIN["u"], 0], 0, atol=1e-12)
    np.testing.assert_allclose(X[:, BAND_BIN["u"], 2], 0, atol=1e-12)


def test_wide_filter_fills_all_its_bins(tmp_path):
    """A band spanning bins 2-4 puts its colour in all three; the rest stay zero."""
    bands = write_bands(tmp_path, {"g": box(bin_edges(2)[0], bin_edges(4)[1]), "i": box(*bin_edges(6))})
    df = pd.DataFrame({"mag_g": [23.5], "mag_i": [22.0]})
    X, _ = catalog_to_XY(df, bands, "i", CONFIG)
    np.testing.assert_allclose(X[0, :, 0], [0, 0, 1.5, 1.5, 1.5, 0, 0, 0], atol=1e-12)
    np.testing.assert_allclose(X[0, :, 2], [0, 0, 1, 1, 1, 0, 1, 0], atol=1e-12)


def test_overlapping_filters_share_their_bin(tmp_path):
    """Two bands covering the same bin each own half of it: the SED there is the mean of their colours."""
    bands = write_bands(
        tmp_path,
        {"g": box(bin_edges(2)[0], bin_edges(3)[1]), "r": box(*bin_edges(3)), "i": box(*bin_edges(6))},
    )
    df = pd.DataFrame({"mag_g": [24.0], "mag_r": [23.0], "mag_i": [22.0]})
    X, _ = catalog_to_XY(df, bands, "i", CONFIG)
    np.testing.assert_allclose(X[0, :, 0], [0, 0, 2.0, 1.5, 0, 0, 0, 0], atol=1e-12)
    np.testing.assert_allclose(X[0, :, 2], [0, 0, 1, 1, 0, 0, 1, 0], atol=1e-12)
