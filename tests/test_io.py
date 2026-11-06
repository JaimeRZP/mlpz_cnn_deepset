import h5py
import numpy as np
import pandas as pd
import pytest

from znn.io import catalog_to_mags, load_filter_curves, read_catalog


@pytest.fixture
def small_df():
    return pd.DataFrame({"mag_g": [22.0, np.nan, 23.0], "mag_r": [21.5, 22.0, np.inf], "redshift": [0.1, 0.5, 1.0]})


def test_read_catalog_hdf5(tmp_path, small_df):
    path = tmp_path / "cat.hdf5"
    with h5py.File(path, "w") as f:
        for col in small_df:
            f[col] = small_df[col].to_numpy()
        f.create_group("meta")  # groups are skipped
    pd.testing.assert_frame_equal(read_catalog(path), small_df, check_like=True)
    assert list(read_catalog(path, columns=["redshift"]).columns) == ["redshift"]


def test_read_catalog_parquet(tmp_path, small_df):
    pytest.importorskip("pyarrow")
    path = tmp_path / "cat.parquet"
    small_df.to_parquet(path)
    pd.testing.assert_frame_equal(read_catalog(path), small_df)
    assert list(read_catalog(path, columns=["mag_g"]).columns) == ["mag_g"]


def test_read_catalog_unknown_format(tmp_path):
    with pytest.raises(ValueError, match="Unsupported"):
        read_catalog(tmp_path / "cat.csv")


def test_load_filter_curves_keeps_band_order(bands, filter_curves):
    curves = load_filter_curves(bands)
    assert list(curves) == list(bands)
    for b in bands:
        np.testing.assert_allclose(curves[b], filter_curves[b])


def test_catalog_to_mags(small_df):
    bands = {"r": ("mag_r", None), "g": ("mag_g", None), "y": ("mag_y", None)}
    mags = catalog_to_mags(small_df, bands)
    assert mags.shape == (3, 3)
    np.testing.assert_array_equal(mags[0], [21.5, 22.0, np.inf])  # rows in the order of bands
    np.testing.assert_array_equal(mags[1], [22.0, np.inf, 23.0])  # NaN non-detection -> inf (unobserved)
    assert np.all(np.isinf(mags[2]))  # missing column -> unobserved
    mags = catalog_to_mags(small_df, bands, nondetect_value=30.0)
    assert mags[1, 1] == 30.0
