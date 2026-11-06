import numpy as np
import pytest
import tensorflow as tf
from scipy import stats
from tensorflow.keras import layers, models

import znn
from znn.data import DEFAULT_FEATURE_CONFIG, catalog_to_XY
from znn.models import ToRedshift, _student_t_nll, build_model, load_model_file, predict, save_model_file

qp = pytest.importorskip("qp")


def _make_model(n_bins, mean=0.5, std=0.3, nu=5.0, seed=0):
    """An untrained (F, G, nu) with the layout train_model returns: F ends in ToRedshift."""
    tf.keras.utils.set_random_seed(seed)
    base = build_model(input_shape=(n_bins, 3))
    F = models.Model(base.inputs[0], ToRedshift(mean, std)(base.outputs[0]))
    G = build_model(input_shape=(n_bins, 3))
    return F, G, nu


# --- network dimensions ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("n_bins", [8, 32])
def test_build_model_dimensions(n_bins):
    """(n_bins, 3) features in, one number out; 3 stages of 32/64/128 channels, halving the length after two."""
    model = build_model(input_shape=(n_bins, 3))
    assert model.input_shape == (None, n_bins, 3)
    assert model.output_shape == (None, 1)

    pools = [layer for layer in model.layers if isinstance(layer, layers.MaxPooling1D)]
    assert [tuple(p.output.shape[1:]) for p in pools] == [(n_bins // 2, 32), (n_bins // 4, 64)]
    adds = [layer for layer in model.layers if isinstance(layer, layers.Add)]
    assert [a.output.shape[-1] for a in adds] == [32, 32, 64, 64, 128, 128]  # 2 residual blocks per stage
    assert isinstance(model.layers[-1], layers.Dense) and model.layers[-1].units == 1

    x = np.zeros((5, n_bins, 3), dtype=np.float32)
    assert model.predict(x, verbose=0).shape == (5, 1)


def test_parameter_count_independent_of_n_bins():
    """Global pooling before the head: the weights do not depend on the number of wavelength bins."""
    assert build_model((8, 3)).count_params() == build_model((32, 3)).count_params() == 273105


def test_pretrained_matches_default_feature_config():
    F = znn.load_pretrained()
    assert F.input_shape == (None, DEFAULT_FEATURE_CONFIG["n_bins"], 3)
    assert F.output_shape == (None, 1)
    assert isinstance(F.layers[-1], ToRedshift)
    assert F.count_params() == build_model((DEFAULT_FEATURE_CONFIG["n_bins"], 3)).count_params()


# --- building blocks ---------------------------------------------------------------------------------------------


def test_to_redshift():
    layer = ToRedshift(0.5, 0.2)
    y = np.array([[-1.0], [0.0], [2.0]], dtype=np.float32)
    np.testing.assert_allclose(layer(y).numpy(), np.expm1(y * 0.2 + 0.5), rtol=1e-6)
    clone = ToRedshift.from_config(layer.get_config())
    assert (clone.mean, clone.std) == (0.5, 0.2)


def test_student_t_nll_is_scipy_logpdf_up_to_constant():
    nu = 5.0
    rng = np.random.default_rng(0)
    r = rng.normal(0, 1, 50).astype(np.float32)
    s = rng.uniform(0.1, 2.0, 50).astype(np.float32)
    loss = _student_t_nll(nu)
    ours = np.array([loss(tf.constant([ri]), tf.constant([np.log(si**2)])).numpy() for ri, si in zip(r, s)])
    scipy_nll = -stats.t.logpdf(r, nu, scale=s)
    offset = scipy_nll - ours
    np.testing.assert_allclose(offset, offset[0], atol=1e-5)  # differ only by the normalising constant
    expected = -stats.t.logpdf(0.0, nu)  # the constant: the NLL of a zero residual at unit scale
    assert offset[0] == pytest.approx(expected, abs=1e-5)


def test_student_t_nll_minimised_at_true_scale():
    """Over many residuals drawn with scale s, the mean NLL is lowest when G predicts log s^2."""
    nu, s = 5.0, 0.3
    r = tf.constant(stats.t.rvs(nu, scale=s, size=200_000, random_state=1).astype(np.float32))
    loss = _student_t_nll(nu)
    grid = np.linspace(0.2, 0.4, 41)
    values = [loss(r, tf.fill(r.shape, float(np.log(g**2)))).numpy() for g in grid]
    assert grid[np.argmin(values)] == pytest.approx(s, abs=0.01)


# --- save / load -------------------------------------------------------------------------------------------------


def test_save_load_roundtrip(tmp_path, catalog, bands, feature_config):
    model = _make_model(feature_config["n_bins"])
    path = tmp_path / "model.pkl"
    save_model_file(path, model, bands, "i", feature_config)
    loaded, features = load_model_file(path)

    assert loaded[2] == model[2]
    assert set(features) == {"bands", "ref_band", "config", "filter_curves"}
    assert features["ref_band"] == "i" and features["config"] == feature_config
    assert list(features["filter_curves"]) == list(bands)

    X, _ = catalog_to_XY(catalog, bands, "i", feature_config)
    for orig, back in zip(model[:2], loaded[:2]):
        np.testing.assert_allclose(back.predict(X, verbose=0), orig.predict(X, verbose=0), rtol=1e-6)
    assert isinstance(loaded[0].layers[-1], ToRedshift)


def test_loaded_model_predicts_without_filter_files(tmp_path, catalog, bands, feature_config):
    """The filter curves are stored in the model file: predicting must not need the original files."""
    model = _make_model(feature_config["n_bins"])
    path = tmp_path / "model.pkl"
    save_model_file(path, model, bands, "i", feature_config)
    for _, filter_file in bands.values():
        (tmp_path / filter_file).unlink()

    loaded, features = load_model_file(path)
    pz = predict(loaded, catalog, **features, ids=np.arange(len(catalog)) + 1000)
    expected = predict(model, catalog, bands, "i", feature_config, filter_curves=features["filter_curves"])

    assert isinstance(pz, qp.Ensemble) and pz.npdf == len(catalog)
    np.testing.assert_array_equal(pz.ancil["object_id"], np.arange(len(catalog)) + 1000)
    np.testing.assert_allclose(pz.ancil["zmean"], expected.ancil["zmean"], rtol=1e-5)
    np.testing.assert_allclose(pz.objdata["xvals"], expected.objdata["xvals"], rtol=1e-5)
