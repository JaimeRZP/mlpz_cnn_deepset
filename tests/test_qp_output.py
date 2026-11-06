"""
Tests of znn.qp_output: the p(z) is a Student-t in y = log(1 + z), truncated to z in [0, zmax]. Each test compares
what package_pdfs stores against an independent reference: direct numerical integration of the exact density, or
Monte Carlo samples drawn from the Student-t with scipy.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import integrate, stats

from znn import qp_output
from znn.qp_output import DEFAULT_QUANTS, DEFAULT_ZMAX, package_pdfs, predictions_mean_std

qp = pytest.importorskip("qp")

NU = 5.0
ZMAX = DEFAULT_ZMAX
# (median z, scale in log(1 + z)): narrow and wide, mid-range and close to each edge, so truncation matters
CASES = [(0.05, 0.02), (0.5, 0.05), (1.0, 0.1), (2.0, 0.3), (2.9, 0.2), (0.0, 0.4), (3.5, 0.3)]
MU = np.log1p(np.array([c[0] for c in CASES]))
SCALE = np.array([c[1] for c in CASES])


class TruncatedLogT:
    """Reference p(z), written independently of znn: t(log(1 + z); nu, mu, s) / (1 + z), renormalised on [0, zmax]."""

    def __init__(self, mu, scale, nu=NU, zmax=ZMAX):
        self.mu, self.scale, self.nu, self.zmax = mu, scale, nu, zmax
        self.t = stats.t(nu, loc=mu, scale=scale)
        y_hi = np.log1p(zmax)
        # probability mass in [0, zmax]: from the survival function when that range is in the upper tail, where the
        # CDF would lose it to round-off
        if mu < 0.5 * y_hi:
            self.norm = self.t.sf(0.0) - self.t.sf(y_hi)
        else:
            self.norm = self.t.cdf(y_hi) - self.t.cdf(0.0)

    def pdf(self, z):
        return self.t.pdf(np.log1p(z)) / (1 + z) / self.norm

    def cdf(self, z):
        if self.mu < 0.5 * np.log1p(self.zmax):
            return (self.t.sf(0.0) - self.t.sf(np.log1p(z))) / self.norm
        return (self.t.cdf(np.log1p(z)) - self.t.cdf(0.0)) / self.norm

    def moment(self, k):
        peak = np.clip(np.expm1(self.mu), 0, self.zmax)
        return integrate.quad(lambda z: z**k * self.pdf(z), 0, self.zmax, points=[peak], limit=500)[0]

    def sample(self, n, rng):
        """Rejection sampling: draw y from the untruncated Student-t, keep 0 <= z <= zmax."""
        out = np.empty(0)
        while out.size < n:
            z = np.expm1(self.t.rvs(size=4 * n, random_state=rng))
            out = np.concatenate([out, z[(z >= 0) & (z <= self.zmax)]])
        return out[:n]


def grid(ens):
    """Per-object grid, (npdf, nquants): qp stores a single object's grid as 1-D."""
    return np.atleast_2d(ens.objdata["xvals"])


@pytest.fixture(scope="module")
def ensemble():
    return package_pdfs(MU, SCALE, NU, ids=np.arange(100, 100 + len(MU)))


@pytest.fixture(scope="module")
def refs():
    return [TruncatedLogT(m, s) for m, s in zip(MU, SCALE)]


# --- structure ---------------------------------------------------------------------------------------------------


def test_ensemble_layout(ensemble):
    assert isinstance(ensemble, qp.Ensemble)
    assert ensemble.npdf == len(MU)
    xvals = ensemble.objdata["xvals"]
    assert xvals.dtype == np.float32
    assert xvals.shape == (len(MU), len(DEFAULT_QUANTS))
    assert np.all(np.diff(xvals, axis=1) > 0), "per-object grids must be strictly increasing"
    assert np.all(xvals >= 0) and np.all(xvals <= ZMAX + 1e-5)
    np.testing.assert_array_equal(ensemble.ancil["object_id"], np.arange(100, 100 + len(MU)))
    assert set(ensemble.ancil) >= {"object_id", "zmode", "zmean", "zstd"}


def test_default_ids_are_sample_index():
    ens = package_pdfs(MU[:3], SCALE[:3], NU)
    np.testing.assert_array_equal(ens.ancil["object_id"], [0, 1, 2])


# --- moments and quantiles against the exact p(z) ----------------------------------------------------------------


@pytest.mark.parametrize("i", range(len(CASES)))
def test_mean_and_std_match_exact_integral(ensemble, refs, i):
    ref = refs[i]
    m1, m2 = ref.moment(1), ref.moment(2)
    std = np.sqrt(m2 - m1**2)
    assert ensemble.ancil["zmean"][i] == pytest.approx(m1, rel=1e-4, abs=1e-6)
    assert ensemble.ancil["zstd"][i] == pytest.approx(std, rel=1e-3)


@pytest.mark.parametrize("i", range(len(CASES)))
def test_grid_sits_on_exact_quantiles(ensemble, refs, i):
    """The stored grid points are the p(z)'s quantiles at DEFAULT_QUANTS: the exact CDF there gives back the levels."""
    x = ensemble.objdata["xvals"][i].astype(float)
    np.testing.assert_allclose(refs[i].cdf(x), DEFAULT_QUANTS, atol=2e-5)


@pytest.mark.parametrize("i", range(len(CASES)))
def test_qp_cdf_and_ppf_are_exact_at_grid(ensemble, refs, i):
    x = ensemble.objdata["xvals"][i]
    np.testing.assert_allclose(ensemble[i].cdf(x).ravel(), DEFAULT_QUANTS, atol=1e-6)
    q = DEFAULT_QUANTS[1:-1]
    np.testing.assert_allclose(ensemble[i].ppf(q).ravel(), x[1:-1], rtol=1e-3, atol=1e-5)


@pytest.mark.parametrize("i", range(len(CASES)))
def test_mode_is_peak_of_exact_density(ensemble, refs, i):
    z = np.linspace(0, ZMAX, 300001)
    expected = z[np.argmax(refs[i].pdf(z))]
    assert ensemble.ancil["zmode"][i] == pytest.approx(expected, abs=2e-5)


def test_untruncated_limit_quantiles_are_shifted_t():
    """Far from the edges the truncation is negligible: quantiles are expm1(mu + scale * t.ppf(q))."""
    mu, scale = np.log1p([1.0]), np.array([0.01])
    ens = package_pdfs(mu, scale, NU)
    q = DEFAULT_QUANTS[1:-1]
    expected = np.expm1(mu + scale * stats.t.ppf(q, NU))
    np.testing.assert_allclose(grid(ens)[0, 1:-1], expected, rtol=1e-6)
    assert ens.ancil["zmode"][0] < np.expm1(mu[0]) < ens.ancil["zmean"][0]  # the 1/(1 + z) Jacobian skews it right


# --- Monte Carlo: samples from the Student-t against the stored p(z) ---------------------------------------------


@pytest.mark.parametrize("i", range(len(CASES)))
def test_student_t_samples_match_stored_pdf(ensemble, refs, i):
    """Rejection-sampled Student-t redshifts have the stored mean, std and quantiles (within Monte Carlo error)."""
    n = 200_000
    z = refs[i].sample(n, np.random.default_rng(i))
    mean, std = ensemble.ancil["zmean"][i], ensemble.ancil["zstd"][i]
    assert abs(z.mean() - mean) < 5 * std / np.sqrt(n)
    assert z.std() == pytest.approx(std, rel=0.02)

    # the fraction of samples below each stored quantile is its level, to within binomial error
    x = ensemble.objdata["xvals"][i, 1:-1]
    q = DEFAULT_QUANTS[1:-1]
    frac = (z[:, None] <= x[None, :]).mean(axis=0)
    np.testing.assert_array_less(np.abs(frac - q), 5 * np.sqrt(q * (1 - q) / n) + 1e-4)


@pytest.mark.parametrize("i", range(len(CASES)))
def test_qp_draws_follow_exact_cdf(ensemble, refs, i):
    """Draws from the stored ensemble are distributed as the exact p(z) (KS test)."""
    draws = ensemble[i].rvs(size=20_000, random_state=i).ravel()
    assert np.all((draws >= 0) & (draws <= ZMAX + 1e-5))
    assert stats.kstest(draws, refs[i].cdf).pvalue > 1e-3


# --- I/O ---------------------------------------------------------------------------------------------------------


def test_write_and_read_roundtrip(ensemble, tmp_path):
    path = tmp_path / "pz.hdf5"
    ensemble.write_to(str(path))
    back = qp.read(str(path))
    np.testing.assert_array_equal(back.objdata["xvals"], ensemble.objdata["xvals"])
    np.testing.assert_allclose(back.objdata["yvals"], ensemble.objdata["yvals"], rtol=1e-6)
    for key in ("object_id", "zmode", "zmean", "zstd"):
        np.testing.assert_array_equal(np.asarray(back.ancil[key]).ravel(), np.asarray(ensemble.ancil[key]).ravel())
    # qp's CDF of the read-back ensemble agrees to float32 round-off
    np.testing.assert_allclose(back.cdf(np.array([0.5, 1.0])), ensemble.cdf(np.array([0.5, 1.0])), atol=1e-6)


# --- edge cases --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("mu", [-3.0, -1.0, 2.0, 5.0])
def test_extreme_locations_match_exact_tail(mu):
    """
    Objects far outside [0, zmax]: the p(z) is the Student-t's power-law tail on [0, zmax] (not a spike at the edge),
    which must stay resolved, with a valid grid and the exact moments.
    """
    ens = package_pdfs(np.array([mu]), np.array([0.1]), NU)
    ref = TruncatedLogT(mu, 0.1)
    x, y = grid(ens)[0], np.atleast_2d(ens.objdata["yvals"])[0]
    assert np.all(np.isfinite(x)) and np.all(np.isfinite(y)) and np.all(y > 0)
    assert np.all(np.diff(x) > 0)
    np.testing.assert_allclose(ref.cdf(x.astype(float)), DEFAULT_QUANTS, atol=2e-5)
    m1, m2 = ref.moment(1), ref.moment(2)
    assert ens.ancil["zmean"][0] == pytest.approx(m1, rel=1e-4)
    assert ens.ancil["zstd"][0] == pytest.approx(np.sqrt(m2 - m1**2), rel=1e-3)


@pytest.mark.parametrize("z", [0.01, 1.5, 2.99])
def test_narrowest_predicted_pdf_is_exact(z):
    """predict floors the scale at 1e-3: down to that width qp's CDF still passes through the quantile levels."""
    ens = package_pdfs(np.log1p([z]), np.array([1e-3]), NU)
    np.testing.assert_allclose(ens[0].cdf(grid(ens)[0]).ravel(), DEFAULT_QUANTS, atol=1e-4)


def test_tiny_scale_gives_strictly_increasing_grid():
    """float32 rounding ties the quantiles of a p(z) narrower than its resolution; the grid must stay valid."""
    ens = package_pdfs(np.log1p([1.5]), np.array([1e-7]), NU)
    x = grid(ens)[0]
    assert np.all(np.diff(x) > 0)
    assert np.all(np.isfinite(ens.objdata["yvals"]))
    assert np.all(np.diff(ens[0].cdf(x).ravel()) >= 0)


@pytest.mark.xfail(
    strict=True,
    reason="known limitation: below float32 resolution (scale << 1e-5) the _MIN_STEP spreading makes qp's CDF "
    "overshoot 1 (1.025 here); predict avoids it by flooring the scale at 1e-3, package_pdfs does not",
)
def test_tiny_scale_cdf_normalised():
    ens = package_pdfs(np.log1p([1.5]), np.array([1e-7]), NU)
    assert ens[0].cdf(grid(ens)[0]).ravel()[-1] == pytest.approx(1.0, abs=1e-4)


def test_chunking_gives_same_result(monkeypatch):
    rng = np.random.default_rng(0)
    mu, scale = rng.uniform(0, 1.3, 50), rng.uniform(0.01, 0.3, 50)
    whole = package_pdfs(mu, scale, NU)
    monkeypatch.setattr(qp_output, "_CHUNK", 7)
    chunked = package_pdfs(mu, scale, NU)
    np.testing.assert_array_equal(chunked.objdata["xvals"], whole.objdata["xvals"])
    np.testing.assert_allclose(chunked.ancil["zmean"], whole.ancil["zmean"])


# --- fallback and helpers ----------------------------------------------------------------------------------------


def test_fallback_dataframe_without_qp(monkeypatch, ensemble):
    monkeypatch.setattr(qp_output, "QP_INSTALLED", False)
    with pytest.warns(UserWarning, match="qp is not installed"):
        df = package_pdfs(MU, SCALE, NU)
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["object_id", "zmode", "mean", "std"]
    np.testing.assert_allclose(df["mean"], ensemble.ancil["zmean"])
    np.testing.assert_allclose(df["std"], ensemble.ancil["zstd"])


def test_predictions_mean_std_from_ensemble_and_dataframe(ensemble):
    mean, std = predictions_mean_std(ensemble)
    assert mean.shape == std.shape == (len(MU), 1)
    np.testing.assert_array_equal(mean.ravel(), ensemble.ancil["zmean"])
    df = pd.DataFrame({"mean": [1.0, 2.0], "std": [0.1, 0.2]})
    mean, std = predictions_mean_std(df)
    np.testing.assert_array_equal(mean, [[1.0], [2.0]])
    np.testing.assert_array_equal(std, [[0.1], [0.2]])


def test_predictions_mean_std_other_ensemble_uses_qp_moments():
    ens = qp.Ensemble(qp.stats.norm, data={"loc": np.array([[1.0], [2.0]]), "scale": np.array([[0.1], [0.3]])})
    mean, std = predictions_mean_std(ens)
    np.testing.assert_allclose(mean.ravel(), [1.0, 2.0])
    np.testing.assert_allclose(std.ravel(), [0.1, 0.3])
