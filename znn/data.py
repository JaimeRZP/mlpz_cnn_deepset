import numpy as np

from .io import catalog_to_mags, load_filter_curves

__all__ = [
    "DEFAULT_FEATURE_CONFIG",
    "rebin_filter",
    "get_bin_edges",
    "stretch",
    "interpolate_filter_curves",
    "make_lambda_bins",
    "bin_filters",
    "convert_data_format",
    "transform_data_to_XY",
    "make_incomplete_nir_data",
    "catalog_to_XY",
    "has_errors",
    "resample_photometry",
]

# Processing settings for catalog_to_XY; the bands themselves are always supplied by the user.
DEFAULT_FEATURE_CONFIG = {
    "nondetect_value": np.inf,  # np.inf treats non-detections as unobserved
    "n_lambda": 1000,  # points in the common wavelength grid
    # (min, max) of the grid, in Å: the LSST ugrizy + Roman Y106/J129/H158 span, which the bundled pre-trained
    # weights use. None spans the bands' filter curves instead.
    "lambda_range": (3000.0, 18650.0),
    "n_bins": 32,  # wavelength bins fed to the CNN
    "apply_stretch": False,
    "c": 0.8,
    "k": 20,
}


def rebin_filter(bin_edges, counts, new_edges):
    new_counts = []
    for i in range(len(new_edges) - 1):
        lo, hi = new_edges[i], new_edges[i + 1]
        total = 0.0
        for j, (b0, b1, c) in enumerate(zip(bin_edges[:-1], bin_edges[1:], counts)):
            overlap = max(0, min(hi, b1) - max(lo, b0))
            total += c * overlap  # /(b1 - b0)  # proportional share
        new_counts.append(total)
    return np.array(new_counts)


def get_bin_edges(centres):
    step = centres[1] - centres[0]  # uniform spacing
    inner_edges = (centres[:-1] + centres[1:]) / 2
    first_edge = centres[0] - step / 2
    last_edge = centres[-1] + step / 2
    return np.concatenate([[first_edge], inner_edges, [last_edge]])


def stretch(x, c=0.5, k=10):
    """
    x: array of values in [0,1]
    c: center of your typical fluctuation band
    k: stretch strength (try 5-50; higher = more aggressive)
    """
    x = np.asarray(x)
    u = np.where(x >= c, (x - c) / (1 - c), (x - c) / c)  # -> [-1, 1], c -> 0
    stretched = np.arcsinh(k * u) / np.arcsinh(k)  # amplify near u=0
    y = np.where(stretched >= 0, c + stretched * (1 - c), c + stretched * c)
    return np.clip(y, 0, 1)


def interpolate_filter_curves(filter_curves, lambda_common):
    """
    Interpolate each band's raw (wavelength, transmission) curve onto lambda_common and
    combine them into a filter bank that avoids double-counting overlapping filters.

    filter_curves: dict[str -> (M, 2) array] raw curves keyed by band letter, in the row
        order to use for every downstream array (mags, filters_array, ...).
    lambda_common: (L,) shared wavelength grid.

    Math
    ----
    Let T_b(lambda) be band b's raw transmission curve, interpolated onto lambda_common
    (zero outside its native support). The filter bank is its "ownership" share:

        filters_array_b(lambda) = T_b(lambda) / sum_b' T_b'(lambda)

    (0 where the denominator is 0).

    Returns filters_array, shape (n_bands, L).
    """
    raw_filters = np.array(
        [
            np.interp(lambda_common, filter_curves[b][:, 0], filter_curves[b][:, 1], left=0, right=0)
            for b in filter_curves.keys()
        ]
    )

    total_transmission = raw_filters.sum(axis=0)
    return np.divide(raw_filters, total_transmission, out=np.zeros_like(raw_filters), where=total_transmission > 0)


def make_lambda_bins(lambda_common, n_bins):
    """
    Build n_bins equal-width wavelength bins spanning lambda_common's range.

    Returns (lambda_bin_centers, bin_idx):
      - lambda_bin_centers: (n_bins,) bin center wavelengths.
      - bin_idx: (L,) int array mapping each lambda_common point to its bin, in [0, n_bins).
    """
    lambda_bin_edges = np.linspace(lambda_common[0], lambda_common[-1], n_bins + 1)
    lambda_bin_centers = (lambda_bin_edges[:-1] + lambda_bin_edges[1:]) / 2
    bin_idx = np.clip(np.digitize(lambda_common, lambda_bin_edges) - 1, 0, n_bins - 1)
    return lambda_bin_centers, bin_idx


def bin_filters(filters_array, n_bins, lambda_common):
    """
    Turn per-lambda-point filter curves into a fixed (n_bands, n_bins) linear operator
    that averages each band's curve within each wavelength bin. Folding this into a
    matrix (rather than binning per source) lets the whole dataset be transformed with a
    single matmul: mags (n_sources, n_bands) @ filters_binned -> (n_sources, n_bins).

    Returns filters_binned, shape (n_bands, n_bins).
    """
    L = filters_array.shape[1]

    _, bin_idx = make_lambda_bins(lambda_common, n_bins)
    counts = np.bincount(bin_idx, minlength=n_bins)
    W = np.zeros((n_bins, L))
    W[bin_idx, np.arange(L)] = 1.0
    W = W / np.maximum(counts, 1)[:, None]

    return filters_array @ W.T


def convert_data_format(mags, mag_i, filters_array, n_bins, lambda_common):
    """
    Vectorized over all sources.

    mags: (n_filters, n_sources) raw per-band magnitudes, row order matching
        filters_array's band rows. Missing values are encoded as np.nan (no detection) or
        np.inf (no observation), same convention as the rest of the module.
    mag_i: (n_sources,) i-band magnitude, subtracted from each band's magnitude before
        binning.
    filters_array: (n_filters, n_lambda) from interpolate_filter_curves(...). Used both to
        build the curve and, via which bands were actually observed, the coverage channel.
    n_bins: number of wavelength bins to bin down to.
    lambda_common: (n_lambda,) wavelength grid matching filters_array.

    Returns X, shape (n_sources, n_bins, 3):
      channel 0: binned, i-band-normalized, amplitude-weighted curve value -- the i-band
        subtraction is applied per band before binning (mags - mag_i), not to the binned
        curve.
      channel 1: wavelength-bin position label (0..1), identical across sources.
      channel 2: coverage/availability mask in [0, 1] (1 = bin fully backed by observed
        bands, 0 = bin has no observed-band support, fractional at the boundary between
        an observed and unobserved band).
    """
    n_sources = mags.shape[1]

    mags_normed = mags - mag_i[None, :]  # normalize by i-band mag, per band, before binning

    ind_nan = np.isnan(mags_normed)  # no detection
    ind_inf = np.isinf(mags_normed)  # no observation

    mags_clean = np.where(ind_nan | ind_inf, 0.0, mags_normed)  # (n_filters, n_sources)

    filters_binned = bin_filters(filters_array, n_bins, lambda_common)  # (n_filters, n_bins)

    mags_data = mags_clean.T @ filters_binned  # (n_sources, n_bins)

    observed = (~ind_inf).T.astype(float)  # (n_sources, n_filters); 1 if band was observed
    coverage = np.clip(observed @ filters_binned, 0, 1)  # (n_sources, n_bins)

    wave_labels = np.arange(n_bins)
    wave_labels = wave_labels / wave_labels[-1]
    lambda_labels = np.outer(np.ones(n_sources), wave_labels)

    return np.stack([mags_data, lambda_labels, coverage], axis=-1)


def transform_data_to_XY(
    mags,
    redshift,
    filters_array,
    n_bins,
    lambda_common,
    mag_i=0.0,
    apply_stretch=True,
    c=0.8,
    k=20,
    missingY=False,
):
    X = convert_data_format(mags, mag_i, filters_array, n_bins, lambda_common)
    Y = 0 if missingY else redshift
    if apply_stretch:
        X[:, :, 0] = stretch(X[:, :, 0], c=c, k=k)
    return X, Y


def make_incomplete_nir_data(
    mags,
    redshift,
    filters_array,
    n_bins,
    lambda_common,
    nir_idx,
    mag_i=0.0,
    frac=0.5,
    apply_stretch=False,
):
    """
    nir_idx: row indices into mags (matching filters_array's band order) of the bands to
        drop for a random subset of sources, e.g. the J/H rows, so their contribution is
        treated as unobserved (np.inf) for that subset.
    """
    n_sources = mags.shape[1]
    idx = np.random.choice(n_sources, size=int(round(frac * n_sources)), replace=False)
    mags_copy = mags.copy()
    mags_copy[np.ix_(nir_idx, idx)] = np.inf
    return transform_data_to_XY(
        mags_copy, redshift, filters_array, n_bins, lambda_common, mag_i=mag_i, apply_stretch=apply_stretch
    )


def catalog_to_XY(df, bands, ref_band, config=DEFAULT_FEATURE_CONFIG, filter_curves=None):
    """
    Turn a catalogue DataFrame (see znn.io.read_catalog) into CNN inputs.

    bands: dict band -> (magnitude column, filter file path), see znn.io.
    ref_band: key of `bands` whose magnitude normalizes the others (e.g. "i").
    config: processing settings, as DEFAULT_FEATURE_CONFIG.
    filter_curves: optional pre-loaded curves, as returned by znn.io.load_filter_curves; when
        None they are read from the paths in `bands`.

    The common wavelength grid spans config["lambda_range"], in the filter curves' wavelength
    units. Fix it to compare or combine catalogues observed in different filters: each bin then
    covers the same wavelengths whatever the bands, and filter curves outside the range are
    ignored. When it is None (or missing, as in older configs) the grid spans the full range of
    the bands' filter curves.

    Returns (X, Y): X of shape (n_sources, n_bins, 3), and Y the "redshift" column, or None
    when df has none (e.g. test catalogues).
    """
    if filter_curves is None:
        filter_curves = load_filter_curves(bands)

    lambda_range = config.get("lambda_range")
    if lambda_range is None:
        lambda_min = min(curve[:, 0].min() for curve in filter_curves.values())
        lambda_max = max(curve[:, 0].max() for curve in filter_curves.values())
    else:
        lambda_min, lambda_max = lambda_range
    lambda_common = np.linspace(lambda_min, lambda_max, config["n_lambda"])
    filters_array = interpolate_filter_curves(filter_curves, lambda_common)

    mags = catalog_to_mags(df, bands, config["nondetect_value"])
    mag_ref = mags[list(bands).index(ref_band)]

    has_redshift = "redshift" in df
    X, Y = transform_data_to_XY(
        mags,
        df["redshift"].to_numpy() if has_redshift else None,
        filters_array,
        config["n_bins"],
        lambda_common,
        mag_i=mag_ref,
        apply_stretch=config["apply_stretch"],
        c=config["c"],
        k=config["k"],
        missingY=not has_redshift,
    )
    return X, (Y if has_redshift else None)


def has_errors(df, errors):
    """
    Whether a catalogue has any photometric errors to sample over: at least one of the error columns in
    `errors` (dict magnitude column -> error column, or None) is in df with a finite value.
    """
    return any(
        err_column in df and np.isfinite(df[err_column].to_numpy(dtype=float)).any() for err_column in (errors or {}).values()
    )


def resample_photometry(df, errors, rng):
    """
    Return a copy of a catalogue with its magnitudes redrawn from their photometric errors: one noise
    realization of the photometry.

    errors: dict magnitude column -> its error column, e.g. {"mag_u_lsst": "mag_u_lsst_err", ...}.
    rng: numpy Generator, e.g. np.random.default_rng(seed).

    The noise is Gaussian in flux, f = 10**(-0.4 m) with sigma_f = 0.4 ln(10) f sigma_m, as photometric
    noise is; draws with f <= 0 become non-detections (np.nan). Magnitudes or errors that are not finite
    (non-detections, unobserved bands), and bands whose error column is not in df, are left unchanged.
    """
    out = df.copy()
    for mag_column, err_column in errors.items():
        if err_column not in df:
            continue
        mag = df[mag_column].to_numpy(dtype=float)
        err = df[err_column].to_numpy(dtype=float)
        ok = np.isfinite(mag) & np.isfinite(err)

        flux = 10 ** (-0.4 * mag[ok])
        flux_new = flux + 0.4 * np.log(10) * flux * err[ok] * rng.standard_normal(ok.sum())

        mag_new = mag.copy()
        mag_new[ok] = np.nan
        detected = flux_new > 0
        mag_new[np.flatnonzero(ok)[detected]] = -2.5 * np.log10(flux_new[detected])
        out[mag_column] = mag_new
    return out

