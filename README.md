# znn

Photometric redshifts from an ensemble of 1-D convolutional neural networks.

Each galaxy's photometry is turned into a short "spectrum" on a common wavelength grid by spreading
every band's magnitude over its filter curve. An ensemble of CNNs predicts a redshift from it; the
members share one training/validation split and can each be trained on their own noise realization of
the training photometry. The per-object p(z) is the histogram of the members' predictions, optionally
over many noise realizations of the photometry to include the photometric errors.

## Installation

```bash
pip install -e .          # numpy, pandas, scikit-learn, tensorflow, h5py, scipy, matplotlib
pip install -e ".[qp]"    # also qp-prob, to get p(z) as qp ensembles
```

Without `qp`, the prediction functions return a DataFrame with `object_id`, `mean` and `std`
columns instead of a `qp.Ensemble`.

## Quick start

```python
import numpy as np
import znn

# band -> (catalogue magnitude column, filter curve file), ordered by wavelength.
# You always supply these: the package ships no filters.
BANDS = {
    "u": ("mag_u_lsst", "filters/DC2LSST_u.res"),
    "g": ("mag_g_lsst", "filters/DC2LSST_g.res"),
    "r": ("mag_r_lsst", "filters/DC2LSST_r.res"),
    "i": ("mag_i_lsst", "filters/DC2LSST_i.res"),
    "z": ("mag_z_lsst", "filters/DC2LSST_z.res"),
    "y": ("mag_y_lsst", "filters/DC2LSST_y.res"),
}
REF_BAND = "i"  # every magnitude is taken relative to this band

train = znn.read_catalog("training.hdf5")  # needs a "redshift" column
test = znn.read_catalog("test.hdf5")

# Features and training
X, z = znn.catalog_to_XY(train, BANDS, REF_BAND)
models, histories = znn.train_ensembles(znn.build_model, X, z, N_SPLITS=5)  # starts from the pop-cosmos weights
# models, histories = znn.train_ensembles(znn.build_model, X, z, N_SPLITS=5, pretraining=None)  # from scratch

# p(z) from the spread of the ensemble members
X_test, _ = znn.catalog_to_XY(test, BANDS, REF_BAND)
pz = znn.ensemble_predict(models, X_test, ids=test["object_id"].to_numpy())

# p(z) with the photometric errors propagated: 50 noise draws, each through every member
errors = {column: f"{column}_err" for column, _ in BANDS.values()}
pz = znn.ensemble_predict_resampled(
    models, test, errors, 50, np.random.default_rng(42), BANDS, REF_BAND, ids=test["object_id"].to_numpy()
)
pz.write_to("pz_estimate.hdf5")

# Save the ensemble with everything needed to rebuild its inputs, then reload it
znn.save_ensemble_file("model.pkl", models, BANDS, REF_BAND, znn.DEFAULT_FEATURE_CONFIG)
models, features = znn.load_ensemble_file("model.pkl")
X_test, _ = znn.catalog_to_XY(test, **features)
```

## How it works

1. **Photometry to a wavelength-binned vector** (`catalog_to_XY`)
   - Each band's magnitude, minus the reference band's, is spread over that band's filter curve.
   - The curves sit on a common grid of `n_lambda` points spanning `lambda_range`, by default the LSST
     ugrizy + Roman Y106/J129/H158 range (3000–18650 Å). Catalogues observed in different filters then
     share the same bins, and filters outside the range are ignored. `lambda_range=None` spans the bands'
     own filter curves instead.
   - The grid is then averaged down to `n_bins` wavelength bins.
   - Each object becomes an array of shape `(n_bins, 3)` with three channels:

     | channel | content |
     |---|---|
     | 0 | the binned, reference-normalised magnitude curve |
     | 1 | the bin's position along the wavelength axis, from 0 to 1 |
     | 2 | coverage: the fraction of the bin backed by observed bands, from 0 to 1 |

2. **An ensemble of CNNs** (`train_ensembles`)
   - `N_SPLITS` members share one split of the training set: they train on `1 - VAL_FRAC` of it and use
     the other `VAL_FRAC` (default 0.2) as their validation set.
   - Each member can be given its own inputs: `resampled_inputs` draws one noise realization of the
     training photometry per member from its errors (just the photometry as given if it has none), so
     the members differ by noise realization rather than by data cut.
   - Each member predicts a single redshift, trained with an MSE loss. Targets are scaled as z / 3.
   - Training uses early stopping and reduces the learning rate when the validation loss plateaus.
   - **Pre-training (default):** every member starts from a copy of the same pre-trained member
     (`pretrained_member`, default 0; `None` cycles through them, member k starting from pre-trained
     member k mod K), with its first two convolutional blocks frozen, and trains with the same settings
     as from scratch.
     - `pretraining="popcosmos"` (default): a 5-member ensemble shipped with znn
       (`znn/pretrained/popcosmos.{json,npz}`), trained on 200k galaxies of the pop-cosmos mock
       (Zenodo v1.1.0, i < 25.5) in 23 COSMOS2020 bands at their native depth, on the default grid. Its
       inputs used the default feature config, so yours must too. `znn.load_pretrained()` returns it with
       its config and provenance.
     - `pretraining=None`: train every member from scratch.
     - Your own: a list of `(model, Y_mean, Y_std)`, a `save_ensemble_file` `.pkl` path, or a
       `save_pretrained` path. Its inputs must be on the same grid as yours.
     - Pre-training checks the input shape against the pre-trained models but cannot check the grid itself.

3. **p(z)** (`ensemble_predict`, `ensemble_predict_resampled`)
   - `ensemble_predict` gives each object the histogram of the K members' predictions (bins
     `DEFAULT_Z_BINS`, z = 0 to 3 in steps of 0.01; `zmode` is its peak).
   - `ensemble_predict_resampled` also redraws the photometry S times from its errors and runs every
     member on every draw; the p(z) is the histogram of all S × K predictions, so it includes the
     photometric errors as well as the disagreement between members. A catalogue with no errors gets
     a single draw, the photometry as given.
   - With `return_components=True`, it also returns the two parts of the width, "model" and
     "photometric". They add in quadrature to the total.

## Package layout

| Module | Purpose | Main functions |
|---|---|---|
| `znn/io.py` | Reading catalogues and filter curves | `read_catalog`, `load_filter_curves`, `catalog_to_mags` |
| `znn/data.py` | Building CNN inputs and resampling photometry | `catalog_to_XY`, `resample_photometry`, `resampled_inputs`, `DEFAULT_FEATURE_CONFIG` |
| `znn/models.py` | Architectures, training, prediction, saving and loading | `build_model`, `train_ensembles`, `ensemble_predict`, `ensemble_predict_resampled`, `save_ensemble_file`, `load_ensemble_file`, `save_pretrained`, `load_pretrained` |
| `znn/qp_output.py` | Packaging predictions as p(z) | `package_samples`, `save_predictions`, `DEFAULT_Z_BINS` |
| `znn/stats.py` | Robust point-estimate statistics | `get_biweight_mean_sigma_outlier`, `get_all_stats`, `stats_to_markdown` |
| `znn/plotting.py` | Diagnostic plots | `plot_stats`, `compare_binned_stats`, `plot_ensemble_losses`, `visualize_the_data` |

Everything is importable from the top level, e.g. `znn.catalog_to_XY`.

The lower-level steps behind `catalog_to_XY` live in `data.py` and are useful on their own:
- `interpolate_filter_curves` puts the curves on a common grid;
- `bin_filters` averages them into wavelength bins;
- `convert_data_format` builds the three channels;
- `make_incomplete_nir_data` drops near-infrared bands for a random subset of objects.

## Conventions

**Catalogues**
- `read_catalog` reads `.parquet` files, and flat HDF5 files with one 1-D dataset per column.
- Training catalogues need a `redshift` column.

**`bands`**
- A dict `band -> (magnitude column, filter file)`. Its order fixes the order of every
  per-band array.
- Filter files are two-column text files of wavelength and transmission.
- `ref_band` is one of its keys.

**Missing photometry**
- `NaN` means not detected; `inf` means not observed.
- By default (`nondetect_value=np.inf`), non-detections are treated as unobserved: they get zero
  coverage.
- A band whose column is missing from the catalogue is unobserved for every object.

**`errors`**
- A dict `magnitude column -> error column`, used by `resample_photometry` and
  `ensemble_predict_resampled`.
- Noise is drawn in flux. A draw with negative flux becomes a non-detection.

**Feature settings**
- `DEFAULT_FEATURE_CONFIG` sets the wavelength grid (`n_lambda`, and `lambda_range`, the LSST+Roman span by
  default; `None` spans the bands' filter curves), the number of bins (`n_bins`) and how non-detections are
  handled. The default pre-training assumes the default grid and `n_bins`.
- Pass a modified copy as `config` to change them.

**Ensembles**
- A trained ensemble is a list of `(model, Y_mean, Y_std)` tuples, one per member. Predictions are
  `model.predict(X) * Y_std + Y_mean`.

**Saved models**
- `save_ensemble_file` writes one pickle holding each member's architecture and weights, plus the
  feature settings and the filter curves themselves. Estimation then does not need the filter
  files.
- `save_ensemble` / `load_ensemble` write the older layout: a directory of `.keras` files and
  `norm_params.json`.

**Statistics**
- `get_biweight_mean_sigma_outlier(dz)` returns a 5-tuple for dz = (z_pred − z_true) / (1 + z_true),
  computed after 3σ clipping:
  1. biweight bias;
  2. its uncertainty;
  3. biweight scatter;
  4. 3σ outlier rate;
  5. |dz| > 0.2 outlier rate.

## Notebooks

The analysis notebooks live on the `notebooks` branch. They cover the Cardinal and BPZ-template
tests and the DESC PZ data challenge task sets.

