# znn

Photometric redshifts from an ensemble of 1-D convolutional neural networks.

Each galaxy's photometry is turned into a short "spectrum" on a common wavelength grid by spreading
every band's magnitude over its filter curve. An ensemble of CNNs, one per K-fold split of the
training set, predicts a redshift from it. The per-object p(z) is a Gaussian whose mean and width
come from the spread of the ensemble's predictions, optionally including the photometric errors.

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
models, histories = znn.train_ensembles(znn.build_model, X, z, N_SPLITS=5)
# or the residual squeeze-excitation net with a log(1 + z) target and a Huber loss
models, histories = znn.train_ensembles(znn.build_model_res_se, X, z, N_SPLITS=5, target="log1pz", loss="huber")

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
   - The curves sit on a common grid of `n_lambda` points that spans all the filters.
   - The grid is then averaged down to `n_bins` wavelength bins.
   - Each object becomes an array of shape `(n_bins, 3)` with three channels:

     | channel | content |
     |---|---|
     | 0 | the binned, reference-normalised magnitude curve |
     | 1 | the bin's position along the wavelength axis, from 0 to 1 |
     | 2 | coverage: the fraction of the bin backed by observed bands, from 0 to 1 |

2. **An ensemble of CNNs** (`train_ensembles`)
   - The training set is split K ways (`N_SPLITS`), and one model is trained per fold, using the
     other folds as its validation set.
   - Each member predicts a single redshift, trained with an MSE loss. Targets are scaled as z / 3.
   - Training uses early stopping and reduces the learning rate when the validation loss plateaus.
   - Options (the defaults reproduce the above):
     - `target="log1pz"` trains on log(1 + z), standardised with each fold's training mean and std.
       A `ToRedshift` layer is appended to each trained model, so it still outputs z.
     - `loss="huber"` uses a Huber loss; `huber_delta` (default 0.05) is given in dz / (1 + z).

   | Architecture | Parameters | Description |
   |---|---|---|
   | `build_model` | 1.34M | 6 convs, 3 poolings, global average pooling, dense head |
   | `build_model_v2` | 556k | 5 convs, 4 poolings, same head |
   | `build_model_v2_k7` | 557k | `build_model_v2` with a first kernel of 7 |
   | `build_model_res_se` | 273k | residual net with squeeze-excitation: kernel-7 stem, 3 stages of 2 blocks, same head |

   The best setups found were `build_model_res_se` with `target="log1pz"` (MSE or Huber), then
   `build_model_v2_k7` with `target="log1pz"`. The log(1 + z) target helped every architecture.

3. **p(z)** (`ensemble_predict`, `ensemble_predict_resampled`)
   - `ensemble_predict` gives each object a Gaussian with the mean and standard deviation of the K
     members' predictions.
   - `ensemble_predict_resampled` also redraws the photometry S times from its errors and runs every
     member on every draw. The width then includes the photometric errors as well as the
     disagreement between members.
   - With `return_components=True`, it also returns the two parts of the width, "model" and
     "photometric". They add in quadrature to the total.

## Package layout

| Module | Purpose | Main functions |
|---|---|---|
| `znn/io.py` | Reading catalogues and filter curves | `read_catalog`, `load_filter_curves`, `catalog_to_mags` |
| `znn/data.py` | Building CNN inputs and resampling photometry | `catalog_to_XY`, `resample_photometry`, `DEFAULT_FEATURE_CONFIG` |
| `znn/models.py` | Architectures, target scaling, losses, training, prediction, saving and loading | `build_model`, `build_model_v2_k7`, `build_model_res_se`, `train_ensembles`, `ensemble_predict`, `ensemble_predict_resampled`, `save_ensemble_file`, `load_ensemble_file`, `fine_tune_pre_trained_model` |
| `znn/qp_output.py` | Packaging predictions as p(z) | `package_predictions`, `save_predictions` |
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
- `DEFAULT_FEATURE_CONFIG` sets the wavelength grid (`n_lambda`), the number of bins (`n_bins`)
  and how non-detections are handled.
- Pass a modified copy as `config` to change them.

**Ensembles**
- A trained ensemble is a list of `(model, Y_mean, Y_std)` tuples, one per member. Predictions are
  `model.predict(X) * Y_std + Y_mean`.
- With `target="log1pz"` the model ends in a `ToRedshift` layer that outputs z, and the member is
  `(model, 0, 1)`, so every prediction and saving function works unchanged.

**Saved models**
- `save_ensemble_file` writes one pickle holding each member's architecture and weights, plus the
  feature settings and the filter curves themselves. Estimation then does not need the filter
  files.
- `ToRedshift` is registered with Keras when `znn` is imported, so log(1 + z) ensembles load like
  any other.
- `save_ensemble` / `load_ensemble` write the older layout: a directory of `.keras` files and
  `norm_params.json`. `fine_tune_pre_trained_model` reads this layout.

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

