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

# Features and training, starting from the pop-cosmos pre-trained weights (pretraining=None: from scratch)
X, z = znn.catalog_to_XY(train, BANDS, REF_BAND)
models, histories = znn.train_ensembles(znn.build_model_res_se, X, z, N_SPLITS=5)

# p(z) from the spread of the ensemble members
X_test, _ = znn.catalog_to_XY(test, BANDS, REF_BAND)
pz = znn.ensemble_predict(models, X_test, ids=test["object_id"].to_numpy())

# p(z) with the photometric errors propagated: 50 noise draws, each through every member
errors = {column: f"{column}_err" for column, _ in BANDS.values()}
pz = znn.ensemble_predict_resampled(
    models, test, errors, 50, np.random.default_rng(42), BANDS, REF_BAND, ids=test["object_id"].to_numpy()
)
pz.write_to("pz_estimate.hdf5")

# Or train from the catalogue, each member fine-tuned on its own K-fold cut from a base model trained on
# all the objects
models, histories, base_history = znn.train_from_catalog(
    znn.build_model_res_se, train, train["redshift"].to_numpy(), BANDS, REF_BAND, N_MEMBERS=10, from_base=True
)

# Save the ensemble with everything needed to rebuild its inputs, then reload it
znn.save_ensemble_file("model.pkl", models, BANDS, REF_BAND, znn.DEFAULT_FEATURE_CONFIG)
models, features = znn.load_ensemble_file("model.pkl")
X_test, _ = znn.catalog_to_XY(test, **features)
```

## How it works

1. **Photometry to a wavelength-binned vector** (`catalog_to_XY`)
   - Each band's magnitude, minus the reference band's, is spread over that band's filter curve.
   - The curves sit on a common grid of `n_lambda` points spanning `lambda_range`, by default the LSST
     ugrizy + Roman Y106/J129/H158 range (3000–18650 Å), so catalogues observed in different filters
     share the same bins. `lambda_range=None` spans the bands' own filter curves instead.
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
   - The model is `build_model_res_se`, a residual net with squeeze-excitation (273k parameters: a
     kernel-7 stem, 3 stages of 2 blocks, global average pooling and a dense head).
   - Each member predicts a single redshift, trained with an MSE loss on log(1 + z), standardised with
     the training mean and std. A `ToRedshift` layer appended to the trained model maps its output back to z.
   - Training uses early stopping and reduces the learning rate when the validation loss plateaus.
   - **Pre-training** (`pretraining`): by default every member starts from `"popcosmos"`, the model
     shipped in `znn/pretrained/popcosmos.keras` (`znn.load_pretrained`); nothing is frozen, it only sets
     the starting weights. It was trained with `train_model` on 160k galaxies of the pop-cosmos mock
     (Zenodo 10.5281/zenodo.17132433, v1.1.0, HSC i < 25.5; 40k more held out for early stopping) in its
     23 COSMOS2020 bands within the default grid, at their native depth. On 50k other mock galaxies:
     bias −0.0001, scatter 0.017, 3σ outliers 4.8%. Your inputs must use the default feature config.
     `pretraining=None` trains from scratch; a `.keras` path or an ensemble (its first member) also works.
   - `train_from_catalog` trains from the catalogue, on the photometry as given (the errors are only
     used in prediction). With `from_base=True`, a base model is first trained on all the objects, then
     each member is a copy of it fine-tuned on its own fold (`FINETUNE_EPOCHS` at `FINETUNE_LR`,
     no early stopping), with the base's stem (first `N_FROZEN_LAYERS` layers) frozen.

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
| `znn/models.py` | The model, training, prediction, saving and loading | `build_model_res_se`, `train_ensembles`, `train_from_catalog`, `train_model`, `load_pretrained`, `ensemble_predict`, `ensemble_predict_resampled`, `save_ensemble_file`, `load_ensemble_file` |
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
  `model.predict(X) * Y_std + Y_mean`. The models end in `ToRedshift` and output z, so members are
  `(model, 0, 1)`; `ToRedshift` is registered with Keras when `znn` is imported, so they load as usual.

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

