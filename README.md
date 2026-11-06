# znn

Photometric redshifts from a pair of 1-D convolutional neural networks.

Each galaxy's photometry is turned into a short "spectrum" on a common wavelength grid by spreading
every band's magnitude over its filter curve. One CNN, F, predicts a redshift from it. A second CNN, G,
trained on F's residuals, predicts how far off F is likely to be. The per-object p(z) is a Student-t
in log(1 + z) centred on F's prediction, with G's predicted scale.

## Installation

```bash
pip install -e .          # numpy, pandas, scikit-learn, tensorflow, h5py, scipy, matplotlib
pip install -e ".[qp]"    # also qp-prob, to get p(z) as qp ensembles
```

Without `qp`, the prediction functions return a DataFrame with `object_id`, `mean` and `std`
columns instead of a `qp.Ensemble`.

## Quick start

```python
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

# Training, with F starting from the pop-cosmos pre-trained weights (pretraining=None: from scratch); the features
# are built from the catalogue, and the redshifts are its "redshift" column unless given as Y
model, histories = znn.train_model(znn.build_model, train, BANDS, REF_BAND)

# p(z) of the test catalogue
pz = znn.predict(model, test, BANDS, REF_BAND, ids=test["object_id"].to_numpy())
pz.write_to("pz_estimate.hdf5")

# Save the model with everything needed to rebuild its inputs, then reload it
znn.save_model_file("model.pkl", model, BANDS, REF_BAND, znn.DEFAULT_FEATURE_CONFIG)
model, features = znn.load_model_file("model.pkl")
pz = znn.predict(model, test, **features)
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

2. **Two networks** (`train_model`), after the moment networks of Jeffrey & Wandelt
   (2020, arXiv:2011.05991)
   - Both are `build_model`, a residual net with squeeze-excitation (273k parameters: a kernel-7 stem,
     3 stages of 2 blocks, global average pooling and a dense head), as separate instances.
   - **F** (`train_mean_model`) predicts a single redshift, trained with an MSE loss on log(1 + z), standardised
     with the training mean and std. A `ToRedshift` layer appended to the trained network maps its output back to z.
   - **G** (`train_scale_model`) is trained afterwards, from scratch, on F's residuals on the same objects, with F
     fixed. It predicts the log squared scale of a Student-t with `nu` degrees of freedom (default 5), trained by
     its negative log-likelihood. The Student-t keeps the rare catastrophic residuals from setting everyone's
     width, which a Gaussian (the paper's eq. 5, its `nu` → ∞ limit) does not, and gives the p(z) heavy tails.
   - Both hold out a random `VAL_FRAC` (default 20%) for early stopping, and reduce the learning rate when the
     validation loss plateaus. Training uses the photometry as given; the errors are not used.
   - **Pre-training** (`pretraining`): by default F starts from `"popcosmos"`, the network shipped in
     `znn/pretrained/popcosmos.keras` (`znn.load_pretrained`); nothing is frozen, it only sets the starting
     weights. It was trained with `train_mean_model` on 160k galaxies of the pop-cosmos mock
     (Zenodo 10.5281/zenodo.17132433, v1.1.0, HSC i < 25.5; 40k more held out for early stopping) in its
     23 COSMOS2020 bands within the default grid, at their native depth. On 50k other mock galaxies:
     bias −0.0001, scatter 0.017, 3σ outliers 4.8%. Your inputs must use the default feature config.
     `pretraining=None` trains F from scratch; a `.keras` path or a trained model (its F) also works. G always
     starts from scratch.

3. **p(z)** (`predict`)
   - Each object gets a Student-t in log(1 + z), centred on log(1 + F's prediction), with G's scale and `nu`
     degrees of freedom, converted to z and evaluated on a grid (`DEFAULT_ZGRID`, 0 to 3 in steps of 0.005).
   - The p(z) is predicted on the photometry as given. G's scale already includes the photometric noise of the
     data it was trained on, so drawing noise realizations on top would count it twice.

## Package layout

| Module | Purpose | Main functions |
|---|---|---|
| `znn/io.py` | Reading catalogues and filter curves | `read_catalog`, `load_filter_curves`, `catalog_to_mags` |
| `znn/data.py` | Building CNN inputs | `catalog_to_XY`, `DEFAULT_FEATURE_CONFIG` |
| `znn/models.py` | The model, training, prediction, saving and loading | `build_model`, `train_model`, `train_mean_model`, `train_scale_model`, `load_pretrained`, `predict`, `save_model_file`, `load_model_file` |
| `znn/qp_output.py` | Packaging predictions as p(z) | `package_pdfs`, `predictions_mean_std` |
| `znn/stats.py` | Robust point-estimate statistics | `get_biweight_mean_sigma_outlier`, `get_all_stats`, `stats_to_markdown` |

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

**Feature settings**
- `DEFAULT_FEATURE_CONFIG` sets the wavelength grid (`n_lambda`), the number of bins (`n_bins`)
  and how non-detections are handled.
- Pass a modified copy as `config` to change them.

**Models**
- A trained model is a tuple `(F, G, nu)`. F outputs z (it ends in `ToRedshift`, registered with Keras when
  `znn` is imported, so it loads as usual); G outputs the log squared scale of the Student-t in log(1 + z).

**Saved models**
- `save_model_file` writes one pickle holding F's and G's architecture and weights, `nu`, the feature
  settings and the filter curves themselves. Estimation then does not need the filter files.
- `load_model_file` also reads the files of the earlier layout of the same model, a two-member list
  `[(F, 0, 1), (G, log s², nu)]`.

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

