<img width="482" height="209" alt="Screenshot From 2026-11-06 10-07-49" src="https://github.com/user-attachments/assets/b54d7f5a-5fd5-4f76-b35f-e08769af48c0" />


ZNN is a machine learning photometric redshift estimator. ZNN constructs as coarse approximation to the SED of source based on the observed photometric filters magnitudes. This representation of the data allows us to establish physical relationships between the observed filters' magnitudes, as probing different wavelength bands of a continuos physical process. Moreover, it allows us to consistently combine information from overlapping photometric filters as well as from different surveys. ZNN leverages this continuous representation of the photometry to learn its relationship to redshift using a convolutional neural network. This approach is effectively a data-driven template fitting methodology, where the templates are the learnt filters of the network. To estimate the uncertainty in its redshift estimates, ZNN uses second network that learns the residual of the first.

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

