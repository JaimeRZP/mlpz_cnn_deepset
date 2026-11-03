import os
import pickle

import numpy as np
import tensorflow as tf
from scipy import stats
from sklearn.model_selection import train_test_split
from tensorflow.keras import callbacks, layers, models

from .data import DEFAULT_FEATURE_CONFIG, catalog_to_XY
from .io import load_filter_curves
from .qp_output import package_pdfs

__all__ = [
    "build_model",
    "ToRedshift",
    "train_mean_model",
    "train_scale_model",
    "train_model",
    "predict",
    "save_model_file",
    "load_model_file",
    "load_pretrained",
    "PRETRAINED_DIR",
    "DEFAULT_PRETRAINING",
    "DEFAULT_NU",
    "DEFAULT_ZGRID",
]

# The model is a pair of networks after the moment networks of Jeffrey & Wandelt 2020 (arXiv:2011.05991): F predicts
# the redshift, and G, trained on F's residuals with F fixed, predicts their spread. G predicts the log squared scale
# of a Student-t with nu degrees of freedom, trained by its negative log-likelihood: eq. 5 of the paper (MSE on the
# squared residuals, i.e. the variance) is the nu -> infinity limit, but is dominated by the rare catastrophic
# residuals and gives too wide p(z)s. A trained model is the tuple (F, G, nu): F outputs z (it ends in ToRedshift),
# G the log squared scale in log(1 + z), and the p(z) is that Student-t in log(1 + z) around log(1 + F).

# Pre-trained models shipped with znn, as <name>.keras: "popcosmos" is an F trained on the pop-cosmos mock, on the
# default feature grid (see README)
PRETRAINED_DIR = os.path.join(os.path.dirname(__file__), "pretrained")
DEFAULT_PRETRAINING = "popcosmos"
DEFAULT_NU = 5.0  # degrees of freedom of the Student-t
DEFAULT_ZGRID = np.linspace(0, 3, 601)  # where predict evaluates p(z); finer than the narrowest p(z)s


def _squeeze_excite(x, ratio=4):
    """Channel attention: reweight each channel by a gate computed from its average over wavelength."""
    channels = x.shape[-1]
    s = layers.GlobalAveragePooling1D()(x)
    s = layers.Dense(max(channels // ratio, 4), activation="relu")(s)
    s = layers.Dense(channels, activation="sigmoid")(s)
    s = layers.Reshape((1, channels))(s)
    return layers.Multiply()([x, s])


def _residual_se_block(x, width, kernel_size=3):
    """Two conv-BN layers and a squeeze-excitation gate, added to an identity skip (1x1 conv if the width changes)."""
    shortcut = x
    if x.shape[-1] != width:
        shortcut = layers.Conv1D(width, kernel_size=1, use_bias=False)(x)
        shortcut = layers.BatchNormalization()(shortcut)

    y = layers.Conv1D(width, kernel_size=kernel_size, padding="same", use_bias=False)(x)
    y = layers.BatchNormalization()(y)
    y = layers.Activation("relu")(y)
    y = layers.Conv1D(width, kernel_size=kernel_size, padding="same", use_bias=False)(y)
    y = layers.BatchNormalization()(y)
    y = _squeeze_excite(y)

    y = layers.Add()([shortcut, y])
    return layers.Activation("relu")(y)


def build_model(input_shape):
    # The model: residual squeeze-excitation net with a kernel-7 stem (conv, batch norm, ReLU), 3 stages
    # (32/64/128 filters) of 2 blocks each, pooling after the first two stages (32 -> 16 -> 8 positions), then a
    # dense head. ~270k parameters. Used for both F and G (see train_model)
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv1D(32, kernel_size=7, padding="same", use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    for stage, width in enumerate([32, 64, 128]):
        x = _residual_se_block(x, width)
        x = _residual_se_block(x, width)
        if stage < 2:
            x = layers.MaxPooling1D()(x)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1)(x)
    return models.Model(inputs, outputs)


@tf.keras.utils.register_keras_serializable(package="znn")
class ToRedshift(layers.Layer):
    """Map a standardised log(1 + z) output back to redshift: z = exp(y * std + mean) - 1."""

    def __init__(self, mean, std, **kwargs):
        super().__init__(**kwargs)
        self.mean = float(mean)
        self.std = float(std)

    def call(self, y):
        return tf.math.expm1(y * self.std + self.mean)

    def get_config(self):
        config = super().get_config()
        config.update({"mean": self.mean, "std": self.std})
        return config


def _callbacks():
    """Early stopping and learning-rate reduction on the validation loss."""
    early_stop = callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    reduce_lr = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1)
    return [early_stop, reduce_lr]


def _start(pretraining):
    """The model F starts from: None (from scratch), load_pretrained(pretraining), or a trained model's F."""
    if isinstance(pretraining, str):
        return load_pretrained(pretraining)
    return pretraining[0] if isinstance(pretraining, tuple) else pretraining


def train_mean_model(
    build_model_func, X, Y, VAL_FRAC=0.2, EPOCHS=100, BATCH_SIZE=256, random_state=42, pretraining=DEFAULT_PRETRAINING
):
    """
    Train F, the network predicting the redshift, with MSE on log(1 + z), holding out a random VAL_FRAC of the
    objects for early stopping. Returns (F, history).

    The target is standardised with the training redshifts; a ToRedshift layer appended to the trained network maps
    its output back to z. pretraining: what F starts from: "popcosmos" (default, shipped with znn; X must be built
    with the default feature config), the path of a .keras file (see load_pretrained), a trained model (its F), or
    None (from scratch). From a pre-training, F keeps its ToRedshift normalisation; nothing is frozen.
    """
    tf.keras.backend.clear_session()

    idx_train, idx_val = train_test_split(np.arange(len(Y)), test_size=VAL_FRAC, random_state=random_state)
    T_train, T_val = np.log1p(Y[idx_train]), np.log1p(Y[idx_val])

    start = _start(pretraining)
    if start is None:
        model = build_model_func(input_shape=X.shape[1:])
        Y_mean, Y_std = T_train.mean(), T_train.std()
    else:
        start_model = tf.keras.models.clone_model(start)  # a copy, so start is left untouched
        start_model.set_weights(start.get_weights())
        to_z = start_model.layers[-1]
        Y_mean, Y_std = to_z.mean, to_z.std
        model = models.Model(start_model.inputs[0], to_z.input)

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    history = model.fit(
        X[idx_train],
        (T_train - Y_mean) / Y_std,
        validation_data=(X[idx_val], (T_val - Y_mean) / Y_std),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=_callbacks(),
        verbose=2,
    )
    return models.Model(model.inputs[0], ToRedshift(Y_mean, Y_std)(model.outputs[0])), history


def _student_t_nll(nu):
    """Keras loss: the Student-t negative log-likelihood (no constants) of residuals y, given the log squared scale."""

    def loss(y, log_scale2):
        return tf.reduce_mean(
            0.5 * log_scale2 + 0.5 * (nu + 1) * tf.math.log1p(tf.square(y) * tf.exp(-log_scale2) / nu)
        )

    return loss


def train_scale_model(
    build_model_func, X, Y, mean_model, nu=DEFAULT_NU, VAL_FRAC=0.2, EPOCHS=100, BATCH_SIZE=256, random_state=42
):
    """
    Train G, the network predicting the spread of F's residuals, with F (mean_model) fixed: G learns the log squared
    scale of a Student-t with nu degrees of freedom around F's prediction, by its negative log-likelihood. G is a new
    instance of build_model_func, from scratch, holding out a random VAL_FRAC for early stopping. Returns (G, history).

    G is trained in the standardised log(1 + z) of F; a fixed offset appended to the trained network puts its output
    in log(1 + z).
    """
    to_z = mean_model.layers[-1]
    T_pred = models.Model(mean_model.inputs[0], to_z.input).predict(X, batch_size=4096, verbose=0).ravel()
    R = (np.log1p(Y) - to_z.mean) / to_z.std - T_pred  # F's residuals, in its training units
    idx_train, idx_val = train_test_split(np.arange(len(Y)), test_size=VAL_FRAC, random_state=random_state)

    model = build_model_func(input_shape=X.shape[1:])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=_student_t_nll(nu))
    history = model.fit(
        X[idx_train],
        R[idx_train],
        validation_data=(X[idx_val], R[idx_val]),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=_callbacks(),
        verbose=2,
    )
    # a new (uncompiled) model, so its architecture (save_model_file) does not refer to the custom loss
    offset = layers.Rescaling(1.0, offset=float(np.log(to_z.std**2)))
    return models.Model(model.inputs[0], offset(model.outputs[0])), history


def _train_model(
    build_model_func,
    X,
    Y,
    nu=DEFAULT_NU,
    VAL_FRAC=0.2,
    EPOCHS=100,
    BATCH_SIZE=256,
    random_state=42,
    pretraining=DEFAULT_PRETRAINING,
):
    """
    Train the model on features X and redshifts Y: F (train_mean_model, from pretraining), then G on F's residuals on
    the same objects (train_scale_model, from scratch), both with the same VAL_FRAC held out for early stopping.

    Returns ((F, G, nu), (F's history, G's history)).
    """
    mean_model, mean_history = train_mean_model(
        build_model_func, X, Y, VAL_FRAC, EPOCHS, BATCH_SIZE, random_state, pretraining
    )
    scale_model, scale_history = train_scale_model(
        build_model_func, X, Y, mean_model, nu, VAL_FRAC, EPOCHS, BATCH_SIZE, random_state
    )
    return (mean_model, scale_model, float(nu)), (mean_history, scale_history)


def train_model(
    build_model_func,
    df,
    bands,
    ref_band,
    config=DEFAULT_FEATURE_CONFIG,
    filter_curves=None,
    Y=None,
    nu=DEFAULT_NU,
    VAL_FRAC=0.2,
    EPOCHS=100,
    BATCH_SIZE=256,
    random_state=42,
    pretraining=DEFAULT_PRETRAINING,
):
    """
    Train the model on a catalogue: its features (catalog_to_XY, on the photometry as given; the photometric errors
    are not used), then F and G (see the top of this module).

    df: the training catalogue (see znn.io.read_catalog).
    bands, ref_band, config, filter_curves: feature settings, as for catalog_to_XY.
    Y: reference redshifts, in the order of df; by default its "redshift" column.
    nu: degrees of freedom of the Student-t G learns.
    VAL_FRAC: random fraction held out for early stopping (the same objects for F and G).
    pretraining: what F starts from: "popcosmos" (default, shipped with znn; needs the default feature config), the
        path of a .keras file, a trained model (its F), or None (from scratch). G always starts from scratch.

    Returns ((F, G, nu), (F's history, G's history)).
    """
    X, Y_catalog = catalog_to_XY(df, bands, ref_band, config, filter_curves=filter_curves)
    Y = Y_catalog if Y is None else np.asarray(Y)
    return _train_model(build_model_func, X, Y, nu, VAL_FRAC, EPOCHS, BATCH_SIZE, random_state, pretraining)


def _predict(model, X, ids=None, zgrid=DEFAULT_ZGRID, min_scale=1e-3):
    """
    p(z) of features X with a trained model (F, G, nu): a Student-t in log(1 + z) with F's prediction as location,
    G's scale (floored at min_scale) and nu degrees of freedom, converted to z (Jacobian 1 / (1 + z)) and evaluated
    on zgrid.
    """
    mean_model, scale_model, nu = model
    mu = np.log1p(mean_model.predict(X, batch_size=4096, verbose=0).ravel())
    scale = np.maximum(np.exp(0.5 * scale_model.predict(X, batch_size=4096, verbose=0).ravel()), min_scale)
    pdfs = stats.t.pdf(np.log1p(zgrid), nu, loc=mu[:, None], scale=scale[:, None]) / (1 + zgrid)
    return package_pdfs(zgrid, pdfs, ids=ids)


def predict(
    model, df, bands, ref_band, config=DEFAULT_FEATURE_CONFIG, filter_curves=None, ids=None, zgrid=DEFAULT_ZGRID
):
    """
    Predict p(z) for a catalogue with a trained model (F, G, nu): a Student-t in log(1 + z) around F's prediction,
    with G's scale and nu degrees of freedom, converted to z and evaluated on zgrid.

    df: the catalogue (see znn.io.read_catalog).
    bands, ref_band, config, filter_curves: the feature settings the model was trained with, as for catalog_to_XY;
        load_model_file returns them as a dict to pass as **features.
    ids: per-object identifiers for the p(z). Defaults to the sample order index (0..N-1).

    The p(z) is predicted on the photometry as given: G's scale already includes the photometric noise of the data
    it was trained on.

    Returns the p(z) on zgrid, as a qp interp Ensemble (or the fallback DataFrame, see package_pdfs).
    """
    X, _ = catalog_to_XY(df, bands, ref_band, config, filter_curves=filter_curves)
    return _predict(model, X, ids=ids, zgrid=zgrid)


def save_model_file(path, model, bands, ref_band, config):
    """
    Save a trained model and everything needed to rebuild its inputs into a single pickle file.

    model: (F, G, nu), as returned by train_model.
    bands, ref_band, config: the feature settings the model was trained with (see znn.catalog_to_XY). The filter
        curves are loaded and stored too, so estimation does not depend on the filter files still existing.

    F and G are stored as their architecture (JSON) and weights rather than pickled Keras objects.
    """
    mean_model, scale_model, nu = model
    payload = {
        "mean": {"architecture": mean_model.to_json(), "weights": mean_model.get_weights()},
        "scale": {"architecture": scale_model.to_json(), "weights": scale_model.get_weights()},
        "nu": float(nu),
        "features": {
            "bands": bands,
            "ref_band": ref_band,
            "config": config,
            "filter_curves": load_filter_curves(bands),
        },
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def _from_json(network):
    model = tf.keras.models.model_from_json(network["architecture"])
    model.set_weights(network["weights"])
    return model


def load_model_file(path):
    """
    Load a model saved with save_model_file.

    Returns (model, features): model is (F, G, nu), and features holds the stored settings as keyword arguments for
    predict (and catalog_to_XY), i.e. ``pz = predict(model, df, **features)``.
    """
    with open(path, "rb") as f:
        payload = pickle.load(f)

    if "members" in payload:
        # the earlier layout: [(F, 0, 1), (G, log s**2, nu)], G's output in F's standardised units
        assert len(payload["members"]) == 2, f"{path} holds an ensemble, not a model"
        mean, scale = payload["members"]
        scale_model = _from_json(scale)
        scale_model = models.Model(
            scale_model.inputs[0], layers.Rescaling(1.0, offset=scale["Y_mean"])(scale_model.outputs[0])
        )
        return (_from_json(mean), scale_model, scale["Y_std"]), payload["features"]

    return (_from_json(payload["mean"]), _from_json(payload["scale"]), payload["nu"]), payload["features"]


def load_pretrained(name=DEFAULT_PRETRAINING):
    """
    A pre-trained F: one shipped with znn (name, e.g. "popcosmos", in PRETRAINED_DIR) or the path of a .keras file.
    Inputs must be built with the feature config it was trained with, the default one for "popcosmos".
    """
    path = name if name.endswith(".keras") else os.path.join(PRETRAINED_DIR, f"{name}.keras")
    return tf.keras.models.load_model(path)
