import json
import os
import pickle

import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold, train_test_split
from tensorflow.keras import callbacks, layers, models

from .data import DEFAULT_FEATURE_CONFIG, catalog_to_XY, resample_photometry
from .io import load_filter_curves
from .qp_output import package_predictions

__all__ = [
    "build_model",
    "ToRedshift",
    "ensemble_predict",
    "ensemble_predict_resampled",
    "save_ensemble",
    "load_ensemble",
    "save_ensemble_file",
    "load_ensemble_file",
    "load_pretrained",
    "train_ensembles",
    "train_model",
    "train_from_catalog",
    "PRETRAINED_DIR",
    "DEFAULT_PRETRAINING",
]

# Pre-trained models shipped with znn, as <name>.keras: "popcosmos" is build_model trained on the pop-cosmos
# mock, on the default feature grid (see README)
PRETRAINED_DIR = os.path.join(os.path.dirname(__file__), "pretrained")
DEFAULT_PRETRAINING = "popcosmos"
# Layers kept fixed when the members are fine-tuned from a base model: the stem (input, conv, batch norm, ReLU)
N_FROZEN_LAYERS = 4


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
    # dense head. ~270k parameters. Trained on log(1 + z) (see train_ensembles)
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


def _inputs_per_member(X, n_members):
    """One input array per ensemble member: X itself for every member, or X's entries if it is a list."""
    if isinstance(X, (list, tuple)):
        if len(X) != n_members:
            raise ValueError(f"got {len(X)} input arrays for {n_members} ensemble members")
        return list(X)
    return [X] * n_members


# --- Prediction: average across all models ---
def ensemble_predict(trained_models, X_test, ids=None):
    """
    Predict redshifts with an ensemble of models and package the result as a
    per-object p(z).

    Parameters
    ----------
    trained_models : list of (model, Y_mean, Y_std)
        Ensemble members, as produced by :func:`train_ensembles`.
    X_test : array-like
        Input features to predict on.
    ids : array-like, optional
        Per-object identifiers. Defaults to the sample order index (0..N-1).

    Returns
    -------
    qp.Ensemble or pandas.DataFrame
        A qp Ensemble of per-object Gaussians (mean/std from the ensemble) if
        qp is installed, otherwise a DataFrame with "object_id", "mean", "std"
        columns. See :func:`znn.qp_output.package_predictions`.
    """
    predictions = []

    for model, Y_mean, Y_std in trained_models:
        # predict in normalised space, then denormalise
        y_pred_norm = model.predict(X_test)
        y_pred = y_pred_norm * Y_std + Y_mean  # denormalise
        predictions.append(y_pred)

    # average predictions across all models
    y_pred_mean = np.mean(predictions, axis=0)
    y_pred_std = np.std(predictions, axis=0)

    return package_predictions(y_pred_mean, y_pred_std, ids=ids)


def ensemble_predict_resampled(
    trained_models,
    df,
    errors,
    n_samples,
    rng,
    bands,
    ref_band,
    config=DEFAULT_FEATURE_CONFIG,
    filter_curves=None,
    ids=None,
    return_components=False,
):
    """
    Predict p(z) with the photometric errors propagated by Monte Carlo.

    Draws n_samples noise realizations of the catalogue's photometry (see resample_photometry), runs
    every ensemble member on every realization, and packages the n_samples x n_members predictions per
    object as a Gaussian p(z) with their mean and std. The number of realizations is independent of the
    number of members.

    trained_models: list of (model, Y_mean, Y_std), as returned by train_ensembles.
    df: the catalogue to predict on (see znn.io.read_catalog).
    errors: dict magnitude column -> its error column, as for resample_photometry.
    n_samples: number of noise realizations.
    rng: numpy Generator, e.g. np.random.default_rng(seed).
    bands, ref_band, config, filter_curves: feature settings, as for catalog_to_XY; a dict of these as
        returned by load_ensemble_file can be passed as **features.
    ids: per-object identifiers for the p(z), as for ensemble_predict.
    return_components: also return the split of the predicted std into its two sources.

    Returns the p(z) as from ensemble_predict. With return_components, returns (p(z), components), where
    components holds per-object stds that add in quadrature to the total (law of total variance):
      - "model": spread between members on the same realization, averaged over realizations;
      - "photometric": spread of the ensemble mean across realizations.
    """
    predictions = np.empty((n_samples, len(trained_models), len(df)))
    for s in range(n_samples):
        X, _ = catalog_to_XY(resample_photometry(df, errors, rng), bands, ref_band, config, filter_curves=filter_curves)
        for k, (model, Y_mean, Y_std) in enumerate(trained_models):
            predictions[s, k] = model.predict(X, batch_size=4096, verbose=0).ravel() * Y_std + Y_mean

    pz = package_predictions(predictions.mean(axis=(0, 1)), predictions.std(axis=(0, 1)), ids=ids)
    if not return_components:
        return pz

    components = {
        "model": np.sqrt(predictions.var(axis=1).mean(axis=0)),
        "photometric": predictions.mean(axis=1).std(axis=0),
    }
    return pz, components


def save_ensemble(trained_models, save_dir=""):
    """
    Save all ensemble models and their normalisation parameters.
    trained_models: list of (model, Y_mean, Y_std) tuples
    """
    os.makedirs(save_dir, exist_ok=True)

    # Save each model separately
    norm_params = []
    for fold, (model, Y_mean, Y_std) in enumerate(trained_models):
        # Save model weights
        model_path = os.path.join(save_dir, f"model_fold_{fold+1}.keras")
        model.save(model_path)
        print(f"Saved fold {fold+1} model to {model_path}")

        # Collect normalisation parameters
        norm_params.append({"fold": fold + 1, "Y_mean": float(Y_mean), "Y_std": float(Y_std)})

    # Save normalisation parameters as a single JSON file
    norm_path = os.path.join(save_dir, "norm_params.json")
    with open(norm_path, "w") as f:
        json.dump(norm_params, f, indent=2)
    print(f"Saved normalisation parameters to {norm_path}")


def load_ensemble(save_dir=""):
    """
    Load all ensemble models and their normalisation parameters.
    Returns: list of (model, Y_mean, Y_std) tuples
    """
    # Load normalisation parameters
    norm_path = os.path.join(save_dir, "norm_params.json")
    with open(norm_path, "r") as f:
        norm_params = json.load(f)

    # Load each model
    trained_models = []
    for params in norm_params:
        fold = params["fold"]
        model_path = os.path.join(save_dir, f"model_fold_{fold}.keras")

        model = tf.keras.models.load_model(model_path)
        Y_mean = params["Y_mean"]
        Y_std = params["Y_std"]

        trained_models.append((model, Y_mean, Y_std))
        print(f"Loaded fold {fold} model from {model_path}")

    return trained_models


def save_ensemble_file(path, trained_models, bands, ref_band, config):
    """
    Save an ensemble and everything needed to rebuild its inputs into a single pickle file.

    trained_models: list of (model, Y_mean, Y_std) tuples, as returned by train_ensembles.
    bands, ref_band, config: the feature settings the ensemble was trained with (see
        znn.catalog_to_XY). The filter curves are loaded and stored too, so estimation does not
        depend on the filter files still existing.

    Each member is stored as its architecture (JSON) and weights rather than a pickled Keras object.
    """
    payload = {
        "members": [
            {
                "architecture": model.to_json(),
                "weights": model.get_weights(),
                "Y_mean": float(Y_mean),
                "Y_std": float(Y_std),
            }
            for model, Y_mean, Y_std in trained_models
        ],
        "features": {
            "bands": bands,
            "ref_band": ref_band,
            "config": config,
            "filter_curves": load_filter_curves(bands),
        },
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def load_ensemble_file(path):
    """
    Load an ensemble saved with save_ensemble_file.

    Returns (trained_models, features): trained_models is a list of (model, Y_mean, Y_std) tuples
    for ensemble_predict, and features holds the stored settings as keyword arguments for
    catalog_to_XY, i.e. ``X, _ = catalog_to_XY(df, **features)``.
    """
    with open(path, "rb") as f:
        payload = pickle.load(f)

    trained_models = []
    for member in payload["members"]:
        model = tf.keras.models.model_from_json(member["architecture"])
        model.set_weights(member["weights"])
        trained_models.append((model, member["Y_mean"], member["Y_std"]))

    return trained_models, payload["features"]


def load_pretrained(name=DEFAULT_PRETRAINING):
    """
    A pre-trained model as an ensemble member (model, 0, 1): one shipped with znn (name, e.g. "popcosmos", in
    PRETRAINED_DIR) or the path of a .keras file. Inputs must be built with the feature config it was trained with,
    the default one for "popcosmos".
    """
    path = name if name.endswith(".keras") else os.path.join(PRETRAINED_DIR, f"{name}.keras")
    return tf.keras.models.load_model(path), 0.0, 1.0


def _start(pretraining):
    """The member training starts from: None (from scratch), load_pretrained(pretraining), or an ensemble's first."""
    if isinstance(pretraining, str):
        return load_pretrained(pretraining)
    return None if pretraining is None else pretraining[0]


def _train_member(
    build_model_func,
    X,
    Y,
    idx_train,
    idx_val,
    start,
    EPOCHS,
    BATCH_SIZE,
    learning_rate=1e-3,
    monitor=True,
    freeze_stem=False,
):
    """
    Train one member on X[idx_train] and return ((model, 0, 1), history).

    The target is log(1 + z), standardised with the training redshifts; a ToRedshift layer appended to the trained
    model maps its output back to z. From start = (model, 0, 1), the network under start's ToRedshift layer is
    copied, keeps that layer's normalisation and trains with every layer trainable, but, with freeze_stem, its
    first N_FROZEN_LAYERS. X[idx_val] is used for early stopping and learning-rate reduction, or, with monitor
    False, only goes in the history (all EPOCHS at a constant learning_rate).
    """
    tf.keras.backend.clear_session()

    X_train, X_val = X[idx_train], X[idx_val]
    T_train, T_val = np.log1p(Y[idx_train]), np.log1p(Y[idx_val])

    if start is None:
        model = build_model_func(input_shape=X_train.shape[1:])
        Y_mean, Y_std = T_train.mean(), T_train.std()
    else:
        start_model = tf.keras.models.clone_model(start[0])  # a copy, so start is left untouched
        start_model.set_weights(start[0].get_weights())
        to_z = start_model.layers[-1]
        Y_mean, Y_std = to_z.mean, to_z.std
        model = models.Model(start_model.inputs[0], to_z.input)
        for k, layer in enumerate(model.layers):
            layer.trainable = not (freeze_stem and k < N_FROZEN_LAYERS)

    # ── Callbacks ─────────────────────────────────────────────────────────────────
    early_stop = callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    reduce_lr = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss="mse")
    history = model.fit(
        X_train,
        (T_train - Y_mean) / Y_std,
        validation_data=(X_val, (T_val - Y_mean) / Y_std),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop, reduce_lr] if monitor else [],
        verbose=2,
    )
    model = models.Model(model.inputs[0], ToRedshift(Y_mean, Y_std)(model.outputs[0]))
    return (model, 0.0, 1.0), history


def train_ensembles(
    build_model_func, X, Y, N_SPLITS=10, EPOCHS=100, BATCH_SIZE=256, random_state=42, pretraining=DEFAULT_PRETRAINING
):
    """
    Train one model per K-fold split, each predicting a point redshift trained with MSE on log(1 + z).

    X is either one feature array shared by all members, or a list of N_SPLITS arrays (same objects, in
    the same order as Y), one per member, e.g. a different photometric noise realization for each.

    pretraining: what every member starts from: "popcosmos" (default, shipped with znn; X must be built with the
    default feature config), the path of a .keras file (see load_pretrained), an ensemble (its first member), or
    None (from scratch). Nothing is frozen: the pre-training only sets the starting weights.

    Each member is (model, 0, 1): the model ends in a ToRedshift layer and outputs z.
    """
    start = _start(pretraining)
    X_members = _inputs_per_member(X, N_SPLITS)
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=random_state)

    trained_models = []
    histories = []
    for fold, (idx_train, idx_val) in enumerate(kf.split(np.arange(len(Y)))):
        print(f"\n--- Fold {fold+1}/{N_SPLITS} ---")
        trained, history = _train_member(
            build_model_func, X_members[fold], Y, idx_train, idx_val, start, EPOCHS, BATCH_SIZE
        )
        trained_models.append(trained)
        histories.append(history)

    return trained_models, histories


def train_model(
    build_model_func, X, Y, VAL_FRAC=0.2, EPOCHS=100, BATCH_SIZE=256, random_state=42, pretraining=DEFAULT_PRETRAINING
):
    """
    Train a single model, e.g. a pre-training, holding out a random VAL_FRAC of the objects for early stopping.
    pretraining as for train_ensembles. Returns ((model, 0, 1), history).
    """
    idx_train, idx_val = train_test_split(np.arange(len(Y)), test_size=VAL_FRAC, random_state=random_state)
    return _train_member(build_model_func, X, Y, idx_train, idx_val, _start(pretraining), EPOCHS, BATCH_SIZE)


def train_from_catalog(
    build_model_func,
    df,
    Y,
    bands,
    ref_band,
    config=DEFAULT_FEATURE_CONFIG,
    filter_curves=None,
    N_MEMBERS=5,
    from_base=False,
    EPOCHS=100,
    FINETUNE_EPOCHS=10,
    FINETUNE_LR=3e-4,
    BATCH_SIZE=256,
    random_state=42,
    pretraining=DEFAULT_PRETRAINING,
    VAL_FRAC=0.2,
):
    """
    Train an ensemble of N_MEMBERS models on a catalogue, as train_ensembles: member k on its own K-fold cut,
    on the photometry as given (the photometric errors are not used in training).

    from_base: first train a base model from `pretraining` on all the objects (a random VAL_FRAC held out for
    early stopping), then fine-tune a copy of it per member for FINETUNE_EPOCHS at a constant FINETUNE_LR, with
    its stem (first N_FROZEN_LAYERS layers) frozen, the same in every member. The base has seen every member's
    validation fold, so the fine-tuning has no early stopping.

    df: the training catalogue (see znn.io.read_catalog), with Y its reference redshifts in the same order.
    bands, ref_band, config, filter_curves: feature settings, as for catalog_to_XY.

    Returns (trained_models, histories, base_history), base_history None without from_base.
    """
    X, _ = catalog_to_XY(df, bands, ref_band, config, filter_curves=filter_curves)
    start = _start(pretraining)
    base_history = None
    epochs, learning_rate, monitor = EPOCHS, 1e-3, True
    if from_base:
        print("\n=== Base model, on all objects ===")
        idx_train, idx_val = train_test_split(np.arange(len(Y)), test_size=VAL_FRAC, random_state=random_state)
        start, base_history = _train_member(build_model_func, X, Y, idx_train, idx_val, start, EPOCHS, BATCH_SIZE)
        epochs, learning_rate, monitor = FINETUNE_EPOCHS, FINETUNE_LR, False

    kf = KFold(n_splits=N_MEMBERS, shuffle=True, random_state=random_state)
    trained_models = []
    histories = []
    for member, (idx_train, idx_val) in enumerate(kf.split(np.arange(len(Y)))):
        print(f"\n--- Member {member+1}/{N_MEMBERS} ---")
        trained, history = _train_member(
            build_model_func, X, Y, idx_train, idx_val, start, epochs, BATCH_SIZE, learning_rate, monitor, from_base
        )
        trained_models.append(trained)
        histories.append(history)

    return trained_models, histories, base_history
