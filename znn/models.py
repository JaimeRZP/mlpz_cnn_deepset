import json
import os
import pickle

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import callbacks, layers, models

from .data import DEFAULT_FEATURE_CONFIG, catalog_to_XY, has_errors, resample_photometry
from .io import load_filter_curves
from .qp_output import DEFAULT_Z_BINS, package_samples

__all__ = [
    "build_model",
    "build_model_v2",
    "ensemble_predict",
    "ensemble_predict_resampled",
    "save_ensemble",
    "load_ensemble",
    "save_ensemble_file",
    "load_ensemble_file",
    "train_ensembles",
    "save_pretrained",
    "load_pretrained",
    "DEFAULT_PRETRAINING",
]

# Pre-trained ensembles shipped with the package, as <name>.json + <name>.npz (see save_pretrained)
PRETRAINED_DIR = os.path.join(os.path.dirname(__file__), "pretrained")
# Used by train_ensembles unless pretraining=None: trained on the pop-cosmos mock, on the default grid
DEFAULT_PRETRAINING = "popcosmos"


def build_model(input_shape):
    # 6 layers
    model = models.Sequential(
        [
            layers.Conv1D(32, kernel_size=3, activation="relu", padding="same", input_shape=input_shape),
            layers.MaxPooling1D(),
            layers.Conv1D(64, kernel_size=3, activation="relu", padding="same"),
            layers.MaxPooling1D(),
            layers.Conv1D(128, kernel_size=3, activation="relu", padding="same"),
            layers.MaxPooling1D(),
            layers.Conv1D(256, kernel_size=3, activation="relu", padding="same"),
            # layers.MaxPooling1D(),
            layers.Conv1D(512, kernel_size=3, activation="relu", padding="same"),
            layers.Conv1D(512, kernel_size=3, activation="relu", padding="same"),
            layers.GlobalAveragePooling1D(),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(1),
        ]
    )
    return model


def build_model_v2(input_shape):
    # 5 layers
    model = models.Sequential(
        [
            layers.Conv1D(32, kernel_size=3, activation="relu", padding="same", input_shape=input_shape),
            layers.MaxPooling1D(),
            layers.Conv1D(64, kernel_size=3, activation="relu", padding="same"),
            layers.MaxPooling1D(),
            layers.Conv1D(128, kernel_size=3, activation="relu", padding="same"),
            layers.MaxPooling1D(),
            layers.Conv1D(256, kernel_size=3, activation="relu", padding="same"),
            layers.MaxPooling1D(),
            layers.Conv1D(512, kernel_size=3, activation="relu", padding="same"),
            layers.GlobalAveragePooling1D(),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(1),
        ]
    )
    return model


def _inputs_per_member(X, n_members):
    """One input array per ensemble member: X itself for every member, or X's entries if it is a list."""
    if isinstance(X, (list, tuple)):
        if len(X) != n_members:
            raise ValueError(f"got {len(X)} input arrays for {n_members} ensemble members")
        return list(X)
    return [X] * n_members


# --- Prediction: pool the predictions of all models ---
def ensemble_predict(trained_models, X_test, ids=None, bins=DEFAULT_Z_BINS):
    """
    Predict redshifts with an ensemble of models and package the result as a
    per-object p(z): the histogram of the members' predictions.

    Parameters
    ----------
    trained_models : list of (model, Y_mean, Y_std)
        Ensemble members, as produced by :func:`train_ensembles`.
    X_test : array-like
        Input features to predict on.
    ids : array-like, optional
        Per-object identifiers. Defaults to the sample order index (0..N-1).
    bins : array-like
        Histogram bin edges of the p(z).

    Returns
    -------
    qp.Ensemble or pandas.DataFrame
        A qp Ensemble of per-object histograms if qp is installed, otherwise a
        DataFrame with "object_id", "mean", "std" columns. See
        :func:`znn.qp_output.package_samples`.
    """
    predictions = []

    for model, Y_mean, Y_std in trained_models:
        # predict in normalised space, then denormalise
        y_pred_norm = model.predict(X_test)
        y_pred = y_pred_norm.ravel() * Y_std + Y_mean  # denormalise
        predictions.append(y_pred)

    # one sample per member and object
    return package_samples(np.stack(predictions, axis=1), ids=ids, bins=bins)


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
    bins=DEFAULT_Z_BINS,
):
    """
    Predict p(z) with the photometric errors propagated by Monte Carlo.

    Draws n_samples noise realizations of the catalogue's photometry (see resample_photometry) and runs
    every ensemble member on every realization. The n_samples x n_members predictions per object are
    pooled, not averaged over members: the p(z) is their histogram on `bins` (see package_samples). The
    number of realizations is independent of the number of members. If df has no errors (see
    has_errors) there is only one realization, the photometry as given, whatever n_samples.

    trained_models: list of (model, Y_mean, Y_std), as returned by train_ensembles.
    df: the catalogue to predict on (see znn.io.read_catalog).
    errors: dict magnitude column -> its error column, as for resample_photometry.
    n_samples: number of noise realizations, when df has errors.
    rng: numpy Generator, e.g. np.random.default_rng(seed).
    bands, ref_band, config, filter_curves: feature settings, as for catalog_to_XY; a dict of these as
        returned by load_ensemble_file can be passed as **features.
    ids: per-object identifiers for the p(z), as for ensemble_predict.
    return_components: also return the split of the predicted std into its two sources.
    bins: histogram bin edges of the p(z).

    Returns the p(z), as a qp Ensemble (or the fallback DataFrame, see package_samples). With
    return_components, returns (p(z), components), where components holds per-object stds that add in
    quadrature to the total (law of total variance):
      - "model": spread between members on the same realization, averaged over realizations;
      - "photometric": spread of the ensemble mean across realizations.
    """
    noisy = has_errors(df, errors)
    if not noisy:
        n_samples = 1
    if filter_curves is None:
        filter_curves = load_filter_curves(bands)

    predictions = np.empty((n_samples, len(trained_models), len(df)))
    for s in range(n_samples):
        photometry = resample_photometry(df, errors, rng) if noisy else df
        X, _ = catalog_to_XY(photometry, bands, ref_band, config, filter_curves=filter_curves)
        for k, (model, Y_mean, Y_std) in enumerate(trained_models):
            predictions[s, k] = model.predict(X, batch_size=4096, verbose=0).ravel() * Y_std + Y_mean

    pz = package_samples(predictions.reshape(-1, len(df)).T, ids=ids, bins=bins)
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


def save_pretrained(path, trained_models, config, info=None):
    """
    Save an ensemble as a pre-training starting point: <path>.json (architecture, Y normalisations, feature
    config, provenance) and <path>.npz (weights). No pickle, so loading it cannot run code.

    trained_models: list of (model, Y_mean, Y_std), as returned by train_ensembles; all members must share
        the architecture.
    config: the feature config the inputs were built with (see znn.catalog_to_XY). Users of the
        pre-training must build their inputs on the same grid.
    info: optional dict of provenance (training data, settings, versions), stored as is.
    """
    meta = {
        "architecture": trained_models[0][0].to_json(),
        "members": [{"Y_mean": float(Y_mean), "Y_std": float(Y_std)} for _, Y_mean, Y_std in trained_models],
        "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in config.items()},
        "info": info or {},
    }
    weights = {f"m{k}_{i}": w for k, (model, _, _) in enumerate(trained_models) for i, w in enumerate(model.get_weights())}
    with open(f"{path}.json", "w") as f:
        json.dump(meta, f, indent=2, default=float)
    np.savez_compressed(f"{path}.npz", **weights)


def load_pretrained(name=DEFAULT_PRETRAINING):
    """
    Load a pre-trained ensemble saved with save_pretrained.

    name: the name of an ensemble shipped with znn (e.g. "popcosmos"), or a path without extension to a
        <path>.json / <path>.npz pair.

    Returns (trained_models, config, info): the members as (model, Y_mean, Y_std) tuples, the feature config
    their inputs were built with, and the provenance dict.
    """
    path = os.path.join(PRETRAINED_DIR, name) if os.path.exists(os.path.join(PRETRAINED_DIR, f"{name}.json")) else name
    with open(f"{path}.json") as f:
        meta = json.load(f)
    weights = np.load(f"{path}.npz")

    trained_models = []
    for k, member in enumerate(meta["members"]):
        model = tf.keras.models.model_from_json(meta["architecture"])
        model.set_weights([weights[f"m{k}_{i}"] for i in range(len(model.get_weights()))])
        trained_models.append((model, member["Y_mean"], member["Y_std"]))

    config = {k: (tuple(v) if isinstance(v, list) else v) for k, v in meta["config"].items()}
    return trained_models, config, meta["info"]


def _resolve_pretraining(pretraining):
    """The pre-trained members as a list of (model, Y_mean, Y_std), or None to train from scratch."""
    if pretraining is None:
        return None
    if isinstance(pretraining, str):
        if pretraining.endswith(".pkl"):
            return load_ensemble_file(pretraining)[0]
        return load_pretrained(pretraining)[0]
    return list(pretraining)


# Layers kept fixed when training from a pre-trained ensemble: the first two conv + pooling blocks
N_FROZEN_LAYERS = 4


def train_ensembles(
    build_model_func,
    X,
    Y,
    N_SPLITS=10,
    EPOCHS=100,
    BATCH_SIZE=256,
    random_state=42,
    pretraining=DEFAULT_PRETRAINING,
    VAL_FRAC=0.2,
    pretrained_member=0,
):
    """
    Train an ensemble of N_SPLITS models, each predicting a point redshift trained with MSE.

    Every member uses the same data cut: one split of the objects (random_state), training on 1 - VAL_FRAC
    of them and using the other VAL_FRAC for early stopping and learning-rate reduction. With
    pre-training, every member also starts from the same pre-trained member (pretrained_member), so
    members differ only through their inputs (and the randomness of training).

    X is either one feature array shared by all members, or a list of N_SPLITS arrays (same objects, in
    the same order as Y), one per member, e.g. a different photometric noise realization for each.

    pretraining: where each member starts.
      - "popcosmos" (default): the ensemble shipped with znn, trained on the pop-cosmos mock (see
        load_pretrained). Its inputs were built with the default feature config, so X must be too (in
        particular the default lambda_range and n_bins).
      - None: every member from scratch, built with build_model_func.
      - your own: the name or path of a save_pretrained pair, the path of a save_ensemble_file .pkl, or a
        list of (model, Y_mean, Y_std) as returned by train_ensembles. Its inputs must be on the same grid
        as X.
      With pre-training, every member starts from a copy of pre-trained member pretrained_member (or, with
      pretrained_member=None, member k from pre-trained member k mod K, for K pre-trained members), with its
      first N_FROZEN_LAYERS layers frozen and its Y normalisation kept;
      everything else (optimizer, learning rate, epochs, callbacks, split) is the same as from scratch, and
      build_model_func is not used. The pre-trained models are not modified.
    """
    pretrained = _resolve_pretraining(pretraining)
    X_members = _inputs_per_member(X, N_SPLITS)
    if pretrained is not None:
        if pretrained_member is not None and not 0 <= pretrained_member < len(pretrained):
            raise ValueError(f"pretrained_member={pretrained_member}, but there are {len(pretrained)} pre-trained members")
        expected = tuple(pretrained[0][0].input_shape[1:])
        if tuple(X_members[0].shape[1:]) != expected:
            raise ValueError(
                f"inputs have shape {X_members[0].shape[1:]} but the pre-trained models expect {expected}; build "
                "them with the same feature config as the pre-training, or pass pretraining=None"
            )

    # One data cut shared by all members
    idx_train, idx_val = train_test_split(np.arange(len(Y)), test_size=VAL_FRAC, random_state=random_state)

    trained_models = []
    histories = []

    for member in range(N_SPLITS):
        print(f"\n--- Member {member+1}/{N_SPLITS} ---")
        X = X_members[member]

        # Clear session before each member
        tf.keras.backend.clear_session()

        # ── Callbacks ─────────────────────────────────────────────────────────────────
        early_stop = callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
        reduce_lr = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1)

        # Split data using the shared indices
        X_train, X_val = X[idx_train], X[idx_val]
        Y_train, Y_val = Y[idx_train], Y[idx_val]

        if pretrained is None:
            # Build fresh model for each member
            model = build_model_func(input_shape=X_train.shape[1:])
            Y_mean = 0
            Y_std = 3
        else:
            # Copy a pre-trained member (so it is left untouched): the same one for every member, or cycling
            # through them; freeze its early layers, and keep the Y normalisation its output layer learned
            start = member % len(pretrained) if pretrained_member is None else pretrained_member
            pretrained_model, Y_mean, Y_std = pretrained[start]
            model = tf.keras.models.clone_model(pretrained_model)
            model.set_weights(pretrained_model.get_weights())
            for layer in model.layers[:N_FROZEN_LAYERS]:
                layer.trainable = False

        # Normalise Y
        Y_train_norm = (Y_train - Y_mean) / Y_std
        Y_val_norm = (Y_val - Y_mean) / Y_std

        model.compile(optimizer="adam", loss="mse")

        # Train
        history = model.fit(
            X_train,
            Y_train_norm,
            validation_data=(X_val, Y_val_norm),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[early_stop, reduce_lr],
            verbose=2,
        )

        trained_models.append((model, Y_mean, Y_std))  # save model + its normalisation
        histories.append(history)

    return trained_models, histories
