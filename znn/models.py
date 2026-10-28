import json
import os
import pickle

import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold, train_test_split
from tensorflow.keras import callbacks, layers, models

from .data import DEFAULT_FEATURE_CONFIG, catalog_to_XY, has_errors, resample_photometry
from .io import load_filter_curves
from .qp_output import package_samples

__all__ = [
    "build_model",
    "build_model_v2",
    "ensemble_predict",
    "predict_from_catalog",
    "ensemble_predict_resampled",
    "save_ensemble",
    "load_ensemble",
    "save_ensemble_file",
    "load_ensemble_file",
    "train_ensembles",
    "train_from_catalog",
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
def _member_predictions(trained_models, X):
    """Every member's denormalised redshift prediction on one input array: shape (n_members, n_objects)."""
    return np.stack(
        [model.predict(X, batch_size=4096, verbose=0).ravel() * Y_std + Y_mean for model, Y_mean, Y_std in trained_models]
    )


def _package_predictions(predictions, ids, return_components):
    """
    The p(z) from predictions of shape (n_realizations, n_members, n_objects): a Gaussian with the mean and std
    of each object's n_realizations x n_members predictions, pooled (see package_samples). With
    return_components, returns (p(z), components), where components holds per-object stds that add in
    quadrature to the total (law of total variance):
      - "model": spread between members on the same realization, averaged over realizations;
      - "photometric": spread of the ensemble mean across realizations.
    """
    pz = package_samples(predictions.reshape(-1, predictions.shape[-1]).T, ids=ids)
    if not return_components:
        return pz
    components = {
        "model": np.sqrt(predictions.var(axis=1).mean(axis=0)),
        "photometric": predictions.mean(axis=1).std(axis=0),
    }
    return pz, components


def ensemble_predict(trained_models, X_test, ids=None):
    """
    Predict redshifts with an ensemble of models from prebuilt inputs and package the result as a
    per-object p(z): a Gaussian with the mean and std of the members' predictions.

    The matrix-level building block, for inputs already built (see catalog_to_XY): one realization of the
    photometry, so the p(z) width is the members' spread only. predict_from_catalog predicts from a
    catalogue, with its photometric errors propagated.

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
        A qp Ensemble of per-object Gaussians if qp is installed, otherwise a
        DataFrame with "object_id", "mean", "std" columns. See
        :func:`znn.qp_output.package_samples`.
    """
    return _package_predictions(_member_predictions(trained_models, X_test)[None], ids, return_components=False)


def predict_from_catalog(
    trained_models,
    df,
    bands,
    ref_band,
    config=DEFAULT_FEATURE_CONFIG,
    filter_curves=None,
    errors=None,
    n_samples=50,
    rng=None,
    ids=None,
    return_components=False,
):
    """
    Predict p(z) for a catalogue, with its photometric errors propagated by Monte Carlo.

    Draws n_samples noise realizations of the catalogue's photometry (see resample_photometry), runs every
    ensemble member on every realization, and pools the n_samples x n_members predictions per object (not
    averaged over members) into a Gaussian p(z) with their mean and std (see package_samples). The number
    of realizations is independent of the number of members. If df has no errors (errors=None, or see
    has_errors) there is only one realization, the photometry as given, whatever n_samples.

    trained_models: list of (model, Y_mean, Y_std), as returned by train_ensembles or train_from_catalog.
    df: the catalogue to predict on (see znn.io.read_catalog).
    bands, ref_band, config, filter_curves: feature settings, as for catalog_to_XY; a dict of these as
        returned by load_ensemble_file can be passed as **features.
    errors: dict magnitude column -> its error column, as for resample_photometry.
    n_samples: number of noise realizations, when df has errors.
    rng: numpy Generator, e.g. np.random.default_rng(seed); needed when df has errors.
    ids: per-object identifiers for the p(z), as for ensemble_predict.
    return_components: also return the split of the predicted std into its two sources ("model" and
        "photometric", adding in quadrature).

    Returns the p(z), as a qp Ensemble (or the fallback DataFrame, see package_samples), or (p(z),
    components) with return_components.
    """
    noisy = has_errors(df, errors)
    if noisy and rng is None:
        raise ValueError("rng is needed to draw the noise realizations of a catalogue with errors")
    if filter_curves is None:
        filter_curves = load_filter_curves(bands)

    predictions = np.stack(
        [
            _member_predictions(
                trained_models,
                catalog_to_XY(resample_photometry(df, errors, rng) if noisy else df, bands, ref_band, config, filter_curves=filter_curves)[0],
            )
            for _ in range(n_samples if noisy else 1)
        ]
    )
    return _package_predictions(predictions, ids, return_components)


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
    """predict_from_catalog with its arguments in the older order, kept for the notebooks that use it."""
    return predict_from_catalog(
        trained_models, df, bands, ref_band, config, filter_curves, errors=errors, n_samples=n_samples, rng=rng,
        ids=ids, return_components=return_components,
    )


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
    """The pre-trained start as (model, Y_mean, Y_std): the first member of the given ensemble; None from scratch."""
    if pretraining is None:
        return None
    if isinstance(pretraining, str):
        if pretraining.endswith(".pkl"):
            return load_ensemble_file(pretraining)[0][0]
        return load_pretrained(pretraining)[0][0]
    return list(pretraining)[0]


# Layers kept fixed when training from a pre-trained model: the first two conv + pooling blocks
N_FROZEN_LAYERS = 4


def _check_inputs(start, input_shape):
    """Raise if inputs of input_shape do not fit the pre-trained start (model, Y_mean, Y_std); None is from scratch."""
    if start is None:
        return
    expected = tuple(start[0].input_shape[1:])
    if tuple(input_shape) != expected:
        raise ValueError(
            f"inputs have shape {tuple(input_shape)} but the pre-trained model expects {expected}; build them "
            "with the same feature config as the pre-training, or pass pretraining=None"
        )


def _kfold_cuts(n_objects, n_members, random_state):
    """(training, validation) indices per member: a K-fold split with K = n_members, member k validating on fold k."""
    if n_members < 2:
        raise ValueError(f"a K-fold ensemble needs at least 2 members, not {n_members}")
    return list(KFold(n_splits=n_members, shuffle=True, random_state=random_state).split(np.arange(n_objects)))


def _train_member(build_model_func, X, Y, idx_train, idx_val, start, EPOCHS, BATCH_SIZE, LEARNING_RATE):
    """
    Train one member on X[idx_train], with X[idx_val] for early stopping and learning-rate reduction: from
    scratch with build_model_func (start None), or from a copy of start = (model, Y_mean, Y_std) with its first
    N_FROZEN_LAYERS layers frozen and its Y normalisation kept. Returns ((model, Y_mean, Y_std), history).
    """
    # Clear session before each member
    tf.keras.backend.clear_session()

    # ── Callbacks ─────────────────────────────────────────────────────────────────
    early_stop = callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    reduce_lr = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1)

    X_train, X_val = X[idx_train], X[idx_val]
    Y_train, Y_val = Y[idx_train], Y[idx_val]

    if start is None:
        # Build a fresh model
        model = build_model_func(input_shape=X_train.shape[1:])
        Y_mean = 0
        Y_std = 3
    else:
        # Copy the start (so it is left untouched), freeze its early layers, and keep the Y normalisation its
        # output layer learned
        start_model, Y_mean, Y_std = start
        model = tf.keras.models.clone_model(start_model)
        model.set_weights(start_model.get_weights())
        for layer in model.layers[:N_FROZEN_LAYERS]:
            layer.trainable = False

    # Normalise Y
    Y_train_norm = (Y_train - Y_mean) / Y_std
    Y_val_norm = (Y_val - Y_mean) / Y_std

    model.compile(optimizer=tf.keras.optimizers.Adam(LEARNING_RATE), loss="mse")

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
    return (model, Y_mean, Y_std), history  # model + its normalisation


def train_ensembles(
    build_model_func,
    X,
    Y,
    N_SPLITS=10,
    EPOCHS=100,
    BATCH_SIZE=256,
    random_state=42,
    pretraining=DEFAULT_PRETRAINING,
    LEARNING_RATE=1e-3,
):
    """
    Train an ensemble of N_SPLITS models from prebuilt inputs, each predicting a point redshift trained with
    MSE.

    The matrix-level building block, for inputs already built (see catalog_to_XY). train_from_catalog trains
    from a catalogue, with a noise realization of its photometry per member.

    X is either one feature array shared by all members, or a list of N_SPLITS arrays (same objects, in
    the same order as Y), one per member.

    The objects are split K-fold, K = N_SPLITS, by random_state: member k trains on all folds but fold k, and
    uses fold k for early stopping and learning-rate reduction, so the members differ by data cut.

    LEARNING_RATE: initial Adam learning rate (1e-3 is the Adam default), reduced on validation plateaus.

    pretraining: the one model every member starts from.
      - "popcosmos" (default): the model shipped with znn, trained on the pop-cosmos mock (see
        load_pretrained). Its inputs were built with the default feature config, so X must be too (in
        particular the default lambda_range and n_bins).
      - None: every member from scratch, built with build_model_func.
      - your own: the name or path of a save_pretrained pair, the path of a save_ensemble_file .pkl, or a
        list of (model, Y_mean, Y_std) as returned by train_ensembles; of an ensemble, the first member is
        used. Its inputs must be on the same grid as X.
      With pre-training, every member starts from a copy of that model, with its first N_FROZEN_LAYERS
      layers frozen and its Y normalisation kept; everything else (optimizer, learning rate, epochs,
      callbacks, split) is as from scratch, and build_model_func is not used. The pre-trained model is not
      modified.

    Returns (trained_models, histories): the members as (model, Y_mean, Y_std) and their Keras histories.
    """
    start = _resolve_pretraining(pretraining)
    X_members = _inputs_per_member(X, N_SPLITS)
    _check_inputs(start, X_members[0].shape[1:])

    trained_models = []
    histories = []
    for member, (idx_train, idx_val) in enumerate(_kfold_cuts(len(Y), N_SPLITS, random_state)):
        print(f"\n--- Member {member+1}/{N_SPLITS} ---")
        trained, history = _train_member(
            build_model_func, X_members[member], Y, idx_train, idx_val, start, EPOCHS, BATCH_SIZE, LEARNING_RATE
        )
        trained_models.append(trained)
        histories.append(history)

    return trained_models, histories


def train_from_catalog(
    build_model_func,
    df,
    Y,
    bands,
    ref_band,
    config=DEFAULT_FEATURE_CONFIG,
    filter_curves=None,
    errors=None,
    rng=None,
    N_MEMBERS=5,
    from_base=False,
    EPOCHS=100,
    FINETUNE_EPOCHS=10,
    FINETUNE_LR=1e-4,
    BATCH_SIZE=256,
    random_state=42,
    pretraining=DEFAULT_PRETRAINING,
    VAL_FRAC=0.2,
):
    """
    Train an ensemble of N_MEMBERS models on a catalogue, each on its own noise realization of the
    photometry and its own K-fold data cut.

    Member k is trained on a noise realization of the catalogue's photometry drawn from its errors (see
    resample_photometry), just before it is trained, so N_MEMBERS can be large without holding all the
    realizations in memory. If df has no errors (errors=None, or see has_errors), every member is trained on
    the photometry as given and the members differ only by data cut (and the randomness of training).

    The objects are split K-fold, K = N_MEMBERS, as in train_ensembles. pretraining and random_state are as
    for train_ensembles.

    from_base: train in two stages, cheaper per member:
      1. a base model, from `pretraining`, on all the objects and the photometry as given (a random VAL_FRAC
         of them is held out for its early stopping), up to EPOCHS epochs;
      2. every member is a copy of the base model fine-tuned on its own noise realization and data cut, at
         the smaller learning rate FINETUNE_LR for up to FINETUNE_EPOCHS epochs, with the base model's first
         N_FROZEN_LAYERS layers frozen and its Y normalisation.
      The base model has seen every member's validation objects, so the members' early stopping is not on
      unseen data. Without from_base every member is trained in full from `pretraining` (up to EPOCHS, at the
      Adam default learning rate).

    df: the training catalogue (see znn.io.read_catalog), with Y its reference redshifts in the same order.
    bands, ref_band, config, filter_curves: feature settings, as for catalog_to_XY.
    errors: dict magnitude column -> its error column, as for resample_photometry.
    rng: numpy Generator, e.g. np.random.default_rng(seed); needed when df has errors.

    Returns (trained_models, histories, base_history): the members as (model, Y_mean, Y_std), their Keras
    histories, and that of the base model (None without from_base).
    """
    noisy = has_errors(df, errors)
    if noisy and rng is None:
        raise ValueError("rng is needed to draw the noise realizations of a catalogue with errors")
    if filter_curves is None:
        filter_curves = load_filter_curves(bands)

    def inputs(photometry):
        return catalog_to_XY(photometry, bands, ref_band, config, filter_curves=filter_curves)[0]

    X_given = inputs(df)
    start = _resolve_pretraining(pretraining)
    _check_inputs(start, X_given.shape[1:])

    base_history = None
    epochs, learning_rate = EPOCHS, 1e-3
    if from_base:
        print("\n=== Base model, on all objects with the photometry as given ===")
        idx_train, idx_val = train_test_split(np.arange(len(Y)), test_size=VAL_FRAC, random_state=random_state)
        start, base_history = _train_member(build_model_func, X_given, Y, idx_train, idx_val, start, EPOCHS, BATCH_SIZE, 1e-3)
        epochs, learning_rate = FINETUNE_EPOCHS, FINETUNE_LR

    trained_models = []
    histories = []
    for member, (idx_train, idx_val) in enumerate(_kfold_cuts(len(Y), N_MEMBERS, random_state)):
        print(f"\n--- Member {member+1}/{N_MEMBERS}" + (" (its own noise realization)" if noisy else "") + " ---")
        X = inputs(resample_photometry(df, errors, rng)) if noisy else X_given
        trained, history = _train_member(build_model_func, X, Y, idx_train, idx_val, start, epochs, BATCH_SIZE, learning_rate)
        trained_models.append(trained)
        histories.append(history)

    return trained_models, histories, base_history
