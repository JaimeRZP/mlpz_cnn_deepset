import json
import os
import pickle

import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
from tensorflow.keras import callbacks, layers, models

from .data import DEFAULT_FEATURE_CONFIG, catalog_to_XY, resample_photometry
from .io import load_filter_curves
from .qp_output import package_predictions

__all__ = [
    "build_model",
    "build_model_v2",
    "build_model_v2_k7",
    "build_model_res_se",
    "TARGETS",
    "LOSSES",
    "ToRedshift",
    "ensemble_predict",
    "ensemble_predict_resampled",
    "save_ensemble",
    "load_ensemble",
    "save_ensemble_file",
    "load_ensemble_file",
    "train_ensembles",
    "fine_tune_pre_trained_model",
]


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


def build_model_v2_k7(input_shape):
    # build_model_v2 with a first kernel of 7, so the first layer sees the broad shape of the SED
    model = models.Sequential(
        [
            layers.Conv1D(32, kernel_size=7, activation="relu", padding="same", input_shape=input_shape),
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


def build_model_res_se(input_shape):
    # residual squeeze-excitation net: kernel-7 stem, 3 stages (32/64/128 filters) of 2 blocks each,
    # pooling after the first two stages (32 -> 16 -> 8 positions), then the same head as build_model.
    # ~270k parameters, a fifth of build_model; train it with target="log1pz" (see train_ensembles)
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


# --- Target scaling and loss ---
# Targets: "z3" scales the redshift as z / 3 (the original scheme); "log1pz" standardises log(1 + z) with
# the mean and std of each fold's training redshifts.
TARGETS = ("z3", "log1pz")
# Losses: "mse", or "huber" with its delta given in dz / (1 + z) (see _make_loss).
LOSSES = ("mse", "huber")


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


def _make_loss(loss, target, Y_train, Y_std, huber_delta):
    """
    Keras loss in the normalised units the model is trained in.

    The Huber delta is given in log(1 + z), i.e. roughly in dz / (1 + z), so it means the same for both targets:
      - log1pz: delta / Y_std;
      - z3: an error of delta in log(1 + z) is delta * (1 + z) in z, taken at the median training redshift.
    """
    if loss == "mse":
        return "mse"
    if loss == "huber":
        if target == "log1pz":
            return tf.keras.losses.Huber(delta=huber_delta / Y_std)
        return tf.keras.losses.Huber(delta=huber_delta * (1 + np.median(Y_train)) / Y_std)
    raise ValueError(f"unknown loss '{loss}'. Available: {LOSSES}")


def _as_ensemble_member(model, target, Y_mean, Y_std):
    """
    Package a trained model as a (model, Y_mean, Y_std) ensemble member.

    Members are denormalised linearly, as model.predict(X) * Y_std + Y_mean. z3 is linear, so the member is
    (model, Y_mean, Y_std). log1pz is not, so a ToRedshift layer is appended and the member is (model, 0, 1):
    the model then outputs z, and ensemble_predict, ensemble_predict_resampled and saving/loading work as is.
    """
    if target == "z3":
        return model, Y_mean, Y_std
    outputs = ToRedshift(Y_mean, Y_std)(model.outputs[0])
    return models.Model(model.inputs[0], outputs), 0.0, 1.0


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


def train_ensembles(
    build_model_func,
    X,
    Y,
    N_SPLITS=10,
    EPOCHS=100,
    BATCH_SIZE=256,
    random_state=42,
    target="z3",
    loss="mse",
    huber_delta=0.05,
):
    """
    Train one model per K-fold split, each predicting a point redshift.

    X is either one feature array shared by all members, or a list of N_SPLITS arrays (same objects, in
    the same order as Y), one per member, e.g. a different photometric noise realization for each.

    target: how the redshift is scaled for training (see TARGETS).
      - "z3" (default): z / 3, the original scheme.
      - "log1pz": log(1 + z), standardised with each fold's training mean and std. A ToRedshift layer is
        appended to each trained model so it outputs z, and the member is (model, 0, 1).
    loss: "mse" (default) or "huber" (see LOSSES).
    huber_delta: the Huber delta in dz / (1 + z); only used with loss="huber".

    The defaults reproduce the original training. The best setups from the architecture tests were
    build_model_res_se and build_model_v2_k7 with target="log1pz", and build_model_res_se with
    target="log1pz", loss="huber".
    """
    if target not in TARGETS:
        raise ValueError(f"unknown target '{target}'. Available: {TARGETS}")
    if loss not in LOSSES:
        raise ValueError(f"unknown loss '{loss}'. Available: {LOSSES}")
    X_members = _inputs_per_member(X, N_SPLITS)
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=random_state)

    trained_models = []
    histories = []

    for fold, (idx_train, idx_val) in enumerate(kf.split(np.arange(len(Y)))):
        print(f"\n--- Fold {fold+1}/{N_SPLITS} ---")
        X = X_members[fold]

        # Clear session before each fold
        tf.keras.backend.clear_session()

        # ── Callbacks ─────────────────────────────────────────────────────────────────
        early_stop = callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
        reduce_lr = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1)

        # Split data using fold indices
        X_train, X_val = X[idx_train], X[idx_val]
        Y_train, Y_val = Y[idx_train], Y[idx_val]

        # Normalise Y using train statistics only
        # Y_mean, Y_std = Y_train.mean(), Y_train.std()
        if target == "z3":
            Y_mean = 0
            Y_std = 3
            Y_train_norm = (Y_train - Y_mean) / Y_std
            Y_val_norm = (Y_val - Y_mean) / Y_std
        else:  # "log1pz"
            Y_mean, Y_std = float(np.log1p(Y_train).mean()), float(np.log1p(Y_train).std())
            Y_train_norm = (np.log1p(Y_train) - Y_mean) / Y_std
            Y_val_norm = (np.log1p(Y_val) - Y_mean) / Y_std

        # Build fresh model for each fold
        model = build_model_func(input_shape=X_train.shape[1:])
        model.compile(optimizer="adam", loss=_make_loss(loss, target, Y_train, Y_std, huber_delta))

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

        trained_models.append(_as_ensemble_member(model, target, Y_mean, Y_std))  # save model + its normalisation
        histories.append(history)

    return trained_models, histories


def fine_tune_pre_trained_model(X_fortrain, Y_fortrain, pretrained_models=None, model_root="", nlayers_forzen=4):
    # need to provide either the model or the model_dir to load the model
    # will always load model if both are provided

    # Each member keeps the Y normalisation it was pre-trained with, so its output layer stays on the
    # scale it learned
    if model_root != "":
        with open(os.path.join(model_root, "norm_params.json"), "r") as f:
            pretrained_norms = [(p["Y_mean"], p["Y_std"]) for p in json.load(f)]
    elif pretrained_models is not None:
        pretrained_norms = [(Y_mean, Y_std) for _, Y_mean, Y_std in pretrained_models]
    else:
        raise ValueError("provide either pretrained_models or model_root")

    # one fold per pre-trained member
    n_splits = len(pretrained_norms)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    trained_models = []
    histories = []

    for fold, (idx_train, idx_val) in enumerate(kf.split(X_fortrain)):
        print(f"\n--- Fold {fold+1}/{n_splits} ---")
        tf.keras.backend.clear_session()

        # ── Callbacks ─────────────────────────────────────────────────────────────────
        early_stop = callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
        reduce_lr = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1)

        # Split data using fold indices
        X_train, X_val = X_fortrain[idx_train], X_fortrain[idx_val]
        Y_train, Y_val = Y_fortrain[idx_train], Y_fortrain[idx_val]

        if model_root != "":
            model = tf.keras.models.load_model(os.path.join(model_root, f"model_fold_{fold+1}.keras"))
        else:
            model = pretrained_models[fold][0]

        # Step 2: optionally freeze early layers (keep low-level features fixed)
        for layer in model.layers[:nlayers_forzen]:  # freeze first 4 layers
            layer.trainable = False

        # same normalization as the pre-training
        Y_mean, Y_std = pretrained_norms[fold]
        Y_train_norm = (Y_train - Y_mean) / Y_std
        Y_val_norm = (Y_val - Y_mean) / Y_std

        # Step 3: recompile with a LOWER learning rate — important
        # too high a learning rate will destroy what the model already learned
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),  # much lower than default 1e-3
            loss="mse",
            metrics=["mae"],
        )

        # Step 4: train on new data
        history = model.fit(
            X_train,
            Y_train_norm,
            validation_data=(X_val, Y_val_norm),
            epochs=60,  # fewer epochs than original training
            batch_size=256,
            callbacks=[early_stop, reduce_lr],
            verbose=2,
        )

        trained_models.append((model, Y_mean, Y_std))  # save model + its normalisation
        histories.append(history)

    return trained_models, histories
