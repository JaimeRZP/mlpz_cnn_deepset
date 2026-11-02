import glob
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
    "build_model_v2_k7",
    "build_model_res_se",
    "TARGETS",
    "LOSSES",
    "ToRedshift",
    "ensemble_predict",
    "predict_from_catalog",
    "ensemble_predict_resampled",
    "save_ensemble",
    "load_ensemble",
    "save_ensemble_file",
    "load_ensemble_file",
    "train_ensembles",
    "train_model",
    "train_from_catalog",
    "save_pretrained",
    "load_pretrained",
    "pretrained_name",
    "PRETRAINED_DIR",
    "DEFAULT_PRETRAINING",
]

# Pre-trained ensembles shipped with the package, as <name>_<build_model>_<target>.json + .npz, one per
# architecture and target (see pretrained_name and save_pretrained)
PRETRAINED_DIR = os.path.join(os.path.dirname(__file__), "pretrained")
# Used by train_ensembles unless pretraining=None: trained on the pop-cosmos mock, on the default grid; the variant
# of the architecture and target being trained is used
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


def build_model_v2_k7(input_shape):
    # 5 convs, 4 poolings, with a first kernel of 7, so the first layer sees the broad shape of the SED
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
      - z3 (or any linear normalisation): an error of delta in log(1 + z) is delta * (1 + z) in z, taken at the
        median training redshift.
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
    the model then outputs z, and prediction, saving/loading and pre-training work as is.
    """
    if target == "z3":
        return model, Y_mean, Y_std
    outputs = ToRedshift(Y_mean, Y_std)(model.outputs[0])
    return models.Model(model.inputs[0], outputs), 0.0, 1.0


def _model_target(model):
    """The target a trained model was trained with: log1pz models end in ToRedshift, z3 ones do not."""
    return "log1pz" if isinstance(model.layers[-1], ToRedshift) else "z3"


def _check_target_loss(target, loss):
    if target not in TARGETS:
        raise ValueError(f"unknown target '{target}'. Available: {TARGETS}")
    if loss not in LOSSES:
        raise ValueError(f"unknown loss '{loss}'. Available: {LOSSES}")


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


def pretrained_name(name, build_model_func, target):
    """
    The file name (no extension) of the pre-training called name of build_model_func's architecture trained with
    target, e.g. popcosmos_build_model_res_se_log1pz: the pre-trainings shipped in PRETRAINED_DIR are named so,
    and pretraining=name picks the one matching the build_model_func and target it trains with.
    """
    return f"{name}_{build_model_func.__name__}_{target}"


def save_pretrained(path, trained_models, config, info=None, *, build_model_func, frozen_until=None):
    """
    Save an ensemble as a pre-training starting point: <path>.json (architecture, Y normalisations, feature
    config, provenance) and <path>.npz (weights). No pickle, so loading it cannot run code.

    trained_models: list of (model, Y_mean, Y_std), as returned by train_ensembles. Members whose
        architecture differs from the first's (e.g. log1pz members, whose ToRedshift layer holds their own
        normalisation) store their own.
    config: the feature config the inputs were built with (see znn.catalog_to_XY). Users of the
        pre-training must build their inputs on the same grid.
    info: optional dict of provenance (training data, settings, versions), stored as is.
    build_model_func: the function the members were built with. Its name is stored with the target the members
        were trained with, and both are checked against those of any training started from this pre-training.
    frozen_until: the name of a layer of the members. Training started from this pre-training keeps that layer and
        every layer it depends on frozen (see _freeze_until), e.g. "activation_2", the end of the first residual
        block of build_model_res_se. None (default): the first N_FROZEN_LAYERS layers.

    To ship a pre-training with znn, save it to os.path.join(PRETRAINED_DIR, pretrained_name(name,
    build_model_func, target)).
    """
    if frozen_until is not None:
        trained_models[0][0].get_layer(frozen_until)  # raises if the members have no such layer
    architecture = trained_models[0][0].to_json()
    members = []
    for model, Y_mean, Y_std in trained_models:
        member = {"Y_mean": float(Y_mean), "Y_std": float(Y_std)}
        if model.to_json() != architecture:
            member["architecture"] = model.to_json()
        members.append(member)
    meta = {
        "build_model": build_model_func.__name__,
        "target": _model_target(trained_models[0][0]),
        "frozen_until": frozen_until,
        "architecture": architecture,
        "members": members,
        "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in config.items()},
        "info": info or {},
    }
    weights = {f"m{k}_{i}": w for k, (model, _, _) in enumerate(trained_models) for i, w in enumerate(model.get_weights())}
    with open(f"{path}.json", "w") as f:
        json.dump(meta, f, indent=2, default=float)
    np.savez_compressed(f"{path}.npz", **weights)


def _shipped_variants(name):
    """(build_model, target) of every pre-training called name shipped in PRETRAINED_DIR (see pretrained_name)."""
    variants = []
    for path in sorted(glob.glob(os.path.join(PRETRAINED_DIR, f"{name}_*.json"))):
        with open(path) as f:
            meta = json.load(f)
        if os.path.basename(path) == f"{name}_{meta['build_model']}_{meta['target']}.json":
            variants.append((meta["build_model"], meta["target"]))
    return variants


def _pretrained_path(name, build_model_func, target):
    """
    The <path>.json / <path>.npz pair to load for pretraining=name: the variant of a shipped pre-training matching
    build_model_func and target, or name itself if no pre-training of that name is shipped (a path).
    """
    variants = _shipped_variants(name)
    if not variants:
        return name
    if build_model_func is None:
        raise ValueError(f"pass build_model_func to choose which '{name}' pre-training to use")
    if (build_model_func.__name__, target) not in variants:
        available = ", ".join(f"{b} with target='{t}'" for b, t in variants)
        raise ValueError(
            f"znn ships no '{name}' pre-training of {build_model_func.__name__} with target='{target}' (available: "
            f"{available}); train one (see save_pretrained) or pass pretraining=None"
        )
    return os.path.join(PRETRAINED_DIR, pretrained_name(name, build_model_func, target))


def _read_pretrained(path):
    """The members saved with save_pretrained at path, as (model, Y_mean, Y_std) tuples, and the stored metadata."""
    with open(f"{path}.json") as f:
        meta = json.load(f)
    weights = np.load(f"{path}.npz")

    trained_models = []
    for k, member in enumerate(meta["members"]):
        model = tf.keras.models.model_from_json(member.get("architecture", meta["architecture"]))
        model.set_weights([weights[f"m{k}_{i}"] for i in range(len(model.get_weights()))])
        trained_models.append((model, member["Y_mean"], member["Y_std"]))
    return trained_models, meta


def load_pretrained(name=DEFAULT_PRETRAINING, build_model_func=build_model, target="z3"):
    """
    Load a pre-trained ensemble saved with save_pretrained.

    name: the name of a pre-training shipped with znn (e.g. "popcosmos"), of which the variant of
        build_model_func's architecture trained with target is loaded (see pretrained_name); or a path without
        extension to a <path>.json / <path>.npz pair, loaded as is.

    Returns (trained_models, config, info): the members as (model, Y_mean, Y_std) tuples, the feature config
    their inputs were built with, and the provenance dict.
    """
    trained_models, meta = _read_pretrained(_pretrained_path(name, build_model_func, target))
    config = {k: (tuple(v) if isinstance(v, list) else v) for k, v in meta["config"].items()}
    return trained_models, config, meta["info"]


def _resolve_pretraining(pretraining, build_model_func, target):
    """
    The pre-trained start as (model, Y_mean, Y_std): the first member of the given ensemble; None from scratch. A
    shipped pre-training is the variant of build_model_func's architecture and target; one given by path must have
    been saved with the same build_model_func. The target of any start is checked by _check_target. The layers a
    save_pretrained pair names in frozen_until are marked frozen (see _freeze_until).
    """
    if pretraining is None:
        return None
    if isinstance(pretraining, str):
        if pretraining.endswith(".pkl"):
            return load_ensemble_file(pretraining)[0][0]
        trained_models, meta = _read_pretrained(_pretrained_path(pretraining, build_model_func, target))
        if build_model_func is not None and meta.get("build_model", build_model_func.__name__) != build_model_func.__name__:
            raise ValueError(
                f"the pre-training '{pretraining}' is a {meta['build_model']}, not a {build_model_func.__name__}; "
                "use a pre-training of this architecture, or pass pretraining=None"
            )
        if meta.get("frozen_until"):
            _freeze_until(trained_models[0][0], meta["frozen_until"])
        return trained_models[0]
    return list(pretraining)[0]


# Layers kept fixed when training from a pre-trained model: the first two conv + pooling blocks of build_model. A
# pre-training can set its own instead (frozen_until, see save_pretrained)
N_FROZEN_LAYERS = 4


def _freeze_until(model, name):
    """Freeze the layer called name and every layer its output depends on, whatever order model.layers lists them in."""
    for layer in models.Model(model.inputs, model.get_layer(name).output).layers:
        layer.trainable = False


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


def _check_target(start, target):
    """
    Raise if the pre-trained start (model, Y_mean, Y_std) was trained with another target than target; None is from
    scratch. A log1pz model ends in ToRedshift, a z3 one does not.
    """
    if start is None:
        return
    start_target = _model_target(start[0])
    if start_target != target:
        raise ValueError(
            f"the pre-trained model was trained with target='{start_target}', not '{target}'; pass the same target, "
            "use a pre-training with this target, or pass pretraining=None"
        )


def _kfold_cuts(n_objects, n_members, random_state):
    """(training, validation) indices per member: a K-fold split with K = n_members, member k validating on fold k."""
    if n_members < 2:
        raise ValueError(f"a K-fold ensemble needs at least 2 members, not {n_members}")
    return list(KFold(n_splits=n_members, shuffle=True, random_state=random_state).split(np.arange(n_objects)))


def _train_member(
    build_model_func,
    X,
    Y,
    idx_train,
    idx_val,
    start,
    EPOCHS,
    BATCH_SIZE,
    LEARNING_RATE,
    monitor=True,
    target="z3",
    loss="mse",
    huber_delta=0.05,
    freeze=True,
):
    """
    Train one member on X[idx_train], with X[idx_val] for early stopping and learning-rate reduction: from
    scratch with build_model_func (start None), or from a copy of start = (model, Y_mean, Y_std) with its first
    N_FROZEN_LAYERS layers frozen, as well as any layer frozen in start (e.g. by a pre-training's frozen_until, and
    so in every model trained from it), and its Y normalisation kept. With freeze False, nothing of start is frozen:
    it is only the starting point of every layer. With monitor False, X[idx_val] only goes in the
    history: the member trains for all EPOCHS at a constant LEARNING_RATE. Returns ((model, Y_mean, Y_std), history).

    target and loss are as for train_ensembles; a start must have been trained with the same target. A log1pz
    start is trained without its ToRedshift layer, in the standardised log(1 + z) that layer holds, and gets it
    back afterwards.
    """
    _check_target(start, target)
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
        if target == "z3":
            Y_mean = 0
            Y_std = 3
        else:  # "log1pz": standardised with this member's training redshifts
            Y_mean, Y_std = float(np.log1p(Y_train).mean()), float(np.log1p(Y_train).std())
    else:
        # Copy the start (so it is left untouched), freeze its early layers (unless freeze is False), and keep the Y
        # normalisation its output layer learned
        start_model, Y_mean, Y_std = start
        model = tf.keras.models.clone_model(start_model)
        model.set_weights(start_model.get_weights())
        # what is frozen in start stays so, or with freeze False nothing is (layers without weights, e.g. the input,
        # which cloning may rename, aside)
        for start_layer in start_model.layers:
            if start_layer.weights:
                model.get_layer(start_layer.name).trainable = start_layer.trainable if freeze else True
        if target == "log1pz":
            # train the network under ToRedshift, in the standardised log(1 + z) it was pre-trained in
            to_z = model.layers[-1]
            Y_mean, Y_std = to_z.mean, to_z.std
            model = models.Model(model.inputs[0], to_z.input)
        if freeze:
            for layer in model.layers[:N_FROZEN_LAYERS]:
                layer.trainable = False

    T_train, T_val = (Y_train, Y_val) if target == "z3" else (np.log1p(Y_train), np.log1p(Y_val))

    # Normalise Y
    Y_train_norm = (T_train - Y_mean) / Y_std
    Y_val_norm = (T_val - Y_mean) / Y_std

    model.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE), loss=_make_loss(loss, target, Y_train, Y_std, huber_delta)
    )

    # Train
    history = model.fit(
        X_train,
        Y_train_norm,
        validation_data=(X_val, Y_val_norm),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop, reduce_lr] if monitor else [],
        verbose=2,
    )
    return _as_ensemble_member(model, target, Y_mean, Y_std), history  # model + its normalisation


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
    target="z3",
    loss="mse",
    huber_delta=0.05,
    freeze=True,
):
    """
    Train an ensemble of N_SPLITS models from prebuilt inputs, each predicting a point redshift.

    The matrix-level building block, for inputs already built (see catalog_to_XY). train_from_catalog trains
    from a catalogue, with a noise realization of its photometry per member.

    X is either one feature array shared by all members, or a list of N_SPLITS arrays (same objects, in
    the same order as Y), one per member.

    The objects are split K-fold, K = N_SPLITS, by random_state: member k trains on all folds but fold k, and
    uses fold k for early stopping and learning-rate reduction, so the members differ by data cut.

    LEARNING_RATE: initial Adam learning rate (1e-3 is the Adam default), reduced on validation plateaus.

    pretraining: the one model every member starts from.
      - "popcosmos" (default): the model shipped with znn, trained on the pop-cosmos mock (see
        load_pretrained), of the architecture of build_model_func trained with target: the file
        PRETRAINED_DIR/popcosmos_<build_model_func>_<target> (see pretrained_name). An error lists the variants
        shipped if there is none for these options. Its inputs were built with the default feature config, so X
        must be too (in particular the default lambda_range and n_bins).
      - None: every member from scratch, built with build_model_func.
      - your own: the path of a save_pretrained pair, the path of a save_ensemble_file .pkl, or a list of
        (model, Y_mean, Y_std) as returned by train_ensembles; of an ensemble, the first member is used. Its
        inputs must be on the same grid as X, and it must have been trained with target; a save_pretrained pair
        must also have been saved with build_model_func.
      With pre-training, every member starts from a copy of that model, with its early layers frozen and its Y
      normalisation kept. The frozen layers are those the pre-training names (frozen_until, see save_pretrained;
      e.g. the whole first residual block of build_model_res_se), or else the first N_FROZEN_LAYERS; everything else (optimizer, learning rate, epochs,
      callbacks, split) is as from scratch, and build_model_func only selects and checks the pre-training. The
      pre-trained model is not modified.
    freeze: with pre-training, freeze its early layers as above (True, default), or nothing (False): the pre-trained
      model is then only the starting point of every layer, and the whole network trains.

    target: how the redshift is scaled for training (see TARGETS).
      - "z3" (default): z / 3, the original scheme.
      - "log1pz": log(1 + z), standardised with each member's training mean and std. A ToRedshift layer is
        appended to each trained model so it outputs z, and the member is (model, 0, 1).
      With pre-training, the pre-trained model must have been trained with the same target (an error is raised
      otherwise), and its Y normalisation is kept: a log1pz model is fine-tuned in the standardised log(1 + z)
      its ToRedshift layer holds.
    loss: "mse" (default) or "huber" (see LOSSES).
    huber_delta: the Huber delta in dz / (1 + z); only used with loss="huber".

    The new architectures (build_model_res_se, build_model_v2_k7) need pretraining=None or a pre-training of
    their own, since the shipped pre-training is a build_model (trained with target="z3"). The best setups from the architecture tests were build_model_res_se and
    build_model_v2_k7 with target="log1pz", and build_model_res_se with target="log1pz", loss="huber".

    Returns (trained_models, histories): the members as (model, Y_mean, Y_std) and their Keras histories.
    """
    _check_target_loss(target, loss)
    start = _resolve_pretraining(pretraining, build_model_func, target)
    X_members = _inputs_per_member(X, N_SPLITS)
    _check_inputs(start, X_members[0].shape[1:])
    _check_target(start, target)

    trained_models = []
    histories = []
    for member, (idx_train, idx_val) in enumerate(_kfold_cuts(len(Y), N_SPLITS, random_state)):
        print(f"\n--- Member {member+1}/{N_SPLITS} ---")
        trained, history = _train_member(
            build_model_func, X_members[member], Y, idx_train, idx_val, start, EPOCHS, BATCH_SIZE, LEARNING_RATE,
            target=target, loss=loss, huber_delta=huber_delta, freeze=freeze,
        )
        trained_models.append(trained)
        histories.append(history)

    return trained_models, histories


def train_model(
    build_model_func,
    X,
    Y,
    VAL_FRAC=0.2,
    EPOCHS=100,
    BATCH_SIZE=256,
    random_state=42,
    pretraining=DEFAULT_PRETRAINING,
    LEARNING_RATE=1e-3,
    target="z3",
    loss="mse",
    huber_delta=0.05,
    freeze=True,
):
    """
    Train a single model from prebuilt inputs, e.g. a pre-training (see save_pretrained).

    A random VAL_FRAC of the objects (by random_state) is held out for early stopping and learning-rate reduction,
    and the model trains on the rest, as the base model of train_from_catalog(from_base=True). Everything else
    (pretraining, LEARNING_RATE, target, loss, huber_delta, freeze) is as for train_ensembles.

    Returns ((model, Y_mean, Y_std), history): a one-member ensemble is [(model, Y_mean, Y_std)].
    """
    _check_target_loss(target, loss)
    start = _resolve_pretraining(pretraining, build_model_func, target)
    _check_inputs(start, X.shape[1:])
    _check_target(start, target)

    idx_train, idx_val = train_test_split(np.arange(len(Y)), test_size=VAL_FRAC, random_state=random_state)
    return _train_member(
        build_model_func, X, Y, idx_train, idx_val, start, EPOCHS, BATCH_SIZE, LEARNING_RATE,
        target=target, loss=loss, huber_delta=huber_delta, freeze=freeze,
    )


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
    FINETUNE_LR=3e-4,
    BATCH_SIZE=256,
    random_state=42,
    pretraining=DEFAULT_PRETRAINING,
    VAL_FRAC=0.2,
    target="z3",
    loss="mse",
    huber_delta=0.05,
    freeze=True,
):
    """
    Train an ensemble of N_MEMBERS models on a catalogue, each on its own noise realization of the
    photometry and its own K-fold data cut.

    Member k is trained on a noise realization of the catalogue's photometry drawn from its errors (see
    resample_photometry), just before it is trained, so N_MEMBERS can be large without holding all the
    realizations in memory. If df has no errors (errors=None, or see has_errors), every member is trained on
    the photometry as given and the members differ only by data cut (and the randomness of training).

    The objects are split K-fold, K = N_MEMBERS, as in train_ensembles. pretraining, random_state, target,
    loss and huber_delta are as for train_ensembles. With from_base, the base model and the members fine-tuned
    from it are all trained with target.

    freeze: as for train_ensembles, for what is trained from `pretraining`: the base model with from_base, every
    member without. freeze=False with from_base uses the pre-training only as the starting point of the base model's
    every layer; the members are then fine-tuned from the base as from a base trained from scratch (its first
    N_FROZEN_LAYERS layers frozen, the same in every member), so the run differs from one from scratch only by the
    base model's starting point.

    from_base: train in two stages, cheaper per member:
      1. a base model, from `pretraining`, on all the objects and the photometry as given (a random VAL_FRAC
         of them is held out for its early stopping), up to EPOCHS epochs;
      2. every member is a copy of the base model fine-tuned on its own noise realization and data cut, at
         the learning rate FINETUNE_LR for FINETUNE_EPOCHS epochs, with the base model's first N_FROZEN_LAYERS
         layers frozen, and those frozen in the base model (by the pre-training it started from), and its Y
         normalisation.
      The base model has seen every member's validation objects, so their validation loss cannot tell a better
      member from one that stays close to the base: the fine-tuning has no early stopping or learning-rate
      reduction, and runs all FINETUNE_EPOCHS at a constant FINETUNE_LR. Without from_base every member is trained in full from `pretraining` (up to EPOCHS, at the
      Adam default learning rate).

    df: the training catalogue (see znn.io.read_catalog), with Y its reference redshifts in the same order.
    bands, ref_band, config, filter_curves: feature settings, as for catalog_to_XY.
    errors: dict magnitude column -> its error column, as for resample_photometry.
    rng: numpy Generator, e.g. np.random.default_rng(seed); needed when df has errors.

    Returns (trained_models, histories, base_history): the members as (model, Y_mean, Y_std), their Keras
    histories, and that of the base model (None without from_base).
    """
    _check_target_loss(target, loss)
    noisy = has_errors(df, errors)
    if noisy and rng is None:
        raise ValueError("rng is needed to draw the noise realizations of a catalogue with errors")
    if filter_curves is None:
        filter_curves = load_filter_curves(bands)

    def inputs(photometry):
        return catalog_to_XY(photometry, bands, ref_band, config, filter_curves=filter_curves)[0]

    X_given = inputs(df)
    start = _resolve_pretraining(pretraining, build_model_func, target)
    _check_inputs(start, X_given.shape[1:])
    _check_target(start, target)

    base_history = None
    epochs, learning_rate, monitor = EPOCHS, 1e-3, True
    if from_base:
        print("\n=== Base model, on all objects with the photometry as given ===")
        idx_train, idx_val = train_test_split(np.arange(len(Y)), test_size=VAL_FRAC, random_state=random_state)
        start, base_history = _train_member(
            build_model_func, X_given, Y, idx_train, idx_val, start, EPOCHS, BATCH_SIZE, 1e-3,
            target=target, loss=loss, huber_delta=huber_delta, freeze=freeze,
        )
        epochs, learning_rate, monitor = FINETUNE_EPOCHS, FINETUNE_LR, False

    trained_models = []
    histories = []
    for member, (idx_train, idx_val) in enumerate(_kfold_cuts(len(Y), N_MEMBERS, random_state)):
        print(f"\n--- Member {member+1}/{N_MEMBERS}" + (" (its own noise realization)" if noisy else "") + " ---")
        X = inputs(resample_photometry(df, errors, rng)) if noisy else X_given
        # fine-tuned from the base: frozen as from any base; without from_base, from `pretraining`, as freeze says
        trained, history = _train_member(
            build_model_func, X, Y, idx_train, idx_val, start, epochs, BATCH_SIZE, learning_rate, monitor,
            target=target, loss=loss, huber_delta=huber_delta, freeze=True if from_base else freeze,
        )
        trained_models.append(trained)
        histories.append(history)

    return trained_models, histories, base_history
