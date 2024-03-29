import os
from parallelism import Parallelism
from contextlib import nullcontext
import logging
from bias_types import BiasTypes
from activation import Activation, make_activation
from enum import Enum
import flax.linen as nn
from fork_on_parallelism import fork_on_parallelism
from create_filtered_audio import create_biquad_coefficients
import yaml

# import logging
# logging.basicConfig(level=logging.INFO)
import soundfile
from types import SimpleNamespace
import local_env
import time
from loss import LossFn, Loss_fn_to_loss, LogCoshLoss, ESRLoss
from dataclasses import dataclass


INPUT_IS = "input"
TARGET_IS = "target"


@dataclass(frozen=True)
class ConversionConfig:
    fft_size: int
    hop_size: int
    window_size: int
    sample_rate: int
    amps_log_min: float
    amps_log_max: float
    amps_epsilon: float
    freqs_min: float
    freqs_max: float


start_time = time.time()

IS_CPU = local_env.parallelism == Parallelism.NONE
if IS_CPU:
    print("no gpus found")
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

from typing import Any
from flax import struct
from comet_ml import Experiment, Artifact
from pvc import PVC, do_conversion, normalize, denormalize
from fouriax.pvc import noscbank
from clu import metrics
from functools import partial
import jax.numpy as jnp
import flax.jax_utils as jax_utils
import numpy as np
import flax.linen as nns
import math
from flax.training import train_state
import optax
import jax
from data import make_data
import orbax.checkpoint
from tqdm import tqdm
import sys
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax.lax import with_sharding_constraint
from jax.experimental import mesh_utils
import subprocess

RESTORE = None


def checkpoint_walker(ckpt):
    def _cmp(i):
        try:
            o = jax.device_get(i)
            return o
        except Exception as e:
            return i

    return jax.tree_map(_cmp, ckpt)


PRNGKey = jax.Array


def trim_batch(tensor, batch_size):
    """
    Truncates the tensor to the largest multiple of batch_size.

    Parameters:
    tensor (jax.numpy.ndarray): The input tensor with a leading batch dimension.
    batch_size (int): The batch size to truncate to.

    Returns:
    jax.numpy.ndarray: The truncated tensor.
    """
    # Get the size of the leading dimension (batch dimension)
    batch_dim = tensor.shape[0]

    # Calculate the size of the truncated dimension
    truncated_size = (batch_dim // batch_size) * batch_size

    # Truncate the tensor
    truncated_tensor = tensor[:truncated_size]

    return truncated_tensor


@struct.dataclass
class Metrics(metrics.Collection):
    loss: metrics.Average.from_output("loss")
    input_max_amp: metrics.LastValue.from_output("input_max_amp")
    input_min_amp: metrics.LastValue.from_output("input_min_amp")
    input_max_freq: metrics.LastValue.from_output("input_max_freq")
    input_min_freq: metrics.LastValue.from_output("input_min_freq")
    input_normalized_max_amp: metrics.LastValue.from_output("input_normalized_max_amp")
    input_normalized_min_amp: metrics.LastValue.from_output("input_normalized_min_amp")
    input_normalized_max_freq: metrics.LastValue.from_output(
        "input_normalized_max_freq"
    )
    input_normalized_min_freq: metrics.LastValue.from_output(
        "input_normalized_min_freq"
    )
    pred_max_amp: metrics.LastValue.from_output("pred_max_amp")
    pred_min_amp: metrics.LastValue.from_output("pred_min_amp")
    pred_max_freq: metrics.LastValue.from_output("pred_max_freq")
    pred_min_freq: metrics.LastValue.from_output("pred_min_freq")
    pred_normalized_max_amp: metrics.LastValue.from_output("pred_normalized_max_amp")
    pred_normalized_min_amp: metrics.LastValue.from_output("pred_normalized_min_amp")
    pred_normalized_max_freq: metrics.LastValue.from_output("pred_normalized_max_freq")
    pred_normalized_min_freq: metrics.LastValue.from_output("pred_normalized_min_freq")
    target_max_amp: metrics.LastValue.from_output("target_max_amp")
    target_min_amp: metrics.LastValue.from_output("target_min_amp")
    target_max_freq: metrics.LastValue.from_output("target_max_freq")
    target_min_freq: metrics.LastValue.from_output("target_min_freq")
    target_normalized_max_amp: metrics.LastValue.from_output(
        "target_normalized_max_amp"
    )
    target_normalized_min_amp: metrics.LastValue.from_output(
        "target_normalized_min_amp"
    )
    target_normalized_max_freq: metrics.LastValue.from_output(
        "target_normalized_max_freq"
    )
    target_normalized_min_freq: metrics.LastValue.from_output(
        "target_normalized_min_freq"
    )


class TrainState(train_state.TrainState):
    metrics: Metrics
    batch_stats: Any


def create_train_state(rng: PRNGKey, x, module, tx, conversion_config) -> TrainState:
    print("creating train state", rng.shape, x.shape)
    x = normalize(
        do_conversion(conversion_config, x),
        amps_log_max=conversion_config.amps_log_max,
        amps_log_min=conversion_config.amps_log_min,
        amps_epsilon=conversion_config.amps_epsilon,
        freqs_max=conversion_config.freqs_max,
        freqs_min=conversion_config.freqs_min,
    )
    variables = module.init(rng, x, train=False)
    params = variables["params"]
    # if we are not doing batch norm, there won't be any batch_stats
    batch_stats = variables["batch_stats"] if "batch_stats" in variables else {}
    return TrainState.create(
        apply_fn=module.apply,
        params=params,
        tx=tx,
        batch_stats=batch_stats,
        metrics=Metrics.empty(),
    )


def train_step(state, input_raw, target_raw, conversion_config):
    """Train for a single step."""

    input_c = do_conversion(conversion_config, input_raw)
    input_c_amp = input_c[:, :, ::2]
    input_c_freq = input_c[:, :, 1::2]
    target_c = do_conversion(conversion_config, target_raw)
    target_c_amp = target_c[:, :, ::2]
    target_c_freq = target_c[:, :, 1::2]
    input = normalize(
        input_c,
        amps_log_max=conversion_config.amps_log_max,
        amps_log_min=conversion_config.amps_log_min,
        amps_epsilon=conversion_config.amps_epsilon,
        freqs_max=conversion_config.freqs_max,
        freqs_min=conversion_config.freqs_min,
    )
    input = jax.lax.stop_gradient(input)
    input_amp = input[:, :, ::2]
    input_freq = input[:, :, 1::2]
    target = normalize(
        target_c,
        amps_log_max=conversion_config.amps_log_max,
        amps_log_min=conversion_config.amps_log_min,
        amps_epsilon=conversion_config.amps_epsilon,
        freqs_max=conversion_config.freqs_max,
        freqs_min=conversion_config.freqs_min,
    )
    target = jax.lax.stop_gradient(target)
    target_amp = target[:, :, ::2]
    target_freq = target[:, :, 1::2]

    def loss_fn(i, t):
        targ = t

        def _ret(params):
            pred, updates = state.apply_fn(
                {"params": params, "batch_stats": state.batch_stats},
                i,
                train=True,
                mutable=["batch_stats"],
            )
            reach_back = min(pred.shape[1], targ.shape[1]) // 2
            pred, t = pred[:, -reach_back:, :], targ[:, -reach_back:, :]
            pred_a = pred[:, :, ::2]
            pred_f = pred[:, :, 1::2]
            p = denormalize(
                pred,
                amps_log_max=conversion_config.amps_log_max,
                amps_log_min=conversion_config.amps_log_min,
                amps_epsilon=conversion_config.amps_epsilon,
                freqs_max=conversion_config.freqs_max,
                freqs_min=conversion_config.freqs_min,
            )
            p_a = p[:, :, ::2]
            p_f = p[:, :, 1::2]
            loss = optax.l2_loss(pred, t).mean()
            return loss, (
                updates,
                jnp.max(pred_a),
                jnp.min(pred_a),
                jnp.max(pred_f),
                jnp.min(pred_f),
                jnp.max(p_a),
                jnp.min(p_a),
                jnp.max(p_f),
                jnp.min(p_f),
            )

        return _ret

    grad_fn = jax.value_and_grad(loss_fn(input, target), has_aux=True)
    (
        loss,
        (
            updates,
            pred_normalized_amp_max,
            pred_normalized_amp_min,
            pred_normalized_freq_max,
            pred_normalized_freq_min,
            pred_amp_max,
            pred_amp_min,
            pred_freq_max,
            pred_freq_min,
        ),
    ), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    state = state.replace(batch_stats=updates["batch_stats"])
    return (
        state,
        loss,
        jnp.max(input_c_amp),
        jnp.min(input_c_amp),
        jnp.max(input_c_freq),
        jnp.min(input_c_freq),
        jnp.max(input_amp),
        jnp.min(input_amp),
        jnp.max(input_freq),
        jnp.min(input_freq),
        pred_amp_max,
        pred_amp_min,
        pred_freq_max,
        pred_freq_min,
        pred_normalized_amp_max,
        pred_normalized_amp_min,
        pred_normalized_freq_max,
        pred_normalized_freq_min,
        jnp.max(target_c_amp),
        jnp.min(target_c_amp),
        jnp.max(target_c_freq),
        jnp.min(target_c_freq),
        jnp.max(target_amp),
        jnp.min(target_amp),
        jnp.max(target_freq),
        jnp.min(target_freq),
    )


def _replace_metrics(state):
    return state.replace(metrics=state.metrics.empty())


def do_inference(state, input, conversion_config: ConversionConfig):
    input = normalize(
        do_conversion(conversion_config, input),
        amps_log_max=conversion_config.amps_log_max,
        amps_log_min=conversion_config.amps_log_min,
        amps_epsilon=conversion_config.amps_epsilon,
        freqs_max=conversion_config.freqs_max,
        freqs_min=conversion_config.freqs_min,
    )
    input = jax.lax.stop_gradient(input)
    o, _ = state.apply_fn(
        {"params": state.params, "batch_stats": state.batch_stats},
        input,
        train=False,
        mutable=["batch_stats"],
    )
    p_inc = 1.0 / conversion_config.sample_rate
    i_inv = 1.0 / conversion_config.hop_size
    batch_size = o.shape[0]
    lastval = np.zeros((batch_size, o.shape[-1] // 2, 2))
    index = np.zeros((batch_size, o.shape[-1] // 2))
    o = denormalize(
        o,
        amps_log_max=conversion_config.amps_log_max,
        amps_log_min=conversion_config.amps_log_min,
        amps_epsilon=conversion_config.amps_epsilon,
        freqs_max=conversion_config.freqs_max,
        freqs_min=conversion_config.freqs_min,
    )
    o = jax.vmap(
        partial(
            noscbank,
            nw=conversion_config.window_size,
            p_inc=p_inc,
            i_inv=i_inv,
            rg=jnp.arange(conversion_config.hop_size),
        ),
        in_axes=0,
        out_axes=0,
    )((lastval, index), o)
    o = jnp.reshape(o[1], (batch_size, -1, 1))
    return o, lastval, index


replace_metrics = fork_on_parallelism(jax.jit, jax.pmap)(_replace_metrics)


def compute_loss(state, input_raw, target_raw, conversion_config):
    input_c = do_conversion(conversion_config, input_raw)
    input_c_amp = input_c[:, :, ::2]
    input_c_freq = input_c[:, :, 1::2]
    target_c = do_conversion(conversion_config, target_raw)
    target_c_amp = target_c[:, :, ::2]
    target_c_freq = target_c[:, :, 1::2]
    input = normalize(
        input_c,
        amps_log_max=conversion_config.amps_log_max,
        amps_log_min=conversion_config.amps_log_min,
        amps_epsilon=conversion_config.amps_epsilon,
        freqs_max=conversion_config.freqs_max,
        freqs_min=conversion_config.freqs_min,
    )
    input = jax.lax.stop_gradient(input)
    input_amp = input[:, :, ::2]
    input_freq = input[:, :, 1::2]
    target = normalize(
        target_c,
        amps_log_max=conversion_config.amps_log_max,
        amps_log_min=conversion_config.amps_log_min,
        amps_epsilon=conversion_config.amps_epsilon,
        freqs_max=conversion_config.freqs_max,
        freqs_min=conversion_config.freqs_min,
    )
    target = jax.lax.stop_gradient(target)
    target_amp = target[:, :, ::2]
    target_freq = target[:, :, 1::2]

    pred, _ = state.apply_fn(
        {"params": state.params, "batch_stats": state.batch_stats},
        input,
        train=False,
        mutable=["batch_stats"],
    )
    reach_back = min(pred.shape[1], target.shape[1]) // 2
    pred, target = pred[:, -reach_back:, :], target[:, -reach_back:, :]
    pred_a = pred[:, :, ::2]
    pred_f = pred[:, :, 1::2]
    p = denormalize(
        pred,
        amps_log_max=conversion_config.amps_log_max,
        amps_log_min=conversion_config.amps_log_min,
        amps_epsilon=conversion_config.amps_epsilon,
        freqs_max=conversion_config.freqs_max,
        freqs_min=conversion_config.freqs_min,
    )
    p_a = p[:, :, ::2]
    p_f = p[:, :, 1::2]
    loss = optax.l2_loss(pred, target).mean()
    pred_normalized_amp_max = jnp.max(pred_a)
    pred_normalized_amp_min = jnp.min(pred_a)
    pred_normalized_freq_max = jnp.max(pred_f)
    pred_normalized_freq_min = jnp.min(pred_f)
    pred_amp_max = jnp.max(p_a)
    pred_amp_min = jnp.min(p_a)
    pred_freq_max = jnp.max(p_f)
    pred_freq_min = jnp.min(p_f)

    return (
        loss,
        jnp.max(input_c_amp),
        jnp.min(input_c_amp),
        jnp.max(input_c_freq),
        jnp.min(input_c_freq),
        jnp.max(input_amp),
        jnp.min(input_amp),
        jnp.max(input_freq),
        jnp.min(input_freq),
        pred_amp_max,
        pred_amp_min,
        pred_freq_max,
        pred_freq_min,
        pred_normalized_amp_max,
        pred_normalized_amp_min,
        pred_normalized_freq_max,
        pred_normalized_freq_min,
        jnp.max(target_c_amp),
        jnp.min(target_c_amp),
        jnp.max(target_c_freq),
        jnp.min(target_c_freq),
        jnp.max(target_amp),
        jnp.min(target_amp),
        jnp.max(target_freq),
        jnp.min(target_freq),
    )


def _add_losses_to_metrics(
    state,
    loss,
    input_max_amp,
    input_min_amp,
    input_max_freq,
    input_min_freq,
    input_normalized_max_amp,
    input_normalized_min_amp,
    input_normalized_max_freq,
    input_normalized_min_freq,
    pred_max_amp,
    pred_min_amp,
    pred_max_freq,
    pred_min_freq,
    pred_normalized_max_amp,
    pred_normalized_min_amp,
    pred_normalized_max_freq,
    pred_normalized_min_freq,
    target_max_amp,
    target_min_amp,
    target_max_freq,
    target_min_freq,
    target_normalized_max_amp,
    target_normalized_min_amp,
    target_normalized_max_freq,
    target_normalized_min_freq,
):
    metric_updates = state.metrics.single_from_model_output(
        loss=loss,
        input_max_amp=input_max_amp,
        input_min_amp=input_min_amp,
        input_max_freq=input_max_freq,
        input_min_freq=input_min_freq,
        input_normalized_max_amp=input_normalized_max_amp,
        input_normalized_min_amp=input_normalized_min_amp,
        input_normalized_max_freq=input_normalized_max_freq,
        input_normalized_min_freq=input_normalized_min_freq,
        pred_max_amp=pred_max_amp,
        pred_min_amp=pred_min_amp,
        pred_max_freq=pred_max_freq,
        pred_min_freq=pred_min_freq,
        pred_normalized_max_amp=pred_normalized_max_amp,
        pred_normalized_min_amp=pred_normalized_min_amp,
        pred_normalized_max_freq=pred_normalized_max_freq,
        pred_normalized_min_freq=pred_normalized_min_freq,
        target_max_amp=target_max_amp,
        target_min_amp=target_min_amp,
        target_max_freq=target_max_freq,
        target_min_freq=target_min_freq,
        target_normalized_max_amp=target_normalized_max_amp,
        target_normalized_min_amp=target_normalized_min_amp,
        target_normalized_max_freq=target_normalized_max_freq,
        target_normalized_min_freq=target_normalized_min_freq,
    )
    metrics = state.metrics.merge(metric_updates)
    state = state.replace(metrics=metrics)
    return state


add_losses_to_metrics = fork_on_parallelism(jax.jit, jax.pmap)(_add_losses_to_metrics)

maybe_replicate = fork_on_parallelism(lambda x: x, jax_utils.replicate)
maybe_unreplicate = fork_on_parallelism(lambda x: x, jax_utils.unreplicate)
maybe_device_put = fork_on_parallelism(jax.device_put, lambda x, _: x)


def mesh_sharding(pspec: PartitionSpec) -> NamedSharding:
    return NamedSharding(mesh, pspec)


def run_inference(
    inference_dataset,
    epoch,
    config,
    x_sharding,
    state_sharding,
    state,
    conversion_config,
    run,
):
    for inference_batch_ix, inference_batch in tqdm(
        enumerate(
            inference_dataset.take(config.inference_artifacts_per_batch_per_epoch).iter(
                batch_size=config.inference_batch_size
            )
        ),
        total=config.inference_artifacts_per_batch_per_epoch,
    ):
        input_ = trim_batch(
            jnp.array(inference_batch[INPUT_IS]), config.inference_batch_size
        )
        if input_.shape[0] == 0:
            continue
        target_ = trim_batch(
            jnp.array(inference_batch[TARGET_IS]), config.inference_batch_size
        )
        input = maybe_replicate(input_)
        input = maybe_device_put(input, x_sharding)
        logging.warning(f"input shape for inference is is {input.shape}")

        jit_do_inference = fork_on_parallelism(
            partial(
                jax.jit,
                static_argnums=(2,),
                in_shardings=(state_sharding, x_sharding),
                out_shardings=(x_sharding, x_sharding, x_sharding),
            ),
            partial(jax.pmap, static_broadcasted_argnums=(2,)),
        )(do_inference)

        o, _, _ = jit_do_inference(state, input, conversion_config)
        o = maybe_unreplicate(o)
        assert o.shape[-1] == 1
        # logging.info(f"shape of batch is {input.shape}")

        for i in range(o.shape[0]):
            audy = np.squeeze(np.array(o[i]))
            run.log_audio(
                audy,
                sample_rate=44100,
                step=epoch,
                file_name=f"audio_{epoch}_{inference_batch_ix}_{i}_prediction.wav",
            )
            audy = np.squeeze(np.array(input_[i, :, :1]))
            run.log_audio(
                audy,
                sample_rate=44100,
                step=epoch,
                file_name=f"audio_{epoch}_{inference_batch_ix}_{i}_input.wav",
            )
            audy = np.squeeze(np.array(target_[i]))
            run.log_audio(
                audy,
                sample_rate=44100,
                step=epoch,
                file_name=f"audio_{epoch}_{inference_batch_ix}_{i}_target.wav",
            )


if __name__ == "__main__":
    from get_files import FILES

    logging.basicConfig(level=logging.WARN)
    if local_env.parallelism == Parallelism.PMAP:
        if local_env.do_manual_parallelism_setup:
            jax.distributed.initialize(
                coordinator_address=local_env.coordinator_address,
                num_processes=local_env.num_processes,
                process_id=local_env.process_id,
            )
        else:
            jax.distributed.initialize()

    checkpoint_dir = local_env.checkpoint_dir

    if os.path.exists(checkpoint_dir):
        logging.warn(f"consisder clearing checkpoint dir first: {checkpoint_dir}")
    else:
        os.makedirs(checkpoint_dir)

    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=1, create=True)
    checkpoint_manager = orbax.checkpoint.CheckpointManager(
        checkpoint_dir, orbax_checkpointer, options
    )

    device_len = len(jax.devices())

    print(f"Using {device_len} devices")

    if (device_len != 1) and (device_len % 2 == 1):
        raise ValueError("not ")

    run = Experiment(
        api_key=local_env.comet_ml_api_key,
        project_name="jax-pvc",
    )
    ## defaults
    _config = {}
    _config["seed"] = 42
    _config["batch_size"] = 2**4
    _config["inference_batch_size"] = 2**3
    _config["inference_artifacts_per_batch_per_epoch"] = (
        _config["inference_batch_size"] * 4
    )
    _config["validation_split"] = 0.2
    _config["learning_rate"] = 1e-4
    _config["epochs"] = 2**7
    _config["window"] = 2**13
    _config["inference_window"] = 2**13
    _config["stride"] = 2**8
    _config["step_freq"] = 2**6
    _config["test_size"] = 0.1
    _config["mesh_x_div"] = 1
    #
    _config["fft_size"] = 1024
    _config["hop_size"] = 128
    _config["window_size"] = 2048
    _config["sample_rate"] = 44100
    _config["kernel_size"] = 7
    _config["n_phasors"] = 512
    _config["conv_depth"] = 16
    _config["attn_depth"] = 16
    _config["heads"] = 32
    _config["expand_factor"] = 2.0
    _config["amps_log_min"] = -73.682724
    _config["amps_log_max"] = -6.9077554
    _config["amps_log_epsilon"] = -73.682724
    _config["freqs_min"] = 0.0
    _config["freqs_max"] = 11025.0
    with open(local_env.config_file, "r") as f:
        in_config = yaml.safe_load(f)["config"]
        for k, v in in_config.items():
            if k not in _config:
                raise ValueError(f"Unknown config key {k}")
        for k, v in _config.items():
            if k not in in_config:
                raise ValueError(f"Requires key {k}")
        _config = in_config
        _config["mesh_x"] = device_len // _config["mesh_x_div"]
        _config["mesh_y"] = _config["mesh_x_div"]
        del _config["mesh_x_div"]
    run.log_parameters(_config)
    if local_env.parallelism == Parallelism.PMAP:
        run.log_parameter("run_id", sys.argv[1])
    config = SimpleNamespace(**_config)

    conversion_config = ConversionConfig(
        fft_size=config.fft_size,
        hop_size=config.hop_size,
        window_size=config.window_size,
        sample_rate=config.sample_rate,
        amps_log_min=config.amps_log_min,
        amps_log_max=config.amps_log_max,
        amps_epsilon=np.exp(config.amps_log_epsilon),
        freqs_min=config.freqs_min,
        freqs_max=config.freqs_max,
    )
    device_mesh = None
    mesh = None
    x_sharding = None
    state_sharding = None
    old_state_sharding = None
    # messshhh
    if local_env.parallelism == Parallelism.SHARD:
        device_mesh = mesh_utils.create_device_mesh((config.mesh_x, config.mesh_y))
        mesh = Mesh(devices=device_mesh, axis_names=("data", "model"))
        print(mesh)
        x_sharding = mesh_sharding(PartitionSpec("data", None))
    ###

    len_files = len(FILES)
    test_files = (
        FILES[: int(len_files * config.test_size)] if not IS_CPU else FILES[0:1]
    )
    train_files = (
        FILES[int(len_files * config.test_size) :] if not IS_CPU else FILES[1:2]
    )
    print("making datasets")
    proto_train_dataset, train_dataset_total = make_data(
        # channels=config.conv_depth[0],
        # afstart=config.afstart,
        # afend=config.afend,
        # qstart=config.qstart,
        # qend=config.qend,
        naug=0,
        paths=train_files,
        window=config.window,
        stride=config.stride,  # , shift=config.shift, dilation=config.dilation, features=config.features, feature_dim=-1, shuffle=True
        # shuffle=fork_on_parallelism(True, False),
    )
    proto_test_dataset, test_dataset_total = make_data(
        # channels=config.conv_depth[0],
        # afstart=config.afstart,
        # afend=config.afend,
        # qstart=config.qstart,
        # qend=config.qend,
        paths=test_files,
        window=config.window,
        stride=config.stride,  # , shift=config.shift, dilation=config.dilation, features=config.features, feature_dim=-1, shuffle=True
        # shuffle=fork_on_parallelism(True, False),
    )
    proto_inference_dataset, inference_dataset_total = make_data(
        # channels=config.conv_depth[0],
        # afstart=config.afstart,
        # afend=config.afend,
        # qstart=config.qstart,
        # qend=config.qend,
        paths=test_files,
        window=config.inference_window,
        stride=config.stride,  # , shift=config.shift, dilation=config.dilation, features=config.features, feature_dim=-1, shuffle=True
        # shuffle=fork_on_parallelism(True, False),
    )
    print("datasets generated")
    init_rng = jax.random.PRNGKey(config.seed)
    onez = jnp.ones([config.batch_size, config.window, 1])  # 1,

    def array_to_tuple(arr):
        if isinstance(arr, np.ndarray):
            return tuple(array_to_tuple(a) for a in arr)
        else:
            return arr

    module = PVC(
        fft_size=config.fft_size,
        hop_size=config.hop_size,
        window_size=config.window_size,
        sample_rate=config.sample_rate,
        kernel_size=config.kernel_size,
        n_phasors=config.n_phasors,
        conv_depth=config.conv_depth,
        attn_depth=config.attn_depth,
        heads=config.heads,
        expand_factor=config.expand_factor,
    )
    tx = optax.adam(config.learning_rate)

    if local_env.parallelism == Parallelism.SHARD:
        abstract_variables = jax.eval_shape(
            partial(
                create_train_state,
                module=module,
                tx=tx,
                conversion_config=conversion_config,
            ),
            init_rng,
            onez,
        )

        state_sharding = nn.get_sharding(abstract_variables, mesh)

    jit_create_train_state = fork_on_parallelism(
        partial(
            jax.jit,
            static_argnums=(2, 3, 4),
            in_shardings=(
                (
                    mesh_sharding(None)
                    if local_env.parallelism == Parallelism.SHARD
                    else None
                ),
                x_sharding,
            ),  # PRNG key and x
            out_shardings=state_sharding,
        ),
        partial(jax.pmap, static_broadcasted_argnums=(2, 3, 4)),
    )(create_train_state)
    rng_for_train_state = (
        init_rng
        if local_env.parallelism == Parallelism.SHARD
        else jax.random.split(
            init_rng, 8
        )  ### #UGH we hardcode 8, not sure why this worked before :-/
    )
    state = jit_create_train_state(
        rng_for_train_state,
        fork_on_parallelism(onez, onez),
        module,
        tx,
        conversion_config,
    )

    target = {"model": state, "config": None}

    if RESTORE is not None:
        CKPT = checkpoint_manager.restore(RESTORE, target)

        state = CKPT["model"]

    jit_train_step = fork_on_parallelism(
        partial(
            jax.jit,
            static_argnums=(3,),
            in_shardings=(state_sharding, x_sharding, x_sharding),
            out_shardings=(
                state_sharding,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            ),
        ),
        partial(jax.pmap, static_broadcasted_argnums=(3,)),
    )(train_step)

    jit_compute_loss = fork_on_parallelism(
        partial(
            jax.jit,
            static_argnums=(3,),
            in_shardings=(state_sharding, x_sharding, x_sharding),
        ),
        partial(jax.pmap, static_broadcasted_argnums=(3,)),
    )(compute_loss)

    del init_rng  # Must not be used anymore.
    step_ctr = 0
    for epoch in range(config.epochs):
        # ugggh
        # commenting out for now
        epoch_is_0 = False  # epoch == 0
        to_take_in_0_epoch = 16
        train_dataset = (
            proto_train_dataset
            if not epoch_is_0
            else proto_train_dataset.take(config.batch_size * to_take_in_0_epoch)
        )
        test_dataset = (
            proto_test_dataset
            if not epoch_is_0
            else proto_test_dataset.take(config.batch_size * to_take_in_0_epoch)
        )
        inference_dataset = (
            proto_inference_dataset
            if not epoch_is_0
            else proto_inference_dataset.take(config.batch_size * to_take_in_0_epoch)
        )

        # log the epoch
        run.log_current_epoch(epoch)
        train_dataset.set_epoch(epoch)
        inference_dataset.set_epoch(epoch)

        # train
        train_total = train_dataset_total // config.batch_size
        with tqdm(
            enumerate(
                train_dataset.iter(batch_size=config.batch_size, drop_last_batch=False)
            ),
            total=train_total if not epoch_is_0 else to_take_in_0_epoch,
            unit="batch",
        ) as train_loop:
            for train_batch_ix, train_batch in train_loop:
                should_use_gen = train_batch_ix % 2 == 1
                input = trim_batch(jnp.array(train_batch[INPUT_IS]), config.batch_size)
                if input.shape[0] == 0:
                    continue
                assert input.shape[1] == config.window
                input = maybe_replicate(input)
                input = maybe_device_put(input, x_sharding)
                target = trim_batch(
                    jnp.array(train_batch[TARGET_IS]), config.batch_size
                )
                assert target.shape[1] == config.window
                target = maybe_replicate(target)
                with fork_on_parallelism(mesh, nullcontext()):
                    (
                        state,
                        loss,
                        input_max_amp,
                        input_min_amp,
                        input_max_freq,
                        input_min_freq,
                        input_normalized_max_amp,
                        input_normalized_min_amp,
                        input_normalized_max_freq,
                        input_normalized_min_freq,
                        pred_max_amp,
                        pred_min_amp,
                        pred_max_freq,
                        pred_min_freq,
                        pred_normalized_max_amp,
                        pred_normalized_min_amp,
                        pred_normalized_max_freq,
                        pred_normalized_min_freq,
                        target_max_amp,
                        target_min_amp,
                        target_max_freq,
                        target_min_freq,
                        target_normalized_max_amp,
                        target_normalized_min_amp,
                        target_normalized_max_freq,
                        target_normalized_min_freq,
                    ) = jit_train_step(state, input, target, conversion_config)

                    state = add_losses_to_metrics(
                        state=state,
                        loss=loss,
                        input_max_amp=input_max_amp,
                        input_min_amp=input_min_amp,
                        input_max_freq=input_max_freq,
                        input_min_freq=input_min_freq,
                        input_normalized_max_amp=input_normalized_max_amp,
                        input_normalized_min_amp=input_normalized_min_amp,
                        input_normalized_max_freq=input_normalized_max_freq,
                        input_normalized_min_freq=input_normalized_min_freq,
                        pred_max_amp=pred_max_amp,
                        pred_min_amp=pred_min_amp,
                        pred_max_freq=pred_max_freq,
                        pred_min_freq=pred_min_freq,
                        pred_normalized_max_amp=pred_normalized_max_amp,
                        pred_normalized_min_amp=pred_normalized_min_amp,
                        pred_normalized_max_freq=pred_normalized_max_freq,
                        pred_normalized_min_freq=pred_normalized_min_freq,
                        target_max_amp=target_max_amp,
                        target_min_amp=target_min_amp,
                        target_max_freq=target_max_freq,
                        target_min_freq=target_min_freq,
                        target_normalized_max_amp=target_normalized_max_amp,
                        target_normalized_min_amp=target_normalized_min_amp,
                        target_normalized_max_freq=target_normalized_max_freq,
                        target_normalized_min_freq=target_normalized_min_freq,
                    )

                if train_batch_ix % config.step_freq == 0:
                    metrics = maybe_unreplicate(state.metrics).compute()
                    run.log_metrics(
                        {
                            "train_loss": metrics["loss"],
                            "train_input_max_amp": metrics["input_max_amp"],
                            "train_input_min_amp": metrics["input_min_amp"],
                            "train_input_max_freq": metrics["input_max_freq"],
                            "train_input_min_freq": metrics["input_min_freq"],
                            "train_input_normalized_max_amp": metrics[
                                "input_normalized_max_amp"
                            ],
                            "train_input_normalized_min_amp": metrics[
                                "input_normalized_min_amp"
                            ],
                            "train_input_normalized_max_freq": metrics[
                                "input_normalized_max_freq"
                            ],
                            "train_input_normalized_min_freq": metrics[
                                "input_normalized_min_freq"
                            ],
                            "train_pred_max_amp": metrics["pred_max_amp"],
                            "train_pred_min_amp": metrics["pred_min_amp"],
                            "train_pred_max_freq": metrics["pred_max_freq"],
                            "train_pred_min_freq": metrics["pred_min_freq"],
                            "train_pred_normalized_max_amp": metrics[
                                "pred_normalized_max_amp"
                            ],
                            "train_pred_normalized_min_amp": metrics[
                                "pred_normalized_min_amp"
                            ],
                            "train_pred_normalized_max_freq": metrics[
                                "pred_normalized_max_freq"
                            ],
                            "train_pred_normalized_min_freq": metrics[
                                "pred_normalized_min_freq"
                            ],
                            "train_target_max_amp": metrics["target_max_amp"],
                            "train_target_min_amp": metrics["target_min_amp"],
                            "train_target_max_freq": metrics["target_max_freq"],
                            "train_target_min_freq": metrics["target_min_freq"],
                            "train_target_normalized_max_amp": metrics[
                                "target_normalized_max_amp"
                            ],
                            "train_target_normalized_min_amp": metrics[
                                "target_normalized_min_amp"
                            ],
                            "train_target_normalized_max_freq": metrics[
                                "target_normalized_max_freq"
                            ],
                            "train_target_normalized_min_freq": metrics[
                                "target_normalized_min_freq"
                            ],
                        },
                        step=step_ctr,
                    )
                    step_ctr += 1
                    train_loop.set_postfix(loss=metrics["loss"])
                    state = replace_metrics(state)
                    current_time = time.time()
                    elapsed_time = current_time - start_time
                if elapsed_time > (60 * 60 * 2):
                    run_inference(
                        inference_dataset,
                        epoch,
                        config,
                        x_sharding,
                        state_sharding,
                        state,
                        conversion_config,
                        run,
                    )
                    # we test checkpointing early just to make sure it
                    # works so there aren't any nasty surprises
                    # checkpoint
                    ckpt_model = state
                    # needs to use underscore config
                    # becuase otherwise it doesn't serialize correctly
                    ckpt = {"model": ckpt_model, "config": _config}
                    if local_env.parallelism == Parallelism.PMAP:
                        ckpt = checkpoint_walker(ckpt)

                    checkpoint_manager.save(step_ctr, ckpt)
                    logging.warning(
                        f"saved checkpoint for epoch {epoch} filenmae {step_ctr} in {os.listdir(checkpoint_dir)}"
                    )
                    try:
                        subprocess.run(
                            f'gsutil -m cp -r {os.path.join(checkpoint_dir, f"{step_ctr}")} gs://meeshkan-experiments/jax-pvc/{run.id}/{step_ctr}/',
                            check=True,
                            shell=True,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                        )
                    except ValueError as e:
                        logging.warning(f"checkpoint artifact did not work {e}")
                    start_time = current_time
                    elapsed_time = 0
        test_dataset.set_epoch(epoch)
        with tqdm(
            enumerate(
                test_dataset.iter(batch_size=config.batch_size, drop_last_batch=False)
            ),
            total=(
                (test_dataset_total // config.batch_size)
                if not epoch_is_0
                else to_take_in_0_epoch
            ),
            unit="batch",
        ) as validation_loop:
            for val_batch_ix, val_batch in validation_loop:
                input = maybe_replicate(
                    trim_batch(jnp.array(val_batch[INPUT_IS]), config.batch_size)
                )
                if input.shape[0] == 0:
                    continue
                input = maybe_device_put(input, x_sharding)
                target = maybe_replicate(
                    trim_batch(jnp.array(val_batch[TARGET_IS]), config.batch_size)
                )
                (
                    loss,
                    input_max_amp,
                    input_min_amp,
                    input_max_freq,
                    input_min_freq,
                    input_normalized_max_amp,
                    input_normalized_min_amp,
                    input_normalized_max_freq,
                    input_normalized_min_freq,
                    pred_max_amp,
                    pred_min_amp,
                    pred_max_freq,
                    pred_min_freq,
                    pred_normalized_max_amp,
                    pred_normalized_min_amp,
                    pred_normalized_max_freq,
                    pred_normalized_min_freq,
                    target_max_amp,
                    target_min_amp,
                    target_max_freq,
                    target_min_freq,
                    target_normalized_max_amp,
                    target_normalized_min_amp,
                    target_normalized_max_freq,
                    target_normalized_min_freq,
                ) = jit_compute_loss(
                    state,
                    input,
                    target,
                    conversion_config,
                )
                state = add_losses_to_metrics(
                    state=state,
                    loss=loss,
                    input_max_amp=input_max_amp,
                    input_min_amp=input_min_amp,
                    input_max_freq=input_max_freq,
                    input_min_freq=input_min_freq,
                    input_normalized_max_amp=input_normalized_max_amp,
                    input_normalized_min_amp=input_normalized_min_amp,
                    input_normalized_max_freq=input_normalized_max_freq,
                    input_normalized_min_freq=input_normalized_min_freq,
                    pred_max_amp=pred_max_amp,
                    pred_min_amp=pred_min_amp,
                    pred_max_freq=pred_max_freq,
                    pred_min_freq=pred_min_freq,
                    pred_normalized_max_amp=pred_normalized_max_amp,
                    pred_normalized_min_amp=pred_normalized_min_amp,
                    pred_normalized_max_freq=pred_normalized_max_freq,
                    pred_normalized_min_freq=pred_normalized_min_freq,
                    target_max_amp=target_max_amp,
                    target_min_amp=target_min_amp,
                    target_max_freq=target_max_freq,
                    target_min_freq=target_min_freq,
                    target_normalized_max_amp=target_normalized_max_amp,
                    target_normalized_min_amp=target_normalized_min_amp,
                    target_normalized_max_freq=target_normalized_max_freq,
                    target_normalized_min_freq=target_normalized_min_freq,
                )
        metrics = maybe_unreplicate(state.metrics).compute()
        run.log_metrics(
            {
                "val_loss": metrics["loss"],
                "val_input_max_amp": metrics["input_max_amp"],
                "val_input_min_amp": metrics["input_min_amp"],
                "val_input_max_freq": metrics["input_max_freq"],
                "val_input_min_freq": metrics["input_min_freq"],
                "val_input_normalized_max_amp": metrics["input_normalized_max_amp"],
                "val_input_normalized_min_amp": metrics["input_normalized_min_amp"],
                "val_input_normalized_max_freq": metrics["input_normalized_max_freq"],
                "val_input_normalized_min_freq": metrics["input_normalized_min_freq"],
                "val_pred_max_amp": metrics["pred_max_amp"],
                "val_pred_min_amp": metrics["pred_min_amp"],
                "val_pred_max_freq": metrics["pred_max_freq"],
                "val_pred_min_freq": metrics["pred_min_freq"],
                "val_pred_normalized_max_amp": metrics["pred_normalized_max_amp"],
                "val_pred_normalized_min_amp": metrics["pred_normalized_min_amp"],
                "val_pred_normalized_max_freq": metrics["pred_normalized_max_freq"],
                "val_pred_normalized_min_freq": metrics["pred_normalized_min_freq"],
                "val_target_max_amp": metrics["target_max_amp"],
                "val_target_min_amp": metrics["target_min_amp"],
                "val_target_max_freq": metrics["target_max_freq"],
                "val_target_min_freq": metrics["target_min_freq"],
                "val_target_normalized_max_amp": metrics["target_normalized_max_amp"],
                "val_target_normalized_min_amp": metrics["target_normalized_min_amp"],
                "val_target_normalized_max_freq": metrics["target_normalized_max_freq"],
                "val_target_normalized_min_freq": metrics["target_normalized_min_freq"],
            },
            step=epoch,
        )
        state = replace_metrics(state)
        # inference
        inference_dataset.set_epoch(epoch)
        run_inference(
            inference_dataset,
            epoch,
            config,
            x_sharding,
            state_sharding,
            state,
            conversion_config,
            run,
        )
