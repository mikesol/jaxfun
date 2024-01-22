import os
from parallelism import Parallelism
from contextlib import nullcontext
import logging
import crop
from bias_types import BiasTypes
from activation import Activation, make_activation
from enum import Enum
from loss import LossFn, Loss_fn_to_loss, LogCoshLoss, ESRLoss

import flax.linen as nn
from fork_on_parallelism import fork_on_parallelism
from fade_in import apply_fade_in
import yaml

# import logging
# logging.basicConfig(level=logging.INFO)
import soundfile
from types import SimpleNamespace
import local_env
import time

start_time = time.time()

IS_CPU = local_env.parallelism == Parallelism.NONE
if IS_CPU:
    print("no gpus found")
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

from typing import Any
from flax import struct
from comet_ml import Experiment, Artifact
from clu import metrics
from sineconv import SineconvNetwork
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

RESTORE = None


def batchify(f, n):
    to_concat = tuple([f() for _ in range(n.shape[0])])
    return jnp.concatenate(to_concat, axis=0)


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


class TrainState(train_state.TrainState):
    metrics: Metrics


def make_phases(features_list, sines_per_window, shp):
    return [
        batchify(lambda: jnp.array(np.random.randn(1, x * sines_per_window, y)), shp)
        for x, y in zip((1,) + features_list[:-1], features_list)
    ]


def create_train_state(rng: PRNGKey, x, sine_range, phases, module, tx) -> TrainState:
    print("creating train state", rng.shape, x.shape)
    variables = module.init(rng, x, sine_range=sine_range, phases=phases)
    params = variables["params"]
    return TrainState.create(
        apply_fn=module.apply,
        params=params,
        tx=tx,
        metrics=Metrics.empty(),
    )



def interleave_jax(input_array, trained_output):
    input_expanded = jnp.expand_dims(input_array, axis=3)
    trained_output_expanded = jnp.expand_dims(trained_output, axis=3)
    concatenated = jnp.concatenate([input_expanded, trained_output_expanded], axis=3)
    interleaved = concatenated.reshape(
        trained_output.shape[0],
        input_array.shape[1] + trained_output.shape[1],
        trained_output.shape[2],
    )
    return interleaved


def train_step(
    state, input, target, sine_range, phases, lossy_loss_loss, croppy_crop_crop
):
    """Train for a single step."""

    def loss_fn(params):
        pred = state.apply_fn(
            {"params": params}, input, sine_range=sine_range, phases=phases
        )
        loss = crop.cropping_to_function(croppy_crop_crop)(
            pred, target, Loss_fn_to_loss(lossy_loss_loss)
        )
        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss


def _replace_metrics(state):
    return state.replace(metrics=state.metrics.empty())


def do_inference(state, input, sine_range, phases):
    o = state.apply_fn(
        {"params": state.params},
        input,
        sine_range=sine_range,
        phases=phases,
    )
    return o


replace_metrics = fork_on_parallelism(jax.jit, jax.pmap)(_replace_metrics)


def compute_loss(
    state, input, target, sine_range, phases, lossy_loss_loss, croppy_crop_crop
):
    pred = state.apply_fn(
        {"params": state.params}, input, sine_range=sine_range, phases=phases
    )
    loss = crop.cropping_to_function(croppy_crop_crop)(
        pred, target, Loss_fn_to_loss(lossy_loss_loss)
    )
    return loss


def _add_losses_to_metrics(state, loss):
    metric_updates = state.metrics.single_from_model_output(loss=loss)
    metrics = state.metrics.merge(metric_updates)
    state = state.replace(metrics=metrics)
    return state


add_losses_to_metrics = fork_on_parallelism(jax.jit, jax.pmap)(_add_losses_to_metrics)

maybe_replicate = fork_on_parallelism(lambda x: x, jax_utils.replicate)
maybe_unreplicate = fork_on_parallelism(lambda x: x, jax_utils.unreplicate)
maybe_device_put = fork_on_parallelism(jax.device_put, lambda x, _: x)


def mesh_sharding(pspec: PartitionSpec) -> NamedSharding:
    return NamedSharding(mesh, pspec)


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
    options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=2, create=True)
    checkpoint_manager = orbax.checkpoint.CheckpointManager(
        checkpoint_dir, orbax_checkpointer, options
    )

    device_len = len(jax.devices())

    print(f"Using {device_len} devices")

    if (device_len != 1) and (device_len % 2 == 1):
        raise ValueError("not ")

    run = Experiment(
        api_key=local_env.comet_ml_api_key,
        project_name="jax-sine-conv",
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
    # _config["features"] = 2**7
    _config["features_list"] = tuple(
        2**n for n in [11, 11, 10, 10, 9, 9, 8, 8, 7, 7, 6, 6, 5, 5, 4, 4, 3, 3]
    )
    _config["sine_window"] = 2**8 - 1
    _config["sines_per_window"] = 2**5
    _config["cropping"] = crop.Cropping.CAUSAL
    _config["mesh_x_div"] = 1
    _config["loss_fn"] = LossFn.LOGCOSH
    with open(local_env.config_file, "r") as f:
        in_config = yaml.safe_load(f)["config"]
        for k, v in in_config.items():
            if k not in _config:
                raise ValueError(f"Unknown config key {k}")
        for k, v in _config.items():
            if k not in in_config:
                raise ValueError(f"Requires key {k}")
        _config = in_config
        _config["loss_fn"] = LossFn(_config["loss_fn"])
        _config["mesh_x"] = device_len // _config["mesh_x_div"]
        _config["mesh_y"] = _config["mesh_x_div"]
        _config["features_list"] = tuple(_config["features_list"])
        _config["cropping"] = crop.Cropping(_config["cropping"])
        del _config["mesh_x_div"]
    run.log_parameters(_config)
    if local_env.parallelism == Parallelism.PMAP:
        run.log_parameter("run_id", sys.argv[1])
    config = SimpleNamespace(**_config)

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
        paths=train_files,
        window=config.window,
        stride=config.stride,
        naug=0
    )
    proto_test_dataset, test_dataset_total = make_data(
        paths=test_files,
        window=config.window,
        stride=config.stride,        naug=0

    )
    proto_inference_dataset, inference_dataset_total = make_data(
        paths=test_files,
        window=config.inference_window,
        stride=config.stride,        naug=0

    )
    print("datasets generated")
    init_rng = jax.random.PRNGKey(config.seed)
    onez = jnp.ones([config.batch_size, config.window, 1])  # 1,

    def array_to_tuple(arr):
        if isinstance(arr, np.ndarray):
            return tuple(array_to_tuple(a) for a in arr)
        else:
            return arr

    module = SineconvNetwork(
        features_list=config.features_list,
        sines_per_window=config.sines_per_window,
        sine_window=config.sine_window,
        cropping=lambda x, y: crop.cropping_to_function(config.cropping)(
            x, y, lambda a, b: a + b
        ),
    )
    tx = optax.adam(config.learning_rate)
    sine_range = jnp.expand_dims(
        jnp.expand_dims(jnp.arange(config.window) / 44100, axis=0), axis=-1
    )

    if local_env.parallelism == Parallelism.SHARD:
        abstract_variables = jax.eval_shape(
            partial(
                create_train_state,
                module=module,
                tx=tx,
            ),
            init_rng,
            onez,
            batchify(lambda: sine_range, onez),
            make_phases(config.features_list, config.sines_per_window, onez),
        )

        state_sharding = nn.get_sharding(abstract_variables, mesh)

    jit_create_train_state = fork_on_parallelism(
        partial(
            jax.jit,
            static_argnums=(4, 5),
            in_shardings=(
                mesh_sharding(None)
                if local_env.parallelism == Parallelism.SHARD
                else None,
                x_sharding,
                x_sharding,
                x_sharding,
            ),  # PRNG key and x
            out_shardings=state_sharding,
        ),
        partial(jax.pmap, static_broadcasted_argnums=(4, 5)),
    )(create_train_state)
    rng_for_train_state = (
        init_rng
        if local_env.parallelism == Parallelism.SHARD
        else jax.random.split(
            init_rng, 8
        )  ### #UGH we hardcode 8, not sure why this worked before :-/
    )
    print("will call jit_create_train_state", rng_for_train_state.shape, onez)
    state = jit_create_train_state(
        rng_for_train_state,
        fork_on_parallelism(onez, onez),
        batchify(lambda: sine_range, onez),
        make_phases(config.features_list, config.sines_per_window, onez),
        module,
        tx,
    )

    target = {"model": state, "config": None}

    if RESTORE is not None:
        CKPT = checkpoint_manager.restore(RESTORE, target)

        state = CKPT["model"]

    jit_train_step = fork_on_parallelism(
        partial(
            jax.jit,
            static_argnums=(5, 6),
            in_shardings=(
                state_sharding,
                x_sharding,
                x_sharding,
                x_sharding,
                x_sharding,
            ),
            out_shardings=(state_sharding, None),
        ),
        partial(jax.pmap, static_broadcasted_argnums=(5, 6)),
    )(train_step)

    jit_compute_loss = fork_on_parallelism(
        partial(
            jax.jit,
            static_argnums=(5, 6),
            in_shardings=(
                state_sharding,
                x_sharding,
                x_sharding,
                x_sharding,
                x_sharding,
            ),
        ),
        partial(jax.pmap, static_broadcasted_argnums=(5, 6)),
    )(compute_loss)

    del init_rng  # Must not be used anymore.
    for epoch in range(config.epochs):
        # ugggh
        epoch_is_0 = epoch == 0
        to_take_in_0_epoch = 104
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

        # train
        train_total = train_dataset_total // config.batch_size
        with tqdm(
            enumerate(
                train_dataset.iter(batch_size=config.batch_size, drop_last_batch=False)
            ),
            total=train_total if not epoch_is_0 else to_take_in_0_epoch,
            unit="batch",
        ) as loop:
            for batch_ix, batch in loop:
                should_use_gen = batch_ix % 2 == 1
                input = trim_batch(jnp.array(batch["input"]), config.batch_size)
                if input.shape[0] == 0:
                    continue
                assert input.shape[1] == config.window
                input = maybe_replicate(input)
                input = maybe_device_put(input, x_sharding)
                target = trim_batch(jnp.array(batch["target"]), config.batch_size)
                assert target.shape[1] == config.window
                target = maybe_replicate(target)
                with fork_on_parallelism(mesh, nullcontext()):
                    state, loss = jit_train_step(
                        state,
                        input,
                        target,
                        batchify(lambda: sine_range, input),
                        make_phases(
                            config.features_list, config.sines_per_window, input
                        ),
                        config.loss_fn,
                        config.cropping,
                    )

                    state = add_losses_to_metrics(state=state, loss=loss)

                if batch_ix % config.step_freq == 0:
                    metrics = maybe_unreplicate(state.metrics).compute()
                    run.log_metrics({"train_loss": metrics["loss"]}, step=batch_ix)
                    loop.set_postfix(loss=metrics["loss"])
                    state = replace_metrics(state)
                    current_time = time.time()
                    elapsed_time = current_time - start_time
                    if elapsed_time >= 3600:
                        # checkpoint
                        ckpt_model = state
                        # needs to use underscore config
                        # becuase otherwise it doesn't serialize correctly
                        ckpt = {"model": ckpt_model, "config": _config}
                        if local_env.parallelism == Parallelism.PMAP:
                            ckpt = checkpoint_walker(ckpt)
                        CHECK_NAME = (
                            epoch * train_total
                            + batch_ix
                            + (RESTORE if RESTORE is not None else 0)
                        )
                        checkpoint_manager.save(CHECK_NAME, ckpt)
                        logging.warning(
                            f"saved checkpoint for epoch {epoch} in {os.listdir(checkpoint_dir)}"
                        )
                        try:
                            artifact = Artifact("checkpoint", artifact_type="model")
                            artifact.add(os.path.join(checkpoint_dir, f"{CHECK_NAME}"))
                            run.log_artifact(artifact)
                        except ValueError as e:
                            logging.warning(f"checkpoint artifact did not work {e}")
                        start_time = current_time
        test_dataset.set_epoch(epoch)
        with tqdm(
            enumerate(
                test_dataset.iter(batch_size=config.batch_size, drop_last_batch=False)
            ),
            total=(test_dataset_total // config.batch_size)
            if not epoch_is_0
            else to_take_in_0_epoch,
            unit="batch",
        ) as loop:
            for batch_ix, batch in loop:
                input = maybe_replicate(
                    trim_batch(jnp.array(batch["input"]), config.batch_size)
                )
                if input.shape[0] == 0:
                    continue
                input = maybe_device_put(input, x_sharding)
                target = maybe_replicate(
                    trim_batch(jnp.array(batch["target"]), config.batch_size)
                )
                loss = jit_compute_loss(
                    state,
                    input,
                    target,
                    batchify(lambda: sine_range, input),
                    make_phases(config.features_list, config.sines_per_window, input),
                    config.loss_fn,
                    config.cropping,
                )
                loop.set_postfix(loss=loss)
                state = add_losses_to_metrics(state=state, loss=loss)
        metrics = maybe_unreplicate(state.metrics).compute()
        run.log_metrics({"val_loss": metrics["loss"]}, step=batch_ix)
        state = replace_metrics(state)
        # inference
        inference_dataset.set_epoch(epoch)
        for batch_ix, batch in tqdm(
            enumerate(
                inference_dataset.take(
                    config.inference_artifacts_per_batch_per_epoch
                ).iter(batch_size=config.inference_batch_size)
            ),
            total=config.inference_artifacts_per_batch_per_epoch,
        ):
            input_ = trim_batch(jnp.array(batch["input"]), config.inference_batch_size)
            if input_.shape[0] == 0:
                continue
            target_ = trim_batch(
                jnp.array(batch["target"]), config.inference_batch_size
            )
            input = maybe_replicate(input_)
            input = maybe_device_put(input, x_sharding)
            logging.warning(f"input shape for inference is is {input.shape}")
            jit_do_inference = fork_on_parallelism(
                partial(
                    jax.jit,
                    # static_argnums=(2, 3),
                    in_shardings=(state_sharding, x_sharding, x_sharding, x_sharding),
                    out_shardings=x_sharding,
                ),
                partial(
                    jax.pmap,
                    # static_broadcasted_argnums=(2, 3)
                ),
            )(do_inference)

            o = jit_do_inference(
                state,
                input,
                batchify(lambda: sine_range, input),
                make_phases(config.features_list, config.sines_per_window, input),
            )
            o = maybe_unreplicate(o)
            assert o.shape[-1] == 1
            # logging.info(f"shape of batch is {input.shape}")

            for i in range(o.shape[0]):
                audy = np.squeeze(np.array(o[i]))
                run.log_audio(
                    audy,
                    sample_rate=44100,
                    step=batch_ix,
                    file_name=f"audio_{epoch}_{batch_ix}_{i}_prediction.wav",
                )
                audy = np.squeeze(np.array(input_[i, :, :1]))
                run.log_audio(
                    audy,
                    sample_rate=44100,
                    step=batch_ix,
                    file_name=f"audio_{epoch}_{batch_ix}_{i}_input.wav",
                )
                audy = np.squeeze(np.array(target_[i]))
                run.log_audio(
                    audy,
                    sample_rate=44100,
                    step=batch_ix,
                    file_name=f"audio_{epoch}_{batch_ix}_{i}_target.wav",
                )
