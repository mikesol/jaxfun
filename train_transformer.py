import os
from parallelism import Parallelism
from contextlib import nullcontext
import logging
import flax.linen as nn
from fork_on_parallelism import fork_on_parallelism
import yaml

# import logging
# logging.basicConfig(level=logging.INFO)
from types import SimpleNamespace
import local_env
import time
from loss import LossFn, Loss_fn_to_loss, LogCoshLoss, ESRLoss

start_time = time.time()

IS_CPU = local_env.parallelism == Parallelism.NONE
if IS_CPU:
    print("no gpus found")
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

import subprocess
from typing import Any
from flax import struct
from comet_ml import Experiment, Artifact
from transformer import TransformerNetwork
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
from data import make_data_16
import orbax.checkpoint
from tqdm import tqdm
import sys
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax.lax import with_sharding_constraint
from jax.experimental import mesh_utils

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


class TrainState(train_state.TrainState):
    metrics: Metrics
    key: Any


def create_train_state(rng: PRNGKey, dropout_key: PRNGKey, x, module, tx) -> TrainState:
    print("creating train state", rng.shape, x.shape)
    variables = module.init(rng, x, x, train=False)
    params = variables["params"]
    return TrainState.create(
        apply_fn=module.apply,
        params=params,
        tx=tx,
        key=dropout_key,
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


def train_step(state, input, target, dropout_key):
    """Train for a single step."""
    dropout_train_key = jax.random.fold_in(key=dropout_key, data=state.step)

    def loss_fn(params):
        pred = state.apply_fn(
            {"params": params},
            input[:, :-1, :],
            target[:, :-1, :],
            train=True,
            rngs={"dropout": dropout_train_key},
        )
        loss = optax.softmax_cross_entropy_with_integer_labels(
            pred, jnp.squeeze(target[:, 1:, :], axis=-1)
        ).mean()
        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss


def _replace_metrics(state):
    return state.replace(metrics=state.metrics.empty())


# todo: should we use scan to reduce compilation time?
def do_inference(state, input, w_size):
    B, T, C = input.shape
    input = input # input = jnp.pad(input, ((0, 0), (w_size, 0), (0, 0)))
    output = input[:, :w_size, :]
    to_loop = 1 # T - 1
    oo = None
    for x in range(to_loop):
        o = state.apply_fn(
            {"params": state.params},
            input[:, x : x + w_size, :],
            output,
            train=False,
        )
        # output is B, T, 1
        # o is (B, T, Logits)
        oo = o if x == 0 else jnp.concatenate([oo, o[:, -1:, :]], axis=1)
        output = jnp.concatenate(
            [output, jnp.expand_dims(jnp.argmax(o[:, -1:, :], axis=-1), axis=-1)],
            axis=1,
        )[:, 1:, :]
    return oo


replace_metrics = fork_on_parallelism(jax.jit, jax.pmap)(_replace_metrics)


def compute_loss(state, input, target, w_size):
    B, T, C = input.shape
    # find a way to make jitting practical
    # and then we can do the whole shebang
    input = input # input = jnp.pad(input, ((0, 0), (w_size, 0), (0, 0)))
    to_loop = 1 # T - 1
    output = input[:, :w_size, :]
    oo = None
    for x in range(to_loop):
        o = state.apply_fn(
            {"params": state.params},
            input[:, x : x + w_size, :],
            output,
            train=False,
        )
        # output is B, T, 1
        # o is (B, T, Logits)
        oo = o if x == 0 else jnp.concatenate([oo, o[:, -1:, :]], axis=1)
        output = jnp.concatenate(
            [output, jnp.expand_dims(jnp.argmax(o[:, -1:, :], axis=-1), axis=-1)],
            axis=1,
        )[:, 1:, :]
    loss = optax.softmax_cross_entropy_with_integer_labels(
        oo, jnp.reshape(target[:, 1:w_size + to_loop], (-1, w_size + to_loop - 1))
    ).mean()
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
        project_name="jax-transformer",
    )
    ## defaults
    _config = {}
    _config["mesh_x_div"] = 1
    _config["seed"] = 42
    _config["batch_size"] = 2**4
    _config["inference_batch_size"] = 2**3
    _config["inference_artifacts_per_batch_per_epoch"] = (
        _config["inference_batch_size"] * 4
    )
    _config["learning_rate"] = 1e-4
    _config["epochs"] = 2**7
    _config["window_plus_one"] = 2**10 + 1
    _config["val_window_plus_one"] = 2**11 + 1
    _config["inference_window"] = 2**15
    _config["stride"] = 2**8
    _config["step_freq"] = 2**6
    _config["test_size"] = 0.1
    _config["vocab_size"] = 2**16
    _config["n_embed"] = 2**10
    _config["n_heads"] = 2**5
    _config["dff"] = 2**11
    _config["depth"] = 2**4
    _config["dropout_rate"] = 0.2
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
    proto_train_dataset, train_dataset_total = make_data_16(
        paths=train_files,
        window=config.window_plus_one,
        stride=config.stride,
    )
    proto_test_dataset, test_dataset_total = make_data_16(
        paths=test_files,
        window=config.val_window_plus_one,
        stride=config.stride,
    )
    proto_inference_dataset, inference_dataset_total = make_data_16(
        paths=test_files,
        window=config.inference_window,
        stride=config.stride,
    )
    print("datasets generated")
    init_rng = jax.random.PRNGKey(config.seed)
    init_rng, dropout_rng = jax.random.split(init_rng, 2)
    onez = jnp.ones(
        [config.batch_size, config.window_plus_one - 1, 1], dtype=jnp.int32
    )  # 1,
    module = TransformerNetwork(
        vocab_size=config.vocab_size,
        block_size=config.window_plus_one - 1,
        n_embed=config.n_embed,
        num_heads=config.n_heads,
        dff=config.dff,
        depth=config.depth,
        dropout_rate=config.dropout_rate,
    )
    tx = optax.adamw(config.learning_rate)

    if local_env.parallelism == Parallelism.SHARD:
        abstract_variables = jax.eval_shape(
            partial(create_train_state, module=module, tx=tx),
            init_rng,
            dropout_rng,
            onez,
        )

        state_sharding = nn.get_sharding(abstract_variables, mesh)

    jit_create_train_state = fork_on_parallelism(
        partial(
            jax.jit,
            static_argnums=(3, 4),
            in_shardings=(
                (
                    mesh_sharding(None)
                    if local_env.parallelism == Parallelism.SHARD
                    else None
                ),
                (
                    mesh_sharding(None)
                    if local_env.parallelism == Parallelism.SHARD
                    else None
                ),
                x_sharding,
            ),  # PRNG key and x
            out_shardings=state_sharding,
        ),
        partial(jax.pmap, static_broadcasted_argnums=(3, 4)),
    )(create_train_state)
    init_rng, dropout_rng, loop_rng = jax.random.split(init_rng, 3)
    rng_for_train_state = (
        init_rng
        if local_env.parallelism == Parallelism.SHARD
        else jax.random.split(
            init_rng, 8
        )  ### #UGH we hardcode 8, not sure why this worked before :-/
    )
    dropout_rng_for_train_state = (
        dropout_rng
        if local_env.parallelism == Parallelism.SHARD
        else jax.random.split(
            dropout_rng, 8
        )  ### #UGH we hardcode 8, not sure why this worked before :-/
    )
    state = jit_create_train_state(
        rng_for_train_state,
        dropout_rng_for_train_state,
        onez,
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
            in_shardings=(
                state_sharding,
                x_sharding,
                x_sharding,
                (
                    mesh_sharding(None)
                    if local_env.parallelism == Parallelism.SHARD
                    else None
                ),
            ),
            out_shardings=(state_sharding, None),
        ),
        jax.pmap,
    )(train_step)

    jit_compute_loss = fork_on_parallelism(
        partial(
            jax.jit,
            static_argnums=(3,),
            in_shardings=(state_sharding, x_sharding, x_sharding),
        ),
        partial(jax.pmap, static_broadcasted_argnums=(3,)),
    )(compute_loss)
    jit_do_inference = fork_on_parallelism(
        partial(
            jax.jit,
            in_shardings=(state_sharding, x_sharding),
            out_shardings=x_sharding,
            static_argnums=(2,),
        ),
        partial(jax.pmap, static_broadcasted_argnums=(2,)),
    )(do_inference)
    del init_rng  # Must not be used anymore.
    for epoch in range(config.epochs):
        # ugggh
        # commenting out for now
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
                assert input.shape[1] == config.window_plus_one
                input = maybe_replicate(input)
                input = maybe_device_put(input, x_sharding)
                target = trim_batch(jnp.array(batch["target"]), config.batch_size)
                assert target.shape[1] == config.window_plus_one
                target = maybe_replicate(target)
                with fork_on_parallelism(mesh, nullcontext()):
                    loop_rng, new_dropout_rng = jax.random.split(loop_rng, 2)

                    dropout_rng_for_train_state = (
                        new_dropout_rng
                        if local_env.parallelism == Parallelism.SHARD
                        else jax.random.split(
                            new_dropout_rng, 8
                        )  ### #UGH we hardcode 8, not sure why this worked before :-/
                    )
                    state, loss = jit_train_step(
                        state, input, target, dropout_rng_for_train_state
                    )

                    state = add_losses_to_metrics(state=state, loss=loss)

                if batch_ix % config.step_freq == 0:
                    metrics = maybe_unreplicate(state.metrics).compute()
                    run.log_metrics({"train_loss": metrics["loss"]}, step=batch_ix)
                    loop.set_postfix(loss=metrics["loss"])
                    state = replace_metrics(state)
                    current_time = time.time()
                    elapsed_time = current_time - start_time
        # temporarily move checkpoint to after the first epoch as it crashes otherwise
        if True:
            # we test checkpointing early just to make sure it
            # works so there aren't any nasty surprises
            # checkpoint
            ckpt_model = state
            # needs to use underscore config
            # becuase otherwise it doesn't serialize correctly
            ckpt = {"model": ckpt_model, "config": _config}
            if local_env.parallelism == Parallelism.PMAP:
                ckpt = checkpoint_walker(ckpt)

            CHECK_NAME = (
                epoch * train_total
                # uncomment when we move back
                # + batch_ix
                + (RESTORE if RESTORE is not None else 0)
            )
            checkpoint_manager.save(CHECK_NAME, ckpt)
            logging.warning(
                f"saved checkpoint for epoch {epoch} in {os.listdir(checkpoint_dir)}"
            )
            try:
                subprocess.run(
                    f"gsutil -m rsync -r {os.path.join(checkpoint_dir)} gs://meeshkan-experiments/jax-transformers/{run.id}",
                    check=True,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
            except ValueError as e:
                logging.warning(f"checkpoint artifact did not work {e}")
            start_time = current_time
            # hack suggested on https://github.com/google/flax/discussions/1690
            # print(state.params)
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
                    config.window_plus_one - 1,
                )
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
            o = jit_do_inference(state, input, config.window_plus_one - 1)
            o = maybe_unreplicate(o)
            # this will squeeze out the logit dimension
            o = jnp.argmax(o, axis=-1)
            assert len(o.shape) == 2
            # logging.info(f"shape of batch is {input.shape}")

            for i in range(o.shape[0]):
                audy = np.array(o[i])
                # vocab to float
                audy = audy.astype(np.float32) - 32768
                audy = audy / 32768
                print("prediction dimension", audy.shape)
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
