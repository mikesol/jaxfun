import os
from parallelism import Parallelism
from contextlib import nullcontext
import logging
import librosa
from bias_types import BiasTypes
from activation import Activation, make_activation
from enum import Enum
from fork_on_parallelism import fork_on_parallelism
from fade_in import apply_fade_in
from create_filtered_audio import create_biquad_coefficients
import soundfile
from flax.training import orbax_utils
from loss import LossFn, Loss_fn_to_loss, LogCoshLoss, ESRLoss

# import logging
# logging.basicConfig(level=logging.INFO)
import soundfile
from types import SimpleNamespace
import local_env
import time
import yaml

start_time = time.time()

IS_CPU = local_env.parallelism == Parallelism.NONE
if IS_CPU:
    print("no gpus found")
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

from typing import Any
from flax import struct
from rnn import LSTMDrivingSines2
from clu import metrics
from functools import partial
import jax.numpy as jnp
import flax.jax_utils as jax_utils
import numpy as np
import flax.linen as nn
import math
from flax.training import train_state
import optax
import jax
import orbax.checkpoint
from tqdm import tqdm
import sys
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax.experimental import mesh_utils
import scipy

PRNGKey = jax.Array


@struct.dataclass
class Metrics(metrics.Collection):
    loss: metrics.Average.from_output("loss")


class TrainState(train_state.TrainState):
    metrics: Metrics


def create_train_state(rng: PRNGKey, x, module, tx) -> TrainState:
    print("creating train state", rng.shape, x.shape)
    variables = module.init(rng, x)
    params = variables["params"]
    return TrainState.create(
        apply_fn=module.apply,
        params=params,
        tx=tx,
        metrics=Metrics.empty(),
    )


def do_inference(state, input):
    o, _ = state.apply_fn(
        {"params": state.params},
        input,
    )
    return o


maybe_replicate = fork_on_parallelism(lambda x: x, jax_utils.replicate)
maybe_unreplicate = fork_on_parallelism(lambda x: x, jax_utils.unreplicate)
maybe_device_put = fork_on_parallelism(jax.device_put, lambda x, _: x)


def mesh_sharding(pspec: PartitionSpec) -> NamedSharding:
    return NamedSharding(mesh, pspec)


if __name__ == "__main__":
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

    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=2, create=True)
    checkpoint_manager = orbax.checkpoint.CheckpointManager(
        checkpoint_dir, orbax_checkpointer, options
    )

    device_len = len(jax.devices())

    print(f"Using {device_len} devices")

    if (device_len != 1) and (device_len % 2 == 1):
        raise ValueError("not ")

    _config = {}
    with open(local_env.config_file, "r") as f:
        in_config = yaml.safe_load(f)["config"]
        _config = in_config
        _config["loss_fn"] = LossFn(_config["loss_fn"])
        _config["mesh_x"] = device_len // _config["mesh_x_div"]
        _config["mesh_y"] = _config["mesh_x_div"]
        del _config["mesh_x_div"]

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

    init_rng = jax.random.PRNGKey(config.seed)
    onez = jnp.ones([config.batch_size, config.window, 1])  # 1,

    def array_to_tuple(arr):
        if isinstance(arr, np.ndarray):
            return tuple(array_to_tuple(a) for a in arr)
        else:
            return arr

    module = LSTMDrivingSines2(
        features=config.features,
        skip=config.skip,
        levels=config.levels,
        end_features=config.end_features,
        end_levels=config.end_levels,
    )
    tx = optax.adam(config.learning_rate)

    if local_env.parallelism == Parallelism.SHARD:
        abstract_variables = jax.eval_shape(
            partial(create_train_state, module=module, tx=tx),
            init_rng,
            onez,
        )

        state_sharding = nn.get_sharding(abstract_variables, mesh)

    jit_create_train_state = fork_on_parallelism(
        partial(
            jax.jit,
            static_argnums=(2, 3),
            in_shardings=(
                mesh_sharding(None)
                if local_env.parallelism == Parallelism.SHARD
                else None,
                x_sharding,
            ),  # PRNG key and x
            out_shardings=state_sharding,
        ),
        partial(jax.pmap, static_broadcasted_argnums=(2, 3)),
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
        module,
        tx,
    )

    __config = {**_config}
    __config["loss_fn"] = _config["loss_fn"].value
    target = {"model": state, "config": __config}
    restore_args = orbax_utils.restore_args_from_target(target)

    CKPT = checkpoint_manager.restore(
        local_env.restore, target, restore_kwargs={"restore_args": restore_args}
    )

    state = CKPT["model"]

    del init_rng  # Must not be used anymore.

    # ugggh

    input_, _ = librosa.load(local_env.inference_file_source, sr=44100)
    print("s1", input_.shape)
    target_, _ = librosa.load(local_env.inference_file_target, sr=44100)
    ##### ugggggggggh
    # input_ = jnp.concatenate([input_ for _ in range(device_len)], axis=0)
    input = input_
    input = jnp.expand_dims(input, axis=0)
    input = jnp.expand_dims(input, axis=-1)
    target = target_
    assert len(input.shape) == 3

    jit_do_inference = fork_on_parallelism(
        partial(
            jax.jit,
            in_shardings=(state_sharding, x_sharding),
            out_shardings=x_sharding,
        ),
        jax.pmap,
    )(do_inference)
    # jit_do_inference = jax.jit(do_inference)
    print("input shape", input.shape)
    stride = config.window + 1
    o = jit_do_inference(state, input[:, :stride, :])
    o_len = o.shape[1]
    print("olen", o_len)
    assert o_len % 2 == 0
    half_o_len = o_len // 2
    hann = scipy.signal.windows.hann(o_len)
    print("starting inference with size", o_len)
    offset = 0
    zzz = np.zeros((44100 * 11,))
    a = []
    while offset < (44100 * 10):
        o = jit_do_inference(state, input[:, offset : offset + stride, :])
        loss = Loss_fn_to_loss(config.loss_fn)(
            o, input[:, offset : offset + o.shape[1], :]
        )
        print("on second", offset / 44100, loss)
        o = jnp.squeeze(o)
        o = np.array(o)
        assert o.shape == (o_len,)
        o = o * hann
        zzz[offset : offset + o_len] += o
        offset += half_o_len
        zzz[offset : offset + o_len] += o
        offset += o_len

    print("o after concat", o.shape)
    print("o after squeeze", o.shape)
    soundfile.write("/tmp/input.wav", input_, 44100)
    soundfile.write("/tmp/prediction.wav", zzz, 44100)
    soundfile.write("/tmp/target.wav", target_, 44100)
