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




maybe_replicate = fork_on_parallelism(lambda x: x, jax_utils.replicate)
maybe_unreplicate = fork_on_parallelism(lambda x: x, jax_utils.unreplicate)
maybe_device_put = fork_on_parallelism(jax.device_put, lambda x, _: x)


def mesh_sharding(pspec: PartitionSpec) -> NamedSharding:
    return NamedSharding(mesh, pspec)


def find_near_zero_weights(params, near_zero_weights, epsilon=1e-6, prefix=""):
    for layer_name, weights in params.items():
        if type(weights) == type({}):
            find_near_zero_weights(weights, near_zero_weights, epsilon=epsilon, prefix=os.path.join(prefix, layer_name))
            continue
        # Assuming weights are stored in NumPy arrays
        near_zero_count = (np.abs(weights) < epsilon).sum()
        total_count = weights.size
        near_zero_weights[os.path.join(prefix,layer_name)] = (near_zero_count, total_count)

    return near_zero_weights

def total_weights(params, epsilon=1e-6, prefix=""):
    total = 0
    for layer_name, weights in params.items():
        if type(weights) == type({}):
            total += total_weights(weights, epsilon=epsilon, prefix=os.path.join(prefix, layer_name))
            continue
        # Assuming weights are stored in NumPy arrays
        total += weights.size

    return total

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
        # _config["loss_fn"] = LossFn(_config["loss_fn"])
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

    target = {"model": state, "config":_config}
    restore_args = orbax_utils.restore_args_from_target(target)

    CKPT = checkpoint_manager.restore(
        local_env.restore, target, restore_kwargs={"restore_args": restore_args}
    )

    state = CKPT["model"]

    del init_rng  # Must not be used anymore.

    nz_weights = find_near_zero_weights(state.params, {}, epsilon=1e-6)
    for layer, (near_zero, total) in nz_weights.items():
        print(f"Layer {layer}: {near_zero} out of {total} weights are near zero (below epsilon)")

    print(total_weights(params=state.params))