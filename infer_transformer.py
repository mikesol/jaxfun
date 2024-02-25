import os
from parallelism import Parallelism
from contextlib import nullcontext
import logging
import librosa
from bias_types import BiasTypes
from activation import Activation, make_activation
from enum import Enum
import flax.linen as nn
from fork_on_parallelism import fork_on_parallelism
from create_filtered_audio import create_biquad_coefficients
import yaml
from train_transformer import (
    create_train_state,
    do_inference,
    mesh_sharding,
    maybe_replicate,
    maybe_device_put,
)


# import logging
# logging.basicConfig(level=logging.INFO)
import soundfile
from types import SimpleNamespace
import local_env
import time
from flax.training import orbax_utils
from loss import LossFn, Loss_fn_to_loss, LogCoshLoss, ESRLoss
from dataclasses import dataclass


start_time = time.time()

IS_CPU = local_env.parallelism == Parallelism.NONE
if IS_CPU:
    print("no gpus found")
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

from typing import Any
from flax import struct
from transformer import TransformerNetwork
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

PRNGKey = jax.Array

def do_inference_(state, input, w_size):
    return do_inference(state, input, w_size)

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
    options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=1, create=True)
    checkpoint_manager = orbax.checkpoint.CheckpointManager(
        checkpoint_dir, orbax_checkpointer, options
    )

    device_len = len(jax.devices())

    print(f"Using {device_len} devices")

    if (device_len != 1) and (device_len % 2 == 1):
        raise ValueError("not ")

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
    _config["mask_encoder"] = True
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
    config = SimpleNamespace(**_config)
    device_mesh = mesh_utils.create_device_mesh((config.mesh_x, config.mesh_y))
    mesh = Mesh(devices=device_mesh, axis_names=("data", "model"))
    print(mesh)
    x_sharding = NamedSharding(mesh, PartitionSpec("data", None))

    init_rng = jax.random.PRNGKey(config.seed)
    init_rng, dropout_rng = jax.random.split(init_rng, 2)
    onez = jnp.ones([config.batch_size, config.window, 1])  # 1,

    def array_to_tuple(arr):
        if isinstance(arr, np.ndarray):
            return tuple(array_to_tuple(a) for a in arr)
        else:
            return arr

    module = TransformerNetwork(
        vocab_size=config.vocab_size,
        block_size=config.window_plus_one - 1,
        n_embed=config.n_embed,
        num_heads=config.n_heads,
        dff=config.dff,
        depth=config.depth,
        dropout_rate=config.dropout_rate,
        mask_encoder=config.mask_encoder,
    )
    tx = optax.adam(config.learning_rate)

    abstract_variables = jax.eval_shape(
        partial(
            create_train_state,
            module=module,
            tx=tx,
        ),
        init_rng,
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

    ### data
    input, _ = librosa.load(local_env.inference_file_source, sr=44100)
    target, _ = librosa.load(local_env.inference_file_target, sr=44100)
    input, target = jnp.reshape(input, (-1,)), jnp.reshape(target, (-1,))
    # for now hardcode the length
    # print(config.window * config.batch_size, input[: config.window * config.batch_size].shape)
    input_ = jnp.reshape(input[: config.window * 16 * 8], (8, -1, 1))
    print("input shape is", input_.shape)
    input_ = maybe_replicate(input_)
    input_ = maybe_device_put(input_, x_sharding)

    ###

    state = jit_create_train_state(
        rng_for_train_state,
        dropout_rng_for_train_state,
        onez,
        module,
        tx,
    )
    target = {"model": state, "config": None}
    restore_args = orbax_utils.restore_args_from_target(target)

    CKPT = checkpoint_manager.restore(
        local_env.restore, target, restore_kwargs={"restore_args": restore_args}
    )
    state = CKPT["model"]

    jit_do_inference = fork_on_parallelism(
        partial(
            jax.jit,
            static_argnums=(2,),
            in_shardings=(state_sharding, x_sharding),
            out_shardings=x_sharding,
        ),
        partial(jax.pmap, static_broadcasted_argnums=(2,)),
    )(do_inference_)
    del init_rng  # Must not be used anymore.
    o = jit_do_inference(state, input, config.window_plus_one - 1)
    # # for now, we only keep the second half as we are training it to always have
    # # padding in the beginning
    # # we can alter training later if needed
    # out_length = o.shape[1] // 2
    # # hopefully this is the case, if not figure out something creative!
    # assert o.shape[1] % 2 == 0
    # logging.warning(
    #     f"input shape for inference is is {input_.shape} with output {out_length}"
    # )
    # input_ = jax.lax.conv_general_dilated_patches(
    #     jnp.transpose(
    #         jnp.reshape(input[: config.window * config.batch_size], (1, -1, 1)),
    #         (0, 2, 1),
    #     ),
    #     filter_shape=(config.window,),
    #     window_strides=(out_length // 2,),
    #     padding=((0, 0),),
    # )
    # input_ = jnp.transpose(
    #     # kernel, seq, chan
    #     jnp.reshape(input_, (config.window, -1, 1)),
    #     # seq, kernel, chan
    #     (0, 2, 1),
    # )

    # (o,) = jit_do_inference(state, input, conversion_config)
    # o = maybe_unreplicate(o)
    # assert o.shape[-1] == 1
    # # logging.info(f"shape of batch is {input.shape}")

    audy = np.reshape(o[0], (-1,))
    soundfile.write("/tmp/output.wav", audy, samplerate=44100)
