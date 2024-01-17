import os
from parallelism import Parallelism
from contextlib import nullcontext
import logging
import librosa
from enum import Enum
from fork_on_parallelism import fork_on_parallelism
from fade_in import apply_fade_in
from create_filtered_audio import create_biquad_coefficients
import soundfile
from flax.training import orbax_utils

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
from comet_ml import Experiment, Artifact
from tcn import TCNNetwork, ExperimentalTCNNetwork
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
from jax.lax import with_sharding_constraint
from jax.experimental import mesh_utils

RESTORE = 946991


PRNGKey = jax.Array


@struct.dataclass
class Metrics(metrics.Collection):
    loss: metrics.Average.from_output("loss")


class TrainState(train_state.TrainState):
    metrics: Metrics
    batch_stats: Any


def create_train_state(rng: PRNGKey, x, module, tx) -> TrainState:
    print("creating train state", rng.shape, x.shape)
    variables = module.init(rng, x, train=False)
    params = variables["params"]
    batch_stats = variables["batch_stats"]
    return TrainState.create(
        apply_fn=module.apply,
        params=params,
        tx=tx,
        batch_stats=batch_stats,
        metrics=Metrics.empty(),
    )


class LossFn(Enum):
    LOGCOSH = 1
    ESR = 2
    LOGCOSH_RANGE = 3


def do_inference(state, input):
    o, _ = state.apply_fn(
        {"params": state.params, "batch_stats": state.batch_stats},
        input,
        train=False,
        mutable=["batch_stats"],
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
    # cnn
    _config["seed"] = 42
    _config["batch_size"] = 2**4
    _config["inference_batch_size"] = 2**3
    _config["inference_artifacts_per_batch_per_epoch"] = (
        _config["inference_batch_size"] * 4
    )
    _config["validation_split"] = 0.2
    _config["learning_rate"] = 1e-4
    _config["epochs"] = 2**7
    _config["window"] = 2**11
    _config["inference_window"] = 2**10  # 2**11
    _config["stride"] = 2**8
    _config["step_freq"] = 2**6
    _config["test_size"] = 0.1
    # _config["features"] = 2**7
    _config["kernel_dilation"] = 2**1
    _config["conv_kernel_size"] = 2**3
    _config["attn_kernel_size"] = 2**5  # 2**6
    _config["heads"] = 2**2
    _config["conv_depth"] = tuple(
        2**n for n in (10, 10, 9, 9, 8, 8, 7, 7)
    )  # 2**3  # 2**4
    _config["attn_depth"] = 2**3  # 2**2  # 2**4
    _config["sidechain_modulo_l"] = 2
    _config["sidechain_modulo_r"] = 1214124  # set to high to avoid
    _config["expand_factor"] = 2.0
    _config["positional_encodings"] = True
    _config["kernel_size"] = 7
    _config["mesh_x"] = device_len // 1
    _config["mesh_y"] = 1
    _config["loss_fn"] = LossFn.LOGCOSH
    #
    _config["afstart"] = 100
    _config["afend"] = 19000
    _config["qstart"] = 30
    _config["qend"] = 10
    ###
    # CKPT = checkpoint_manager.restore(RESTORE)
    # if ("config" in CKPT) and (len(CKPT["config"]) > 0):
    #     _config = {**_config, **CKPT["config"]}
    # else:
    #     logging.warning("No config found. Try ing local.")
    #     if os.path.exists("cofig.yaml"):
    #         with open("config.yaml", "r") as yfile:
    #             _config = {**_config, **yaml.load(yfile)}
    #     else:
    #         logging.warning(
    #             "No config file exists. Make sure to set the params manually!"
    #         )

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

    coefficients = create_biquad_coefficients(
        config.conv_depth[0] - 1,
        44100,
        config.afstart,
        config.afend,
        config.qstart,
        config.qend,
    )
    module = ExperimentalTCNNetwork(
        # features=config.features,
        coefficients=array_to_tuple(coefficients),
        kernel_dilation=config.kernel_dilation,
        conv_kernel_size=config.conv_kernel_size,
        attn_kernel_size=config.attn_kernel_size,
        heads=config.heads,
        conv_depth=config.conv_depth,
        attn_depth=config.attn_depth,
        expand_factor=config.expand_factor,
        positional_encodings=config.positional_encodings,
        sidechain_modulo_l=config.sidechain_modulo_l,
        sidechain_modulo_r=config.sidechain_modulo_r,
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

    target = {"model": state, "config": None}
    restore_args = orbax_utils.restore_args_from_target(target)

    CKPT = checkpoint_manager.restore(
        RESTORE, target, restore_kwargs={"restore_args": restore_args}
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


    jit_do_inference = jax.jit(do_inference)
    # jit_do_inference = jax.jit(do_inference)
    print("input shape", input.shape)
    stride = 2**13
    o = jit_do_inference(state, input[:,:stride,:])
    o = np.squeeze(np.array(o))
    size_diff = input.shape[1] - o.shape[1]
    print('starting inference with size_diff', size_diff)
    offset = 0
    a = []
    while offset < (44100 * 10):
        print('on second', offset / 44100)
        o = jit_do_inference(state, input[:,offset:offset+stride,:])
        offset += stride - size_diff
        a.append(o)

    o = np.concatenate(a, axis=1)
    soundfile.write("/tmp/input.wav", input_, 44100)
    soundfile.write("/tmp/prediction.wav", o, 44100)
    soundfile.write("/tmp/target.wav", target_, 44100)
