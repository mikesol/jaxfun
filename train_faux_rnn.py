import os
import GPUtil
import truncate_if_odd
from parallelism import Parallelism
from contextlib import nullcontext
import logging
from enum import Enum
from fork_on_parallelism import fork_on_parallelism

# import logging
# logging.basicConfig(level=logging.INFO)
import soundfile
from types import SimpleNamespace
import local_env

IS_CPU = local_env.parallelism == Parallelism.NONE
if IS_CPU:
    print("no gpus found")
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

from typing import Any
from flax import struct
from comet_ml import Experiment, Artifact
from cnn import ConvFauxLarsen
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
from data import make_2d_data
import orbax.checkpoint
from tqdm import tqdm
import sys
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax.lax import with_sharding_constraint
from jax.experimental import mesh_utils

logging.basicConfig(level=logging.WARN)
logging.warn("logging works")
if local_env.parallelism == Parallelism.PMAP:
    if local_env.do_manual_parallelism_setup:
        jax.distributed.initialize(
            coordinator_address=local_env.coordinator_address,
            num_processes=local_env.num_processes,
            process_id=local_env.process_id,
        )
    else:
        jax.distributed.initialize()


def LogCoshLoss(input, target, a=1.0, eps=1e-8):
        losses = jnp.mean(
            (1 / a) * jnp.log(jnp.cosh(a * (input - target)) + eps), axis=-2
        )
        losses = jnp.mean(losses)
        return losses


def ESRLoss(input, target):
    eps = 1e-8
    num = jnp.sum(((target - input) ** 2), axis=1)
    denom = jnp.sum(target**2, axis=1) + eps
    losses = num / denom
    losses = jnp.mean(losses)
    return losses


checkpoint_dir = "/tmp/flax_ckpt/orbax/managed"

if os.path.exists(checkpoint_dir):
    raise ValueError(f"clear checkpoint dir first: {checkpoint_dir}")
else:
    os.makedirs(checkpoint_dir)

orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=2, create=True)
checkpoint_manager = orbax.checkpoint.CheckpointManager(
    checkpoint_dir, orbax_checkpointer, options
)


def checkpoint_walker(ckpt):
    def _cmp(i):
        # logging.warning(f"considering {type(i)}")
        # if type(i) == type(jnp.ones((1, 1))):
        #     logging.warning(
        #         f"info for type {i.is_fully_addressable} {i.is_fully_replicated} {i.shape} {i.sharding}"
        #     )
        try:
            # orbax.checkpoint.utils.fully_replicated_host_local_array_to_global_array
            o = jax.device_get(i)
            # logging.warning("found a fully replicated local array")
            return o
        except Exception as e:
            # logging.warning(f"could not make global {e}")
            return i

    # logging.warning("attempting treemap")
    return jax.tree_map(_cmp, ckpt)


PRNGKey = jax.Array


@struct.dataclass
class Metrics(metrics.Collection):
    loss: metrics.Average.from_output("loss")


class TrainState(train_state.TrainState):
    metrics: Metrics
    batch_stats: Any


def create_train_state(rng: PRNGKey, x, module, tx) -> TrainState:
    print("creating train state", rng.shape, x.shape)
    variables = module.init(rng, x, train=False, to_mask=x.shape[1] // 8)
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

def train_step(state, input, target, to_mask, comparable_field, loss_fn):
    """Train for a single step."""

    def loss_fn(params):
        pred, updates = state.apply_fn(
            {"params": params, "batch_stats": state.batch_stats},
            input,
            train=True,
            to_mask=to_mask,
            mutable=["batch_stats"],
        )
        loss = (ESRLoss if loss_fn == LossFn.ESR else LogCoshLoss)(pred[:, -comparable_field:, :], target[:, -comparable_field:, :])
        return loss, updates

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, updates), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    state = state.replace(batch_stats=updates["batch_stats"])
    return state, loss


def _replace_metrics(state):
    return state.replace(metrics=state.metrics.empty())


def do_inference(state, input, to_mask):
    o, _ = state.apply_fn(
        {"params": state.params, "batch_stats": state.batch_stats},
        input,
        train=False,
        to_mask=to_mask,
        mutable=["batch_stats"],
    )
    return o


replace_metrics = fork_on_parallelism(jax.jit, jax.pmap)(_replace_metrics)


def compute_loss(state, input, target, to_mask, comparable_field, loss_fn):
    pred, _ = state.apply_fn(
        {"params": state.params, "batch_stats": state.batch_stats},
        input,
        train=False,
        to_mask=to_mask,
        mutable=["batch_stats"],
    )
    loss = (ESRLoss if loss_fn == LossFn.ESR else LogCoshLoss)(pred[:, -comparable_field:, :], target[:, -comparable_field:, :])
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

    device_len = len(jax.devices())

    print(f"Using {device_len} devices")

    if (device_len != 1) and (device_len % 2 == 1):
        raise ValueError("not ")

    run = Experiment(
        api_key=local_env.comet_ml_api_key,
        project_name="jax-faux-rnn",
    )
    _config = {}
    # cnn
    _config["seed"] = 42
    _config["inference_artifacts_per_batch_per_epoch"] = 2**2
    _config["batch_size"] = 2**4
    _config["validation_split"] = 0.2
    _config["learning_rate"] = 1e-4
    _config["epochs"] = 2**7
    _config["window"] = 2**11
    _config["inference_window"] = 2**17
    _config["stride"] = 2**8
    _config["step_freq"] = 100
    _config["test_size"] = 0.1
    _config["channels"] = 2**4
    _config["depth"] = 2**3
    _config["to_mask"] = 2**10
    _config["comparable_field"] = _config["to_mask"] // 2
    _config["kernel_size"] = 7
    _config["skip_freq"] = 1
    _config["norm_factor"] = math.sqrt(_config["channels"])
    _config["inner_skip"] = True
    _config["shift"] = 2**4
    _config["dilation"] = 2**0
    _config["modulo_lhs"] = 4
    _config["modulo_rhs"] = 2
    _config["mesh_x"] = 2
    _config["mesh_y"] = device_len // _config["mesh_x"]
    _config["loss_fn"] = LossFn.LOGCOSH
    _config["do_progressive_masking"] = False
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
    # can't use make_2d_data_with_delays_and_dilations because the RNN becomes too dicey :-(
    proto_train_dataset, train_dataset_total = make_2d_data(
        paths=train_files,
        window=config.window,
        stride=config.stride,  # , shift=config.shift, dilation=config.dilation, channels=config.channels, feature_dim=-1, shuffle=True
        # shuffle=fork_on_parallelism(True, False),
    )
    proto_test_dataset, test_dataset_total = make_2d_data(
        paths=test_files,
        window=config.window,
        stride=config.stride,  # , shift=config.shift, dilation=config.dilation, channels=config.channels, feature_dim=-1, shuffle=True
        # shuffle=fork_on_parallelism(True, False),
    )
    proto_inference_dataset, inference_dataset_total = make_2d_data(
        paths=test_files,
        window=config.inference_window,
        stride=config.stride,  # , shift=config.shift, dilation=config.dilation, channels=config.channels, feature_dim=-1, shuffle=True
        # shuffle=fork_on_parallelism(True, False),
    )
    print("datasets generated")
    init_rng = jax.random.PRNGKey(config.seed)
    onez = jnp.ones(
        [
            config.batch_size,
            config.window * 2,
            1,
        ]
    )
    par_onez = maybe_replicate(jnp.ones([config.batch_size, config.window * 2, 1]))

    module = ConvFauxLarsen(
        channels=config.channels,
        depth=config.depth,
        kernel_size=config.kernel_size,
        skip_freq=config.skip_freq,
        norm_factor=config.norm_factor,
        inner_skip=config.inner_skip,
        modulo_lhs=config.modulo_lhs,
        modulo_rhs=config.modulo_rhs,
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
    print("will call jit_create_train_state", rng_for_train_state.shape, par_onez.shape)
    state = jit_create_train_state(
        rng_for_train_state,
        fork_on_parallelism(onez, par_onez),
        module,
        tx,
    )

    jit_train_step = fork_on_parallelism(
        partial(
            jax.jit,
            static_argnums=(3, 4, 5),
            in_shardings=(state_sharding, x_sharding, x_sharding),
            out_shardings=(state_sharding, None),
        ),
        partial(jax.pmap, static_broadcasted_argnums=(3, 4, 5)),
    )(train_step)

    jit_compute_loss = fork_on_parallelism(
        partial(
            jax.jit,
            static_argnums=(3, 4, 5),
            in_shardings=(state_sharding, x_sharding, x_sharding),
        ),
        partial(jax.pmap, static_broadcasted_argnums=(3, 4, 5)),
    )(compute_loss)

    to_mask = config.to_mask
    comparable_field = to_mask // 2
    del init_rng  # Must not be used anymore.
    for epoch in range(config.epochs):
        # ugggh
        epoch_is_0 = epoch == 0
        train_dataset = (
            proto_train_dataset
            if not epoch_is_0
            else proto_train_dataset.take(config.batch_size * 2)
        )
        test_dataset = (
            proto_test_dataset
            if not epoch_is_0
            else proto_test_dataset.take(config.batch_size * 2)
        )
        inference_dataset = (
            proto_inference_dataset
            if not epoch_is_0
            else proto_inference_dataset.take(config.batch_size * 2)
        )

        # log the epoch
        run.log_current_epoch(epoch)
        train_dataset.set_epoch(epoch)
        # train
        for batch_ix, batch in tqdm(
            enumerate(
                train_dataset.iter(batch_size=config.batch_size, drop_last_batch=True)
            ),
            total=train_dataset_total // config.batch_size if not epoch_is_0 else 2,
        ):
            input = maybe_replicate(jnp.array(batch["input"]))
            input = maybe_device_put(input, x_sharding)
            target = maybe_replicate(jnp.array(batch["target"]))
            with fork_on_parallelism(mesh, nullcontext()):
                state, loss = jit_train_step(
                    state, input, target, to_mask, comparable_field, config.loss_fn
                )
                state = add_losses_to_metrics(state=state, loss=loss)

            if batch_ix % config.step_freq == 0:
                metrics = maybe_unreplicate(state.metrics).compute()
                run.log_metrics({"train_loss": metrics["loss"]}, step=batch_ix)
                state = replace_metrics(state)
        test_dataset.set_epoch(epoch)
        for batch_ix, batch in tqdm(
            enumerate(
                test_dataset.iter(batch_size=config.batch_size, drop_last_batch=True)
            ),
            total=test_dataset_total // config.batch_size if not epoch_is_0 else 2,
        ):
            input = maybe_replicate(jnp.array(batch["input"]))
            input = maybe_device_put(input, x_sharding)
            target = maybe_replicate(jnp.array(batch["target"]))
            loss = jit_compute_loss(state, input, target, to_mask, comparable_field, config.loss_fn)
            state = add_losses_to_metrics(state=state, loss=loss)
        metrics = maybe_unreplicate(state.metrics).compute()
        run.log_metrics({"val_loss": metrics["loss"]}, step=batch_ix)
        state = replace_metrics(state)

        if not epoch_is_0 and config.do_progressive_masking:
            to_mask += config.to_mask
            comparable_field = config.to_mask // 2

        # checkpoint
        ckpt_model = state
        ckpt = {"model": ckpt_model, "config": config}
        if local_env.parallelism == Parallelism.PMAP:
            ckpt = checkpoint_walker(ckpt)
        checkpoint_manager.save(epoch, ckpt)
        logging.warning(
            f"saved checkpoint for epoch {epoch} in {os.listdir(checkpoint_dir)}"
        )
        try:
            artifact = Artifact("checkpoint", artifact_type="model")
            artifact.add(os.path.join(checkpoint_dir, f"{epoch}"))
            run.log_artifact(artifact)
        except ValueError as e:
            logging.warning(f"checkpoint artifact did not work {e}")
        # inference
        inference_dataset.set_epoch(epoch)
        for batch_ix, batch in tqdm(
            enumerate(
                inference_dataset.take(
                    config.inference_artifacts_per_batch_per_epoch
                ).iter(batch_size=config.batch_size)
            ),
            total=config.inference_artifacts_per_batch_per_epoch
            if not epoch_is_0
            else 2,
        ):
            input_ = truncate_if_odd.truncate_if_odd(jnp.array(batch["input"]))
            target_ = truncate_if_odd.truncate_if_odd(jnp.array(batch["input"]))
            input = maybe_replicate(input_)
            input = maybe_device_put(input, x_sharding)
            logging.warning(f"input shape for inference is is {input.shape}")
            jit_do_inference = fork_on_parallelism(
                partial(
                    jax.jit,
                    static_argnums=(2,),
                    in_shardings=(state_sharding, x_sharding),
                    out_shardings=x_sharding,
                ),
                partial(jax.pmap, static_broadcasted_argnums=(2,)),
            )(do_inference)
            o = jit_do_inference(ckpt_model, input, config.to_mask)
            o = maybe_unreplicate(o)
            # logging.info(f"shape of batch is {input.shape}")

            for i in range(o.shape[0]):
                audy = np.squeeze(np.array(o[i]))
                run.log_audio(
                    audy,
                    sample_rate=44100,
                    step=batch_ix,
                    file_name=f"audio_{epoch}_{batch_ix}_{i}_prediction.wav",
                )
                audy = np.squeeze(np.array(input_[i]))[::2]
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
