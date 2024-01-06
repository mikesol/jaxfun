import os
import GPUtil

IS_CPU = len(GPUtil.getAvailable()) == 0
if IS_CPU:
    print("in cpu land")
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

import soundfile
from types import SimpleNamespace
import local_env
from typing import Any
from flax import struct
from comet_ml import Experiment, Artifact
from cnn import ConvFauxLarsen
from clu import metrics
from functools import partial
import jax.numpy as jnp
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

PRNGKey = jax.Array


@struct.dataclass
class Metrics(metrics.Collection):
    loss: metrics.Average.from_output("loss")
    # long_loss: metrics.Average.from_output("long_loss")


class TrainState(train_state.TrainState):
    metrics: Metrics
    batch_stats: Any


def create_train_state(rng: PRNGKey, x, module, tx) -> TrainState:
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


def update_train_state(state: TrainState, model: nn.Module) -> TrainState:
    state = state.replace(apply_fn=model.apply)
    return state


def train_step(state, input, target, to_mask, comparable_field):
    """Train for a single step."""

    def loss_fn(params):
        pred, updates = state.apply_fn(
            {"params": params, "batch_stats": state.batch_stats},
            input,
            train=True,
            to_mask=to_mask,
            mutable=["batch_stats"],
        )
        loss = ESRLoss(pred[:, -comparable_field:, :], target[:, -comparable_field:, :])
        return loss, updates

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, updates), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    state = state.replace(batch_stats=updates["batch_stats"])
    return state, loss


@jax.jit
def replace_metrics(state):
    return state.replace(metrics=state.metrics.empty())


def compute_loss(state, input, target, to_mask, comparable_field):
    pred, _ = state.apply_fn(
        {"params": state.params, "batch_stats": state.batch_stats},
        input,
        train=False,
        to_mask=to_mask,
        mutable=["batch_stats"],
    )
    loss = ESRLoss(pred[:, -comparable_field:, :], target[:, -comparable_field:, :])
    return loss


@jax.jit
def add_losses_to_metrics(
    state,
    loss
    # , long_loss
):
    metric_updates = state.metrics.single_from_model_output(
        loss=loss  # , long_loss=long_loss
    )
    metrics = state.metrics.merge(metric_updates)
    state = state.replace(metrics=metrics)
    return state


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
    _config["batch_size"] = 2**8
    _config["validation_split"] = 0.2
    _config["learning_rate"] = 1e-4
    _config["epochs"] = 2**7
    _config["window"] = 2**11
    _config["inference_window"] = 2**17
    _config["stride"] = 2**8
    _config["step_freq"] = 100
    _config["test_size"] = 0.1
    _config["channels"] = 2**6
    _config["depth"] = 2**4
    _config["to_mask"] = 2**5
    _config["comparable_field"] = _config["to_mask"] // 2
    _config["kernel_size"] = 7
    _config["skip_freq"] = 1
    _config["norm_factor"] = math.sqrt(_config["channels"])
    _config["inner_skip"] = True
    _config["shift"] = 2**4
    _config["dilation"] = 2**0
    _config["mesh_x"] = 2
    _config["mesh_y"] = device_len // _config["mesh_x"]
    run.log_parameters(_config)
    config = SimpleNamespace(**_config)

    # messshhh
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
    )
    proto_test_dataset, test_dataset_total = make_2d_data(
        paths=test_files,
        window=config.window,
        stride=config.stride,  # , shift=config.shift, dilation=config.dilation, channels=config.channels, feature_dim=-1, shuffle=True
    )
    proto_inference_dataset, inference_dataset_total = make_2d_data(
        paths=test_files,
        window=config.inference_window,
        stride=config.stride,  # , shift=config.shift, dilation=config.dilation, channels=config.channels, feature_dim=-1, shuffle=True
    )
    print("datasets generated")
    init_rng = jax.random.PRNGKey(config.seed)
    onez = jnp.ones([config.batch_size, config.window * 2, 1])

    module = ConvFauxLarsen(
        channels=config.channels,
        depth=config.depth,
        kernel_size=config.kernel_size,
        skip_freq=config.skip_freq,
        norm_factor=config.norm_factor,
        inner_skip=config.inner_skip,
    )
    tx = optax.adam(config.learning_rate)

    abstract_variables = jax.eval_shape(
        partial(create_train_state, module=module, tx=tx),
        init_rng,
        onez,
    )

    state_sharding = nn.get_sharding(abstract_variables, mesh)

    jit_create_train_state = jax.jit(
        create_train_state,
        static_argnums=(2, 3),
        in_shardings=(mesh_sharding(None), x_sharding),  # PRNG key and x
        out_shardings=state_sharding,
    )
    state = jit_create_train_state(init_rng, onez, module, tx)

    jit_train_step = partial(
        jax.jit,
        static_argnums=(3, 4),
        in_shardings=(state_sharding, x_sharding, x_sharding),
        out_shardings=(state_sharding, None),
    )(train_step)

    jit_compute_loss = partial(
        jax.jit,
        static_argnums=(3, 4),
        in_shardings=(state_sharding, x_sharding, x_sharding),
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

        init_rng = jax.random.PRNGKey(config.seed)
        onez = jnp.ones([config.batch_size, config.window * 2, 1])

        module = ConvFauxLarsen(
            channels=config.channels,
            depth=config.depth,
            kernel_size=config.kernel_size,
            skip_freq=config.skip_freq,
            norm_factor=config.norm_factor,
            inner_skip=config.inner_skip,
        )

        abstract_variables = jax.eval_shape(
            partial(create_train_state, module=module, tx=tx),
            init_rng,
            onez,
        )
        old_state_sharding = state_sharding
        state_sharding = nn.get_sharding(abstract_variables, mesh)

        jit_update_train_state = jax.jit(
            update_train_state,
            static_argnums=(1,),
            in_shardings=(old_state_sharding,),
            out_shardings=state_sharding,
        )
        state = jit_update_train_state(state, module)

        jit_train_step = partial(
            jax.jit,
            static_argnums=(3, 4),
            in_shardings=(state_sharding, x_sharding, x_sharding),
            out_shardings=(state_sharding, None),
        )(train_step)

        jit_compute_loss = partial(
            jax.jit,
            static_argnums=(3, 4),
            in_shardings=(state_sharding, x_sharding, x_sharding),
        )(compute_loss)
        del init_rng
        # end uggggh

        # log the epoch
        run.log_metrics({"epoch": epoch})
        train_dataset.set_epoch(epoch)
        # train
        for batch_ix, batch in tqdm(
            enumerate(
                train_dataset.iter(batch_size=config.batch_size, drop_last_batch=True)
            ),
            total=train_dataset_total // config.batch_size if not epoch_is_0 else 2,
        ):
            input = batch["input"]
            input = jax.device_put(input, x_sharding)
            target = batch["target"]
            with mesh:
                state, loss = jit_train_step(
                    state, input, target, to_mask, comparable_field
                )
                state = add_losses_to_metrics(
                    state=state,
                    loss=loss
                    #     , long_loss=0.0
                )

            if batch_ix % config.step_freq == 0:
                metrics = state.metrics.compute()
                run.log_metrics({"train_loss": metrics["loss"]}, step=batch_ix)
                state = replace_metrics(state)
        test_dataset.set_epoch(epoch)
        for batch_ix, batch in tqdm(
            enumerate(
                test_dataset.iter(batch_size=config.batch_size, drop_last_batch=True)
            ),
            total=test_dataset_total // config.batch_size if not epoch_is_0 else 2,
        ):
            input = batch["input"]
            input = jax.device_put(input, x_sharding)
            target = batch["target"]
            loss = jit_compute_loss(state, input, target, to_mask, comparable_field)
            # full_length = input.shape[1]
            # long_loss = jit_compute_loss(
            #     state,
            #     jnp.pad(input, ((0, 0), (module.get_zlen(), 0), (0, 0))),
            #     target,
            #     full_length * 7 // 8,
            #     full_length * 3 // 4,
            # )
            state = add_losses_to_metrics(
                state=state,
                loss=loss
                # , long_loss=long_loss
            )
        metrics = state.metrics.compute()
        run.log_metrics(
            {
                "val_loss": metrics["loss"]
                # , "val_long_loss": metrics["long_loss"]
            },
            step=batch_ix,
        )
        state = replace_metrics(state)

        if not epoch_is_0:
            to_mask += config.to_mask
            comparable_field = config.to_mask // 2

        # checkpoint
        ckpt_model = state
        ckpt = {"model": ckpt_model, "config": config}
        checkpoint_manager.save(epoch, ckpt)
        artifact = Artifact("checkpoint", type="model")
        artifact.add(os.path.join(checkpoint_dir, f"{epoch}"))
        run.log_artifact(artifact)
        # inference
        artifact = Artifact("inference", type="audio")
        inference_dataset.set_epoch(epoch)
        for batch_ix, batch in tqdm(
            enumerate(
                inference_dataset.take(
                    config.inference_artifacts_per_batch_per_epoch
                ).iter(batch_size=config.batch_size, drop_last_batch=True)
            ),
            total=config.inference_artifacts_per_batch_per_epoch
            if not epoch_is_0
            else 2,
        ):
            input = batch["input"]
            input = jax.device_put(input, x_sharding)
            o, _ = pred, updates = state.apply_fn(
                {"params": ckpt_model.params, "batch_stats": state.batch_stats},
                input,
                train=False,
                to_mask=config.to_mask,
                mutable=["batch_stats"],
            )
            for i in range(len(o)):
                audy = np.squeeze(np.array(o[i]))
                apath = f"/tmp/audio_{batch_ix}_{i}.wav"
                soundfile.write(apath, audy, 44100)
                artifact.add(apath)
            # full_length = input.shape[1]
            # o, _ = pred, updates = state.apply_fn(
            #     {"params": ckpt_model.params, "batch_stats": state.batch_stats},
            #     jnp.pad(input, ((0, 0), (module.get_zlen(), 0), (0, 0))),
            #     train=False,
            #     to_mask=full_length * 7 // 8,
            #     mutable=["batch_stats"],
            # )
            # for i in range(len(o)):
            #     audy = np.squeeze(np.array(o[i]))
            #     audio = wandb.Audio(audy, sample_rate=44100)
            #     artifact.add(audio, f"audio_with_long_mask_{batch_ix}_{i}")
        run.log_artifact(artifact)
