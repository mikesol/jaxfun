import os
import GPUtil

IS_CPU = len(GPUtil.getAvailable()) == 0
if IS_CPU:
    print("in cpu land")
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

from flax import struct
import wandb
from cnn_attn import ConvAttnFauxLarsen
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


class TrainState(train_state.TrainState):
    metrics: Metrics


def create_train_state(rng: PRNGKey, x, module, tx) -> TrainState:
    # window is 2x'd because input is interleaved
    params = module.init(rng, x)["params"]
    return TrainState.create(
        apply_fn=module.apply, params=params, tx=tx, metrics=Metrics.empty()
    )


def train_step(state, input, target, comparable_field):
    """Train for a single step."""

    def loss_fn(params):
        pred = state.apply_fn({"params": params}, input)
        loss = ESRLoss(pred[:, -comparable_field:, :], target[:, -comparable_field:, :])
        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss


@jax.jit
def replace_metrics(state):
    return state.replace(metrics=state.metrics.empty())


def compute_loss(state, input, target, comparable_field):
    pred = state.apply_fn({"params": state.params}, input)
    loss = ESRLoss(pred[:, -comparable_field:, :], target[:, -comparable_field:, :])
    return loss


@jax.jit
def compute_metrics(state, loss):
    metric_updates = state.metrics.single_from_model_output(loss=loss)
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

    run = wandb.init(
        project="jax-cnn-faux-rnn",
    )
    config = wandb.config
    # cnn
    config.seed = 42
    config.inference_artifacts_per_batch_per_epoch = 2**2
    config.batch_size = 2**7
    config.validation_split = 0.2
    config.learning_rate = 1e-4
    config.epochs = 2**7
    config.window = 2**11
    config.inference_window = 2**11
    config.stride = 2**8
    config.step_freq = 100
    config.test_size = 0.1
    config.channels = 2**5
    config.depth = 2**3
    config.to_mask = config.inference_window * 3 // 4
    config.comparable_field = 2**10
    config.kernel_size = 7
    config.skip_freq = 1
    config.norm_factor = math.sqrt(config.channels)
    config.layernorm = True
    config.inner_skip = True
    config.shift = 2**4
    config.dilation = 2**0
    config.mesh_x = 2
    config.mesh_y = device_len // config.mesh_x
    # messshhh
    device_mesh = mesh_utils.create_device_mesh((config.mesh_x, config.mesh_y))
    mesh = Mesh(devices=device_mesh, axis_names=("data", "model"))
    print(mesh)
    x_sharding = mesh_sharding(
        PartitionSpec("data", None)
    )
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
    train_dataset, train_dataset_total = make_2d_data(
        paths=train_files,
        window=config.window,
        stride=config.stride,  # , shift=config.shift, dilation=config.dilation, channels=config.channels, feature_dim=-1, shuffle=True
    )
    test_dataset, test_dataset_total = make_2d_data(
        paths=test_files,
        window=config.window,
        stride=config.stride,  # , shift=config.shift, dilation=config.dilation, channels=config.channels, feature_dim=-1, shuffle=True
    )
    inference_dataset, inference_dataset_total = make_2d_data(
        paths=test_files[:1],
        window=config.window,
        stride=config.stride,  # , shift=config.shift, dilation=config.dilation, channels=config.channels, feature_dim=-1, shuffle=True
    )
    print("datasets generated")
    init_rng = jax.random.PRNGKey(config.seed)
    onez = jnp.ones([config.batch_size, config.window * 2, 1])

    module = ConvAttnFauxLarsen(
        to_mask=config.to_mask,
        channels=config.channels,
        depth=config.depth,
        kernel_size=config.kernel_size,
        skip_freq=config.skip_freq,
        norm_factor=config.norm_factor,
        layernorm=config.layernorm,
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
        static_argnums=(3,),
        in_shardings=(state_sharding, x_sharding, x_sharding),
        out_shardings=(state_sharding, None),
    )(train_step)

    jit_compute_loss = partial(
        jax.jit,
        static_argnums=(3,),
        in_shardings=(state_sharding, x_sharding, x_sharding),
    )(compute_loss)

    del init_rng  # Must not be used anymore.
    for epoch in range(config.epochs):
        # log the epoch
        wandb.log({"epoch": epoch})
        train_dataset.set_epoch(epoch)
        # train
        for batch_ix, batch in tqdm(
            enumerate(train_dataset.iter(batch_size=config.batch_size)),
            total=train_dataset_total // config.batch_size,
        ):
            input = batch["input"]
            input = jax.device_put(input, x_sharding)
            target = batch["target"]
            with mesh:
                state, loss = jit_train_step(
                    state, input, target, config.comparable_field
                )
                state = compute_metrics(state=state, loss=loss)

            if batch_ix % config.step_freq == 0:
                metrics = state.metrics.compute()
                wandb.log({"train_loss": metrics["loss"]})
                state = replace_metrics(state)
        for batch_ix, batch in tqdm(
            enumerate(test_dataset.iter(batch_size=config.batch_size)),
            total=test_dataset_total // config.batch_size,
        ):
            input = batch["input"]
            input = jax.device_put(input, x_sharding)
            target = batch["target"]
            loss = jit_compute_loss(state, input, target, config.comparable_field)
            state = compute_metrics(state=state, loss=loss)
        # checkpoint
        ckpt_model = state
        ckpt = {"model": ckpt_model, "config": config}
        checkpoint_manager.save(epoch, ckpt)
        artifact = wandb.Artifact("checkpoint", type="model")
        artifact.add_dir(os.path.join(checkpoint_dir, f"{epoch}"))
        run.log_artifact(artifact)
        # inference
        artifact = wandb.Artifact("inference", type="audio")
        for batch_ix, batch in tqdm(
            enumerate(
                inference_dataset.take(
                    config.inference_artifacts_per_batch_per_epoch
                ).iter(batch_size=config.batch_size)
            ),
            total=config.inference_artifacts_per_batch_per_epoch,
        ):
            input = batch["input"]
            input = jax.device_put(input, x_sharding)
            o = ckpt_model.apply_fn({"params": ckpt_model.params}, input)
            # make it 1d
            audio = wandb.Audio(np.squeeze(np.array(o)), sample_rate=44100)
            artifact.add(audio, f"audio_{batch_ix}")
        run.log_artifact(artifact)

        metrics = state.metrics.compute()
        wandb.log({"val_loss": metrics["loss"]})
        state = replace_metrics(state)
