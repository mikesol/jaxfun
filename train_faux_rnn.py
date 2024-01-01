from flax import struct
import wandb
from cnn_attn import ConvAttnFauxLarsen
from clu import metrics
from functools import partial
import jax.numpy as jnp
import math
import flax.jax_utils as jax_utils
import flax.linen as nn
from flax.training import train_state
import optax
import os
from rnn import LSTM, LSTMCell, SimpleLSTMCombinator, Transformeresque, StackedRNNCell
import jax
from data import make_2d_data
import orbax.checkpoint
from tqdm import tqdm

jax.distributed.initialize()

checkpoint_dir = "/tmp/flax_ckpt/orbax/managed"

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


@partial(jax.pmap, static_broadcasted_argnums=(1, 2))
def create_train_state(
    rng: PRNGKey, config: wandb.Config, learning_rate: float
) -> TrainState:
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
    params = module.init(rng, jnp.ones([1, config.window, 1]))["params"]
    tx = optax.adam(learning_rate)
    return TrainState.create(
        apply_fn=module.apply, params=params, tx=tx, metrics=Metrics.empty()
    )


@partial(
    jax.pmap,
    static_broadcasted_argnums=(
        1,
        2,
    ),
)
def regenerate_model(
    state: TrainState, config: wandb.Config, to_mask: int
) -> TrainState:
    module = ConvAttnFauxLarsen(
        to_mask=to_mask,
        channels=config.channels,
        depth=config.depth,
        kernel_size=config.kernel_size,
        skip_freq=config.skip_freq,
        norm_factor=config.norm_factor,
        layernorm=config.layernorm,
        inner_skip=config.inner_skip,
    )
    return state.replace(apply_fn=module.apply)


@jax.pmap
def train_step(state, input, target):
    """Train for a single step."""

    def loss_fn(params):
        pred = state.apply_fn({"params": params}, input)
        loss = optax.l2_loss(pred, target[:,-pred.shape[1]:,:]).mean()
        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    return loss, grads


@jax.pmap
def update_model(state, grads):
    return state.apply_gradients(grads=grads)


@jax.pmap
def replace_metrics(state):
    return state.replace(metrics=state.metrics.empty())


@jax.pmap
def compute_loss(state, input, target):
    pred = state.apply_fn({"params": state.params}, input)
    loss = optax.l2_loss(pred, target[:,-pred.shape[1]:,:]).mean()
    return loss


@jax.pmap
def compute_metrics(state, loss):
    metric_updates = state.metrics.single_from_model_output(loss=loss)
    metrics = state.metrics.merge(metric_updates)
    state = state.replace(metrics=metrics)
    return state


if __name__ == "__main__":
    from get_files import FILES

    # FILES = FILES[:3]
    run = wandb.init(
        project="jax-cnn-faux-rnn",
    )
    config = wandb.config
    # cnn
    config.seed = 42
    config.batch_size = 2**7
    config.validation_split = 0.2
    config.learning_rate = 1e-4
    config.epochs = 2**8
    config.window = 2**12
    config.stride = 2**8
    config.step_freq = 100
    config.test_size = 0.1
    config.channels = 2**6
    config.depth = 2**3
    config.to_mask = 2**4
    config.kernel_size = 7
    config.skip_freq = 1
    config.norm_factor = math.sqrt(config.channels)
    config.layernorm = True
    config.inner_skip = True
    len_files = len(FILES)
    test_files = FILES[: int(len_files * config.test_size)]
    train_files = FILES[int(len_files * config.test_size) :]
    train_dataset, train_dataset_total = make_2d_data(
        train_files, config.window, config.stride
    )
    test_dataset, test_dataset_total = make_2d_data(
        test_files, config.window, config.stride
    )
    init_rng = jax.random.PRNGKey(config.seed)

    state = create_train_state(
        jax.random.split(init_rng, jax.device_count()), config, config.learning_rate
    )
    del init_rng  # Must not be used anymore.
    to_mask = config.to_mask
    for epoch in range(config.epochs):
        state = regenerate_model(state, config, to_mask)
        to_mask += config.to_mask
        # checkpoint at beginning as sanity check of checkpointing
        ckpt = {"model": jax_utils.unreplicate(state), "config": config}
        checkpoint_manager.save(epoch, ckpt)
        artifact = wandb.Artifact("checkpoint", type="model")
        artifact.add_dir(os.path.join(checkpoint_dir, f"{epoch}"))
        run.log_artifact(artifact)
        # log the epoch
        wandb.log({"epoch": epoch})
        train_dataset.set_epoch(epoch)
        # train
        for batch_ix, batch in tqdm(
            enumerate(train_dataset.iter(batch_size=config.batch_size)),
            total=train_dataset_total // config.batch_size,
        ):
            input = jax_utils.replicate(batch["input"])
            target = jax_utils.replicate(batch["target"])
            loss, grads = train_step(state, input, target)
            state = update_model(state, grads)
            state = compute_metrics(state=state, loss=loss)

            if batch_ix % config.step_freq == 0:
                metrics = jax_utils.unreplicate(state.metrics).compute()
                wandb.log({"train_loss": metrics["loss"]})
                state = replace_metrics(state)
        for batch_ix, batch in tqdm(
            enumerate(test_dataset.iter(batch_size=config.batch_size)),
            total=test_dataset_total // config.batch_size,
        ):
            input = jax_utils.replicate(batch["input"])
            target = jax_utils.replicate(batch["target"])
            loss = compute_loss(state, input, target)
            state = compute_metrics(state=state, loss=loss)

        metrics = jax_utils.unreplicate(state.metrics).compute()
        wandb.log({"val_loss": metrics["loss"]})
        state = replace_metrics(state)
