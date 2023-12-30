from flax import struct
import wandb
from clu import metrics
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import optax
import os
from model import Network
import jax
from data import make_data
import orbax.checkpoint
from tqdm import tqdm

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


def create_train_state(
    module: nn.Module, rng: PRNGKey, learning_rate: float
) -> TrainState:
    params = module.init(rng, jnp.ones([1, 2**16, 1]))["params"]
    tx = optax.adam(learning_rate)
    return TrainState.create(
        apply_fn=module.apply, params=params, tx=tx, metrics=Metrics.empty()
    )


@jax.jit
def train_step(state, batch):
    """Train for a single step."""

    def loss_fn(params):
        pred = state.apply_fn({"params": params}, batch["input"])
        loss = optax.l2_loss(predictions=pred, targets=batch["target"]).mean()
        return loss

    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state


@jax.jit
def compute_metrics(*, state: TrainState, batch):
    pred = state.apply_fn({"params": state.params}, batch["input"])
    loss = optax.l2_loss(predictions=pred, targets=batch["target"]).mean()
    metric_updates = state.metrics.single_from_model_output(loss=loss)
    metrics = state.metrics.merge(metric_updates)
    state = state.replace(metrics=metrics)
    return state


if __name__ == "__main__":
    from get_files import FILES

    # FILES = FILES[:3]
    run = wandb.init(
        project="simple-jax-lstm",
    )
    config = wandb.config
    config.seed = 42
    config.batch_size = 2**4
    config.validation_split = 0.2
    config.learning_rate = 1e-4
    config.epochs = 15
    config.window = 2**14
    config.stride = 2**8
    config.step_freq = 100
    config.test_size = 0.1
    len_files = len(FILES)
    test_files = FILES[: int(len_files * config.test_size)]
    train_files = FILES[int(len_files * config.test_size) :]
    train_dataset, train_dataset_total = make_data(
        train_files, config.window, config.stride
    )
    test_dataset, test_dataset_total = make_data(
        test_files, config.window, config.stride
    )
    init_rng = jax.random.PRNGKey(config.seed)
    lstm = Network()
    state = create_train_state(lstm, init_rng, config.learning_rate)
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
            state = train_step(state, batch)
            state = compute_metrics(state=state, batch=batch)

            if batch_ix % config.step_freq == 0:
                metrics = state.metrics.compute()
                wandb.log({"train_loss": metrics["loss"]})
                state = state.replace(metrics=state.metrics.empty())
        for batch_ix, batch in tqdm(
            enumerate(test_dataset.iter(batch_size=config.batch_size)),
            total=test_dataset_total // config.batch_size,
        ):
            state = compute_metrics(state=state, batch=batch)

        metrics = state.metrics.compute()
        wandb.log({"val_loss": metrics["loss"]})
        state = state.replace(metrics=state.metrics.empty())
        # checkpoint at beginning as sanity check of checkpointing
        ckpt = {"model": state, "config": config}
        checkpoint_manager.save(epoch, ckpt)
        artifact = wandb.Artifact("checkpoint", type="model")
        artifact.add_dir(os.path.join(checkpoint_dir, f"{epoch}"))
        run.log_artifact(artifact)
