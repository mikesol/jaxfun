from flax import struct
import wandb
from clu import metrics
from functools import partial
import jax.numpy as jnp
import flax.jax_utils as jax_utils
import flax.linen as nn
from flax.training import train_state
import optax
import os
from model import LSTM, LSTMCell, SimpleLSTMCombinator
import jax
from data import make_data
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
    module = LSTM(
        features=config.n_features,
        levels=config.n_levels,
        skip=True,
        projection=1,
        name="lstm",
        cell=partial(LSTMCell, combinator=SimpleLSTMCombinator),
    )
    params = module.init(rng, jnp.ones([1, config.window, 1]))["params"]
    tx = optax.adam(learning_rate)
    return TrainState.create(
        apply_fn=module.apply, params=params, tx=tx, metrics=Metrics.empty()
    )


@jax.pmap
def train_step(state, input, target):
    """Train for a single step."""

    def loss_fn(params):
        pred = state.apply_fn({"params": params}, input)
        print("PRED_SHAPE IN LOSS_FN", pred.shape)
        loss = optax.l2_loss(predictions=pred, targets=target).mean()
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
    loss = optax.l2_loss(pred, target).mean()
    print("LOSS_SHAPE IN COMPUTE_LOSS", loss.shape)
    return loss


@jax.pmap
def compute_metrics(state, loss):
    print("LOSS_SHAPE IN COMPUTE_METRICS", loss)
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
    config.batch_size = 2**3
    config.validation_split = 0.2
    config.learning_rate = 1e-4
    config.epochs = 15
    config.window = 2**13
    config.stride = 2**8
    config.step_freq = 100
    config.test_size = 0.1
    config.n_features = 2**4
    config.n_levels = 2**4
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

    state = create_train_state(
        jax.random.split(init_rng, jax.device_count()), config, config.learning_rate
    )
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
            print("STARTING BATCH")
            input = jax_utils.replicate(batch["input"])
            target = jax_utils.replicate(batch["target"])
            print("DIMS", input.shape, target.shape)
            loss, grads = train_step(state, input, target)
            state = update_model(state, grads)
            print("BEFORE_UNREPLICATED", loss.shape, jax_utils.unreplicate(loss))
            state = compute_metrics(state=state, loss=loss)

            if batch_ix % config.step_freq == 0:
                print("BATCH_INDEX FOUND")
                metrics = jax_utils.unreplicate(state.metrics).compute()
                print("GOT METRICS")
                wandb.log({"train_loss": metrics["loss"]})
                state = replace_metrics(state)
                print("REPLACING")
        for batch_ix, batch in tqdm(
            enumerate(test_dataset.iter(batch_size=config.batch_size)),
            total=test_dataset_total // config.batch_size,
        ):
            print("VAL")
            input = jax_utils.replicate(batch["input"])
            target = jax_utils.replicate(batch["target"])
            loss = compute_loss(state, input, target)
            state = compute_metrics(state=state, loss=loss)

        metrics = jax_utils.unreplicate(state.metrics).compute()
        wandb.log({"val_loss": metrics["loss"]})
        state = replace_metrics(state)
        # checkpoint at beginning as sanity check of checkpointing
        ckpt = {"model": state, "config": config}
        checkpoint_manager.save(epoch, ckpt)
        artifact = wandb.Artifact("checkpoint", type="model")
        artifact.add_dir(os.path.join(checkpoint_dir, f"{epoch}"))
        run.log_artifact(artifact)
