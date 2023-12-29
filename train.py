from flax import struct
from clu import metrics
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import optax
from model import Network
import jax
from data import make_data

PRNGKey = jax.Array


@struct.dataclass
class Metrics(metrics.Collection):
    loss: metrics.Average.from_output("loss")


class TrainState(train_state.TrainState):
    metrics: Metrics


def create_train_state(
    module: nn.Module, rng: PRNGKey, learning_rate: float
) -> TrainState:
    params = module.init(rng, jnp.ones([1, 2**16, 1]))['params']
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

    window = 2**16
    stride = 2**8
    dataset = make_data(FILES, window, stride)
    N_EPOCHS = 10
    init_rng = jax.random.PRNGKey(42)
    learning_rate = 0.01
    lstm = Network()
    state = create_train_state(lstm, init_rng, learning_rate)
    del init_rng  # Must not be used anymore.
    batch_n = 0
    for epoch in range(N_EPOCHS):
        for batch in dataset.iter(batch_size=4):
            # Run optimization steps over training batches and compute batch metrics
            state = train_step(
                state, batch
            )
            state = compute_metrics(state=state, batch=batch)

            for metric, value in state.metrics.compute().items():
                print("METRICS", metric, value)
            state = state.replace(
                metrics=state.metrics.empty()
            )
