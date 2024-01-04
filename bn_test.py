import flax.linen as nn
import jax.numpy as jnp
import jax


class MLP(nn.Module):
    @nn.compact
    def __call__(self, x, train: bool):
        x = nn.Dense(features=4)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
        x = nn.Dense(features=1)(x)
        return x


class Foo(nn.Module):
    @nn.compact
    def __call__(self, x, train: bool):
        x = nn.scan(MLP)(x, train=train)
        return x


mlp = MLP()
x = jnp.ones((1, 100, 3))
variables = mlp.init(jax.random.key(0), x, train=False)
params = variables["params"]
batch_stats = variables["batch_stats"]

y, updates = mlp.apply(
    {"params": params, "batch_stats": batch_stats},
    x,
    train=True,
    mutable=["batch_stats"],
)
batch_stats = updates["batch_stats"]
