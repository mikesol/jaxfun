from cnn import ConvFauxLarsen
from cnn_att import ConvAttnFauxLarsen
import jax
import flax.linen as nn
import jax.numpy as jnp
import pytest


def c1d(i, p, d, k, s):
    return ((i + (2 * p) - d * (k - 1) - 1) / s) + 1


def test_cnn_faux_larsen_with_variable_depth():
    batch_size = 2**2
    window = 2**9
    depth = [2**1, 2**2, 2**3, 2**4, 2**5, 2**6, 2**7, 2**8, 2**8, 2**7, 2**6, 2**5, 2**4, 2**3, 2**2, 2**1]
    channels = None
    kernel_size = 7
    skip_freq = 1
    inner_skip = True
    to_mask = window // 2
    model = ConvFauxLarsen(
        to_mask=to_mask,
        channels=channels,
        depth=depth,
        kernel_size=kernel_size,
        skip_freq=skip_freq,
        inner_skip=inner_skip,
    )
    i = jnp.ones((batch_size, window * 2, 1))
    rng = jax.random.PRNGKey(0)
    variables = model.init(rng, i, train=False)
    params = variables["params"]
    batch_stats = variables["batch_stats"]
    o, updates = model.apply(
        {"params": params, "batch_stats": batch_stats},
        i,
        train=True,
        mutable=["batch_stats"],
    )
    batch_stats = updates["batch_stats"]
    l = i.shape[1]
    l = c1d(l, 0, 1, kernel_size * 2, 2)
    for _ in range(len(depth) - 1):
        l = c1d(l, 0, 1, kernel_size, 1)
    assert o.shape == (batch_size, int(l), 1)

def test_cnn_faux_larsen():
    batch_size = 2**2
    window = 2**9
    depth = 2**4
    channels = 2**6
    kernel_size = 7
    norm_factor = 1.0
    skip_freq = 1
    inner_skip = True
    to_mask = window // 2
    model = ConvFauxLarsen(
        to_mask=to_mask,
        channels=channels,
        depth=depth,
        kernel_size=kernel_size,
        skip_freq=skip_freq,
        inner_skip=inner_skip,
    )
    i = jnp.ones((batch_size, window * 2, 1))
    rng = jax.random.PRNGKey(0)
    variables = model.init(rng, i, train=False)
    params = variables["params"]
    batch_stats = variables["batch_stats"]
    o, updates = model.apply(
        {"params": params, "batch_stats": batch_stats},
        i,
        train=True,
        mutable=["batch_stats"],
    )
    batch_stats = updates["batch_stats"]
    l = i.shape[1]
    l = c1d(l, 0, 1, kernel_size * 2, 2)
    for _ in range(depth - 1):
        l = c1d(l, 0, 1, kernel_size, 1)
    assert o.shape == (batch_size, int(l), 1)


@pytest.mark.skip(reason="stopped maintaining this so it is buggy, should be removed soon")
def test_cnn_attn_faux_larsen():
    batch_size = 2**2
    window = 2**9
    depth = 2**4
    channels = 2**6
    kernel_size = 7
    norm_factor = 1.0
    skip_freq = 1
    inner_skip = True
    to_mask = window // 2
    model = ConvAttnFauxLarsen(
        to_mask=to_mask,
        channels=channels,
        depth=depth,
        kernel_size=kernel_size,
        skip_freq=skip_freq,
        norm_factor=norm_factor,
        inner_skip=inner_skip,
    )
    i = jnp.ones((batch_size, window * 2, 1))
    rng = jax.random.PRNGKey(0)
    variables = model.init(rng, i, train=False)
    params = variables["params"]
    batch_stats = variables["batch_stats"]
    o, updates = model.apply(
        {"params": params, "batch_stats": batch_stats},
        i,
        train=True,
        mutable=["batch_stats"],
    )
    batch_stats = updates["batch_stats"]
    l = i.shape[1]
    l = c1d(l, 0, 1, kernel_size * 2, 2)
    for _ in range(depth - 1):
        l = c1d(l, 0, 1, kernel_size, 1)
    assert o.shape == (batch_size, int(l), 1)
