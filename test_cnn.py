from cnn_attn import ConvAttnFauxLarsen
from cnn import ConvFauxLarsen
import jax
import flax.linen as nn
import jax.numpy as jnp
import pytest


def c1d(i, p, d, k, s):
    return ((i + (2 * p) - d * (k - 1) - 1) / s) + 1


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
        channels=channels,
        depth=depth,
        kernel_size=kernel_size,
        skip_freq=skip_freq,
        norm_factor=norm_factor,
        inner_skip=inner_skip,
    )
    i = jnp.ones((batch_size, window * 2, 1))
    rng = jax.random.PRNGKey(0)
    variables = model.init(rng, i, train=False, to_mask=to_mask)
    params = variables["params"]
    batch_stats = variables["batch_stats"]
    o, updates = model.apply(
        {"params": params, "batch_stats": batch_stats},
        i,
        train=True,
        to_mask=to_mask,
        mutable=["batch_stats"],
    )
    batch_stats = updates["batch_stats"]
    l = i.shape[1]
    l = c1d(l, 0, 1, kernel_size * 2, 2)
    for _ in range(depth - 1):
        l = c1d(l, 0, 1, kernel_size, 1)
    assert o.shape == (batch_size, int(l), 1)
    o, updates = model.apply(
        {"params": params, "batch_stats": batch_stats},
        i,
        to_mask=to_mask,
        train=False,
        mutable=["batch_stats"],
    )
    assert o.shape == (batch_size, int(l), 1)


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
    params = model.init(rng, i)
    o = model.apply(params, i)
    l = i.shape[1]
    l = c1d(l, 0, 1, kernel_size * 2, 2)
    for _ in range(depth - 1):
        l = c1d(l, 0, 1, kernel_size, 1)
    assert o.shape == (batch_size, int(l), 1)
