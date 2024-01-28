import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.linen import initializers
import math
from typing import Tuple
from scipy import signal
from tcn import MultiBiquad

from typing import (
    Any,
    Callable,
    Optional,
    Tuple,
    TypeVar,
)
import math
from jax import random
import pytest
from create_filtered_audio import create_biquad_coefficients


def test_create_biquad_coefficients():
    bc = create_biquad_coefficients(127, 44100, 100, 700, 30, 10)
    assert bc.shape == (5, 127)
    b, a = signal.iirpeak(100, 30, 44100)
    assert np.allclose(np.flip(bc[:3, 0]), b)
    assert np.allclose(-1 * bc[3:, 0], a[1:])


def test_biquad():
    bc = create_biquad_coefficients(127, 44100, 100, 700, 30, 10)
    i_ = np.random.randn(4, 1024, 1)
    i = jnp.array(i_)
    b, a = signal.iirpeak(100, 30, 44100)
    x, y = signal.iirpeak(700, 10, 44100)
    filtered00, _ = signal.lfilter(b, a, i_[0, :, 0], zi=np.zeros_like(a[1:]))
    filtered10, _ = signal.lfilter(b, a, i_[1, :, 0], zi=np.zeros_like(a[1:]))
    filtered01, _ = signal.lfilter(x, y, i_[0, :, 0], zi=np.zeros_like(x[1:]))
    print(filtered00.shape)
    print(filtered00[0], b[0], i_[0, 0, 0])
    assert np.allclose(filtered00[0], i_[0, 0, 0] * b[0])
    assert np.allclose(
        filtered00[1], i_[0, 1, 0] * b[0] + i_[0, 0, 0] * b[1] - a[1] * filtered00[0]
    )
    assert np.allclose(
        filtered00[2],
        i_[0, 2, 0] * b[0]
        + i_[0, 1, 0] * b[1]
        + i_[0, 0, 0] * b[2]
        - a[1] * filtered00[1]
        - a[2] * filtered00[0],
    )
    model = MultiBiquad(coefficients=bc)
    o = np.array(model.apply({}, i))
    assert o.shape == (4, 1024, 127)
    for x in range(o.shape[1]):
        assert np.allclose(o[0][x][0], filtered00[x], atol=1.0e-3)
    for x in range(o.shape[1]):
        assert np.allclose(o[1][x][0], filtered10[x], atol=1.0e-3)
    for x in range(o.shape[1]):
        assert np.allclose(o[0][x][-1], filtered01[x], atol=1.0e-3)
