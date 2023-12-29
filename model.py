from flax import linen as nn
from typing import (
    Any,
    Callable,
    Dict,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)
from functools import partial
import jax
import numpy as np
from absl import logging
from jax import numpy as jnp
from jax import random
from typing_extensions import Protocol

A = TypeVar("A")
PRNGKey = jax.Array
Shape = Tuple[int, ...]
Dtype = Any  # this could be a real type?
Array = jax.Array
Carry = Any
CarryHistory = Any
Output = Any


class LSTMCell(nn.Module):
    features: int
    gate_fn: Callable[..., Any] = nn.sigmoid
    activation_fn: Callable[..., Any] = nn.tanh
    kernel_init: nn.initializers.Initializer = nn.linear.default_kernel_init
    recurrent_kernel_init: nn.initializers.Initializer = nn.initializers.orthogonal()
    bias_init: nn.initializers.Initializer = nn.initializers.zeros_init()
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32
    carry_init: nn.initializers.Initializer = nn.initializers.zeros_init()

    @nn.compact
    def __call__(self, carry, inputs):
        c, h = carry
        hidden_features = h.shape[-1]
        # input and recurrent layers are summed so only one needs a bias.
        dense_h = partial(
            nn.Dense,
            features=hidden_features,
            use_bias=True,
            kernel_init=self.recurrent_kernel_init,
            bias_init=self.bias_init,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        dense_i = partial(
            nn.Dense,
            features=hidden_features,
            use_bias=False,
            kernel_init=self.kernel_init,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        i = self.gate_fn(dense_i(name="ii")(inputs) + dense_h(name="hi")(h))
        f = self.gate_fn(dense_i(name="if")(inputs) + dense_h(name="hf")(h))
        g = self.activation_fn(dense_i(name="ig")(inputs) + dense_h(name="hg")(h))
        o = self.gate_fn(dense_i(name="io")(inputs) + dense_h(name="ho")(h))
        new_c = f * c + i * g
        new_h = o * self.activation_fn(new_c)
        return (new_c, new_h), new_h

    @nn.nowrap
    def initialize_carry(
        self, rng: PRNGKey, input_shape: Tuple[int, ...]
    ) -> Tuple[Array, Array]:
        batch_dims = input_shape[:-1]
        key1, key2 = random.split(rng)
        mem_shape = batch_dims + (self.features,)
        c = self.carry_init(key1, mem_shape, self.param_dtype)
        h = self.carry_init(key2, mem_shape, self.param_dtype)
        return (c, h)

    @property
    def num_feature_axes(self) -> int:
        return 1


class LSTM(nn.Module):
    features: int
    gate_fn: Callable[..., Any] = nn.sigmoid
    activation_fn: Callable[..., Any] = nn.tanh
    kernel_init: nn.initializers.Initializer = nn.linear.default_kernel_init
    recurrent_kernel_init: nn.initializers.Initializer = nn.initializers.orthogonal()
    bias_init: nn.initializers.Initializer = nn.initializers.zeros_init()
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32
    carry_init: nn.initializers.Initializer = nn.initializers.zeros_init()

    @nn.compact
    def __call__(self, x):
        x = nn.RNN(
            LSTMCell(
                features=self.features,
                gate_fn=self.gate_fn,
                activation_fn=self.activation_fn,
                kernel_init=self.kernel_init,
                recurrent_kernel_init=self.recurrent_kernel_init,
                bias_init=self.bias_init,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                carry_init=self.carry_init,
            )
        )(x)
        return x


class Network(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = LSTM(features=32, name="first_lstm")(x)
        x = LSTM(features=32, name="second_lstm")(x)
        x = LSTM(features=32, name="third_lstm")(x)
        x = LSTM(features=1, name="fourth_lstm")(x)
        return x
