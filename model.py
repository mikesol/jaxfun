from flax import linen as nn
from typing import (
    Any,
    Callable,
    Optional,
    Tuple,
    TypeVar,
)
from functools import partial
import jax
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


class SimpleLSTMCombinator(nn.Module):
    kernel_init: nn.initializers.Initializer = nn.linear.default_kernel_init
    recurrent_kernel_init: nn.initializers.Initializer = nn.initializers.orthogonal()
    bias_init: nn.initializers.Initializer = nn.initializers.zeros_init()
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, i, h):
        hidden_features = h.shape[-1]
        return nn.Dense(
            features=hidden_features,
            use_bias=False,
            kernel_init=self.kernel_init,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="i",
        )(i) + nn.Dense(
            features=hidden_features,
            use_bias=True,
            kernel_init=self.recurrent_kernel_init,
            bias_init=self.bias_init,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="h",
        )(
            h
        )


class ComplexLSTMCombinator(nn.Module):
    kernel_init: nn.initializers.Initializer = nn.linear.default_kernel_init
    recurrent_kernel_init: nn.initializers.Initializer = nn.initializers.orthogonal()
    bias_init: nn.initializers.Initializer = nn.initializers.zeros_init()
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, i, h):
        hidden_features = h.shape[-1]
        q = nn.Dense(
            features=hidden_features,
            use_bias=True,
            kernel_init=self.kernel_init,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="q",
        )(i)
        k = nn.Dense(
            features=hidden_features,
            use_bias=True,
            kernel_init=self.recurrent_kernel_init,
            bias_init=self.bias_init,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="k",
        )(h)
        v = nn.Dense(
            features=hidden_features,
            use_bias=True,
            kernel_init=self.recurrent_kernel_init,
            bias_init=self.bias_init,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="v",
        )(h)
        q = jnp.expand_dims(q, axis=-1)
        k = jnp.expand_dims(k, axis=-2)
        v = jnp.expand_dims(v, axis=-1)
        mm = nn.tanh(jnp.matmul(q, k) / hidden_features)
        return jnp.squeeze(jnp.matmul(mm, v), axis=-1)


class LSTMCell(nn.Module):
    features: int
    skip: bool = False
    combinator: Callable[..., Any] = None
    gate_fn: Callable[..., Any] = nn.sigmoid
    activation_fn: Callable[..., Any] = nn.tanh
    kernel_init: nn.initializers.Initializer = nn.linear.default_kernel_init
    recurrent_kernel_init: nn.initializers.Initializer = nn.initializers.orthogonal()
    bias_init: nn.initializers.Initializer = nn.initializers.zeros_init()
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32
    carry_init: nn.initializers.Initializer = nn.initializers.zeros_init()

    def setup(self):
        if self.combinator == None:
            self.combinator = partial(
                SimpleLSTMCombinator,
                kernel_init=self.kernel_init,
                recurrent_kernel_init=self.recurrent_kernel_init,
                bias_init=self.bias_init,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
            )

        self.i = self.combinator(name="i")
        self.f = self.combinator(name="f")
        self.g = self.combinator(name="g")
        self.o = self.combinator(name="o")

    def __call__(self, carry, inputs):
        c, h = carry
        # input and recurrent layers are summed so only one needs a bias.
        i = self.gate_fn(self.i(inputs, h))
        f = self.gate_fn(self.f(inputs, h))
        g = self.activation_fn(self.g(inputs, h))
        o = self.gate_fn(self.o(inputs, h))
        new_c = f * c + i * g
        new_h = o * self.activation_fn(new_c)
        return (new_c, new_h), new_h if not self.skip else new_h + inputs

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


def remove_last_entry_of_second_last_dim(tensor):
    num_dims = tensor.ndim
    slices = [slice(None)] * num_dims
    slices[-2] = slice(None, -1)
    return tensor[tuple(slices)]


class StackedLSTMCell(nn.Module):
    features: int
    skip: bool = False
    combinator: Callable[..., Any] = None
    gate_fn: Callable[..., Any] = nn.sigmoid
    activation_fn: Callable[..., Any] = nn.tanh
    kernel_init: nn.initializers.Initializer = nn.linear.default_kernel_init
    recurrent_kernel_init: nn.initializers.Initializer = nn.initializers.orthogonal()
    bias_init: nn.initializers.Initializer = nn.initializers.zeros_init()
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32
    carry_init: nn.initializers.Initializer = nn.initializers.zeros_init()
    levels: int = 1
    projection: Optional[int] = None

    @nn.compact
    def __call__(self, carry, inputs):
        c, h = carry
        inputs = jnp.expand_dims(inputs, axis=-2)
        # make the inputs match the size of the hidden state
        inputs = nn.Dense(
            features=self.features,
            use_bias=True,
            kernel_init=self.recurrent_kernel_init,
            bias_init=self.bias_init,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )(inputs)
        inputs = jax.lax.concatenate(
            [inputs, remove_last_entry_of_second_last_dim(h)],
            dimension=inputs.ndim - 2,
        )

        carry, out = nn.vmap(
            LSTMCell,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            in_axes=inputs.ndim - 2,
            out_axes=inputs.ndim - 2,
        )(
            features=self.features,
            skip=self.skip,
            combinator=self.combinator,
            gate_fn=self.gate_fn,
            activation_fn=self.activation_fn,
            kernel_init=self.kernel_init,
            recurrent_kernel_init=self.recurrent_kernel_init,
            bias_init=self.bias_init,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            carry_init=self.carry_init,
        )(
            (c, h), inputs
        )

        if self.projection is not None:
            out = nn.Dense(
                features=self.projection,
                use_bias=True,
                kernel_init=self.recurrent_kernel_init,
                bias_init=self.bias_init,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
            )(out[..., -1, :])

        return carry, out

    @nn.nowrap
    def initialize_carry(
        self, rng: PRNGKey, input_shape: Tuple[int, ...]
    ) -> Tuple[Array, Array]:
        batch_dims = input_shape[:-1]
        key1, key2 = random.split(rng)
        levels = self.levels
        mem_shape = batch_dims + (
            levels,
            self.features,
        )
        c = self.carry_init(key1, mem_shape, self.param_dtype)
        h = self.carry_init(key2, mem_shape, self.param_dtype)
        return (c, h)

    @property
    def num_feature_axes(self) -> int:
        return 1


class LSTM(nn.Module):
    features: int
    skip: bool = False
    combinator: Callable[..., Any] = None
    gate_fn: Callable[..., Any] = nn.sigmoid
    activation_fn: Callable[..., Any] = nn.tanh
    kernel_init: nn.initializers.Initializer = nn.linear.default_kernel_init
    recurrent_kernel_init: nn.initializers.Initializer = nn.initializers.orthogonal()
    bias_init: nn.initializers.Initializer = nn.initializers.zeros_init()
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32
    carry_init: nn.initializers.Initializer = nn.initializers.zeros_init()
    projection: Optional[int] = None
    levels: int = 1

    @nn.compact
    def __call__(self, x):
        x = nn.RNN(
            StackedLSTMCell(
                features=self.features,
                levels=self.levels,
                skip=self.skip,
                projection=self.projection,
                gate_fn=self.gate_fn,
                combinator=self.combinator,
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


if __name__ == "__main__":
    model = LSTM(features=2**7, levels=2**5, skip=True, projection=1, combinator=ComplexLSTMCombinator, name="lstm")
    print(model.tabulate(jax.random.key(0), jnp.ones((2**2, 2**8, 1))))
