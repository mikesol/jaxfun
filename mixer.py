import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.linen import initializers
import math
from typing import Tuple
from typing import (
    Any,
    Callable,
    Optional,
    Tuple,
    TypeVar,
)
import math
from jax import random

A = TypeVar("A")
PRNGKey = jax.Array
A = TypeVar("A")
PRNGKey = jax.Array
Shape = Tuple[int, ...]
Dtype = Any  # this could be a real type?
Array = jax.Array
Carry = Any
CarryHistory = Any
Output = Any


class BiquadCell(nn.Module):
    param_dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, carry, inputs):
        yx = jnp.concatenate([inputs, carry], axis=-1)
        weights = self.param(
            "weights",
            initializers.lecun_normal(),
            (1, yx.shape[-1]),
            jnp.float32,
        )
        o = jnp.sum(yx * weights, axis=-1, keepdims=True)
        # skip
        o = nn.tanh(o + inputs[..., -1:])
        return jnp.concatenate([o, carry[..., :-1]], axis=-1), o


class BiquadCellWithSidechain(nn.Module):
    param_dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, carry, inputs, sidechain):
        yx = jnp.concatenate([inputs, carry], axis=-1)
        weights = self.param(
            "weights",
            initializers.lecun_normal(),
            (1, yx.shape[-1]),
            jnp.float32,
        )
        o = jnp.sum(yx * (weights + sidechain), axis=-1, keepdims=True)
        o = nn.tanh(o)
        return jnp.concatenate([o, carry[..., :-1]], axis=-1), o


class MultiBiquadCell(nn.Module):
    channels: int
    carry_init: nn.initializers.Initializer = nn.initializers.zeros_init()
    param_dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, carry, inputs):
        return nn.vmap(
            lambda m, c: m(c, inputs),
            variable_axes={"params": 0},
            split_rngs={"params": True},
            in_axes=-1,
            out_axes=-1,
        )(BiquadCell(), carry)

    @nn.nowrap
    def initialize_carry(
        self, rng: PRNGKey, input_shape: Tuple[int, ...]
    ) -> Tuple[Array, Array]:
        batch_dims = input_shape[:-1]
        mem_shape = batch_dims + (2, self.channels)
        c = self.carry_init(rng, mem_shape, self.param_dtype)
        return c

    @property
    def num_feature_axes(self) -> int:
        return 1


class MultiBiquadCellWithSidechain(nn.Module):
    channels: int
    order: int
    carry_init: nn.initializers.Initializer = nn.initializers.zeros_init()
    param_dtype: Dtype = jnp.float32

    @nn.compact
    # inputs is (b, c, o)
    def __call__(self, carry, inputs):
        weights = self.param(
            "weights",
            initializers.lecun_normal(),
            # 5 for biquad filter
            (self.channels, self.channels, self.order * 2 + 1),
            jnp.float32,
        )
        # the sidechain should only see the present timestamp, so we get rid of the kernels
        to_sidechain = inputs[:, :, -1]
        # x = (b, c) weights = (c, c, 5) w' = (b, c, 5) w = (b, 5, c)
        sidechain = jnp.transpose(
            jnp.einsum("ac,dcg->adg", to_sidechain, weights), (0, 2, 1)
        )
        # add activation to w?
        # inputs for vmap should be (b, 3, c)
        inputs_for_vmap = jnp.transpose(inputs, (0, 2, 1))
        return nn.vmap(
            BiquadCellWithSidechain,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            in_axes=-1,
            out_axes=-1,
        )()(carry, inputs_for_vmap, sidechain)

    @nn.nowrap
    def initialize_carry(
        self, rng: PRNGKey, input_shape: Tuple[int, ...]
    ) -> Tuple[Array, Array]:
        batch_dims = input_shape[:1]
        mem_shape = batch_dims + (self.order, self.channels)
        c = self.carry_init(rng, mem_shape, self.param_dtype)
        return c

    @property
    def num_feature_axes(self) -> int:
        return 2


class MultiBiquad(nn.Module):
    channels: int
    order: int
    carry_init: nn.initializers.Initializer = nn.initializers.zeros_init()

    @nn.compact
    def __call__(self, inputs):
        inputs = jnp.transpose(
            jax.lax.conv_general_dilated_patches(
                jnp.transpose(inputs, (0, 2, 1)),
                filter_shape=(self.order + 1,),
                window_strides=(1,),
                padding=((2, 0),),
            ),
            (0, 2, 1),
        )
        return jnp.squeeze(
            nn.RNN(MultiBiquadCell(channels=self.channels, carry_init=self.carry_init))(
                inputs
            ),
            axis=2,
        )


class MultiBiquadWithSidechain(nn.Module):
    channels: int
    order: int
    carry_init: nn.initializers.Initializer = nn.initializers.zeros_init()

    @nn.compact
    def __call__(self, inputs):
        batch_size = inputs.shape[0]
        inputs = jnp.transpose(
            jax.lax.conv_general_dilated_patches(
                jnp.transpose(inputs, (0, 2, 1)),
                filter_shape=(self.order + 1,),
                window_strides=(1,),
                padding=((2, 0),),
            ),
            (0, 2, 1),
        )
        # (b, c, k, s)
        inputs = jnp.reshape(inputs, (batch_size, self.channels, self.order + 1, -1))
        # (b, c, s, k)
        inputs = jnp.transpose(inputs, (0, 3, 1, 2))
        return jnp.squeeze(
            nn.RNN(
                MultiBiquadCellWithSidechain(
                    channels=self.channels, carry_init=self.carry_init, order=self.order
                )
            )(inputs),
            axis=2,
        )

class MixingBoard(nn.Module):
    channels: int
    depth: int
    order: int
    carry_init: nn.initializers.Initializer = nn.initializers.zeros_init()

    @nn.compact
    def __call__(self, x):
        x = MultiBiquad(channels=self.channels, order=self.order)(x)
        for _ in range(self.depth - 1):
            x = MultiBiquadWithSidechain(channels=self.channels, order=self.order)(x)
        x = nn.Dense(features=1)(x)
        return x
        

if __name__ == "__main__":
    # model = MultiBiquad(channels=2**8)
    # print(model.tabulate(jax.random.key(0), jnp.ones((2**2, 2**14, 2**0))))
    # channels = 2**8
    # model = MultiBiquadWithSidechain(channels=channels)
    # print(model.tabulate(jax.random.key(0), jnp.ones((2**2, 2**14, channels))))
    channels = 2**10
    model = MixingBoard(channels=channels, order=2, depth=8)
    print(model.tabulate(jax.random.key(0), jnp.ones((2**2, 2**14, 1))))
