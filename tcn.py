import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.linen import initializers
from bias_types import BiasTypes
from create_filtered_audio import create_biquad_coefficients
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
import numpy as np


class PELU(nn.Module):
    @nn.compact
    def __call__(self, x):
        # Initialize the alpha and beta parameters
        # These values can be adjusted based on your specific use case
        alpha = self.param("alpha", nn.initializers.ones, (1,))
        beta = self.param("beta", nn.initializers.ones, (1,))

        # PELU activation function
        positive = jnp.where(x > 0, x, 0)
        negative = jnp.where(x <= 0, alpha * (jnp.exp(x / beta) - 1), 0)
        return positive + negative


def array_to_tuple(arr):
    if isinstance(arr, np.ndarray):
        return tuple(array_to_tuple(a) for a in arr)
    else:
        return arr


class Sidechain(nn.Module):
    in_channels: int = 2**6
    out_channels: int = 2**6
    kernel_size: int = 7
    norm_factor: float = 1.0

    @nn.compact
    def __call__(self, x):
        batch_size = x.shape[0]
        weights = self.param(
            "weights",
            nn.with_partitioning(initializers.lecun_normal(), (None, "model")),
            (self.in_channels, self.in_channels, self.kernel_size),
            jnp.float32,
        )
        x_ = x

        def do_unfold(x):
            x = jax.lax.conv_general_dilated_patches(
                jnp.transpose(x, (0, 2, 1)),
                filter_shape=(self.kernel_size,),
                window_strides=(1,),
                padding=((0, 0),),
            )
            x = jnp.transpose(
                jnp.reshape(x, (batch_size, self.in_channels, self.kernel_size, -1)),
                (0, 2, 1, 3),
            )
            return x

        x = do_unfold(x)

        # weights == (b, k, c, s)
        # x = (b, s, c) weights = (x, c, k) w' = (b, x, s, k) w = (b, k, x, s)
        w = jnp.transpose(
            jnp.einsum("abc,dcg->adbg", x_[:, -x.shape[3] :, :], weights), (0, 3, 1, 2)
        )
        w = nn.tanh(w / self.norm_factor)
        x = x * w
        x = jnp.sum(x, axis=1)
        x = jnp.transpose(x, (0, 2, 1))
        if self.out_channels != self.in_channels:
            x = nn.Conv(features=self.out_channels, kernel_size=(1,), use_bias=False)(x)
        return x


class TCN(nn.Module):
    features: int
    kernel_dilation: int
    kernel_size: int
    with_sidechain: bool = True
    activation: callable = nn.gelu
    bias_type: BiasTypes = BiasTypes.BATCH_NORM

    @nn.compact
    def __call__(self, x, train: bool):
        x_ = x
        x = nn.Conv(
            features=self.features,
            kernel_init=nn.with_partitioning(
                initializers.lecun_normal(), (None, "model")
            )
            if x.shape[-1] > 1
            else initializers.lecun_normal(),
            kernel_dilation=(self.kernel_dilation if not self.with_sidechain else 1,),
            kernel_size=(self.kernel_size,),
            padding=((0, 0),),
            use_bias=self.bias_type == BiasTypes.DC,
        )(x_)
        if self.with_sidechain:
            x += Sidechain(
                in_channels=x_.shape[-1],
                out_channels=self.features,
                kernel_size=self.kernel_size,
                norm_factor=math.sqrt(self.features),
            )(x_)
        if self.bias_type == BiasTypes.BATCH_NORM:
            x = nn.BatchNorm(use_running_average=not train)(x)
        x = self.activation()(x)
        x_res = nn.Conv(
            features=self.features,
            kernel_size=(1,),
            # no kernel init as we're going down to 1
            feature_group_count=self.features if x_.shape[-1] > 1 else 1,
            use_bias=False,
        )(x_)
        x = x + x_res[:, -x.shape[-2] :, :]
        return x


class AttnBlock(nn.Module):
    heads: int
    expand_factor: float = 2.0
    activation: Callable = lambda: nn.gelu

    @nn.compact
    def __call__(self, out):
        features = out.shape[-1]
        out_ = out
        out = nn.MultiHeadDotProductAttention(
            self.heads,
            kernel_init=nn.with_partitioning(
                initializers.lecun_normal(), (None, "model")
            ),
        )(out, out)
        out = nn.LayerNorm()(out_ + out)
        out_ = out
        out = nn.Dense(
            features=int(out.shape[-1] * self.expand_factor),
            kernel_init=nn.with_partitioning(
                initializers.lecun_normal(), (None, "model")
            ),
            use_bias=True,
        )(out)
        out = self.activation()(out)
        out = nn.Dense(
            features=features,
            kernel_init=nn.with_partitioning(
                initializers.lecun_normal(), (None, "model")
            ),
            use_bias=True,
        )(out)
        return nn.LayerNorm()(out_ + out)


class PositionalEncoding(nn.Module):
    @nn.compact
    def __call__(self, x):
        seq_length, d_model = x.shape[-2], x.shape[-1]

        position = jnp.arange(seq_length)[:, jnp.newaxis]
        div_term = jnp.exp(jnp.arange(0, d_model, 2) * -(jnp.log(10000.0) / d_model))
        pos_encoding = jnp.zeros((seq_length, d_model))
        pos_encoding = pos_encoding.at[:, 0::2].set(jnp.sin(position * div_term))
        pos_encoding = pos_encoding.at[:, 1::2].set(jnp.cos(position * div_term))

        x += pos_encoding

        return x


class ConvAttnBlock(nn.Module):
    features: int
    kernel_size: int
    heads: int
    depth: int
    positional_encodings: bool = True
    expand_factor: float = 2.0
    activation: Callable = lambda: nn.gelu

    @nn.compact
    def __call__(self, x):
        batch_size = x.shape[0]
        x = jax.lax.conv_general_dilated_patches(
            jnp.transpose(x, (0, 2, 1)),
            filter_shape=(self.kernel_size,),
            window_strides=(1,),
            padding=((0, 0),),
        )
        # (batch, channel, k, seq)
        x = jnp.reshape(x, (batch_size, self.features, self.kernel_size, -1))
        # (batch, k, channel, seq)
        x = jnp.transpose(x, (0, 2, 1, 3))
        if self.positional_encodings:
            x = nn.vmap(
                PositionalEncoding,
                in_axes=-1,
                out_axes=-1,
                variable_axes={"params": None},
                split_rngs={"params": False},
            )()(x)
        for _ in range(self.depth):
            x = nn.vmap(
                AttnBlock,
                in_axes=-1,
                out_axes=-1,
                variable_axes={"params": None},
                split_rngs={"params": False},
            )(
                heads=self.heads,
                expand_factor=self.expand_factor,
                activation=self.activation,
            )(
                x
            )
        # (batch, channel, k, seq)
        x = jnp.transpose(x, (0, 2, 1, 3))
        x = nn.vmap(
            nn.Dense,
            in_axes=-1,
            out_axes=-1,
            variable_axes={"params": None},
            split_rngs={"params": False},
        )(
            features=1,
            # no sharding as we go to 1
        )(
            x
        )
        # (batch, seq, channel, k)
        x = jnp.transpose(x, (0, 3, 1, 2))
        x = jnp.squeeze(x, axis=-1)
        return x


class TCNNetwork(nn.Module):
    features: int
    kernel_dilation: int
    conv_kernel_size: int
    attn_kernel_size: int
    heads: int
    conv_depth: int
    attn_depth: int
    sidechain_modulo_l: int = 2
    sidechain_modulo_r: int = 1
    expand_factor: float = 2.0
    positional_encodings: bool = True
    bias_type: BiasTypes = BiasTypes.BATCH_NORM

    @nn.compact
    def __call__(self, x, train: bool):
        for i in range(self.conv_depth):
            x = TCN(
                bias_type=self.bias_type,
                features=self.features,
                kernel_dilation=self.kernel_dilation,
                kernel_size=self.conv_kernel_size,
                with_sidechain=i % self.sidechain_modulo_l == self.sidechain_modulo_r,
            )(x, train)
        x = ConvAttnBlock(
            features=self.features,
            kernel_size=self.attn_kernel_size,
            heads=self.heads,
            expand_factor=self.expand_factor,
            depth=self.attn_depth,
            positional_encodings=self.positional_encodings,
        )(x)
        return x


class BiquadCell(nn.Module):
    coefficients: Array
    carry_init: nn.initializers.Initializer = nn.initializers.zeros_init()
    param_dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, carry, inputs):
        yx = jnp.concatenate([inputs, carry], axis=-1)
        o = jnp.sum(yx * self.coefficients[None, :], axis=-1, keepdims=True)
        return jnp.concatenate([o, carry[..., :-1]], axis=-1), o

    @nn.nowrap
    def initialize_carry(
        self, rng: PRNGKey, input_shape: Tuple[int, ...]
    ) -> Tuple[Array, Array]:
        batch_dims = input_shape[:-1]
        mem_shape = batch_dims + (2,)
        c = self.carry_init(rng, mem_shape, self.param_dtype)
        return c

    @property
    def num_feature_axes(self) -> int:
        return 1


class Biquad(nn.Module):
    carry_init: nn.initializers.Initializer = nn.initializers.zeros_init()

    @nn.compact
    def __call__(self, inputs, coefficients):
        return nn.RNN(
            BiquadCell(carry_init=self.carry_init, coefficients=coefficients)
        )(inputs)


class MultiBiquad(nn.Module):
    coefficients: Array
    carry_init: nn.initializers.Initializer = nn.initializers.zeros_init()

    @nn.compact
    def __call__(self, inputs):
        inputs = jnp.transpose(
            jax.lax.conv_general_dilated_patches(
                jnp.transpose(inputs, (0, 2, 1)),
                filter_shape=(3,),
                window_strides=(1,),
                padding=((2, 0),),
            ),
            (0, 2, 1),
        )
        vmapped = nn.vmap(
            lambda m, c: m(inputs, c),
            in_axes=-1,
            out_axes=-1,
        )(
            Biquad(carry_init=self.carry_init),
            self.coefficients,
        )
        return jnp.squeeze(vmapped, axis=-2)


class ExperimentalTCNNetwork(nn.Module):
    kernel_dilation: int
    conv_kernel_size: int
    attn_kernel_size: int
    heads: int
    conv_depth: Tuple[int]
    attn_depth: int
    coefficients: Array
    sidechain_modulo_l: int = 2
    sidechain_modulo_r: int = 1
    expand_factor: float = 2.0
    positional_encodings: bool = True
    do_last_activation: bool = True
    do_last_skip: bool = False
    activation: callable = nn.gelu
    bias_type: BiasTypes = BiasTypes.BATCH_NORM

    @nn.compact
    def __call__(self, x, train: bool):
        x_ = x
        mb = MultiBiquad(coefficients=jnp.array(self.coefficients))(x)
        x = jnp.concatenate([x, mb], axis=-1)
        x = jax.lax.stop_gradient(x)
        for i in self.conv_depth:
            x = TCN(
                features=i,
                bias_type=self.bias_type,
                activation=self.activation,
                kernel_dilation=self.kernel_dilation,
                kernel_size=self.conv_kernel_size,
                with_sidechain=i % self.sidechain_modulo_l == self.sidechain_modulo_r,
            )(x, train)
        x = ConvAttnBlock(
            features=self.conv_depth[-1],
            activation=self.activation,
            kernel_size=self.attn_kernel_size,
            heads=self.heads,
            expand_factor=self.expand_factor,
            depth=self.attn_depth,
            positional_encodings=self.positional_encodings,
        )(x)
        if self.do_last_activation:
            x = nn.tanh(x)
        if self.do_last_skip:
            x_ = nn.Conv(
                features=x.shape[-1],
                kernel_init=initializers.lecun_normal(),
                kernel_size=(x_.shape[-2] - x.shape[-2] + 1,),
                padding=((0, 0),),
                use_bias=False,
            )(x_)
            x += x_
        x = nn.Conv(
            features=1,
            kernel_init=initializers.lecun_normal(),
            kernel_size=(1,),
            padding=((0, 0),),
            use_bias=False,
        )(x)
        return x


if __name__ == "__main__":
    coefficients = create_biquad_coefficients(
        2**11 - 1,
        44100,
        300,
        19000,
        30,
        10,
    )
    print(coefficients.shape)
    # model = ExperimentalTCNNetwork(
    #     # features=2**6,
    #     activation=nn.vmap(
    #         PELU,
    #         variable_axes={"params": 0},
    #         split_rngs={"params": True},
    #         in_axes=-1,
    #         out_axes=-1,
    #     ),
    #     coefficients=array_to_tuple(coefficients),
    #     kernel_dilation=2**1,
    #     conv_kernel_size=2**3,
    #     attn_kernel_size=2**7,
    #     heads=2**5,
    #     conv_depth=tuple(2**n for n in (11, 11)),  # 2**4,
    #     attn_depth=2**2,
    #     expand_factor=2.0,
    # )
    model = ExperimentalTCNNetwork(
        # features=2**6,
        activation=lambda: lambda x: x,
        coefficients=array_to_tuple(coefficients),
        kernel_dilation=2**1,
        conv_kernel_size=7,
        attn_kernel_size=2**7,
        heads=2**5,
        do_last_skip=True,
        conv_depth=(
            1024,
            1024,
            512,
            512,
            256,
            256,
            128,
            128,
            64,
            64,
            32,
            32,
            16,
            16,
            8,
            8,
            4,
            4,
            2,
            2,
        ),
        sidechain_modulo_l=1,
        sidechain_modulo_r=0,
        bias_type=False,
        attn_depth=0,
        expand_factor=2.0,
    )
    print(
        model.tabulate(jax.random.key(0), jnp.ones((2**2, 2**14, 1)), train=False)
    )
