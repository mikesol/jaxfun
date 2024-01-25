from flax import linen as nn
from typing import (
    Any,
    Callable,
    Optional,
    Tuple,
    TypeVar,
)
import math
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
    carry_init: nn.initializers.Initializer = nn.initializers.zeros_init()

    def setup(self):
        self._combinator = (
            partial(SimpleLSTMCombinator)
            if self.combinator == None
            else self.combinator
        )
        self.i = self._combinator(name="i")
        self.f = self._combinator(name="f")
        self.g = self._combinator(name="g")
        self.o = self._combinator(name="o")

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


def switch_last_two(i):
    return jnp.transpose(
        i, tuple(x for x in range(i.ndim - 2)) + (i.ndim - 1, i.ndim - 2)
    )


class StackedRNNCell(nn.Module):
    features: int
    cell: Callable[..., Any] = LSTMCell
    skip: bool = False
    dense_init: nn.initializers.Initializer = nn.linear.default_kernel_init
    bias_init: nn.initializers.Initializer = nn.initializers.zeros_init()
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32
    carry_init: nn.initializers.Initializer = nn.initializers.zeros_init()
    is_filter_bank: bool = False
    do_last_skip: bool = False
    dense_across_stack: bool = True
    projection: Optional[int] = None
    only_last: bool = True
    cell_preprocessing: Callable = lambda x: x

    @nn.compact
    def __call__(self, carry, inputs):
        c, h = carry
        # make the inputs match the size of the hidden state
        inputs = nn.Dense(
            features=self.features,
            use_bias=True,
            kernel_init=self.dense_init,
            bias_init=self.bias_init,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )(inputs)
        # in filter bank mode, we treat carry and hidden as outputs
        # of stacked filters. otherwise, we treat them as outputs of a single
        # system that feeds into the next layer of the system.
        if not self.is_filter_bank:

            def _f(ii, hh):
                return jax.lax.concatenate(
                    [ii[..., :1, :], remove_last_entry_of_second_last_dim(hh)],
                    dimension=ii.ndim - 2,
                )

            ff = _f
            for _ in range(len(c.shape) - 3):
                ff = jax.vmap(ff, in_axes=1, out_axes=1)
            inputs = ff(inputs, h)
        carry, out = nn.vmap(
            self.cell,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            in_axes=2,
            out_axes=2,
        )(features=self.features)((c, h), inputs)

        if self.only_last:
            out = out[..., -1, :]
        if self.do_last_skip:
            out += nn.Dense(
                features=self.projection
                if self.projection is not None
                else self.features,
                use_bias=True,
                kernel_init=self.dense_init,
                bias_init=self.bias_init,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
            )(inputs[:, 0, :] if self.only_last else inputs)
        if self.dense_across_stack:
            out = switch_last_two(
                nn.Dense(
                    features=out.shape[-2],
                    use_bias=True,
                    kernel_init=self.dense_init,
                    bias_init=self.bias_init,
                    dtype=self.dtype,
                    param_dtype=self.param_dtype,
                )(switch_last_two(out))
            )

        if self.projection is not None:
            out = nn.Dense(
                features=self.projection,
                use_bias=True,
                kernel_init=self.dense_init,
                bias_init=self.bias_init,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
            )(out)

        return carry, out

    @nn.nowrap
    def initialize_carry(
        self, rng: PRNGKey, input_shape: Tuple[int, ...]
    ) -> Tuple[Array, Array]:
        key1, key2 = random.split(rng)
        mem_shape = input_shape[:-1] + (self.features,)
        c = self.carry_init(key1, mem_shape, self.param_dtype)
        h = self.carry_init(key2, mem_shape, self.param_dtype)
        return (c, h)

    @property
    def num_feature_axes(self) -> int:
        return 1


class StackedRNNCellWithAttn(nn.Module):
    features: int
    cell: Callable[..., Any] = LSTMCell
    skip: bool = False
    dense_init: nn.initializers.Initializer = nn.linear.default_kernel_init
    bias_init: nn.initializers.Initializer = nn.initializers.zeros_init()
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32
    carry_init: nn.initializers.Initializer = nn.initializers.zeros_init()
    levels: int = 2
    attn_levels: int = 1
    projection: Optional[int] = None
    only_last: bool = True

    @nn.compact
    def __call__(self, carry, inputs):
        c, h = carry
        inputs = nn.Dense(
            features=self.features,
            use_bias=True,
            kernel_init=self.dense_init,
            bias_init=self.bias_init,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )(inputs)
        inputs = jax.lax.concatenate(
            [inputs, remove_last_entry_of_second_last_dim(h)],
            dimension=inputs.ndim - 2,
        )
        C = []
        H = []
        I = []
        for LI in range(0, self.levels, self.attn_levels):
            x, y = nn.vmap(
                self.cell,
                variable_axes={"params": 0},
                split_rngs={"params": True},
                in_axes=2,
                out_axes=2,
            )(features=self.features)(
                (
                    c[:, LI : LI + self.attn_levels, :],
                    h[:, LI : LI + self.attn_levels, :],
                ),
                inputs[:, LI : LI + self.attn_levels, :],
            )
            C.append(x[0])
            H.append(x[1])
            I.append(AttnBlock(heads=self.attn_levels // 2)(y))

        out = jnp.concatenate(I, axis=-2)
        c = jnp.concatenate(C, axis=-2)
        h = jnp.concatenate(H, axis=-2)
        if self.only_last:
            out = out[..., -1, :]

        if self.projection is not None:
            out = nn.Dense(
                features=self.projection,
                use_bias=True,
                kernel_init=self.dense_init,
                bias_init=self.bias_init,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
            )(out)

        return (c, h), out

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


class AttnBlock(nn.Module):
    heads: int
    expand_factor: float = 2.0
    dense_init: nn.initializers.Initializer = nn.linear.default_kernel_init
    bias_init: nn.initializers.Initializer = nn.initializers.zeros_init()

    @nn.compact
    def __call__(self, out):
        features = out.shape[-1]
        out_ = out
        out = nn.MultiHeadDotProductAttention(self.heads)(out, out)
        out = nn.LayerNorm()(out_ + out)
        out_ = out
        out = nn.Dense(
            features=int(out.shape[-1] * self.expand_factor),
            kernel_init=self.dense_init,
            bias_init=self.bias_init,
            use_bias=True,
        )(out)
        out = nn.gelu(out)
        out = nn.Dense(
            features=features,
            kernel_init=self.dense_init,
            bias_init=self.bias_init,
            use_bias=True,
        )(out)
        return nn.LayerNorm()(out_ + out)


class Transformeresque(nn.Module):
    to_wrap: Callable[..., Any]
    heads: int
    attn_layers: int = 2**2
    projection: Optional[int] = None
    only_last: bool = True
    positional_encodings: bool = True
    expand_factor: float = 2.0
    param_dtype: Dtype = jnp.float32
    dense_init: nn.initializers.Initializer = nn.linear.default_kernel_init
    bias_init: nn.initializers.Initializer = nn.initializers.zeros_init()

    def setup(self):
        self.wrapped = self.to_wrap()
        if self.positional_encodings:
            self.pe = PositionalEncoding()
        self.attentions = nn.Sequential(
            [
                AttnBlock(
                    heads=self.heads,
                    expand_factor=self.expand_factor,
                    dense_init=self.dense_init,
                    bias_init=self.bias_init,
                )
                for _ in range(self.attn_layers)
            ]
        )
        if self.projection is not None:
            self.proj_dense = nn.Dense(
                features=self.projection,
                use_bias=True,
                kernel_init=self.dense_init,
                bias_init=self.bias_init,
            )

    def __call__(self, carry, inputs):
        carry, out = self.wrapped(carry, inputs)

        if self.positional_encodings:
            out = self.pe(out)
        out = self.attentions(out)

        if self.only_last:
            out = out[..., -1, :]
        if self.projection is not None:
            out = self.proj_dense(out)

        return carry, out

    @nn.nowrap
    def initialize_carry(
        self, rng: PRNGKey, input_shape: Tuple[int, ...]
    ) -> Tuple[Array, Array]:
        return self.wrapped.initialize_carry(rng, input_shape)

    @property
    def num_feature_axes(self) -> int:
        return 1


class LSTM(nn.Module):
    features: int
    cell: Callable[..., Any] = None
    skip: bool = False
    levels: int = 1
    only_last: bool = True
    do_last_skip: bool = False
    projection: Optional[int] = None

    def setup(self):
        self._cell = (
            partial(LSTMCell, features=self.features)
            if self.cell == None
            else self.cell
        )
        self.stack = StackedRNNCell(
            features=self.features,
            levels=self.levels,
            skip=self.skip,
            do_last_skip=self.do_last_skip,
            only_last=self.only_last,
            projection=self.projection,
            cell=self._cell,
        )
        self.rnn = nn.RNN(self.stack)

    def __call__(self, x):
        x = jnp.expand_dims(x, axis=1)
        x = jnp.repeat(x, repeats=self.levels, axis=1)
        x = self.rnn(x)
        return x


class LSTMWithFilterBanks(nn.Module):
    features: int
    cell: Callable[..., Any] = None
    skip: bool = False
    levels: int = 1
    banks: int = 1
    only_last: bool = True
    do_last_skip: bool = False
    projection: Optional[int] = None

    def setup(self):
        self._cell = (
            partial(LSTMCell, features=self.features)
            if self.cell == None
            else self.cell
        )
        self.stack = StackedRNNCell(
            features=self.features,
            skip=self.skip,
            is_filter_bank=False,
            # do_last_skip=self.do_last_skip,
            dense_across_stack=True,
            only_last=self.only_last,
            projection=self.projection,
            cell_preprocessing=lambda x: jnp.expand_dims(x, axis=-2),
            cell=partial(
                StackedRNNCell,
                features=self.features,
                skip=self.skip,
                do_last_skip=False,
                is_filter_bank=True,
                dense_across_stack=True,
                only_last=False,
                projection=None,
                cell=self._cell,
            ),
        )
        self.rnn = nn.RNN(self.stack)

    def __call__(self, x):
        x = jnp.expand_dims(x, axis=1)
        x = jnp.expand_dims(x, axis=1)
        x = jnp.repeat(x, repeats=self.levels, axis=2)
        x = jnp.repeat(x, repeats=self.banks, axis=1)
        x = self.rnn(x)
        x = jnp.sum(x, axis=1)
        x = jnp.transpose(x, (0,2,1))
        return x


if __name__ == "__main__":
    # model = LSTM(
    #     features=2**8,
    #     levels=2**5,
    #     skip=True,
    #     projection=1,
    #     only_last=True,
    #     do_last_skip=True,
    #     name="lstm",
    #     cell=partial(LSTMCell, combinator=ComplexLSTMCombinator),
    # )
    # print(model.tabulate(jax.random.key(0), jnp.ones((2**2, 2**8, 1))))
    model = LSTMWithFilterBanks(
        features=2**8,
        levels=2**3,
        banks=2**5,
        skip=True,
        projection=1,
        only_last=True,
        do_last_skip=True,
        name="lstm",
        cell=partial(LSTMCell, combinator=ComplexLSTMCombinator),
    )
    print(model.tabulate(jax.random.key(0), jnp.ones((2**2, 2**10, 1))))
    # model = nn.RNN(
    #     Transformeresque(
    #         to_wrap=partial(
    #             StackedRNNCell,
    #             features=2**7,
    #             levels=2**5,
    #             skip=True,
    #             only_last=False,
    #             cell=LSTMCell,
    #         ),
    #         heads=2**4,
    #         attn_layers=2**2,
    #     )
    # )
    # print(model.tabulate(jax.random.key(0), jnp.ones((2**2, 2**8, 1))))

    # model = nn.RNN(
    #     StackedRNNCellWithAttn(
    #         features=2**7,
    #         levels=2**5,
    #         attn_levels=2**2,
    #         skip=True,
    #         only_last=False,
    #         cell=LSTMCell,
    #     )
    # )
    # model = nn.RNN(
    #     StackedRNNCell(
    #         features=2**7,
    #         levels=2**5,
    #         do_last_skip=True,
    #         skip=True,
    #         only_last=True,
    #         projection=1,
    #         cell=LSTMCell,
    #     )
    # )

    # print(model.tabulate(jax.random.key(0), jnp.ones((2**2, 2**8, 1))))
