import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.linen import initializers
import math
from typing import Tuple



class Sidechain(nn.Module):
    channels: int = 2**6
    kernel_size: int = 7
    norm_factor: float = 1.0

    @nn.compact
    def __call__(self, x):
        batch_size = x.shape[0]
        weights = self.param(
            "weights",
            nn.with_partitioning(initializers.lecun_normal(), (None, "model")),
            (self.channels, self.channels, self.kernel_size),
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
                jnp.reshape(x, (batch_size, self.channels, self.kernel_size, -1)),
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
        return x


class TCN(nn.Module):
    features: int
    kernel_dilation: int
    kernel_size: int
    with_sidechain: bool = True

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
            use_bias=False,
        )(x_)
        if self.with_sidechain:
            x += Sidechain(
                channels=self.features,
                kernel_size=self.kernel_size,
                norm_factor=math.sqrt(self.features),
            )(x_)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.gelu(x)
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
        out = nn.gelu(out)
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
            )(heads=self.heads, expand_factor=self.expand_factor)(x)
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

    @nn.compact
    def __call__(self, x, train: bool):
        for i in range(self.conv_depth):
            x = TCN(
                features=self.features,
                kernel_dilation=self.kernel_dilation,
                kernel_size=self.conv_kernel_size,
                with_sidechain=i % self.sidechain_modulo_l == self.sidechain_modulo_r
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


class ExperimentalTCNNetwork(nn.Module):
    kernel_dilation: int
    conv_kernel_size: int
    attn_kernel_size: int
    heads: int
    conv_depth: Tuple[int]
    attn_depth: int
    sidechain_modulo_l: int = 2
    sidechain_modulo_r: int = 1
    expand_factor: float = 2.0
    positional_encodings: bool = True

    @nn.compact
    def __call__(self, x, train: bool):
        for i in self.conv_depth:
            x = TCN(
                features=i,
                kernel_dilation=self.kernel_dilation,
                kernel_size=self.conv_kernel_size,
                with_sidechain=False
            )(x, train)
        x = ConvAttnBlock(
            features=self.conv_depth[-1],
            kernel_size=self.attn_kernel_size,
            heads=self.heads,
            expand_factor=self.expand_factor,
            depth=self.attn_depth,
            positional_encodings=self.positional_encodings,
        )(x)
        return x


if __name__ == "__main__":
    model = ExperimentalTCNNetwork(
        # features=2**6,
        kernel_dilation=2**1,
        conv_kernel_size=2**3,
        attn_kernel_size=2**7,
        heads=2**5,
        conv_depth=tuple(2**n for n in (11,11,10,10,9,9,8,8)), # 2**4,
        attn_depth=2**4,
        expand_factor=2.0,
    )
    print(
        model.tabulate(jax.random.key(0), jnp.ones((2**2, 2**14, 2**11)), train=False)
    )
