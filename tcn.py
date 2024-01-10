import flax.linen as nn
import jax
import jax.numpy as jnp
from functools import partial


class TCN(nn.Module):
    features: int
    kernel_dilation: int
    kernel_size: int

    @nn.compact
    def __call__(self, x, train: bool):
        x_ = x
        x = nn.Conv(
            features=self.features,
            kernel_dilation=(self.kernel_dilation,),
            kernel_size=(self.kernel_size,),
            padding=((0, 0),),
        )(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.gelu(x)
        x_res = nn.Conv(
            features=self.features,
            kernel_size=(1,),
            feature_group_count=self.features if x_.shape[-1] > 1 else 1,
        )(x_)
        x = x + x_res[:, -x.shape[-2] :, :]
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
        )(features=1)(x)
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
    expand_factor: float = 2.0
    positional_encodings: bool = True

    @nn.compact
    def __call__(self, x, train: bool):
        for _ in range(self.conv_depth):
            x = TCN(
                features=self.features,
                kernel_dilation=self.kernel_dilation,
                kernel_size=self.conv_kernel_size,
            )(x, train)
        # x = ConvAttnBlock(
        #     features=self.features,
        #     kernel_size=self.attn_kernel_size,
        #     heads=self.heads,
        #     expand_factor=self.expand_factor,
        #     depth=self.attn_depth,
        #     positional_encodings=self.positional_encodings,
        # )(x)
        return x


if __name__ == "__main__":
    model = TCNNetwork(
        features=2**8,
        kernel_dilation=2**1,
        conv_kernel_size=2**4,
        attn_kernel_size=2**8,
        heads=2**6,
        conv_depth=2**5,
        attn_depth=2**4,
        expand_factor=2.0,
    )
    print(
        model.tabulate(jax.random.key(0), jnp.ones((2**2, 2**14, 1)), train=False)
    )
