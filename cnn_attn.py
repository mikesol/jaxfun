import flax.linen as nn
import jax.numpy as jnp
import jax
from flax.linen import initializers


class Convblock(nn.Module):
    channels: int = 2**6
    kernel_size: int = 7
    norm_factor: float = 1.0
    skip: bool = True
    layernorm: bool = True
    inner_skip: bool = True

    @nn.compact
    def __call__(self, x):
        if x.shape[-1] != self.channels:
            x = nn.Conv(
                features=self.channels,
                kernel_size=(1,),
                use_bias=False,
                dtype=None,
                param_dtype=jnp.float32,
            )(x)
            x = nn.PReLU()(x)
        weights = self.param(
            "weights",
            initializers.lecun_normal(),
            (self.channels, self.channels, self.kernel_size),
            jnp.float32,
        )
        # skip
        x_ = x
        # weights == (b, k, c, s)
        w = jnp.transpose(jnp.einsum("abc,dcg->adbg", x, weights), (0, 3, 1, 2))
        w = nn.tanh(w / self.norm_factor)

        def do_unfold(x):
            seq_len = x.shape[-2]
            half_kernel_size = self.kernel_size // 2
            x = jax.lax.conv_general_dilated_patches(
                x,
                filter_shape=(self.kernel_size,),
                window_strides=(1,),
                padding=((half_kernel_size, half_kernel_size),),
            )
            x = jnp.transpose(
                jnp.reshape(x, (-1, self.channels, self.kernel_size, seq_len)),
                (0, 2, 1, 3),
            )
            return x

        x = do_unfold(x)
        x = x * w
        x = jnp.sum(x, axis=1)
        x = jnp.transpose(x, (0, 2, 1))
        x = x_ + x if self.skip and self.inner_skip else x
        if self.layernorm:
            x = nn.LayerNorm()(x)
        x_ = x
        x = nn.Conv(
            features=self.channels,
            kernel_size=(1,),
            use_bias=True,
            dtype=None,
            param_dtype=jnp.float32,
        )(x)
        x = nn.PReLU()(x)
        return x_ + x if self.skip else x


class Convattn(nn.Module):
    channels: int = 2**6
    depth: int = 2**4
    kernel_size: int = 7
    skip_freq: int = 1
    norm_factor: float = 1.0
    layernorm: bool = True
    inner_skip: bool = True

    @nn.compact
    def __call__(self, x):
        for i in range(self.depth):
            x = Convblock(
                channels=self.channels,
                kernel_size=self.kernel_size,
                norm_factor=self.norm_factor,
                skip=(i % self.skip_freq) == (self.skip_freq - 1),
                layernorm=self.layernorm,
                inner_skip=self.inner_skip,
            )(x)
        x = nn.Conv(features=1, kernel_size=(1,), use_bias=True)(x)


if __name__ == "__main__":
    model = Convattn()
    print(model.tabulate(jax.random.key(0), jnp.ones((2**2, 2**8, 1))))
