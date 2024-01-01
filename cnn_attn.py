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
    pad_to_input_size: bool = True

    @nn.compact
    def __call__(self, x):
        batch_size = x.shape[0]
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
        def do_unfold(x):
            half_kernel_size = self.kernel_size // 2
            x = jax.lax.conv_general_dilated_patches(
                jnp.transpose(x, (0, 2, 1)),
                filter_shape=(self.kernel_size,),
                window_strides=(1,),
                padding=((half_kernel_size, half_kernel_size),)
                if self.pad_to_input_size
                else ((0, 0),)
            )
            x = jnp.transpose(
                jnp.reshape(x, (batch_size, self.channels, self.kernel_size, -1)),
                (0, 2, 1, 3),
            )
            return x

        x = do_unfold(x)

        # weights == (b, k, c, s)
        # x = (b, s, c) weights = (c, c, k) w' = (b, c, s, k) w = (b, k, c, s)
        w = jnp.transpose(jnp.einsum("abc,dcg->adbg", x_[:,-x.shape[3]:,:], weights), (0, 3, 1, 2))
        w = nn.tanh(w / self.norm_factor)
        x = x * w
        x = jnp.sum(x, axis=1)
        x = jnp.transpose(x, (0, 2, 1))
        x = x_[:,(-x.shape[1]):,:] + x if self.skip and self.inner_skip else x
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


class ConvblockWithTarget(nn.Module):
    channels: int = 2**6
    kernel_size: int = 7
    norm_factor: float = 1.0
    skip: bool = True
    layernorm: bool = True
    inner_skip: bool = True

    @nn.compact
    def __call__(self, x):
        batch_size = x.shape[0]
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

        def do_unfold(x):
            x = jax.lax.conv_general_dilated_patches(
                jnp.transpose(x, (0, 2, 1)),
                # filter covers interleaved
                filter_shape=(self.kernel_size * 2,),
                # we stride by 2 because of the interleaved
                window_strides=(2,),
                # we don't pad because we're using this for realtime processing
                padding=((0, 0),),
            )
            # the seq len will have the last half lobbed off
            # because we don't right-pad
            # now (batch, channel, k2, seq_shorter)
            x = jnp.reshape(x, (batch_size, self.channels, self.kernel_size * 2, -1))

            # now (batch, channel, seq_shorter, k2)
            x = jnp.transpose(x, (0, 1, 3, 2))
            return x

        x = do_unfold(x)
        # x = (batch_a, seq_b, chan_c) weights = (chan_d, chan_c, k_g) w = (batch_a, chan_d, seq_b, k_g)
        w = jnp.einsum("abc,dcg->adbg", x_[:,-(x.shape[2] * 2):,:], weights)
        even_seq = w[:, :, ::2, :]  # Slices out even-indexed elements
        odd_seq = w[:, :, 1::2, :]  # Slices out odd-indexed elements

        # Stack the even and odd sequences along a new dimension
        # we now have (batch_a, chan_d, seq_shorter, k_g, 2)
        w = jnp.stack([even_seq, odd_seq], axis=-1)
        # interleave
        ws = w.shape
        w = jnp.reshape(w, (ws[0], ws[1], ws[2], ws[3] * 2))
        w = nn.tanh(w / self.norm_factor)

        x = x * w
        # this brings it to (batch, channel, seq)
        x = jnp.sum(x, axis=3)
        # this puts features at the end, so (batch, seq, channel)
        x = jnp.transpose(x, (0, 2, 1))
        x = x_[:,::2,:][:,(-x.shape[1]):,:] + x if self.skip and self.inner_skip else x
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


def c1d(o,k,s):
    return (s*(o - 1)) + 1 + (k-1)

class ConvAttnFauxLarsen(nn.Module):
    to_mask:int = 4
    depth:int = 2**4
    channels: int = 2**6
    kernel_size: int = 7
    norm_factor: float = 1.0
    skip_freq: int = 1
    layernorm: bool = True
    inner_skip: bool = True

    @nn.compact
    def __call__(self, x):
        x_masked = x[:, : -(self.to_mask * 2), :]
        # get rid of the target
        x_final = x[:, -(self.to_mask * 2) :: 2, :]
        z = x_masked
        foundry = x_masked
        zlen = 1
        for j in range(self.depth - 1):
            zlen=c1d(zlen,self.kernel_size,1)
        zlen=c1d(zlen,self.kernel_size*2,2)
        for m in range(self.to_mask):
            z = nn.Conv(features=self.channels, kernel_size=(1,), use_bias=True)(z)
            for i in range(self.depth):
                if i == 0:
                    IZ = z.shape
                    z = ConvblockWithTarget(
                        channels=self.channels,
                        kernel_size=self.kernel_size,
                        norm_factor=self.norm_factor,
                        skip=(i % self.skip_freq) == (self.skip_freq - 1),
                        layernorm=self.layernorm,
                        inner_skip=self.inner_skip,
                    )(z)
                    OZ = z.shape
                    assert IZ[1] == c1d(OZ[1],self.kernel_size*2,2)
                else:
                    IZ = z.shape
                    z = Convblock(
                        channels=self.channels,
                        kernel_size=self.kernel_size,
                        norm_factor=self.norm_factor,
                        skip=(i % self.skip_freq) == (self.skip_freq - 1),
                        layernorm=self.layernorm,
                        inner_skip=self.inner_skip,
                        pad_to_input_size=False
                    )(z)
                    OZ=z.shape
                    assert IZ[1] == c1d(OZ[1],self.kernel_size,1)

            if m != 1:
                z.shape[1] == 1
            z = nn.Conv(features=1, kernel_size=(1,), use_bias=True)(z)
            z = nn.PReLU()(z)
            z = jnp.concatenate([
                foundry[:, -(zlen - 2) :, :],
                x_final[:, m : m + 1, :],
                z[:, -1:, :],
            ], axis=1)
            foundry = z
        return z


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
        return x


if __name__ == "__main__":
    # model = Convattn()
    # print(model.tabulate(jax.random.key(0), jnp.ones((2**2, 2**8, 1))))
    model = ConvAttnFauxLarsen()
    print(model.tabulate(jax.random.key(0), jnp.ones((2**2, 2**8, 1))))