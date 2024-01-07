import flax.linen as nn
import jax.numpy as jnp
import jax
from fork_on_parallelism import fork_on_parallelism
from flax.linen import initializers
maybe_partition = fork_on_parallelism(
    lambda x, y: nn.with_partitioning(x, y), lambda x, _: x
)


class ConvblockNofrills(nn.Module):
    channels: int = 2**6
    kernel_size: int = 7
    norm_factor: float = 1.0
    squeeze: int = 1

    @nn.compact
    def __call__(self, x):
        batch_size = x.shape[0]
        weights = self.param(
            "weights",
            maybe_partition(initializers.lecun_normal(), (None, "model")),
            (self.channels // self.squeeze, self.channels, self.kernel_size),
            jnp.float32,
        )
        # skip
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
        x = x * jnp.repeat(w, repeats=self.squeeze, axis=2)
        x = jnp.sum(x, axis=1)
        x = jnp.transpose(x, (0, 2, 1))
        return x


class Convblock(nn.Module):
    channels: int = 2**6
    kernel_size: int = 7
    norm_factor: float = 1.0
    skip: bool = True
    inner_skip: bool = True
    squeeze: int = 1
    pad_to_input_size: bool = True

    @nn.compact
    def __call__(self, x):
        batch_size = x.shape[0]
        if x.shape[-1] != self.channels:
            x = nn.Conv(
                features=self.channels,
                kernel_size=(1,),
                padding=((0,)),
                use_bias=True,
                dtype=jnp.float32,
                param_dtype=jnp.float32,
                kernel_init=maybe_partition(
                    initializers.lecun_normal(), (None, "model")
                ),
            )(x)
            x = nn.gelu(x)
        weights = self.param(
            "weights",
            maybe_partition(initializers.lecun_normal(), (None, "model")),
            (self.channels // self.squeeze, self.channels, self.kernel_size),
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
                else ((0, 0),),
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
        x = x * jnp.repeat(w, repeats=self.squeeze, axis=2)
        x = jnp.sum(x, axis=1)
        x = jnp.transpose(x, (0, 2, 1))
        x = x_[:, (-x.shape[1]) :, :] + x if self.skip and self.inner_skip else x
        x = nn.LayerNorm()(x)
        x_ = x
        x = nn.Conv(
            features=self.channels,
            kernel_size=(1,),
            padding=((0,)),
            use_bias=True,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            kernel_init=maybe_partition(
                initializers.lecun_normal(), (None, "model")
            ),
        )(x)
        x = nn.gelu(x)
        return x_ + x if self.skip else x


class ConvblockWithTarget(nn.Module):
    channels: int = 2**6
    kernel_size: int = 7
    norm_factor: float = 1.0
    skip: bool = True
    inner_skip: bool = True

    @nn.compact
    def __call__(self, x):
        batch_size = x.shape[0]
        if x.shape[-1] != self.channels:
            x = nn.Conv(
                features=self.channels,
                padding=((0,)),
                kernel_size=(1,),
                use_bias=True,
                dtype=jnp.float32,
                param_dtype=jnp.float32,
                kernel_init=maybe_partition(
                    initializers.lecun_normal(), (None, "model")
                ),
            )(x)
            x = nn.gelu(x)
        weights = self.param(
            "weights",
            maybe_partition(initializers.lecun_normal(), (None, "model")),
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
        w = jnp.einsum("abc,dcg->adbg", x_[:, -(x.shape[2] * 2) :, :], weights)
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
        x = (
            x_[:, ::2, :][:, (-x.shape[1]) :, :] + x
            if self.skip and self.inner_skip
            else x
        )
        x = nn.LayerNorm()(x)
        x_ = x
        x = nn.Conv(
            features=self.channels,
            padding=((0,)),
            kernel_size=(1,),
            use_bias=True,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            kernel_init=maybe_partition(
                initializers.lecun_normal(), (None, "model")
            ),
            # bias_init=maybe_partition(initializers.zeros_init(), (None, "model")),
        )(x)
        x = nn.gelu(x)
        return x_ + x if self.skip else x


def c1d(o, k, s):
    return (s * (o - 1)) + 1 + (k - 1)


class ConvAttnFauxCell(nn.Module):
    to_mask: int = 4
    depth: int = 2**4
    channels: int = 2**6
    kernel_size: int = 7
    norm_factor: float = 1.0
    skip_freq: int = 1
    inner_skip: bool = True

    def setup(self):
        self.start = nn.Conv(
            features=self.channels,
            kernel_size=(1,),
            padding=((0,)),
            use_bias=True,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            # don't shard as it is going from 1 to 32
            # kernel_init=maybe_partition(initializers.lecun_normal(), (None, "model")),
            # bias_init=maybe_partition(initializers.zeros_init(), (None, "model")),
        )
        layers = []
        for i in range(self.depth):
            if i == 0:
                layers.append(
                    ConvblockWithTarget(
                        channels=self.channels,
                        kernel_size=self.kernel_size,
                        norm_factor=self.norm_factor,
                        skip=(i % self.skip_freq) == (self.skip_freq - 1),
                        inner_skip=self.inner_skip,
                    )
                )
            else:
                layers.append(
                    Convblock(
                        channels=self.channels,
                        kernel_size=self.kernel_size,
                        norm_factor=self.norm_factor,
                        skip=(i % self.skip_freq) == (self.skip_freq - 1),
                        inner_skip=self.inner_skip,
                        pad_to_input_size=False,
                    )
                )
        self.layers = nn.Sequential(layers)
        self.end = nn.Conv(
            features=1,
            kernel_size=(1,),
            padding=((0,)),
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            use_bias=True,
            kernel_init=maybe_partition(
                initializers.lecun_normal(), (None, "model")
            ),
            # bias_init=maybe_partition(initializers.zeros_init(), (None, "model")),
        )

    def __call__(self, foundry, ipt, is_first: bool = True):
        foundry_len = foundry.shape[1]
        zlen = 1
        for _ in range(self.depth - 1):
            zlen = c1d(zlen, self.kernel_size, 1)
        zlen = c1d(zlen, self.kernel_size * 2, 2)
        z = None
        if not is_first:
            # the input x needs to be interleaved into the foundry
            foundry = jnp.concatenate(
                [
                    foundry[:, :-1, :],
                    jnp.expand_dims(ipt, axis=1),
                    foundry[:, -1:, :],
                ],
                axis=1,
            )
            z = foundry[:, -zlen:, :]
        else:
            z = ipt
        z = self.start(z)
        z = nn.gelu(z)
        z = self.layers(z)
        if not is_first:
            assert z.shape[1] == 1
        z = self.end(z)
        # no activation at the end
        foundry = jnp.concatenate(
            [
                foundry,
                # we tack z onto the end of the foundry
                z[:, -1:, :],
            ],
            axis=1,
        )
        if not is_first:
            z = jnp.squeeze(z, axis=1)
        return foundry[:, -foundry_len:, :], z


class ConvAttnFauxLarsen(nn.Module):
    to_mask: int = 4
    depth: int = 2**4
    channels: int = 2**6
    kernel_size: int = 7
    norm_factor: float = 1.0
    skip_freq: int = 1
    inner_skip: bool = True

    def setup(self):
        self.cell = ConvAttnFauxCell(
            to_mask=self.to_mask,
            depth=self.depth,
            channels=self.channels,
            kernel_size=self.kernel_size,
            norm_factor=self.norm_factor,
            skip_freq=self.skip_freq,
            inner_skip=self.inner_skip,
        )

    def __call__(self, x):
        x_masked = x[:, : -(self.to_mask * 2), :]
        x_final = x[:, -(self.to_mask * 2) :: 2, :]
        foundry = x_masked
        z = x_masked
        foundry, z0 = self.cell(foundry, z, is_first=True)

        def body_fn(cell, carry, x):
            carry, y = cell(carry, x, is_first=False)
            return carry, y

        foundry, z1 = nn.scan(
            body_fn,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=1,
            out_axes=1,
        )(self.cell, foundry, x_final)
        return jnp.concatenate([z0, z1], axis=1)


class Convattn(nn.Module):
    channels: int = 2**6
    depth: int = 2**4
    kernel_size: int = 7
    skip_freq: int = 1
    norm_factor: float = 1.0
    inner_skip: bool = True

    @nn.compact
    def __call__(self, x):
        for i in range(self.depth):
            x = Convblock(
                channels=self.channels,
                kernel_size=self.kernel_size,
                norm_factor=self.norm_factor,
                skip=(i % self.skip_freq) == (self.skip_freq - 1),
                inner_skip=self.inner_skip,
            )(x)
        x = nn.Conv(
            features=1,
            kernel_size=(1,),
            padding=((0,)),
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            use_bias=True,
            kernel_init=maybe_partition(
                initializers.lecun_normal(), (None, "model")
            ),
        )(x)
        return x


if __name__ == "__main__":
    model = ConvAttnFauxLarsen(to_mask=2**5)
    print(model.tabulate(jax.random.key(0), jnp.ones((2**2, 2**14, 1))))
