import flax.linen as nn
import jax.numpy as jnp
import jax
from flax.linen import initializers
import math


class Convblock(nn.Module):
    channels: int = 2**6
    kernel_size: int = 7
    norm_factor: float = 1.0
    skip: bool = True
    inner_skip: bool = True
    pad_to_input_size: bool = True

    @nn.compact
    def __call__(self, x, train: bool = True):
        batch_size = x.shape[0]
        if x.shape[-1] != self.channels:
            x = nn.Conv(
                features=self.channels,
                kernel_size=(1,),
                padding=((0,)),
                dtype=jnp.float32,
                param_dtype=jnp.float32,
                kernel_init=nn.with_partitioning(
                    initializers.lecun_normal(), (None, "model")
                ),
            )(x)
            x = nn.BatchNorm(use_running_average=not train)(x)
            x = nn.gelu(x)
        weights = self.param(
            "weights",
            nn.with_partitioning(initializers.lecun_normal(), (None, "model")),
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
                else ((0, 0),),
            )
            x = jnp.transpose(
                jnp.reshape(x, (batch_size, self.channels, self.kernel_size, -1)),
                (0, 2, 1, 3),
            )
            return x

        x = do_unfold(x)

        # weights == (b, k, c, s)
        # x = (b, s, c) weights = (c, c, k) w' = (b, c, s, k) w = (b, k, c, s)
        w = jnp.transpose(
            jnp.einsum("abc,dcg->adbg", x_[:, -x.shape[3] :, :], weights), (0, 3, 1, 2)
        )
        w = nn.tanh(w / self.norm_factor)
        x = x * w
        x = jnp.sum(x, axis=1)
        x = jnp.transpose(x, (0, 2, 1))
        x = x_[:, (-x.shape[1]) :, :] + x if self.skip and self.inner_skip else x
        x = nn.BatchNorm(use_running_average=not train)(x)
        x_ = x
        x = nn.Conv(
            features=self.channels,
            kernel_size=(1,),
            padding=((0,)),
            use_bias=False,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            kernel_init=nn.with_partitioning(
                initializers.lecun_normal(), (None, "model")
            ),
        )(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.gelu(x)
        return x_ + x if self.skip else x


class ConvblockWithTarget(nn.Module):
    channels: int = 2**6
    kernel_size: int = 7
    norm_factor: float = 1.0
    skip: bool = True
    inner_skip: bool = True

    @nn.compact
    def __call__(self, x, train: bool = True):
        batch_size = x.shape[0]
        if x.shape[-1] != self.channels:
            x = nn.Conv(
                features=self.channels,
                padding=((0,)),
                kernel_size=(1,),
                use_bias=False,
                dtype=jnp.float32,
                param_dtype=jnp.float32,
                kernel_init=nn.with_partitioning(
                    initializers.lecun_normal(), (None, "model")
                ),
            )(x)
            x = nn.BatchNorm(use_running_average=not train)(x)
            x = nn.gelu(x)
        weights = self.param(
            "weights",
            nn.with_partitioning(initializers.lecun_normal(), (None, "model")),
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
        x = nn.BatchNorm(use_running_average=not train)(x)
        x_ = x
        x = nn.Conv(
            features=self.channels,
            padding=((0,)),
            kernel_size=(1,),
            use_bias=False,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            kernel_init=nn.with_partitioning(
                initializers.lecun_normal(), (None, "model")
            ),
        )(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
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

    @nn.compact
    def __call__(self, foundry, ipt, is_first: bool = True, train: bool = True):
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
        z = nn.Conv(
            features=self.channels,
            kernel_size=(1,),
            padding=((0,)),
            use_bias=False,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
        )(z)
        z = nn.BatchNorm(use_running_average=not train)(z)
        z = nn.gelu(z)
        z = self.layers(z)
        if not is_first:
            assert z.shape[1] == 1
        z = nn.Conv(
            features=1,
            kernel_size=(1,),
            padding=((0,)),
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            use_bias=False,
            kernel_init=nn.with_partitioning(
                initializers.lecun_normal(), (None, "model")
            ),
        )(z)
        z = nn.BatchNorm(use_running_average=not train)(z)
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


class ConvWithSkip(nn.Module):
    channels: int = 2**6
    stride: int = 2
    kernel_size: int = 7
    skip: bool = True

    @nn.compact
    def __call__(self, x, train: bool = True):
        x_ = x
        x = nn.Conv(
            features=self.channels,
            strides=(self.stride,),
            use_bias=False,
            kernel_size=(self.kernel_size,),
            padding=((0,)),
        )(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.gelu(x)
        return (
            x
            if not self.skip
            else nn.Conv(
                features=self.channels,
                feature_group_count=min(x.shape[-1], x_.shape[-1]),
                strides=(1,),
                use_bias=False,
                kernel_size=(1,),
                padding=((0,)),
            )(x_[:, -x.shape[1] :, :])
            + x
        )


class ConvFauxCell(nn.Module):
    to_mask: int = 4
    depth: int = 2**4
    channels: int = 2**6
    kernel_size: int = 7
    skip_freq: int = 1
    inner_skip: bool = True

    @nn.compact
    def __call__(self, foundry, ipt, is_first=True, train: bool = True):
        foundry_len = foundry.shape[1]
        zlen = 1
        for _ in range(
            (self.depth if type(self.depth) == int else len(self.depth)) - 1
        ):
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
        ###
        z = nn.Conv(
            features=self.channels if self.channels is not None else self.depth[-1],
            padding=((0,)),
            kernel_size=(1,),
            use_bias=False,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
        )(z)
        z = nn.BatchNorm(use_running_average=not train)(z)
        z = nn.gelu(z)
        for i in range(self.depth if type(self.depth) == int else len(self.depth)):
            if i == 0:
                z = ConvWithSkip(
                    channels=self.channels
                    if self.channels is not None
                    else self.depth[0],
                    kernel_size=self.kernel_size * 2,
                    stride=2,
                    skip=False,
                )(z, train)
                z = nn.BatchNorm(use_running_average=not train)(z)
                z = nn.gelu(z)
            # elif i == 7:
            #     z = Convblock(
            #         channels=self.channels,
            #         kernel_size=self.kernel_size,
            #         norm_factor=self.norm_factor,
            #         skip=(i % self.skip_freq) == (self.skip_freq - 1),
            #         inner_skip=self.inner_skip,
            #         pad_to_input_size=False,
            #     )(z, train)
            else:
                z = ConvWithSkip(
                    channels=self.channels
                    if self.channels is not None
                    else self.depth[i],
                    kernel_size=self.kernel_size,
                    stride=1,
                )(z, train)
                z = nn.BatchNorm(use_running_average=not train)(z)
                z = nn.gelu(z)
        if not is_first:
            assert z.shape[1] == 1
        z = nn.Conv(
            features=1,
            kernel_size=(1,),
            padding=((0,)),
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            use_bias=False,
            kernel_init=nn.with_partitioning(
                initializers.lecun_normal(), (None, "model")
            ),
        )(z)
        z = nn.BatchNorm(use_running_average=not train)(z)
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

    def __call__(self, x, train: bool = True):
        x_masked = x[:, : -(self.to_mask * 2), :]
        x_final = x[:, -(self.to_mask * 2) :: 2, :]
        foundry = x_masked
        z = x_masked
        foundry, z0 = self.cell(foundry, z, is_first=True, train=train)

        def body_fn(cell, carry, x):
            carry, y = cell(carry, x, is_first=False, train=train)
            return carry, y

        foundry, z1 = nn.scan(
            body_fn,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=1,
            out_axes=1,
        )(self.cell, foundry, x_final)
        return jnp.concatenate([z0, z1], axis=1)


class ConvFauxLarsen(nn.Module):
    to_mask: int = 4
    depth: int = 2**4
    channels: int = 2**6
    kernel_size: int = 7
    skip_freq: int = 1
    inner_skip: bool = True

    def setup(self):
        self.cell = ConvFauxCell(
            to_mask=self.to_mask,
            depth=self.depth,
            channels=self.channels,
            kernel_size=self.kernel_size,
            skip_freq=self.skip_freq,
            inner_skip=self.inner_skip,
        )

    def __call__(self, x, train: bool = True):
        if (self.to_mask >= x.shape[1]) or (type(self.to_mask) == type((1, 2))):
            # from a bug during training
            raise ValueError(
                f"to_mask must be less than the input sequence length: {x.shape[1]} vs {self.to_mask}"
            )
        x_masked = x[:, : -(self.to_mask * 2), :]
        x_final = x[:, -(self.to_mask * 2) :: 2, :]
        foundry = x_masked
        z = x_masked
        foundry, z0 = self.cell(foundry, z, is_first=True, train=train)

        def body_fn(cell, carry, x):
            carry, y = cell(carry, x, is_first=False, train=train)
            return carry, y

        is_initializing = "batch_stats" not in self.variables

        z1 = None
        if is_initializing:
            foundry, z1 = body_fn(self.cell, foundry, x_final)
        else:
            foundry, z1 = nn.scan(
                body_fn,
                variable_carry="batch_stats",
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
    def __call__(self, x, train: bool = True):
        for i in range(self.depth):
            x = Convblock(
                channels=self.channels,
                kernel_size=self.kernel_size,
                norm_factor=self.norm_factor,
                skip=(i % self.skip_freq) == (self.skip_freq - 1),
                inner_skip=self.inner_skip,
            )(x, train)
        x = nn.Conv(
            features=1,
            kernel_size=(1,),
            padding=((0,)),
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            kernel_init=nn.with_partitioning(
                initializers.lecun_normal(), (None, "model")
            ),
        )(x)
        return x


if __name__ == "__main__":
    # model = Convattn()
    # print(model.tabulate(jax.random.key(0), jnp.ones((2**2, 2**8, 1))))
    model = ConvAttnFauxLarsen(to_mask=2**5)
    print(model.tabulate(jax.random.key(0), jnp.ones((2**2, 2**14, 1))))
    # model = ConvFauxLarsen(to_mask=2**5)
    # print(model.tabulate(jax.random.key(0), jnp.ones((2**2, 2**14, 1))))
