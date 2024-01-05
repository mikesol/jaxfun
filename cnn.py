import flax.linen as nn
import jax.numpy as jnp
import jax
from flax.linen import initializers
import math


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

def c1d(o, k, s):
    return (s * (o - 1)) + 1 + (k - 1)
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
