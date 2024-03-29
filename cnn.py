import flax.linen as nn
import jax.numpy as jnp
import jax
import math
from typing import Tuple
from flax.linen import initializers
from cnn_attn import ConvblockNofrills
from fork_on_parallelism import fork_on_parallelism

BatchNorm = nn.BatchNorm
maybe_partition = fork_on_parallelism(
    lambda x, y: nn.with_partitioning(x, y), lambda x, _: x
)


def c1d(o, k, s, d):
    return (s * (o - 1)) + 1 + (d * (k - 1))


class ConvBase(nn.Module):
    depth: int = 2**4
    kernel_size: int = 7
    dilation_layers: Tuple[int] = ()

    def get_zlen(self):
        zlen = 1
        for l in range(self.depth - 1):
            lnum = self.depth - 1 - l
            zlen = c1d(
                zlen, self.kernel_size, 1, 2 if lnum in self.dilation_layers else 1
            )
        # no dilation on the final bloc
        zlen = c1d(zlen, self.kernel_size * 2, 2, 1)
        return zlen


class ConvFauxCell(ConvBase):
    depth: int = 2**4
    channels: int = 2**6
    kernel_size: int = 7
    norm_factor: float = 1.0
    skip_freq: int = 1
    inner_skip: bool = True
    sidechain_layers: Tuple[int] = ()
    dilation_layers: Tuple[int] = ()

    @nn.compact
    def __call__(self, foundry, ipt, is_first=True, train: bool = True):
        foundry_len = foundry.shape[1]
        zlen = self.get_zlen()
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
        z_ = z
        z = nn.Conv(
            features=self.channels,
            padding=((0,)),
            kernel_size=(1,),
            dtype=jnp.float32,
            param_dtype=jnp.float32,
        )(z)
        z = BatchNorm(use_running_average=not train)(z)
        z = nn.gelu(z)
        for i in range(self.depth):
            z_ = z
            z = nn.Conv(
                features=self.channels,
                strides=(2 if i == 0 else 1,),
                kernel_size=(self.kernel_size * 2 if i == 0 else self.kernel_size,),
                kernel_dilation=(2 if i in self.dilation_layers else 1,),
                padding=((0,)),
            )(z)
            if i in self.sidechain_layers:
                z += ConvblockNofrills(
                    channels=self.channels,
                    kernel_size=self.kernel_size,
                    norm_factor=math.sqrt(self.channels),
                    squeeze=1,
                )(z_)

            z = BatchNorm(use_running_average=not train)(z)
            z = nn.gelu(z)
            z = z if not (i % self.skip_freq) == 0 else z_[:, -z.shape[1] :, :] + z
        if not is_first:
            if not (z.shape[1] == 1):
                raise ValueError(
                    f"Inconsistent shape: in-foundry {foundry_len} out-foundry {foundry.shape} input {ipt.shape} z-in {z_.shape} z-out {z.shape} zlen {zlen} is_first {is_first}"
                )
        z = nn.Conv(
            features=1,
            kernel_size=(1,),
            padding=((0,)),
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            use_bias=True,
            kernel_init=maybe_partition(initializers.lecun_normal(), (None, "model")),
        )(z)
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


class ConvFauxLarsen(ConvBase):
    depth: int = 2**4
    channels: int = 2**6
    kernel_size: int = 7
    norm_factor: float = 1.0
    skip_freq: int = 1
    inner_skip: bool = True
    sidechain_layers: Tuple[int] = ()
    dilation_layers: Tuple[int] = ()

    def setup(self):
        self.cell = ConvFauxCell(
            depth=self.depth,
            channels=self.channels,
            kernel_size=self.kernel_size,
            norm_factor=self.norm_factor,
            skip_freq=self.skip_freq,
            inner_skip=self.inner_skip,
            sidechain_layers=self.sidechain_layers,
            dilation_layers=self.dilation_layers,
        )

    def __call__(self, x, train: bool = True, to_mask: int = None):
        if (to_mask > x.shape[1]) or (type(to_mask) == type((1, 2))):
            # from a bug during training
            raise ValueError(
                f"to_mask must be less than the input sequence length: {x.shape[1]} vs {to_mask}"
            )
        # print("to_mask", to_mask, "x.shape", x.shape)
        x_masked = x[:, : -(to_mask * 2), :] if to_mask > 0 else x
        x_final = x[:, -(to_mask * 2) :: 2, :] if to_mask > 0 else []
        foundry = x_masked
        if foundry.shape[1] < self.get_zlen():
            raise ValueError(
                f"Input sequence length {foundry.shape[1]} is too short for the model: {self.get_zlen()} {to_mask} {x.shape[1]}"
            )
        z = x_masked
        z0 = None
        if to_mask < x.shape[1]:
            foundry, z0 = self.cell(foundry, z, is_first=True, train=train)

            if to_mask <= 0:
                return z0

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


if __name__ == "__main__":
    model = ConvFauxLarsen()
    print(
        model.tabulate(
            jax.random.key(0), jnp.ones((2**2, 2**14, 1)), to_mask=2**5
        )
    )
