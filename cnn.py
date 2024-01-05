import flax.linen as nn
import jax.numpy as jnp
import jax
from flax.linen import initializers
from cnn_attn import Convblock

BatchNorm = nn.BatchNorm

def c1d(o, k, s):
    return (s * (o - 1)) + 1 + (k - 1)


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
            kernel_size=(self.kernel_size,),
            padding=((0,)),
        )(x)
        x = BatchNorm(use_running_average=not train)(x)
        x = nn.gelu(x)
        return x if not self.skip else x_[:, -x.shape[1] :, :] + x


class ConvFauxCell(nn.Module):
    depth: int = 2**4
    channels: int = 2**6
    kernel_size: int = 7
    norm_factor: float = 1.0
    skip_freq: int = 1
    inner_skip: bool = True

    def get_zlen(self):
        zlen = 1
        for _ in range(self.depth - 1):
            zlen = c1d(zlen, self.kernel_size, 1)
        zlen = c1d(zlen, self.kernel_size * 2, 2)
        return zlen

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
            if i == 0:
                z = ConvWithSkip(
                    channels=self.channels,
                    kernel_size=self.kernel_size * 2,
                    stride=2,
                    skip=False,
                )(z, train)
                z = BatchNorm(use_running_average=not train)(z)
                z = nn.gelu(z)
            elif i == 5:
                z = Convblock(
                        channels=self.channels,
                        kernel_size=self.kernel_size,
                        norm_factor=self.norm_factor,
                        skip=(i % self.skip_freq) == (self.skip_freq - 1),
                        inner_skip=self.inner_skip,
                        pad_to_input_size=False,
                        squeeze=2**3
                    )
            else:
                z = ConvWithSkip(
                    channels=self.channels, kernel_size=self.kernel_size, stride=1
                )(z, train)
                z = BatchNorm(use_running_average=not train)(z)
                z = nn.gelu(z)
        if not is_first:
            if not (z.shape[1] == 1):
                raise ValueError(f'Inconsistent shape: in-foundry {foundry_len} out-foundry {foundry.shape} input {ipt.shape} z-in {z_.shape} z-out {z.shape} zlen {zlen} is_first {is_first}')
        z = nn.Conv(
            features=1,
            kernel_size=(1,),
            padding=((0,)),
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            use_bias=True,
            kernel_init=nn.with_partitioning(
                initializers.lecun_normal(), (None, "model")
            ),
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


class ConvFauxLarsen(nn.Module):
    depth: int = 2**4
    channels: int = 2**6
    kernel_size: int = 7
    norm_factor: float = 1.0
    skip_freq: int = 1
    inner_skip: bool = True

    # ugh, code dup
    def get_zlen(self):
        zlen = 1
        for _ in range(self.depth - 1):
            zlen = c1d(zlen, self.kernel_size, 1)
        zlen = c1d(zlen, self.kernel_size * 2, 2)
        return zlen

    def setup(self):
        self.cell = ConvFauxCell(
            depth=self.depth,
            channels=self.channels,
            kernel_size=self.kernel_size,
            norm_factor=self.norm_factor,
            skip_freq=self.skip_freq,
            inner_skip=self.inner_skip,
        )

    def __call__(self, x, train: bool = True, to_mask: int = None):
        if (to_mask >= x.shape[1]) or (type(to_mask) == type((1, 2))):
            # from a bug during training
            raise ValueError(
                f"to_mask must be less than the input sequence length: {x.shape[1]} vs {to_mask}"
            )
        # print("to_mask", to_mask, "x.shape", x.shape)
        x_masked = x[:, : -(to_mask * 2), :]
        x_final = x[:, -(to_mask * 2) :: 2, :]
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


if __name__ == "__main__":
    model = ConvFauxLarsen()
    print(
        model.tabulate(
            jax.random.key(0), jnp.ones((2**2, 2**14, 1)), to_mask=2**5
        )
    )
