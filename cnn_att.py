def c1d(o, k, s):
    return (s * (o - 1)) + 1 + (k - 1)




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
