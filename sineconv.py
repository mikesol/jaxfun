import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.linen import initializers
from bias_types import BiasTypes
from functools import partial
from create_filtered_audio import create_biquad_coefficients
import math
from typing import Tuple
from typing import (
    Any,
    Callable,
    Optional,
    Tuple,
    TypeVar,
)
import math
from jax import random

A = TypeVar("A")
PRNGKey = jax.Array
A = TypeVar("A")
PRNGKey = jax.Array
Shape = Tuple[int, ...]
Dtype = Any  # this could be a real type?
Array = jax.Array
Carry = Any
CarryHistory = Any
Output = Any
import numpy as np


class Sineblock(nn.Module):
    sine_window: int
    sines_per_window: int

    @nn.compact
    # (batch, ichan * k, seq) (batch, seq, 1) (batch, ichan * spw)
    def __call__(self, x, sine_range, phases):
        # (batch, 1, ichan)
        in_features = x.shape[1] // self.sine_window
        # (batch, spw, ichan)
        phases = jnp.reshape(phases, (x.shape[0], self.sines_per_window, in_features))
        amplitudes = self.param(
            "amplitude",
            initializers.lecun_normal(),
            (1, self.sines_per_window, in_features),
            jnp.float32,
        )
        frequencies = self.param(
            "frequency",
            initializers.lecun_normal(),
            (1, self.sines_per_window, in_features),
            jnp.float32,
        )
        # something really high
        frequencies *= 19000.0
        phases *= 2 * jnp.pi

        # (1, spw) (1, spw) (batch, spw) -> (batch, seq)
        def sine_me(amp_, freq_, ph_):
            # (1, 1, spw) (1, 1, spw) (batch, 1, spw) -> (batch, seq)
            def sine_me_2(amp, freq, ph):
                return amp * jnp.sin(
                    freq * 2 * jnp.pi * jnp.squeeze(sine_range, axis=-1) + ph
                )

            amp_ = jnp.expand_dims(amp_, axis=1)
            freq_ = jnp.expand_dims(freq_, axis=1)
            ph_ = jnp.expand_dims(ph_, axis=1)

            o = jax.vmap(
                sine_me_2,
                in_axes=-1,
                out_axes=-1,
            )(
                amp_,
                freq_,
                ph_,
            )
            return jnp.sum(o, axis=-1)

        # (b, seq, ch)
        sines = jax.vmap(
            sine_me,
            in_axes=-1,
            out_axes=-1,
        )(
            amplitudes,
            frequencies,
            phases,
        )
        # (batch, k * chan, seq)
        sines = jax.lax.conv_general_dilated_patches(
            jnp.transpose(sines, (0, 2, 1)),
            filter_shape=(self.sine_window,),
            window_strides=(1,),
            padding=((0, 0),),
        )
        # (batch, k * chan, seq)
        conv = x * sines
        # (batch, seq)
        conv = jnp.sum(conv, axis=1)
        conv = nn.tanh(conv)
        return conv


class Sineconv(nn.Module):
    features: int
    sine_window: int
    sines_per_window: int
    cropping: Callable

    # (batch, seq, ichan) (batch, seq, 1) (batch, ichan * spw, ochan)
    @nn.compact
    def __call__(self, x, sine_range, phases):
        assert self.sine_window % 2 == 1
        x_ = x
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        in_features = x.shape[2]
        final_seq_len = (seq_len - self.sine_window) + 1
        # as the convolution gets shorter, we can clip the range
        sine_range = sine_range[:, :seq_len, :]
        assert (x_.shape[0], seq_len, 1) == sine_range.shape
        all_chans = in_features * self.features
        # (b, k*c, s)
        x = jax.lax.conv_general_dilated_patches(
            jnp.transpose(x, (0, 2, 1)),
            filter_shape=(self.sine_window,),
            window_strides=(1,),
            padding=((0, 0),),
        )

        conv = nn.vmap(
            lambda m, ph: m(x, sine_range, ph),
            variable_axes={"params": 0},
            split_rngs={"params": True},
            in_axes=2,
            out_axes=2,
        )(
            Sineblock(
                sine_window=self.sine_window, sines_per_window=self.sines_per_window
            ),
            phases,
        )

        x_res = nn.Conv(
            features=self.features,
            kernel_size=(1,),
            feature_group_count=self.features if x_.shape[-1] > 1 else 1,
            use_bias=False,
        )(x_)
        cropped = self.cropping(conv, x_res)
        return cropped


class SineconvNetwork(nn.Module):
    features_list: Tuple[int]
    sine_window: int
    sines_per_window: int
    cropping: Callable

    @nn.compact
    def __call__(self, x, sine_range, phases):
        for i, features in enumerate(self.features_list):
            x = Sineconv(
                features=features,
                sine_window=self.sine_window,
                sines_per_window=self.sines_per_window,
                cropping=self.cropping,
            )(x=x, sine_range=sine_range, phases=phases[i])
        x = nn.Conv(features=1, kernel_size=(1,))(x)
        return x


if __name__ == "__main__":
    import crop

    window = 2**14
    features_list = tuple(2**n for n in [11, 10, 9, 8, 7, 6, 5, 4, 3, 2])
    sines_per_window = 32
    model = SineconvNetwork(
        features_list=features_list,
        sine_window=2**7 - 1,
        sines_per_window=sines_per_window,
        cropping=partial(crop.center_crop_and_f, f=lambda x, y: x + y),
    )
    batch = 2**2
    print(
        model.tabulate(
            jax.random.key(0),
            jnp.ones((batch, 2**14, 1)),
            jnp.ones((batch, window, 1)),
            [
                jnp.ones((batch, x * sines_per_window, y))
                for x, y in zip((1,) + features_list[:-1], features_list)
            ],
        )
    )
