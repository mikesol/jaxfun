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


class Sineconv(nn.Module):
    features: int
    sine_window: int
    cropping: Callable

    @nn.compact
    def __call__(self, x, sine_range, phases):
        assert self.sine_window % 2 == 1
        x_ = x
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        in_features = x.shape[2]
        final_seq_len = (seq_len - self.sine_window) + 1
        # as the convolution gets shorter, we can clip the range
        sine_range = sine_range[:seq_len]
        assert (seq_len,) == sine_range.shape
        all_chans = in_features * self.features
        # (b, k*c, s)
        x = jax.lax.conv_general_dilated_patches(
            jnp.transpose(x, (0, 2, 1)),
            filter_shape=(self.sine_window,),
            window_strides=(1,),
            padding=((0, 0),),
        )
        # (1, all_chans * k, sine_range)
        x = jnp.repeat(x, repeats=self.features, axis=1)
        amplitudes = self.param(
            "amplitude",
            initializers.lecun_normal(),
            (all_chans, 1),
            jnp.float32,
        )
        frequencies = self.param(
            "frequency",
            initializers.lecun_normal(),
            (all_chans, 1),
            jnp.float32,
        )
        # something really high
        frequencies *= 19000.0
        phases *= 2 * jnp.pi
        # (1, sine_range)
        sine_expanded = jnp.expand_dims(sine_range, axis=0)

        # (all_chans, sine_range)
        def sine_me(freq, amp, ph):
            return amp * jnp.sin(freq * 2 * jnp.pi * sine_expanded + ph)

        sines = jax.vmap(
            sine_me,
            in_axes=0,
            out_axes=0,
        )(amplitudes, frequencies, phases)
        # for some reason this add an axis?
        sines = jnp.squeeze(sines, axis=1)
        # (1, all_chans, sine_range)
        sines = jnp.expand_dims(sines, axis=0)
        # (1, all_chans * k, sine_range)
        sines = jax.lax.conv_general_dilated_patches(
            sines,
            filter_shape=(self.sine_window,),
            window_strides=(1,),
            padding=((0, 0),),
        )
        conv = x * sines
        conv = jnp.reshape(
            conv,
            (batch_size, self.features, in_features * self.sine_window, final_seq_len),
        )
        conv = jnp.sum(conv, axis=2)
        conv = jnp.transpose(conv, (0, 2, 1))
        x_res = nn.Conv(
            features=self.features,
            kernel_size=(1,),
            feature_group_count=self.features if x_.shape[-1] > 1 else 1,
            use_bias=False,
        )(x_)
        return self.cropping(conv, x_res)


class SineconvNetwork(nn.Module):
    features_list: Tuple[int]
    sine_window: int
    cropping: Callable

    @nn.compact
    def __call__(self, x, sine_range, phases):
        for i, features in enumerate(self.features_list):
            x = Sineconv(
                features=features, sine_window=self.sine_window, cropping=self.cropping
            )(x=x, sine_range=sine_range, phases=phases[i])
        x = nn.Conv(features=1, kernel_size=(1,))(x)
        return x


if __name__ == "__main__":
    import crop

    window = 2**14
    features_list = tuple(2**n for n in [11, 10, 9, 8, 7, 6, 5, 4, 3, 2])
    model = SineconvNetwork(
        features_list=features_list,
        sine_window=2**7 - 1,
        cropping=partial(crop.center_crop_and_f, f=lambda x, y: x + y),
    )
    print(
        model.tabulate(
            jax.random.key(0),
            jnp.ones((2**2, 2**14, 1)),
            jnp.ones((window,)),
            [
                jnp.ones((x * y, 1))
                for x, y in zip((1,) + features_list[:-1], features_list)
            ],
        )
    )
