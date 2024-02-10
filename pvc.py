import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.linen import initializers
from functools import partial
from bias_types import BiasTypes
from create_filtered_audio import create_biquad_coefficients
import math
from typing import Tuple
import fouriax.pvc as pvc
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

amps_log_min = jnp.log(1e-20)
amps_log_max = jnp.log(1e-3)
epsilon = 1e-20


def normalize_amps(amps):
    amps_log = jnp.log(amps + epsilon)  # Logarithmic transformation
    # Scale to [0, 1] or [-1, 1] here based on the min-max of the transformed amps
    # Assume amps_log_min and amps_log_max are the global min and max values you've determined
    amps_normalized = (amps_log - amps_log_min) / (
        amps_log_max - amps_log_min
    )  # Example for [0, 1]
    return amps_normalized


def denormalize_amps(amps_normalized):
    amps_log = amps_normalized * (amps_log_max - amps_log_min) + amps_log_min
    amps = jnp.exp(amps_log) - epsilon  # Subtract epsilon if added during normalization
    return amps


def normalize_freqs(freqs, sample_rate):
    freqs_normalized = freqs / (sample_rate / 4) - 1
    return freqs_normalized


def denormalize_freqs(freqs_normalized, sample_rate):
    freqs = (freqs_normalized + 1) * (sample_rate / 4)
    return freqs


class TCN(nn.Module):
    features: int
    kernel_dilation: int
    kernel_size: int
    activation: callable = nn.gelu
    bias_type: BiasTypes = BiasTypes.BATCH_NORM

    @nn.compact
    def __call__(self, x, train: bool):
        x_ = x
        x = nn.Conv(
            features=self.features,
            kernel_init=(
                nn.with_partitioning(initializers.lecun_normal(), (None, "model"))
                if x.shape[-1] > 1
                else initializers.lecun_normal()
            ),
            kernel_dilation=(self.kernel_dilation,),
            kernel_size=(self.kernel_size,),
            padding=((0, 0),),
            use_bias=self.bias_type == BiasTypes.DC,
        )(x_)
        if self.bias_type == BiasTypes.BATCH_NORM:
            x = nn.BatchNorm(use_running_average=not train)(x)
        x = self.activation(x)
        x_res = nn.Conv(
            features=self.features,
            kernel_size=(1,),
            # no kernel init as we're going down to 1
            feature_group_count=self.features if x_.shape[-1] > 1 else 1,
            use_bias=False,
        )(x_)
        x = x + x_res[:, -x.shape[-2] :, :]
        return x


class AttnBlock(nn.Module):
    heads: int
    expand_factor: float = 2.0
    activation: Callable = lambda: nn.gelu

    @nn.compact
    def __call__(self, out):
        features = out.shape[-1]
        out_ = out
        out = nn.MultiHeadDotProductAttention(
            self.heads,
            kernel_init=nn.with_partitioning(
                initializers.lecun_normal(), (None, "model")
            ),
        )(out, out)
        out = nn.LayerNorm()(out_ + out)
        out_ = out
        out = nn.Dense(
            features=int(out.shape[-1] * self.expand_factor),
            kernel_init=nn.with_partitioning(
                initializers.lecun_normal(), (None, "model")
            ),
            use_bias=True,
        )(out)
        out = self.activation()(out)
        out = nn.Dense(
            features=features,
            kernel_init=nn.with_partitioning(
                initializers.lecun_normal(), (None, "model")
            ),
            use_bias=True,
        )(out)
        return nn.LayerNorm()(out_ + out)


class PositionalEncoding(nn.Module):
    @nn.compact
    def __call__(self, x):
        seq_length, d_model = x.shape[-2], x.shape[-1]

        position = jnp.arange(seq_length)[:, jnp.newaxis]
        div_term = jnp.exp(jnp.arange(0, d_model, 2) * -(jnp.log(10000.0) / d_model))
        pos_encoding = jnp.zeros((seq_length, d_model))
        pos_encoding = pos_encoding.at[:, 0::2].set(jnp.sin(position * div_term))
        pos_encoding = pos_encoding.at[:, 1::2].set(jnp.cos(position * div_term))

        x += pos_encoding

        return x


def do_conversion(obj, ipt):
    bitreverse_indices = pvc.precompute_bitreverse_indices(obj.fft_size // 2)
    c = pvc.precompute_cfkt_constants(obj.fft_size // 2)
    ws = pvc.precompute_rfkt_constants(obj.fft_size // 2)
    patches = pvc.make_pvc_patches(ipt, obj.hop_size, obj.window_size)
    window = jnp.ones((obj.window_size,))
    window = pvc.koonce_sinc(window, obj.fft_size, obj.window_size)
    window = pvc.koonce_normalization(window)
    folded = pvc.fold_pvc_patches(
        patches, window, obj.fft_size, obj.hop_size, obj.window_size
    )
    fktd = jax.vmap(
        jax.vmap(
            partial(pvc.rfkt, bitreverse_indices=bitreverse_indices, c=c, ws=ws),
            in_axes=0,
        ),
        in_axes=0,
    )(folded)
    fkt_batch = fktd.shape[0]
    fkt_seq = fktd.shape[1]
    fkt_chan = fktd.shape[2]
    fktd = jnp.reshape(fktd, (fkt_batch, fkt_seq, fkt_chan // 2, 2))
    f_r = fktd[..., 0]
    f_i = fktd[..., 1]
    c_r, c_i = pvc.convert_stft_to_amp_and_freq_using_0_phase(
        f_r, f_i, obj.hop_size, obj.sample_rate
    )
    converted = jnp.reshape(
        jnp.stack((c_r, c_i), axis=-1), (fkt_batch, fkt_seq, fkt_chan)
    )
    # XLC should figure this out anyway
    # but just in case...
    converted = jax.lax.stop_gradient(converted)
    return converted


def normalize(ipt, sample_rate):
    converted_batch = ipt.shape[0]
    converted_seq = ipt.shape[1]
    converted_chan = ipt.shape[2]
    converted_a, converted_f = normalize_amps(ipt[:, :, ::2]), normalize_freqs(
        ipt[:, :, 1::2], sample_rate
    )
    converted = jnp.reshape(
        jnp.stack((converted_a, converted_f), axis=-1),
        (converted_batch, converted_seq, converted_chan),
    )
    # XLC should figure this out anyway
    # but just in case...
    converted = jax.lax.stop_gradient(converted)
    return converted


def denormalize(attended, sample_rate):
    att_batch = attended.shape[0]
    att_seq = attended.shape[1]
    att_chan = attended.shape[2]
    attended_a, attended_f = denormalize_amps(attended[:, :, ::2]), denormalize_freqs(
        attended[:, :, 1::2], sample_rate
    )
    attended = jnp.reshape(
        jnp.stack((attended_a, attended_f), axis=-1),
        (att_batch, att_seq, att_chan),
    )
    return attended


class PVC(nn.Module):
    fft_size: int
    hop_size: int
    window_size: int
    sample_rate: int
    kernel_size: int
    n_phasors: int
    conv_depth: int
    attn_depth: int
    heads: int
    expand_factor: float = 2.0

    @nn.compact
    def __call__(self, ipt, train: bool):
        # XLC should figure this out anyway
        # but just in case...
        # network time!
        features = self.n_phasors * 2
        convolved = ipt
        for _ in range(self.conv_depth):
            convolved = TCN(
                features=features, kernel_dilation=1, kernel_size=self.kernel_size
            )(convolved, train=train)
        encoded = PositionalEncoding()(convolved)
        attended = nn.Sequential(
            [
                AttnBlock(heads=self.heads, expand_factor=self.expand_factor)
                for _ in range(self.attn_depth)
            ]
        )(encoded)

        return attended


if __name__ == "__main__":
    model = PVC(
        fft_size=1024,
        hop_size=128,
        window_size=2048,
        sample_rate=44100,
        kernel_size=7,
        n_phasors=512,
        conv_depth=8,
        attn_depth=8,
        heads=32,
        expand_factor=2.0,
    )
    print(model.tabulate(jax.random.key(0), jnp.ones((2**2, 2**14, 1)), train=False))
