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


def normalize_amps(amps, amps_log_min, amps_log_max, amps_epsilon):
    amps_log = jnp.log(amps + amps_epsilon)  # Logarithmic transformation
    # Scale based on the provided min and max values
    amps_normalized = (amps_log - amps_log_min) / (amps_log_max - amps_log_min) * 2 - 1
    return amps_normalized


def denormalize_amps(amps_normalized, amps_log_min, amps_log_max, amps_epsilon):
    amps_log = (amps_normalized + 1) / 2 * (amps_log_max - amps_log_min) + amps_log_min
    amps = jnp.exp(amps_log) - amps_epsilon
    return amps


def normalize_freqs(freqs, freqs_min, freqs_max):
    freqs_normalized = (freqs - freqs_min) / (freqs_max - freqs_min) * 2 - 1
    return freqs_normalized


def denormalize_freqs(freqs_normalized, freqs_min, freqs_max):
    freqs = (freqs_normalized + 1) / 2 * (freqs_max - freqs_min) + freqs_min
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
    return converted


def lob_padding(o, cc):
    return o[cc.fft_size - cc.hop_size :]


def zero_stuff_out(o, mute_bottom, mute_top, epsilon):
    amps = o[::2]
    freqs = o[1::2]
    amps = jnp.concatenate(
        ([jnp.zeros((mute_bottom,))] if mute_bottom > 0 else [])
        + ([amps[mute_bottom:-mute_top]] if mute_top > 0 else [amps[mute_bottom:]])
        + ([jnp.zeros((mute_top,))] if mute_top > 0 else [])
    )
    amps = jnp.where(amps < epsilon, jnp.zeros_like(amps), amps)
    return jnp.ravel(jnp.column_stack((amps, freqs)))


def do_conversion2(obj, ipt):
    assert len(ipt.shape) == 3
    assert ipt.shape[-1] == 2
    ipt, rnd = ipt[:, :, :1], ipt[:, :, -1:]
    o = do_conversion(obj, ipt)
    o = jax.vmap(
        jax.vmap(
            partial(zero_stuff_out, mute_bottom=0, mute_top=0, epsilon=1e-8),
            in_axes=0,
            out_axes=0,
        ),
        in_axes=0,
        out_axes=0,
    )(o)
    p_inc = 1.0 / obj.sample_rate
    i_inv = 1.0 / obj.hop_size
    batch_size = o.shape[0]
    lastval = np.zeros((batch_size, o.shape[-1] // 2, 2))
    index = np.zeros((batch_size, o.shape[-1] // 2))
    _, o = jax.vmap(
        partial(
            pvc.noscbank,
            nw=obj.window_size,
            p_inc=p_inc,
            i_inv=i_inv,
            rg=jnp.arange(obj.hop_size),
            cell=pvc.noscbank_cell_no_sum,
        ),
        in_axes=0,
        out_axes=0,
    )((lastval, index), o)
    o = jnp.transpose(o, (0, 2, 1, 3))
    o = jnp.reshape(o, (o.shape[0], o.shape[1], -1))
    o = jnp.transpose(o, (0, 2, 1))
    o = jax.vmap(
        partial(lob_padding, cc=obj),
        in_axes=0,
        out_axes=0,
    )(o)
    seq_len = min(o.shape[1], rnd.shape[1])
    full_stack = jnp.concatenate([o[:, :seq_len, :], rnd[:, :seq_len, :]], axis=-1)
    return full_stack


def normalize(ipt, amps_log_min, amps_log_max, amps_epsilon, freqs_min, freqs_max):
    converted_batch = ipt.shape[0]
    converted_seq = ipt.shape[1]
    converted_chan = ipt.shape[2]
    converted_a, converted_f = normalize_amps(
        ipt[:, :, ::2],
        amps_log_max=amps_log_max,
        amps_log_min=amps_log_min,
        amps_epsilon=amps_epsilon,
    ), normalize_freqs(ipt[:, :, 1::2], freqs_max=freqs_max, freqs_min=freqs_min)
    converted = jnp.reshape(
        jnp.stack((converted_a, converted_f), axis=-1),
        (converted_batch, converted_seq, converted_chan),
    )
    # XLC should figure this out anyway
    # but just in case...
    return converted


def denormalize(
    attended, amps_log_min, amps_log_max, amps_epsilon, freqs_min, freqs_max
):
    att_batch = attended.shape[0]
    att_seq = attended.shape[1]
    att_chan = attended.shape[2]
    attended_a, attended_f = denormalize_amps(
        attended[:, :, ::2],
        amps_log_max=amps_log_max,
        amps_log_min=amps_log_min,
        amps_epsilon=amps_epsilon,
    ), denormalize_freqs(attended[:, :, 1::2], freqs_max=freqs_max, freqs_min=freqs_min)
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


class PVCFinal(nn.Module):
    dilation_incr: int
    end_features: int
    conv_depth: int
    attn_depth: int
    kernel_size: int
    heads: int
    expand_factor: float = 2.0

    @nn.compact
    def __call__(self, ipt, train: bool):
        # XLC should figure this out anyway
        # but just in case...
        # network time!
        features = ipt.shape[-1]
        convolved = ipt
        kd = 1
        print('IPT SHAPE', ipt.shape)
        for _ in range(self.conv_depth):
            convolved = TCN(
                features=features,
                kernel_dilation=kd,
                kernel_size=self.kernel_size,
            )(convolved, train=train)
            kd += 8
        reduced = nn.Conv(
            features=self.end_features, kernel_size=(1,), padding=((0, 0),), use_bias=False
        )(convolved)
        encoded = PositionalEncoding()(reduced)
        attended = nn.Sequential(
            [
                AttnBlock(heads=self.heads, expand_factor=self.expand_factor)
                for _ in range(self.attn_depth)
            ]
        )(encoded)

        return nn.Conv(features=1, kernel_size=(1,), padding=((0, 0),), use_bias=False)(
            attended
        )


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
    print(
        model.tabulate(jax.random.key(0), jnp.ones((2**2, 2**14, 1)), train=False)
    )
