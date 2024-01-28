from enum import Enum
import jax.numpy as jnp
from fade_in import apply_fade_in
import stft
import auraloss_jax as auraloss

class LossFn(Enum):
    LOGCOSH = 1
    ESR = 2
    LOGCOSH_RANGE = 3
    MULTI_STFT = 4


def LogCoshLoss(input, target, a=1.0, eps=1e-8):
    losses = jnp.mean((1 / a) * jnp.log(jnp.cosh(a * (input - target)) + eps), axis=-2)
    losses = jnp.mean(losses)
    return losses


def ESRLoss(input, target):
    eps = 1e-8
    num = jnp.sum(((target - input) ** 2), axis=1)
    denom = jnp.sum(target**2, axis=1) + eps
    losses = num / denom
    losses = jnp.mean(losses)
    return losses


def Loss_fn_to_loss(loss_fn):
    if loss_fn == LossFn.ESR:
        return ESRLoss
    if loss_fn == LossFn.LOGCOSH:
        return LogCoshLoss
    if loss_fn == LossFn.LOGCOSH_RANGE:
        return lambda x, y: LogCoshLoss(apply_fade_in(x), apply_fade_in(y))
    if loss_fn == LossFn.MULTI_STFT:
        return MultiResolutionSTFTLoss
    raise ValueError(f"What function? {loss_fn}")

def MultiResolutionSTFTLoss(x, y):
    div_factor = 2
    fft_sizes = [1024, 2048, 512]
    hop_sizes = [120, 240, 50]
    win_lengths = [600, 1200, 240]
    params = [
        stft.init_stft_params(x // div_factor, y // div_factor, z // div_factor)
        for x, y, z in zip(fft_sizes, hop_sizes, win_lengths)
    ]
    return auraloss.multi_resolution_stft_loss(params, x, y)