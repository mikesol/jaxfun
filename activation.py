from enum import Enum
import flax.linen as nn
from tcn import PELU


class Activation(Enum):
    TANH = 1
    PRELU = 2
    ELU = 3
    GELU = 4
    LOTS_OF_PRELUS = 5
    PELU = 6
    LOTS_OF_PELUS = 7
    NADA = 8


def make_activation(activation):
    if activation == Activation.TANH:
        return lambda: nn.tanh
    if activation == Activation.PRELU:
        return lambda: nn.PReLU
    if activation == Activation.ELU:
        return lambda: nn.elu
    if activation == Activation.GELU:
        return lambda: nn.gelu
    if activation == Activation.LOTS_OF_PRELUS:
        return nn.vmap(
            nn.PReLU,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            in_axes=-1,
            out_axes=-1,
        )
    if activation == Activation.PELU:
        return lambda: PELU
    if activation == Activation.LOTS_OF_PELUS:
        return nn.vmap(
            PELU,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            in_axes=-1,
            out_axes=-1,
        )
    if activation == Activation.NADA:
        return lambda: lambda x: x
    raise ValueError(f"What function? {activation}")
