from enum import Enum


class Parallelism(Enum):
    SHARD = 1
    PMAP = 2
    NONE = 3
