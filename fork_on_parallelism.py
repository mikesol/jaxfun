import local_env
from parallelism import Parallelism
def fork_on_parallelism(a, b):
    return a if local_env.parallelism == Parallelism.SHARD else b