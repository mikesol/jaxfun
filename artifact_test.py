import local_env
from comet_ml import Experiment
import orbax.checkpoint

checkpoint_dir = "/Users/mikesol/devel/jaxfun/model/tcn-attn/checkpoints/0678a6909db14cd09f7e0e9d5d06c83c/flax_ckpt/orbax/managed/"

orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=2, create=True)
checkpoint_manager = orbax.checkpoint.CheckpointManager(
    checkpoint_dir, orbax_checkpointer, options
)
ckpt = checkpoint_manager.restore(896276)