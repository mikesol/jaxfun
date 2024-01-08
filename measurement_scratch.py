from types import SimpleNamespace
from cnn import ConvFauxLarsen
import jax.numpy as jnp
import math
import jax

if __name__ == "__main__":
    _config = {}
    # cnn
    _config["seed"] = 42
    _config["inference_artifacts_per_batch_per_epoch"] = 2**2
    _config["batch_size"] = 2**5
    _config["validation_split"] = 0.2
    _config["learning_rate"] = 1e-4
    _config["epochs"] = 2**7
    _config["window"] = 2**11
    _config["inference_window"] = 2**17
    _config["stride"] = 2**8
    _config["step_freq"] = 100
    _config["test_size"] = 0.1
    _config["channels"] = 2**8 # 2**6
    _config["depth"] = 2**5 # 2**4
    _config["sidechain_layers"] = tuple([x for x in range(2, _config["depth"], 2)])
    _config["dilation_layers"] = tuple([x for x in range(1, _config["depth"], 2)])
    _config["do_progressive_masking"] = False
    _config["to_mask"] = 0
    _config["comparable_field"] = None  # _config["to_mask"] // 2
    _config["kernel_size"] = 7
    _config["skip_freq"] = 1
    _config["norm_factor"] = math.sqrt(_config["channels"])
    _config["inner_skip"] = True
    config = SimpleNamespace(**_config)

    module = ConvFauxLarsen(
        channels=config.channels,
        depth=config.depth,
        kernel_size=config.kernel_size,
        skip_freq=config.skip_freq,
        norm_factor=config.norm_factor,
        inner_skip=config.inner_skip,
        sidechain_layers=config.sidechain_layers,
        dilation_layers=config.dilation_layers,
    )
    print(module.tabulate(jax.random.key(0), jnp.ones((2**2, 2**14, 1)), to_mask=0))
