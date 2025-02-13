import jax
jax.config.update("jax_enable_x64", True)

from . import detector, signal, distributions, utils, plotting