import jax.random as jrnd
import jax.numpy as jnp

randy = jrnd.normal(jrnd.PRNGKey(0), (4, 5, 10, 2))
o = jnp.reshape(randy, (4, 5, 20))
assert o[0][0][0] == randy[0][0][0][0]
assert o[0][0][1] == randy[0][0][0][1]
assert o[0][0][2] == randy[0][0][1][0]
