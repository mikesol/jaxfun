import jax.numpy as jnp
a = jnp.ones((2**2, 4, 1))
b = jnp.ones((2**2, 1, 16))
print(jnp.matmul(a,b).shape)