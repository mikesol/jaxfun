import jax
import jax.numpy as jnp
import jax.random as jrnd

randy = jrnd.normal(jrnd.PRNGKey(0), (4, 5, 10))
out = jax.lax.conv_general_dilated_patches(
    randy, filter_shape=(7,), window_strides=(1,), padding=((3, 3),)
)
print(out.shape)
out = jnp.reshape(out, (4, 5, 7, 10))

assert out[0][0][0][0] == 0.0
assert out[0][0][1][0] == 0.0
assert out[0][0][2][0] == 0.0
assert out[0][0][3][0] == randy[0][0][0]
assert out[0][0][4][0] == randy[0][0][1]
assert out[0][0][5][0] == randy[0][0][2]
assert out[0][0][6][0] == randy[0][0][3]
#
assert out[0][0][0][1] == 0.0
assert out[0][0][1][1] == 0.0
assert out[0][0][2][1] == randy[0][0][0]
assert out[0][0][3][1] == randy[0][0][1]
assert out[0][0][4][1] == randy[0][0][2]
assert out[0][0][5][1] == randy[0][0][3]
assert out[0][0][6][1] == randy[0][0][4]
#
assert out[0][0][0][2] == 0.0
assert out[0][0][1][2] == randy[0][0][0]
assert out[0][0][2][2] == randy[0][0][1]
assert out[0][0][3][2] == randy[0][0][2]
assert out[0][0][4][2] == randy[0][0][3]
assert out[0][0][5][2] == randy[0][0][4]
assert out[0][0][6][2] == randy[0][0][5]
##
assert out[0][1][0][0] == 0.0
assert out[0][1][1][0] == 0.0
assert out[0][1][2][0] == 0.0
assert out[0][1][3][0] == randy[0][1][0]
assert out[0][1][4][0] == randy[0][1][1]
assert out[0][1][5][0] == randy[0][1][2]
assert out[0][1][6][0] == randy[0][1][3]
#
assert out[0][1][0][1] == 0.0
assert out[0][1][1][1] == 0.0
assert out[0][1][2][1] == randy[0][1][0]
assert out[0][1][3][1] == randy[0][1][1]
assert out[0][1][4][1] == randy[0][1][2]
assert out[0][1][5][1] == randy[0][1][3]
assert out[0][1][6][1] == randy[0][1][4]


####

randy = jrnd.normal(jrnd.PRNGKey(0), (4, 5, 10, 2))
out = jax.lax.conv_general_dilated_patches(
    randy, filter_shape=(7, 3), window_strides=(1, 1), padding=((3, 3), (1, 1))
)
print(out.shape)
out = jnp.reshape(out, (4, 5, 7, 3, 10, 2))
#  b  c  kh kw h  w
assert out[0][0][0][0][0][0] == 0.0
assert out[0][0][1][0][0][0] == 0.0
assert out[0][0][2][0][0][0] == 0.0
assert out[0][0][3][0][0][0] == 0.0
assert out[0][0][4][0][0][0] == 0.0
assert out[0][0][5][0][0][0] == 0.0
assert out[0][0][6][0][0][0] == 0.0
