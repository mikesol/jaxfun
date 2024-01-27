import jax.numpy as jnp


def advance_sine(prev_pos, delta_t, in_arcsin_range, freq):
    prev_time = jnp.arcsin(prev_pos) / (freq * 2.0 * jnp.pi)
    prev_time = jnp.where(in_arcsin_range, prev_time, (1 / (2.0 * freq)) - prev_time)
    next_time = prev_time + delta_t
    next_pos = jnp.sin(next_time * freq * 2.0 * jnp.pi)
    new_in_arcsin_range = (((next_time + (0.25 / freq)) * 2 * freq) % 2) < 1.0
    return new_in_arcsin_range, next_pos


def advance_sine2(prev_pos, cur_t, d_t, prev_slope, freq, amp):
    cur_slope = amp * freq * 2.0 * jnp.pi * jnp.cos(cur_t * freq * 2.0 * jnp.pi)
    avg_slope = (prev_slope + cur_slope) / 2.0
    next_pos = prev_pos + avg_slope * d_t
    return cur_slope, next_pos

if __name__ == "__main__":
    in_arcsin_range = True
    prev_pos = 0.0
    t = 0.0
    dt = 0.1
    freq = 42.42
    for x in range(100):
        in_arcsin_range, prev_pos = advance_sine(prev_pos, dt, in_arcsin_range, freq)
        t += dt
        should_be = jnp.sin(t * freq * jnp.pi * 2.0)
        assert jnp.allclose(prev_pos, should_be, atol=1e-04)
