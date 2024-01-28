import jax.numpy as jnp
import stft

def spectral_convergence_loss(x_mag, y_mag):
    """
    Calculate the spectral convergence loss.

    Args:
        x_mag (array): The magnitude spectrum of the first signal.
        y_mag (array): The magnitude spectrum of the second signal.

    Returns:
        The spectral convergence loss.
    """
    numerator = jnp.linalg.norm(y_mag - x_mag, ord=2)
    denominator = jnp.linalg.norm(y_mag, ord=2)
    loss = numerator / denominator
    return loss

import jax.numpy as jnp
from jax import vmap

def stft_magnitude_loss(x_mag, y_mag, log=True, distance="L1", reduction="mean"):
    """
    Calculate the STFT magnitude loss.

    Args:
        x_mag (array): The magnitude spectrum of the first signal.
        y_mag (array): The magnitude spectrum of the second signal.
        log (bool): Whether to log-scale the STFT magnitudes.
        distance (str): Distance function ["L1", "L2"].
        reduction (str): Reduction of the loss elements ["mean", "sum", "none"].

    Returns:
        The STFT magnitude loss.
    """
    if log:
        x_mag = jnp.log(x_mag + 1e-8)  # Adding a small value to avoid log(0)
        y_mag = jnp.log(y_mag + 1e-8)

    if distance == "L1":
        # L1 Loss (Mean Absolute Error)
        loss = jnp.abs(x_mag - y_mag)
    elif distance == "L2":
        # L2 Loss (Mean Squared Error)
        loss = (x_mag - y_mag) ** 2
    else:
        raise ValueError(f"Invalid distance: '{distance}'.")

    if reduction == "mean":
        return jnp.mean(loss)
    elif reduction == "sum":
        return jnp.sum(loss)
    elif reduction == "none":
        return loss
    else:
        raise ValueError(f"Invalid reduction: '{reduction}'.")


def stft_loss(input, target, params,
              w_sc, w_log_mag, w_lin_mag, w_phs, sample_rate, scale,
              n_bins, perceptual_weighting, scale_invariance, eps, output, reduction, mag_distance):
    """
    Calculate the STFT loss.
    """
    bs, seq_len, chs = input.shape
    # Compute STFT for input and target
    input_mag, input_phs = stft.stft(params, input)
    target_mag, target_phs = stft.stft(params, target)

    # Apply scaling (e.g., Mel, Chroma) if required
    if scale is not None: pass
        # Apply scaling here

    # Apply perceptual weighting if required
    if perceptual_weighting: pass
        # Apply perceptual weighting here

    # Calculate loss components
    sc_mag_loss = spectral_convergence_loss(input_mag, target_mag) * w_sc if w_sc else 0.0
    log_mag_loss = stft_magnitude_loss(input_mag, target_mag, ...) * w_log_mag if w_log_mag else 0.0
    lin_mag_loss = stft_magnitude_loss(input_mag, target_mag, ...) * w_lin_mag if w_lin_mag else 0.0
    phs_loss = phase_loss(input_phs, target_phs) * w_phs if w_phs else 0.0

    # Combine loss components
    total_loss = sc_mag_loss + log_mag_loss + lin_mag_loss + phs_loss

    # Apply reduction (mean, sum)
    if reduction == 'mean':
        total_loss = jnp.mean(total_loss)
    elif reduction == 'sum':
        total_loss = jnp.sum(total_loss)

    # Return based on the output type
    if output == 'loss':
        return total_loss
    elif output == 'full':
        return total_loss, sc_mag_loss, log_mag_loss, lin_mag_loss, phs_loss
