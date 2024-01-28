import numpy as np
import auraloss_jax as auraloss
import auraloss as auraloss_ref
import pytest
import torch


def test_spectral_convergence_loss():
    x_mag = np.random.rand(1024)
    y_mag = np.random.rand(1024)
    loss = auraloss.spectral_convergence_loss(x_mag, y_mag)
    loss_ref = auraloss_ref.freq.SpectralConvergenceLoss()(torch.from_numpy(x_mag), torch.from_numpy(y_mag))
    assert np.allclose(loss, loss_ref)

# @pytest.mark.only
def test_stft_magnitude_loss():
    x_mag = np.random.rand(1024)
    y_mag = np.random.rand(1024)
    loss = auraloss.stft_magnitude_loss(x_mag, y_mag)
    loss_ref = auraloss_ref.freq.STFTMagnitudeLoss()(torch.from_numpy(x_mag), torch.from_numpy(y_mag))
    assert np.allclose(loss, loss_ref)