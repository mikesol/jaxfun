import numpy as np
import auraloss_jax as auraloss
import auraloss as auraloss_ref
import pytest
import torch
import stft


def test_spectral_convergence_loss():
    x_mag = np.random.rand(1024)
    y_mag = np.random.rand(1024)
    loss = auraloss.spectral_convergence_loss(x_mag, y_mag)
    loss_ref = auraloss_ref.freq.SpectralConvergenceLoss()(
        torch.from_numpy(x_mag), torch.from_numpy(y_mag)
    )
    assert np.allclose(loss, loss_ref)


def test_stft_magnitude_loss():
    x_mag = np.random.rand(1024)
    y_mag = np.random.rand(1024)
    loss = auraloss.stft_magnitude_loss(x_mag, y_mag)
    loss_ref = auraloss_ref.freq.STFTMagnitudeLoss()(
        torch.from_numpy(x_mag), torch.from_numpy(y_mag)
    )
    assert np.allclose(loss, loss_ref)


def test_stft_loss():
    params = stft.init_stft_params(1024, 512)
    x = np.random.randn(4, 1024, 1)
    y = np.random.randn(4, 1024, 1)
    loss = auraloss.stft_loss(params, x, y)
    loss_ref = auraloss_ref.freq.STFTLoss(1024, 512)(
        torch.from_numpy(np.transpose(x, (0, 2, 1))),
        torch.from_numpy(np.transpose(y, (0, 2, 1))),
    )
    assert np.allclose(loss, loss_ref)


def test_multi_resolution_stft_loss():
    fft_sizes = [1024, 2048, 512]
    hop_sizes = [120, 240, 50]
    win_lengths = [600, 1200, 240]
    params = [
        stft.init_stft_params(x, y, z)
        for x, y, z in zip(fft_sizes, hop_sizes, win_lengths)
    ]
    x = np.random.randn(4, 8192, 1)
    y = np.random.randn(4, 8192, 1)
    loss = auraloss.multi_resolution_stft_loss(params, x, y)
    loss_ref = auraloss_ref.freq.MultiResolutionSTFTLoss(
        fft_sizes=fft_sizes, hop_sizes=hop_sizes, win_lengths=win_lengths
    )(
        torch.from_numpy(np.transpose(x, (0, 2, 1))),
        torch.from_numpy(np.transpose(y, (0, 2, 1))),
    )
    assert np.allclose(loss, loss_ref)
