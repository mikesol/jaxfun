import torchlibrosa.stft as stft_ref
import stft as stft
import numpy as np
import pytest
import torch


def test_dft():
    """
    Test the DFT implementation.
    """
    x_real = np.random.randn(128).astype(np.float32)
    x_imag = np.random.randn(128).astype(np.float32)
    params = stft.init_dft_params(128)
    z_real, z_imag = stft.dft(params, x_real, x_imag)
    z_real_ref, z_imag_ref = stft_ref.DFT(128, None).dft(
        torch.from_numpy(x_real), torch.from_numpy(x_imag)
    )
    assert np.allclose(z_real, z_real_ref, atol=1e-3)
    assert np.allclose(z_imag, z_imag_ref, atol=1e-3)


def test_idft():
    """
    Test the DFT implementation.
    """
    x_real = np.random.randn(128).astype(np.float32)
    x_imag = np.random.randn(128).astype(np.float32)
    params = stft.init_dft_params(128)
    z_real, z_imag = stft.idft(params, x_real, x_imag)
    z_real_ref, z_imag_ref = stft_ref.DFT(128, None).idft(
        torch.from_numpy(x_real), torch.from_numpy(x_imag)
    )
    assert np.allclose(z_real, z_real_ref, atol=1e-3)
    assert np.allclose(z_imag, z_imag_ref, atol=1e-3)


def test_rdft():
    """
    Test the DFT implementation.
    """
    x_real = np.random.randn(128).astype(np.float32)
    params = stft.init_dft_params(128)
    z_real = stft.rdft(params, x_real)
    z_real_ref = stft_ref.DFT(128, None).rdft(torch.from_numpy(x_real))
    assert np.allclose(z_real, z_real_ref, atol=1e-3)


def test_idft():
    """
    Test the DFT implementation.
    """
    x_real = np.random.randn(65).astype(np.float32)
    x_imag = np.random.randn(65).astype(np.float32)
    params = stft.init_dft_params(65)
    z_real, z_imag = stft.idft(params, x_real, x_imag)
    z_real_ref, z_imag_ref = stft_ref.DFT(65, None).idft(
        torch.from_numpy(x_real), torch.from_numpy(x_imag)
    )
    assert np.allclose(z_real, z_real_ref, atol=1e-3)
    assert np.allclose(z_imag, z_imag_ref, atol=1e-3)


# @pytest.mark.only
def test_stft():
    for sizes in [
        (128, 64),
        (128, 32),
        (128, 16),
        (256, 128),
        (256, 64),
        (256, 32),
        (256, 16),
    ]:
        x = np.random.randn(8, sizes[0]).astype(np.float32)
        params = stft.init_stft_params(*sizes)
        z_real, z_imag = stft.stft(params, x)
        z_real_ref, z_imag_ref = stft_ref.STFT(*sizes, None)(torch.from_numpy(x))
        assert np.allclose(z_real, z_real_ref, atol=1e-3)
        assert np.allclose(z_imag, z_imag_ref, atol=1e-3)
