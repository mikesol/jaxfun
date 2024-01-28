import fir_filter
import numpy as np
import auraloss.perceptual as fir_filter_ref
import pytest
import torch

# Example Usage


def test_fir_filter_hp():
    signal = np.random.randn(8, 256).astype(np.float32)
    taps = fir_filter.create_fir_filter(
        filter_type="hp", coef=0.85, fs=44100, ntaps=101
    )
    filtered_signal = fir_filter.fir_filter(signal, taps, 101)
    torch_sig = torch.from_numpy(np.expand_dims(signal, axis=-2))
    filtered_signal_ref, _ = fir_filter_ref.FIRFilter(
        filter_type="hp", coef=0.85, fs=44100, ntaps=101
    )(torch_sig, torch_sig)
    assert np.allclose(
        filtered_signal, np.squeeze(np.array(filtered_signal_ref)), atol=1.0e-3
    )
    # just to make sure it's not all zeros
    assert np.any(np.abs(filtered_signal) > 0.0)


def test_fir_filter_fd():
    signal = np.random.randn(8, 256).astype(np.float32)
    taps = fir_filter.create_fir_filter(
        filter_type="fd", coef=0.85, fs=44100, ntaps=101
    )
    filtered_signal = fir_filter.fir_filter(signal, taps, 101)
    torch_sig = torch.from_numpy(np.expand_dims(signal, axis=-2))
    filtered_signal_ref, _ = fir_filter_ref.FIRFilter(
        filter_type="fd", coef=0.85, fs=44100, ntaps=101
    )(torch_sig, torch_sig)
    assert np.allclose(
        filtered_signal, np.squeeze(np.array(filtered_signal_ref)), atol=1.0e-3
    )
    # just to make sure it's not all zeros
    assert np.any(np.abs(filtered_signal) > 0.0)


def test_fir_filter_aw():
    signal = np.random.randn(8, 256).astype(np.float32)
    taps = fir_filter.create_fir_filter(
        filter_type="aw", coef=0.85, fs=44100, ntaps=101
    )
    filtered_signal = fir_filter.fir_filter(signal, taps, 101)
    torch_sig = torch.from_numpy(np.expand_dims(signal, axis=-2))
    filtered_signal_ref, _ = fir_filter_ref.FIRFilter(
        filter_type="aw", coef=0.85, fs=44100, ntaps=101
    )(torch_sig, torch_sig)
    assert np.allclose(
        filtered_signal, np.squeeze(np.array(filtered_signal_ref)), atol=1.0e-3
    )
    # just to make sure it's not all zeros
    assert np.any(np.abs(filtered_signal) > 0.0)
