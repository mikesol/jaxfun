from data import make_2d_data_with_delays_and_dilations, make_2d_data
import librosa
import numpy as np
import pytest

from get_files import FILES


def test_data():
    paths = FILES[:1]
    window = 2**12
    stride = 2**8
    feature_dim = -1
    dataset, _ = make_2d_data(
        paths=paths,
        window=window,
        stride=stride,
        feature_dim=feature_dim,
        shuffle=False,
        normalize=False,
    )

    batch = next(dataset.iter(1, drop_last_batch=True))
    i = np.array(batch["input"])
    o = np.array(batch["target"])
    assert i.shape[1] == o.shape[1] * 2
    i_, _ = librosa.load(FILES[0][0])
    o_, _ = librosa.load(FILES[0][1])
    assert i.shape[0] == 1
    assert i.shape[1] == window * 2
    assert i.shape[2] == 1
    # input is always nudged forward in time by 1
    assert np.isclose(i_[1 + window - 1], i[0][-2][0])
    assert np.isclose(i_[1 + window - 2], i[0][-4][0])
    assert np.isclose(i_[1 + window - 3], i[0][-6][0])
    assert np.isclose(o_[window - 1], i[0][-1][0])
    assert np.isclose(o_[window - 2], i[0][-3][0])
    assert np.isclose(o_[window - 3], i[0][-5][0])


def test_data_with_dilations():
    paths = FILES[:1]
    window = 2**12
    stride = 2**8
    shift = 16
    dilation = 1
    channels = 2**3
    feature_dim = -1
    dataset, _ = make_2d_data_with_delays_and_dilations(
        paths=paths,
        window=window,
        stride=stride,
        shift=shift,
        dilation=dilation,
        channels=channels,
        feature_dim=feature_dim,
        shuffle=False,
        normalize=False,
    )
    zone_size = channels // 4
    sample_width = (window + (zone_size * shift)) * (4 * dilation)

    batch = next(dataset.iter(1, drop_last_batch=True))
    i = batch["input"]
    o = batch["target"]
    assert i.shape[1] == o.shape[1] * 2
    i_, _ = librosa.load(FILES[0][0])
    o_, _ = librosa.load(FILES[0][1])
    assert i.shape[0] == 1
    assert i.shape[1] == window * 2
    assert i.shape[2] == channels
    # input is always nudged forward in time by 1
    assert np.isclose(i_[1 + sample_width - 1], i[0][-2][0])
    assert np.isclose(i_[1 + sample_width - 2], i[0][-4][0])
    assert np.isclose(i_[1 + sample_width - 3], i[0][-6][0])
    assert np.isclose(o_[sample_width - 1], i[0][-1][0])
    assert np.isclose(o_[sample_width - 2], i[0][-3][0])
    assert np.isclose(o_[sample_width - 3], i[0][-5][0])
    ##
    assert np.isclose(i_[1 + sample_width - 1 - shift], i[0][-2][1])
    assert np.isclose(i_[1 + sample_width - 2 - shift], i[0][-4][1])
    assert np.isclose(i_[1 + sample_width - 3 - shift], i[0][-6][1])
    assert np.isclose(o_[sample_width - 1 - shift], i[0][-1][1])
    assert np.isclose(o_[sample_width - 2 - shift], i[0][-3][1])
    assert np.isclose(o_[sample_width - 3 - shift], i[0][-5][1])
    ##
    assert np.isclose(o_[1 + sample_width - 1], o[0][-1][0])
    assert np.isclose(o_[1 + sample_width - 2], o[0][-2][0])
    assert np.isclose(o_[1 + sample_width - 3], o[0][-3][0])
