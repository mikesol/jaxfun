import pytest

import numpy as np

@pytest.mark.only
def test_reshape():
    batch = 4
    in_chan = 7
    out_chan = 5
    seq = 16
    a0 = np.random.random((batch, in_chan, seq))
    a1 = np.repeat(a0, repeats=out_chan, axis=-2)
    a2 = np.reshape(a1, (batch, in_chan, out_chan, seq))
    assert a2[0][0][0][0] == a0[0][0][0]
    assert a2[0][1][0][0] == a0[0][1][0]
    assert a2[0][2][0][0] == a0[0][2][0]
    assert a2[0][0][1][0] == a0[0][0][0]
    assert a2[0][1][1][0] == a0[0][1][0]
    assert a2[0][2][1][0] == a0[0][2][0]
