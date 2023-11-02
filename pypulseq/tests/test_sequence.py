"""Tests for the make_gauss_pulse module

Will Clarke, University of Oxford, 2023
"""


from types import SimpleNamespace

import pytest
from unittest.mock import patch

from pypulseq import Sequence
from pypulseq import make_gauss_pulse


@patch("matplotlib.pyplot.show")
def test_plot(mock_show):
    seq = Sequence()
    seq.add_block(
        make_gauss_pulse(flip_angle=1))
    assert seq.plot() is None

    assert seq.plot(show_blocks=True) is None
