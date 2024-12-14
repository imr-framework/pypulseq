"""Tests for the make_gauss_pulse module

Will Clarke, University of Oxford, 2023
"""

from types import SimpleNamespace

import pytest
from pypulseq import make_gauss_pulse
from pypulseq.supported_labels_rf_use import get_supported_rf_uses


def test_use():
    with pytest.raises(ValueError, match=r'Invalid use parameter. Must be one of'):
        make_gauss_pulse(flip_angle=1, use='invalid')

    for use in get_supported_rf_uses():
        assert isinstance(make_gauss_pulse(flip_angle=1, use=use), SimpleNamespace)
