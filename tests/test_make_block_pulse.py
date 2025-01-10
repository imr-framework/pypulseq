"""Tests for the make_block_pulse.py module

Will Clarke, University of Oxford, 2023
"""

from types import SimpleNamespace

import numpy as np
import pytest
from pypulseq import make_block_pulse


def test_invalid_use_error():
    with pytest.raises(ValueError, match=r'Invalid use parameter.'):
        make_block_pulse(flip_angle=np.pi, duration=1e-3, use='foo')


def test_bandwidth_and_duration_error():
    with pytest.raises(ValueError, match=r'One of bandwidth or duration must be defined, but not both.'):
        make_block_pulse(flip_angle=np.pi, duration=1e-3, bandwidth=1000)


def test_invalid_bandwidth_and_duration_error():
    with pytest.raises(ValueError, match=r'One of bandwidth or duration must be defined and be > 0.'):
        make_block_pulse(flip_angle=np.pi, duration=-1e-3)

    with pytest.raises(ValueError, match=r'One of bandwidth or duration must be defined and be > 0.'):
        make_block_pulse(flip_angle=np.pi, bandwidth=-1e3)


def test_default_duration_warning():
    with pytest.warns(UserWarning, match=r'Using default 4 ms duration for block pulse.'):
        make_block_pulse(flip_angle=np.pi)


def test_generation_methods():
    """Test minimum input cases
    Cover:
        - Just flip_angle
        - duration
        - bandwidth
        - bandwidth + time_bw_product
    """
    # Capture expected warning for default case
    with pytest.warns(UserWarning):
        case1 = make_block_pulse(flip_angle=np.pi)

    assert isinstance(case1, SimpleNamespace)
    assert case1.shape_dur == 4e-3

    case2 = make_block_pulse(flip_angle=np.pi, duration=1e-3)
    assert isinstance(case2, SimpleNamespace)
    assert case2.shape_dur == 1e-3

    case3 = make_block_pulse(flip_angle=np.pi, bandwidth=1e3)
    assert isinstance(case3, SimpleNamespace)
    assert case3.shape_dur == 1 / (4 * 1e3)

    case4 = make_block_pulse(flip_angle=np.pi, bandwidth=1e3, time_bw_product=5)
    assert isinstance(case4, SimpleNamespace)
    assert case4.shape_dur == 5 / 1e3


def test_amp_calculation():
    # A 1 ms 180 degree pulse requires 500 Hz gamma B1
    pulse = make_block_pulse(duration=1e-3, flip_angle=np.pi)
    assert np.isclose(pulse.signal.max(), 500)

    pulse = make_block_pulse(duration=1e-3, flip_angle=np.pi / 2)
    assert np.isclose(pulse.signal.max(), 250)

    pulse = make_block_pulse(duration=2e-3, flip_angle=np.pi / 2)
    assert np.isclose(pulse.signal.max(), 125)
