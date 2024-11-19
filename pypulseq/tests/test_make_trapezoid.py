"""Tests for the make_trapezoid module

Will Clarke, University of Oxford, 2023
"""

from types import SimpleNamespace

import pytest

from pypulseq import make_trapezoid


def test_channel_error():
    with pytest.raises(ValueError, match=r'Invalid channel. Must be one of `x`, `y` or `z`. Passed:'):
        make_trapezoid(channel='p')


def test_falltime_risetime_error():
    with pytest.raises(
        ValueError, match=r'Invalid arguments. Must always supply `rise_time` if `fall_time` is specified explicitly.'
    ):
        make_trapezoid(channel='x', fall_time=10)


def test_area_flatarea_amplitude_error():
    with pytest.raises(ValueError, match=r"Must supply either 'area', 'flat_area' or 'amplitude'."):
        make_trapezoid(channel='x')


def test_flat_time_error():
    errstr = (
        'When `flat_time` is provided, either `flat_area`, '
        'or `amplitude`, or `rise_time` and `area` must be provided as well.'
    )

    with pytest.raises(ValueError, match=errstr):
        make_trapezoid(channel='x', flat_time=10, area=10)


def test_area_too_large_error():
    errstr = 'Requested area is too large for this gradient. Minimum required duration is'

    with pytest.raises(AssertionError, match=errstr):
        make_trapezoid(channel='x', area=1e6, duration=1e-6)


def test_area_too_large_error_rise_time():
    errstr = 'Requested area is too large for this gradient. Probably amplitude is violated'

    with pytest.raises(AssertionError, match=errstr):
        make_trapezoid(channel='x', area=1e6, duration=1e-6, rise_time=1e-7)


def test_no_area_no_duration_error():
    errstr = 'Must supply area or duration.'

    with pytest.raises(ValueError, match=errstr):
        make_trapezoid(channel='x', amplitude=1)


def test_amplitude_too_large_error():
    errstr = r'Refined amplitude \(\d+ Hz/m\) is larger than max \(\d+ Hz/m\).'

    with pytest.raises(AssertionError, match=errstr):
        make_trapezoid(channel='x', amplitude=1e10, duration=1)


def test_generation_methods():
    """Test minimum input cases
    Cover:
        - area
        - amplitude and duration
        - flat_time and flat_area
        - flat_time and amplitude
        - flat_time, area and rise_time
    """

    assert isinstance(make_trapezoid(channel='x', area=1), SimpleNamespace)

    assert isinstance(make_trapezoid(channel='x', amplitude=1, duration=1), SimpleNamespace)

    assert isinstance(make_trapezoid(channel='x', flat_time=1, flat_area=1), SimpleNamespace)

    assert isinstance(make_trapezoid(channel='x', flat_time=1, amplitude=1), SimpleNamespace)

    assert isinstance(make_trapezoid(channel='x', flat_time=0.5, area=1, rise_time=0.1), SimpleNamespace)
