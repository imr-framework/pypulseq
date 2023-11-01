"""Tests for the make_trapezoid module

Will Clarke, University of Oxford, 2023
"""

from types import SimpleNamespace

import pytest

from pypulseq import make_trapezoid


def test_channel_error():

    with pytest.raises(
            ValueError,
            match=r"Invalid channel. Must be one of `x`, `y` or `z`. Passed:"):
        make_trapezoid(channel='p')


def test_falltime_risetime_error():
    with pytest.raises(
            ValueError,
            match=r"Invalid arguments. Must always supply `rise_time` if `fall_time` is specified explicitly."):
        make_trapezoid(channel='x', fall_time=10)


def test_area_flatarea_amplitude_error():
    with pytest.raises(
            ValueError,
            match=r"Must supply either 'area', 'flat_area' or 'amplitude'."):
        make_trapezoid(channel='x')


def test_flat_time_error():
    errstr = "When `flat_time` is provided, either `flat_area` or `amplitude` must be provided as well; you may "\
             "consider providing `duration`, `area` and optionally ramp times instead."

    with pytest.raises(
            ValueError,
            match=errstr):
        make_trapezoid(channel='x', flat_time=10, area=10)


def test_area_too_large_error():
    errstr = "Requested area is too large for this gradient. Minimum required duration is"

    with pytest.raises(
            AssertionError,
            match=errstr):
        make_trapezoid(channel='x', area=1E6, duration=1E-6)


def test_area_too_large_error_rise_time():
    errstr = "Requested area is too large for this gradient. Probably amplitude is violated"

    with pytest.raises(
            AssertionError,
            match=errstr):
        make_trapezoid(channel='x',  area=1E6, duration=1E-6, rise_time=1E-7)


def test_no_area_no_duration_error():
    errstr = "Must supply area or duration."

    with pytest.raises(
            ValueError,
            match=errstr):
        make_trapezoid(channel='x',  amplitude=1)


def test_amplitude_too_large_error():
    errstr = "Amplitude violation."

    with pytest.raises(
            ValueError,
            match=errstr):
        make_trapezoid(channel='x',  amplitude=1E10, duration=1)


def test_generation_methods():

    assert isinstance(
        make_trapezoid(channel='x',  area=1),
        SimpleNamespace)

    assert isinstance(
        make_trapezoid(channel='x', flat_area=1, flat_time=1),
        SimpleNamespace)

    assert isinstance(
        make_trapezoid(channel='x', flat_area=0.5, duration=1, area=1),
        SimpleNamespace)

    assert isinstance(
        make_trapezoid(channel='x',  amplitude=1, duration=1),
        SimpleNamespace)
