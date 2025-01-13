"""Tests for the make_trapezoid module"""

import pytest
from pypulseq import Opts, eps, make_trapezoid


def test_channel_error():
    with pytest.raises(ValueError, match=r'Invalid channel. Must be one of `x`, `y` or `z`. Passed:'):
        make_trapezoid(channel='p')


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

    with pytest.raises(ValueError, match=errstr):
        make_trapezoid(channel='x', amplitude=1e10, duration=1)


def test_duration_too_short_error():
    errstr = 'The `duration` is too short for the given `rise_time`.'

    with pytest.raises(ValueError, match=errstr):
        make_trapezoid(channel='x', area=1, duration=0.1, rise_time=0.1)


def test_notimplemented_input_pairs():
    # flat_area + duration
    with pytest.raises(NotImplementedError, match=r'Flat Area \+ Duration input pair is not implemented yet.'):
        make_trapezoid(channel='x', flat_area=1, duration=1)
    # flat_area + amplitude
    with pytest.raises(NotImplementedError, match=r'Flat Area \+ Amplitude input pair is not implemented yet.'):
        make_trapezoid(channel='x', flat_area=1, amplitude=1)
    # area + amplitude
    with pytest.raises(NotImplementedError, match=r'Amplitude \+ Area input pair is not implemented yet.'):
        make_trapezoid(channel='x', area=1, amplitude=1)
        # compare_trap_out(trap, 1, 2e-5, 0, 2e-5)


def round2raster(x, raster=1e-5):
    return round(x / raster) * raster


def compare_trap_out(trap, amplitude, rise_time, flat_time, fall_time):
    assert abs(trap.amplitude - amplitude) < eps
    assert abs(trap.rise_time - rise_time) < eps
    assert abs(trap.flat_time - flat_time) < eps
    assert abs(trap.fall_time - fall_time) < eps


def test_generation_methods():
    """Test minimum input cases
    Cover:
        - area
        - amplitude and duration
        - flat_time and flat_area
        - flat_time and amplitude
        - flat_time, area and rise_time
    """

    opts = Opts()
    # Amplitude specified
    # amplitude + duration
    trap = make_trapezoid(channel='x', amplitude=1, duration=1)
    compare_trap_out(trap, 1, 1e-5, 1 - 2e-5, 1e-5)

    # flat_time + amplitude
    trap = make_trapezoid(channel='x', flat_time=1, amplitude=1)
    compare_trap_out(trap, 1, 1e-5, 1, 1e-5)

    # Flat area specified
    # flat_area + flat_time
    trap = make_trapezoid(channel='x', flat_time=1, flat_area=1)
    compare_trap_out(trap, 1, 1e-5, 1, 1e-5)

    # Area specified
    # area
    # triangle case
    trap = make_trapezoid(channel='x', area=1)
    compare_trap_out(trap, 50000, 2e-5, 0, 2e-5)
    # trap case
    trap = make_trapezoid(channel='x', area=opts.max_grad * 2)
    time_to_max = round2raster(opts.max_grad / opts.max_slew)
    compare_trap_out(
        trap, opts.max_grad, time_to_max, (opts.max_grad * 2 - time_to_max * opts.max_grad) / opts.max_grad, time_to_max
    )

    # area + duration
    trap = make_trapezoid(channel='x', area=1, duration=1)
    compare_trap_out(trap, 1.00002, 2e-5, 1 - 4e-5, 2e-5)

    # area + duration + rise_time
    trap = make_trapezoid(channel='x', area=1, duration=1, rise_time=0.01)
    compare_trap_out(trap, 1 / 0.99, 0.01, 0.98, 0.01)
    # flat_time + area + rise_time
    trap = make_trapezoid(channel='x', flat_time=0.5, area=1, rise_time=0.1)
    compare_trap_out(trap, 1 / 0.6, 0.1, 0.5, 0.1)
