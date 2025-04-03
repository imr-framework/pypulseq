import math
import random

import numpy as np
import pytest
from pypulseq import Opts, calc_duration, make_extended_trapezoid, make_extended_trapezoid_area
from pypulseq.utils.cumsum import cumsum

system = Opts()

test_zoo = [
    (0, 0, 1),
    (0, 0, 10),
    (0, 0, 100),
    (0, 0, 10000),
    (0, 1000, 100),
    (-1000, 1000, 100),
    (-1000, 0, 100),
    (0, 0, -1),
    (0, 0, -10),
    (0, 0, -100),
    (0, 0, -10000),
    (0, 1000, -100),
    (-1000, 1000, -100),
    (-1000, 0, -100),
    (0, system.max_grad * 0.99, 10000),
    (0, system.max_grad * 0.99, -10000),
    (0, -system.max_grad * 0.99, 1000),
    (0, -system.max_grad * 0.99, -1000),
    (system.max_grad * 0.99, 0, 100),
    (system.max_grad * 0.99, 0, -100),
    (-system.max_grad * 0.99, 0, 1),
    (-system.max_grad * 0.99, 0, -1),
    (0, 100000, 1),
    (0, 100000, -1),
    (0, -100000, 1),
    (0, -100000, -1),
    (0, 90000, 0.45),
    (0, 90000, -0.45),
    (0, -90000, 0.45),
    (0, -90000, -0.45),
    (0, 10000, 0.5 * (10000) ** 2 / (system.max_slew * 0.99)),
    (0, system.max_grad * 0.99, 0.5 * (system.max_grad * 0.99) ** 2 / (system.max_slew * 0.99)),
    (system.max_grad * 0.99, system.max_grad * 0.99, 1),
    (system.max_grad * 0.99, system.max_grad * 0.99, -1),
]


@pytest.mark.parametrize('grad_start, grad_end, area', test_zoo)
def test_make_extended_trapezoid_area(grad_start, grad_end, area):
    g, _, _ = make_extended_trapezoid_area(
        channel='x', grad_start=grad_start, grad_end=grad_end, area=area, system=system
    )

    grad_ok = all(abs(g.waveform) <= system.max_grad)
    slew_ok = all(abs(np.diff(g.waveform) / np.diff(g.tt)) <= system.max_slew)

    assert pytest.approx(g.area) == area, 'Result area is not correct'
    assert grad_ok, 'Maximum gradient strength violated'
    assert slew_ok, 'Maximum slew rate violated'


random.seed(0)
test_zoo_random = [
    (
        (random.random() - 0.5) * 2 * system.max_grad * 0.99,
        (random.random() - 0.5) * 2 * system.max_grad * 0.99,
        (random.random() - 0.5) * 10000,
    )
    for _ in range(100)
]


@pytest.mark.parametrize('grad_start, grad_end, area', test_zoo_random)
def test_make_extended_trapezoid_area_random_cases(grad_start, grad_end, area):
    g, _, _ = make_extended_trapezoid_area(
        channel='x', grad_start=grad_start, grad_end=grad_end, area=area, system=system
    )

    grad_ok = all(abs(g.waveform) <= system.max_grad)
    slew_ok = all(abs(np.diff(g.waveform) / np.diff(g.tt)) <= system.max_slew)

    assert pytest.approx(g.area) == area, 'Result area is not correct'
    assert grad_ok, 'Maximum gradient strength violated'
    assert slew_ok, 'Maximum slew rate violated'


@pytest.mark.parametrize('grad_start, grad_end, area', test_zoo_random)
def test_make_extended_trapezoid_area_convert_to_arb(grad_start, grad_end, area):
    g, _, _ = make_extended_trapezoid_area(
        channel='x', grad_start=grad_start, grad_end=grad_end, area=area, system=system
    )

    g_arb, _, _ = make_extended_trapezoid_area(
        channel='x', grad_start=grad_start, grad_end=grad_end, area=area, convert_to_arbitrary=True, system=system
    )

    grad_ok = all(abs(g.waveform) <= system.max_grad)
    slew_ok = all(abs(np.diff(g.waveform) / np.diff(g.tt)) <= system.max_slew)

    grad_arb_ok = all(abs(g_arb.waveform) <= system.max_grad)
    slew_arb_ok = all(abs(np.diff(g_arb.waveform) / np.diff(g_arb.tt)) <= system.max_slew)

    assert pytest.approx(g.area) == g_arb.area, 'Area of extended trapz and arb gradient do not match'
    assert pytest.approx(g.shape_dur) == g_arb.shape_dur, 'Duration of extended trapz and arb gradient do not match'
    assert g.tt.shape[0] <= g_arb.tt.shape[0], (
        'Extended trapezoid should have less or equal number of points than arb gradient'
    )
    assert g.waveform.shape[0] <= g_arb.waveform.shape[0], (
        'Extended trapezoid should have less or equal number of points than arb gradient'
    )
    assert grad_ok == grad_arb_ok, 'Gradient strength violation between extended trapz and arb gradient'
    assert slew_ok == slew_arb_ok, 'Slew rate violation between extended trapz and arb gradient'


random.seed(0)
test_zoo_random = [
    (
        (random.random() - 0.5) * 3 * system.max_grad,
        (random.random() - 0.5) * 3 * system.max_grad,
        (random.random() - 0.5) * 10 * system.max_grad,
    )
    for _ in range(100)
]


@pytest.mark.parametrize('grad_start, grad_end, grad_amp', test_zoo_random)
def test_make_extended_trapezoid_area_recreate(grad_start, grad_end, grad_amp):
    def _to_raster(time: float) -> float:
        return np.ceil(time / system.grad_raster_time) * system.grad_raster_time

    def _calc_ramp_time(grad_amp, slew_rate, grad_start):
        return _to_raster(abs(grad_amp - grad_start) / slew_rate)

    grad_start = np.clip(grad_start, -system.max_grad, system.max_grad)
    grad_end = np.clip(grad_end, -system.max_grad, system.max_grad)

    # Construct extended gradient based on start, intermediate, and end gradient
    # strengths, assuming maximum slew rates.

    # If grad_amp > max_grad, convert to max_grad, and keep total area
    # approximately equal
    if abs(grad_amp) > system.max_grad * 0.99:
        grad_amp_new = math.copysign(system.max_grad * 0.99, grad_amp)

        flat_time = _to_raster(
            _calc_ramp_time(grad_amp_new, system.max_slew * 0.99, grad_amp)
            * abs(grad_amp_new + grad_amp)
            / abs(grad_amp_new)
        )
        grad_amp = grad_amp_new
    else:
        flat_time = 0

    # Construct extended gradient
    if flat_time == 0:
        times = cumsum(
            0,
            _calc_ramp_time(grad_amp, system.max_slew * 0.99, grad_start),
            _calc_ramp_time(grad_amp, system.max_slew * 0.99, grad_end),
        )
        amplitudes = (grad_start, grad_amp, grad_end)
    else:
        times = cumsum(
            0,
            _calc_ramp_time(grad_amp, system.max_slew * 0.99, grad_start),
            _to_raster(flat_time),
            _calc_ramp_time(grad_amp, system.max_slew * 0.99, grad_end),
        )
        amplitudes = (grad_start, grad_amp, grad_amp, grad_end)

    g_true = make_extended_trapezoid(channel='x', amplitudes=amplitudes, times=times)

    # Recreate gradient using make_extended_trapezoid_area
    g, _, _ = make_extended_trapezoid_area(
        channel='x', grad_start=grad_start, grad_end=grad_end, area=g_true.area, system=system
    )
    grad_ok = all(abs(g.waveform) <= system.max_grad)
    slew_ok = all(abs(np.diff(g.waveform) / np.diff(g.tt)) <= system.max_slew)

    assert pytest.approx(g.area) == g_true.area, 'Result area is not correct'
    assert grad_ok, 'Maximum gradient strength violated'
    assert slew_ok, 'Maximum slew rate violated'

    # Check that new gradient duration is smaller or equal to original gradient duration
    d1 = calc_duration(g)
    d2 = calc_duration(g_true)

    assert pytest.approx(d1) == d2 or d1 < d2
