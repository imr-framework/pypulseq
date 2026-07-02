import warnings
from types import SimpleNamespace

import numpy as np
import pypulseq as pp
import pytest
from pypulseq.supported_labels_rf_use import get_supported_rf_uses


def test_use():
    with pytest.raises(ValueError, match=r'Invalid use parameter\. Must be one of'):
        pp.make_sinc_pulse(flip_angle=1, use='invalid')

    for use in get_supported_rf_uses():
        assert isinstance(pp.make_sinc_pulse(flip_angle=1, use=use), SimpleNamespace)


def test_return_gz_requires_slice_thickness():
    with pytest.raises(ValueError, match=r'Slice thickness must be provided if return_gz is True'):
        pp.make_sinc_pulse(
            flip_angle=1,
            duration=1e-3,
            delay=0,
            return_gz=True,
            slice_thickness=0,
        )


def test_dead_time_warning_without_return_gz_default_delay():
    system = pp.Opts(
        rf_dead_time=100e-6,
        rf_ringdown_time=0,
    )

    with warnings.catch_warnings(record=True) as recorded_warnings:
        warnings.simplefilter('always')
        rf = pp.make_sinc_pulse(
            flip_angle=1,
            duration=1e-3,
            delay=0,
            return_gz=False,
            system=system,
            time_bw_product=4,
        )

    assert len(recorded_warnings) == 1
    assert rf.delay == system.rf_dead_time


def test_dead_time_warning_without_return_gz_nonzero_delay():
    system = pp.Opts(
        rf_dead_time=100e-6,
        rf_ringdown_time=0,
    )

    with warnings.catch_warnings(record=True) as recorded_warnings:
        warnings.simplefilter('always')
        rf = pp.make_sinc_pulse(
            flip_angle=1,
            duration=1e-3,
            delay=10e-6,
            return_gz=False,
            system=system,
            time_bw_product=4,
        )

    assert len(recorded_warnings) == 1
    assert rf.delay == system.rf_dead_time


def test_dead_time_no_spurious_warning_with_return_gz():
    system = pp.Opts(
        max_grad=1e9,
        grad_unit='Hz/m',
        max_slew=1e7,
        slew_unit='Hz/m/s',
        rf_dead_time=100e-6,
        rf_ringdown_time=0,
    )

    with warnings.catch_warnings(record=True) as recorded_warnings:
        warnings.simplefilter('always')
        rf, gz, _ = pp.make_sinc_pulse(
            flip_angle=1,
            duration=1e-3,
            delay=0,
            return_gz=True,
            slice_thickness=1e-3,
            system=system,
            time_bw_product=4,
        )

    assert len(recorded_warnings) == 0
    assert rf.delay >= gz.rise_time + gz.delay
    assert rf.delay >= system.rf_dead_time


def test_flip_angle_normalization():
    flip_angle = 0.9
    rf = pp.make_sinc_pulse(
        flip_angle=flip_angle,
        duration=2e-3,
        delay=0,
        return_gz=False,
        time_bw_product=4,
    )

    dt = rf.t[1] - rf.t[0]
    achieved_flip = np.sum(rf.signal) * dt * 2 * np.pi
    assert np.isclose(achieved_flip, flip_angle, rtol=1e-3, atol=1e-6)


def test_system_not_mutated_by_local_overrides():
    system = pp.Opts(
        max_grad=123,
        grad_unit='Hz/m',
        max_slew=456,
        slew_unit='Hz/m/s',
        rf_dead_time=0,
        rf_ringdown_time=0,
    )

    original_max_grad = system.max_grad
    original_max_slew = system.max_slew

    pp.make_sinc_pulse(
        flip_angle=1,
        duration=1e-3,
        delay=0,
        return_gz=True,
        slice_thickness=1e-3,
        system=system,
        max_grad=1e9,
        max_slew=1e12,
        time_bw_product=4,
    )

    assert system.max_grad == original_max_grad
    assert system.max_slew == original_max_slew
