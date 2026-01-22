from types import SimpleNamespace

import numpy as np
import pypulseq as pp
import pytest


@pytest.fixture
def dummy_rf():
    """
    Fixture factory for creating SimpleNamespace RF pulses.
    Returns a function you can call with parameters.
    """
    def _make_rf(
        amplitude=1.0,
        duration=1.0,
        t_override=None,
        freq_offset=0.0,
        phase_offset=0.0,
        freq_ppm=0.0,
        phase_ppm=0.0,
        delay=0.0,
        use='excitation',
    ):
        system = pp.Opts.default

        if t_override is not None:
            t = np.asarray(t_override)
            duration = t[-1] - t[0]
            signal = amplitude * np.ones_like(t)
        else:
            n = int(duration / system.rf_raster_time) + 1
            t = np.linspace(0, duration, n, endpoint=True)
            signal = amplitude * np.ones_like(t)

        rf = SimpleNamespace()
        rf.type = 'rf'
        rf.signal = signal
        rf.t = t
        rf.shape_dur = duration
        rf.center = 0.5 * duration
        rf.freq_offset = 0.0
        rf.phase_offset = 0.0
        rf.freq_ppm = 0.0
        rf.phase_ppm = 0.0
        rf.dead_time = system.rf_dead_time
        rf.ringdown_time = system.rf_ringdown_time
        rf.delay = 0.0
        rf.use = 'excitation'

        return rf

    return _make_rf


@pytest.fixture
def dummy_sequence():
    """
    Fixture for creating a minimal pypulseq.Sequence with a helper
    to add SimpleNamespace RF pulses as blocks.
    """
    seq = pp.Sequence()

    return seq


def test_constant_rf(dummy_rf):
    """Test constant amplitude pulse against analytic solution."""
    A = 2.0
    T = 4.0e-3
    rf = dummy_rf(amplitude=A, duration=T)

    E, P, rms = pp.calc_rf_power(rf)

    assert np.isclose(E, A**2 * T)
    assert np.isclose(P, A**2)
    assert np.isclose(rms, A)


def test_nonuniform_raster(dummy_rf):
    """Test nonuniformly sampled RF returns correct energy."""
    A = 2.0
    t = [0.0e-3, 2.0e-3]  # nonuniform time samples
    rf = dummy_rf(amplitude=A, t_override=t)

    E, P, rms = pp.calc_rf_power(rf)

    assert np.isclose(E, A**2 * (t[-1] - t[0]))
    assert np.isclose(P, A**2)
    assert np.isclose(rms, A)


def test_resampled_rf(dummy_rf):
    """Test resampling path with dt parameter."""
    A = 2.0
    t = [0.0e-3, 2.0e-3]  # nonuniform time samples
    rf = dummy_rf(amplitude=A, t_override=t)

    E, P, rms = pp.calc_rf_power(rf, dt=pp.Opts.default.rf_raster_time)

    assert np.isclose(E, A**2 * (t[-1] - t[0]), rtol=1e-3)
    assert np.isclose(P, A**2, rtol=1e-3)
    assert np.isclose(rms, A, rtol=1e-3)


def test_sequence_single_block(dummy_rf, dummy_sequence):
    """Test sequence with a single RF block."""
    seq = dummy_sequence
    A = 2.0
    T = 5.0
    rf = dummy_rf(amplitude=A, duration=T)
    seq.add_block(rf)

    mean_pwr, peak, rms, energy = seq.calc_rf_power()

    assert np.isclose(energy, A**2 * T)
    assert np.isclose(mean_pwr, A**2)
    assert np.isclose(rms, A)
    assert np.isclose(peak, A**2)


def test_sequence_partial_overlap(dummy_rf, dummy_sequence):
    """Test sequence block partially overlapping time_range."""
    seq = dummy_sequence
    A = 2.0
    T = 10.0e-3
    rf = dummy_rf(amplitude=A, duration=T)
    seq.add_block(rf)

    mean_pwr, peak, rms, energy = seq.calc_rf_power(time_range=(2.0e-3, 6.0e-3))

    # Only 4s of 10s contribute
    assert np.isclose(energy, A**2 * 4.0e-3)
    assert np.isclose(mean_pwr, A**2)
    assert np.isclose(rms, A)


def test_sequence_sliding_window(dummy_rf, dummy_sequence):
    """
    Test sliding window RF power calculation.

    NOTE:
    The sliding window implementation intentionally follows Pulseq's
    MATLAB behavior, where the accumulated energy is compared against
    the maximum *before* trimming the window back to the nominal
    window_duration.

    This leads to a conservative (slightly overestimated) mean power,
    as explicitly documented in the original Pulseq code and consistent
    with SAR-style worst-case windowing.
    """
    seq = dummy_sequence

    # Two consecutive RF blocks:
    # Block 1: A=1, duration=5s → energy = 1^2 * 5 = 5
    # Block 2: A=3, duration=5s → energy = 3^2 * 5 = 45
    rf1 = dummy_rf(amplitude=1.0, duration=5.0)
    rf2 = dummy_rf(amplitude=3.0, duration=5.0)

    seq.add_block(rf1)
    seq.add_block(rf2)

    mean_pwr, peak, rms, _ = seq.calc_rf_power(window_duration=5.0)

    # During accumulation, both blocks temporarily fall inside the window:
    # total_energy = 5 + 45 = 50
    #
    # mean_pwr is computed using the nominal window duration (5s),
    # resulting in a conservative overestimate:
    # mean_pwr = 50 / 5 = 10
    assert np.isclose(mean_pwr, 10.0)

    # RMS amplitude follows the same conservative windowing logic
    assert np.isclose(rms, np.sqrt(10.0))

    # Peak power is taken from the strongest RF block
    assert np.isclose(peak, 9.0)
