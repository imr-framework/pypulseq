import numpy as np
import pypulseq as pp
from pypulseq.Sequence.sequence import Sequence


def _eig_min(a: np.ndarray) -> float:
    # a is 3x3 symmetric
    w = np.linalg.eigvalsh(0.5 * (a + a.T))
    return float(w.min())


def test_calc_moments_btensor_basic_properties():
    sys = pp.Opts(
        max_grad=np.inf,
        grad_unit='mT/m',
        max_slew=np.inf,
        slew_unit='T/m/s',
        rf_dead_time=0.0,
        rf_ringdown_time=0.0,
        adc_dead_time=0.0,
    )

    # Initialize sequence
    seq = Sequence(sys)

    # ---- Define RF pulses ----
    rf90 = pp.make_block_pulse(
        flip_angle=np.pi / 2,
        duration=1e-3,
        system=sys,
        use='excitation',
    )
    rf180 = pp.make_block_pulse(
        flip_angle=np.pi,
        duration=1e-3,
        system=sys,
        use='refocusing',
    )

    # ---- Target b-value ----
    b_target = 1000e6  # s/m^2 (1000 s/mm^2)
    TE = 80e-3
    delta = 10e-3
    Delta = 30e-3

    # Solve for gradient amplitude (Hz/m)
    G = (1 / (2 * np.pi * delta)) * np.sqrt(b_target / (Delta - delta / 3))

    # ---- Define diffusion gradient lobe ----
    g = pp.make_trapezoid(channel='z', system=sys, amplitude=G, duration=delta)

    # Interval between end of first g and start of second g
    diff_wait = Delta - delta

    # RF free time
    rf_free_time = diff_wait - pp.calc_duration(rf180)

    # Center rf between gradient lobes
    g_wait = 0.5 * rf_free_time

    # Compute te wait time
    texc_center, _ = pp.calc_rf_center(rf90)
    tref_center, _ = pp.calc_rf_center(rf180)
    t_ref = (pp.calc_duration(rf90) - texc_center) + pp.calc_duration(g) + g_wait + tref_center
    te_wait = 0.5 * TE - t_ref

    # ---- Timing layout ----
    seq.add_block(rf90)
    seq.add_block(pp.make_delay(te_wait))
    seq.add_block(g)
    seq.add_block(pp.make_delay(g_wait))
    seq.add_block(rf180)
    seq.add_block(pp.make_delay(g_wait))
    seq.add_block(g)

    # ---- Run moment calculator -----
    B, m1, m2, m3 = seq.calc_moments_btensor()  # defaults: calcB=True only

    # Shapes
    assert B.ndim == 3 and B.shape[1:] == (3, 3)
    assert m1.shape == (B.shape[0], 3)
    assert m2.shape == (B.shape[0], 3)
    assert m3.shape == (B.shape[0], 3)

    # With one excitation this will be one repetition
    assert B.shape[0] == 1

    # Symmetry
    assert np.allclose(B[0], B[0].T, atol=1e-10, rtol=1e-10)

    # PSD (numerical tolerance)
    assert _eig_min(B[0]) > -1e-8

    # Dominant zz for z-only gradients
    assert B[0, 2, 2] >= B[0, 0, 0]
    assert B[0, 2, 2] >= B[0, 1, 1]

    # Off-diagonals should be ~0 (no x/y)
    assert abs(B[0, 0, 1]) < 1e-10
    assert abs(B[0, 0, 2]) < 1e-10
    assert abs(B[0, 1, 2]) < 1e-10

    # Convert to physical b-value (radians)
    b_measured = B[0, 2, 2]

    # Check quantitative correctness
    assert np.isclose(b_measured, b_target, rtol=1e-2)


def test_calc_moments_btensor_moments_flags():
    sys = pp.Opts(max_grad=28, grad_unit='mT/m', max_slew=120, slew_unit='T/m/s')
    seq = Sequence(sys)

    rf90 = pp.make_block_pulse(np.pi / 2, duration=1e-3, system=sys, use='excitation')
    rf180 = pp.make_block_pulse(np.pi, duration=1e-3, system=sys, use='refocusing')

    g = pp.make_trapezoid(channel='x', system=sys, amplitude=5e-3 * sys.gamma, duration=5e-3)

    seq.add_block(rf90)
    seq.add_block(pp.make_delay(2e-3))
    seq.add_block(g)
    seq.add_block(pp.make_delay(1e-3))
    seq.add_block(rf180)
    seq.add_block(pp.make_delay(1e-3))
    seq.add_block(g)
    seq.add_block(pp.make_delay(2e-3))

    B, m1, m2, m3 = seq.calc_moments_btensor(True, True, True, True, 0)

    assert np.any(np.abs(m1) > 0)
    assert np.any(np.abs(m2) > 0)
    assert np.any(np.abs(m3) > 0)
    assert np.any(np.abs(B) > 0)
