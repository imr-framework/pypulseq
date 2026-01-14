import numpy as np
import pypulseq as pp
import pytest
from pypulseq import scale_grad
from pypulseq.opts import Opts

# Updated gradients with realistic hardware limits
grad_list = [
    pp.make_trapezoid(channel='x', amplitude=10, duration=13, max_grad=30, max_slew=200),
    pp.make_trapezoid(channel='y', amplitude=10, duration=13, max_grad=30, max_slew=200),
    pp.make_trapezoid(channel='z', amplitude=10, duration=13, max_grad=30, max_slew=200),
    pp.make_trapezoid(channel='x', amplitude=20, duration=5, max_grad=25, max_slew=150),
    pp.make_trapezoid(channel='y', amplitude=20, duration=5, max_grad=25, max_slew=150),
    pp.make_trapezoid(channel='z', amplitude=20, duration=5, max_grad=25, max_slew=150),
    pp.make_extended_trapezoid(
        'x', [0, 15, 5, 10], convert_to_arbitrary=True, times=[1, 3, 4, 7], max_grad=40, max_slew=300
    ),
    pp.make_extended_trapezoid(
        'y', [0, 15, 5, 10], convert_to_arbitrary=True, times=[1, 3, 4, 7], max_grad=40, max_slew=300
    ),
    pp.make_extended_trapezoid(
        'z', [0, 15, 5, 10], convert_to_arbitrary=True, times=[1, 3, 4, 7], max_grad=40, max_slew=300
    ),
    pp.make_extended_trapezoid(
        'x', [0, 20, 10, 15], convert_to_arbitrary=False, times=[1, 3, 4, 7], max_grad=25, max_slew=150
    ),
    pp.make_extended_trapezoid(
        'y', [0, 20, 10, 15], convert_to_arbitrary=False, times=[1, 3, 4, 7], max_grad=25, max_slew=150
    ),
    pp.make_extended_trapezoid(
        'z', [0, 20, 10, 15], convert_to_arbitrary=False, times=[1, 3, 4, 7], max_grad=25, max_slew=150
    ),
    pp.make_extended_trapezoid(
        'x', [0, 10, 5, 10], convert_to_arbitrary=False, times=[1, 2, 3, 4], max_grad=15, max_slew=80
    ),
    pp.make_extended_trapezoid(
        'y', [0, 10, 5, 10], convert_to_arbitrary=False, times=[1, 2, 3, 4], max_grad=15, max_slew=80
    ),
    pp.make_extended_trapezoid(
        'z', [0, 10, 5, 10], convert_to_arbitrary=False, times=[1, 2, 3, 4], max_grad=15, max_slew=80
    ),
]

# ----------- TEST SCALING IS CORRECT -----------


@pytest.mark.parametrize('grad', grad_list)
def test_scale_grad_correct_scaling(grad):
    scale = 0.5  # Safe scale
    system = Opts(max_grad=40, max_slew=300)
    scaled = scale_grad(grad, scale, system)

    if grad.type == 'trap':
        assert np.isclose(scaled.amplitude, grad.amplitude * scale)
        assert np.isclose(scaled.flat_area, grad.flat_area * scale)
    else:
        assert np.allclose(scaled.waveform, grad.waveform * scale)
        assert np.isclose(scaled.first, grad.first * scale)
        assert np.isclose(scaled.last, grad.last * scale)

    assert np.isclose(scaled.area, grad.area * scale)

    # ID must be removed
    assert not hasattr(scaled, 'id')


# ----------- TEST AMPLITUDE VIOLATION -----------


def test_scale_grad_violates_amplitude():
    scale = 100.0
    system = Opts(max_grad=40, max_slew=999999999)  # make sure we have infinite slew rate

    expected_failures = 0
    actual_failures = 0

    for grad in grad_list:
        if grad.type == 'trap':
            should_fail = abs(grad.amplitude) * scale > system.max_grad
        else:
            should_fail = max(abs(grad.waveform)) * scale > system.max_grad

        if should_fail:
            expected_failures += 1
            with pytest.raises(ValueError, match='maximum amplitude exceeded'):
                scale_grad(grad, scale, system)
            actual_failures += 1
        else:
            scale_grad(grad, scale, system)

    assert expected_failures == actual_failures, (
        f'Expected {expected_failures} amplitude violations, but got {actual_failures}.'
    )


# ----------- TEST SLEW RATE VIOLATION -----------


def test_scale_grad_violates_slew():
    scale = 100.0
    system = Opts(max_grad=999999999, max_slew=300)  # make sure we have infinite gradient amp

    expected_failures = 0
    actual_failures = 0

    for grad in grad_list:
        if grad.type == 'trap':
            if abs(grad.amplitude) > 1e-6:
                approx_slew = abs(grad.amplitude * scale) / min(grad.rise_time, grad.fall_time)
                should_fail = approx_slew > system.max_slew
            else:
                should_fail = False
        else:
            if max(abs(grad.waveform)) > 1e-6:
                waveform = np.array(grad.waveform) * scale
                tt = np.array(grad.tt)
                diffs = np.abs(np.diff(waveform) / np.diff(tt))
                should_fail = max(diffs) > system.max_slew
            else:
                should_fail = False

        if should_fail:
            expected_failures += 1
            with pytest.raises(ValueError, match='maximum slew rate exceeded'):
                scale_grad(grad, scale, system)
            actual_failures += 1
        else:
            scale_grad(grad, scale, system)

    assert expected_failures == actual_failures, (
        f'Expected {expected_failures} slew violations, but got {actual_failures}.'
    )
