"""Simple unit tests for Sequence.plot() method."""

import math

import matplotlib.pyplot as plt
import pypulseq as pp


def create_test_sequence():
    """Create a test sequence with RF, gradients, and ADC - similar to seq5() but simpler."""
    seq = pp.Sequence()
    rf, gz, gzr = pp.make_sinc_pulse(flip_angle=math.pi / 8, duration=1e-3, slice_thickness=3e-3, return_gz=True)
    gx = pp.make_trapezoid('x', flat_area=100, flat_time=5e-3)
    adc = pp.make_adc(num_samples=128, duration=5e-3)

    seq.add_block(rf, gz)
    seq.add_block(gzr)
    seq.add_block(gx, adc)
    return seq


def test_plot_default_behavior_no_figures_provided():
    """Test that default behavior works when no figures are provided."""
    plt.close('all')
    seq = create_test_sequence()

    result = seq.plot(plot_now=False)
    fig1, axes1, fig2, axes2 = result.fig1, result.ax1, result.fig2, result.ax2

    # Should create new figures
    assert isinstance(fig1, plt.Figure)
    assert isinstance(fig2, plt.Figure)
    assert len(axes1) == 3
    assert len(axes2) == 3

    # Figures should have content (subplots)
    assert len(fig1.get_axes()) == 3
    assert len(fig2.get_axes()) == 3

    plt.close('all')


def test_plot_overlay_behavior():
    """Test overlay functionality."""
    plt.close('all')
    seq = create_test_sequence()

    # First plot
    sp1 = seq.plot(plot_now=False)
    fig1_id = id(sp1.fig1)

    # Second plot with overlay
    seq.mod_grad_axis('x', 2.0)
    sp2 = seq.plot(overlay=sp1, plot_now=False)

    # Should reuse the same figure objects
    assert id(sp2.fig1) == fig1_id
    assert id(sp2.fig2) == id(sp1.fig2)

    plt.close('all')


def test_plot_stacked_behavior():
    """Test stacked plotting."""
    plt.close('all')
    seq = create_test_sequence()

    result = seq.plot(stacked=True, plot_now=False)

    assert result.fig1 is not None
    assert result.fig2 is None  # In stacked mode, fig2 is None (or unused)
    assert len(result.ax1) == 6  # 3 for RF/ADC + 3 for Gradients

    plt.close('all')
