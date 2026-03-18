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


def test_plot_rf_plot_modes():
    """Test rf_plot parameter with different modes."""
    plt.close('all')
    seq = create_test_sequence()

    # Test all valid modes without raising
    for mode in ['auto', 'abs', 'real', 'imag']:
        result = seq.plot(plot_now=False, rf_plot=mode)
        assert result.fig1 is not None, f"Failed for rf_plot='{mode}'"
        plt.close('all')


def test_plot_rf_plot_invalid_mode():
    """Test that invalid rf_plot mode raises ValueError."""
    plt.close('all')
    seq = create_test_sequence()

    try:
        seq.plot(plot_now=False, rf_plot='invalid')
        raise AssertionError('Should have raised ValueError for invalid rf_plot mode')
    except ValueError as e:
        assert 'Unsupported RF plot type' in str(e)

    plt.close('all')


def test_plot_rf_plot_auto_default():
    """Test that rf_plot='auto' is the default and produces same result as no argument."""
    plt.close('all')
    seq1 = create_test_sequence()
    seq2 = create_test_sequence()

    result_default = seq1.plot(plot_now=False)
    result_auto = seq2.plot(plot_now=False, rf_plot='auto')

    # Both should succeed and produce figures
    assert result_default.fig1 is not None
    assert result_auto.fig1 is not None

    plt.close('all')


def test_plot_rf_plot_with_overlay():
    """Test that rf_plot parameter works correctly with overlay."""
    plt.close('all')
    seq = create_test_sequence()

    # First plot with rf_plot='real'
    sp1 = seq.plot(plot_now=False, rf_plot='real')

    # Overlay with rf_plot='abs'
    sp2 = seq.plot(overlay=sp1, plot_now=False, rf_plot='abs')

    # Should reuse figures from overlay
    assert id(sp2.fig1) == id(sp1.fig1)

    plt.close('all')
