"""Tests for Sequence.plot() function."""

import math

import matplotlib.pyplot as plt
import pypulseq as pp
import pytest


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
    fig1, axes1, fig2, axes2 = result

    # Should create new figures
    assert isinstance(fig1, plt.Figure)
    assert isinstance(fig2, plt.Figure)
    assert len(axes1) == 3
    assert len(axes2) == 3

    # Figures should have content (subplots)
    assert len(fig1.get_axes()) == 3
    assert len(fig2.get_axes()) == 3

    plt.close('all')


def test_plot_custom_figures_provided():
    """Test that custom figures are used when provided."""
    plt.close('all')
    seq = create_test_sequence()

    # Create custom figures
    custom_fig1 = plt.figure(figsize=(10, 8))
    custom_fig2 = plt.figure(figsize=(12, 6))

    # Store references for comparison
    fig1_id = id(custom_fig1)
    fig2_id = id(custom_fig2)

    result = seq.plot(fig1=custom_fig1, fig2=custom_fig2, plot_now=False)
    returned_fig1, _, returned_fig2, _ = result

    # Should return the same figure objects
    assert id(returned_fig1) == fig1_id
    assert id(returned_fig2) == fig2_id
    assert returned_fig1.get_figwidth() == 10
    assert returned_fig1.get_figheight() == 8
    assert returned_fig2.get_figwidth() == 12
    assert returned_fig2.get_figheight() == 6

    plt.close('all')


def test_plot_mixed_figures_one_provided_one_default():
    """Test providing only one custom figure."""
    plt.close('all')
    seq = create_test_sequence()

    # Create only one custom figure
    custom_fig1 = plt.figure(figsize=(10, 8))
    fig1_id = id(custom_fig1)

    result = seq.plot(fig1=custom_fig1, plot_now=False)
    returned_fig1, _, returned_fig2, _ = result

    # fig1 should be our custom figure
    assert id(returned_fig1) == fig1_id
    assert returned_fig1.get_figwidth() == 10

    # fig2 should be a new default figure
    assert isinstance(returned_fig2, plt.Figure)
    assert id(returned_fig2) != fig1_id

    plt.close('all')


def test_plot_clear_behavior_with_custom_figures():
    """Test that clear parameter works correctly with custom figures."""
    plt.close('all')
    seq = create_test_sequence()

    # Create figures with existing content
    custom_fig1 = plt.figure()
    custom_fig2 = plt.figure()

    # Add some content to the figures
    custom_fig1.add_subplot(111).plot([1, 2, 3], [1, 4, 9])
    custom_fig2.add_subplot(111).plot([1, 2, 3], [2, 4, 6])

    # Test clear=True (should clear existing content)
    result = seq.plot(fig1=custom_fig1, fig2=custom_fig2, clear=True, plot_now=False)
    returned_fig1, _, returned_fig2, _ = result

    # Should have exactly 3 subplots each (our sequence plot structure)
    assert len(returned_fig1.get_axes()) == 3
    assert len(returned_fig2.get_axes()) == 3

    plt.close('all')


def test_plot_overlay_behavior_with_custom_figures():
    """Test overlay functionality with custom figures."""
    plt.close('all')
    seq = create_test_sequence()

    # Create figures and plot first sequence
    custom_fig1 = plt.figure()
    custom_fig2 = plt.figure()

    result1 = seq.plot(fig1=custom_fig1, fig2=custom_fig2, plot_now=False)

    # Modify sequence and overlay
    seq.mod_grad_axis('x', 2.0)
    result2 = seq.plot(fig1=custom_fig1, fig2=custom_fig2, clear=False, plot_now=False)

    # Should still return the same figure objects
    assert id(result1[0]) == id(result2[0])
    assert id(result1[2]) == id(result2[2])

    # Should have 3 subplots each
    assert len(result2[0].get_axes()) == 3
    assert len(result2[2].get_axes()) == 3

    plt.close('all')


def test_plot_figure_numbering_independence():
    """Test that custom figures don't interfere with matplotlib's figure numbering."""
    plt.close('all')
    seq = create_test_sequence()

    # Create some numbered figures first
    plt.figure(1)
    plt.figure(2)
    plt.figure(5)

    # Create custom figures
    custom_fig1 = plt.figure()  # Should get number 6
    custom_fig2 = plt.figure()  # Should get number 7

    # Use custom figures in plot
    result = seq.plot(fig1=custom_fig1, fig2=custom_fig2, plot_now=False)
    returned_fig1, _, returned_fig2, _ = result

    # Should return our custom figures
    assert returned_fig1.number == custom_fig1.number
    assert returned_fig2.number == custom_fig2.number

    # Original numbered figures should still exist
    assert plt.figure(1) is not None
    assert plt.figure(2) is not None
    assert plt.figure(5) is not None

    plt.close('all')


def test_plot_subplot_sharing_with_custom_figures():
    """Test that x-axis sharing works correctly with custom figures."""
    plt.close('all')
    seq = create_test_sequence()

    custom_fig1 = plt.figure()
    custom_fig2 = plt.figure()

    result = seq.plot(fig1=custom_fig1, fig2=custom_fig2, plot_now=False)
    _, (sp11, sp12, sp13), _, (sp21, sp22, sp23) = result

    # Check that subplots share x-axis correctly
    # sp12 and sp13 should share x-axis with sp11
    assert sp12.get_shared_x_axes().joined(sp11, sp12)
    assert sp13.get_shared_x_axes().joined(sp11, sp13)

    # sp21, sp22, sp23 should share x-axis with sp11
    assert sp21.get_shared_x_axes().joined(sp11, sp21)
    assert sp22.get_shared_x_axes().joined(sp11, sp22)
    assert sp23.get_shared_x_axes().joined(sp11, sp23)

    plt.close('all')


def test_plot_figure_size_preservation():
    """Test that custom figure sizes are preserved."""
    plt.close('all')
    seq = create_test_sequence()

    # Create figures with specific sizes
    fig1 = plt.figure(figsize=(15, 10))
    fig2 = plt.figure(figsize=(8, 12))

    result = seq.plot(fig1=fig1, fig2=fig2, plot_now=False)
    returned_fig1, _, returned_fig2, _ = result

    # Sizes should be preserved
    assert returned_fig1.get_figwidth() == 15
    assert returned_fig1.get_figheight() == 10
    assert returned_fig2.get_figwidth() == 8
    assert returned_fig2.get_figheight() == 12

    plt.close('all')


def test_plot_error_handling_invalid_figure_types():
    """Test error handling for invalid figure types."""
    seq = create_test_sequence()

    with pytest.raises(AttributeError):
        # Should fail when trying to call methods on non-Figure objects
        seq.plot(fig1='not_a_figure', plot_now=False)

    with pytest.raises(AttributeError):
        seq.plot(fig2=123, plot_now=False)
