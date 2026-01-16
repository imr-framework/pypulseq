"""
Test functions for the mod_grad_axis method in pypulseq.Sequence.
"""

import numpy as np
import pypulseq as pp
import pytest


def test_mod_grad_axis_trapezoid_amplitude_scaling():
    """Test that trapezoid gradient amplitude scaling works correctly."""
    seq = pp.Sequence()

    # Create trapezoid gradients on different axes
    gx = pp.make_trapezoid('x', amplitude=1000, flat_time=5e-3)
    gy = pp.make_trapezoid('y', amplitude=2000, flat_time=3e-3)
    gz = pp.make_trapezoid('z', amplitude=1500, flat_time=4e-3)

    seq.add_block(gx, gy, gz)

    # Store original values
    original_gx_amp = gx.amplitude
    original_gx_area = gx.area
    original_gx_timing = (gx.rise_time, gx.flat_time, gx.fall_time, gx.delay)

    # Scale x-axis by factor of 2
    seq.mod_grad_axis('x', 2.0)

    # Get the modified gradient
    block = seq.get_block(1)
    modified_gx = block.gx

    # Check amplitude scaling
    assert abs(modified_gx.amplitude - 2 * original_gx_amp) < 1e-10, (
        f'Expected amplitude {2 * original_gx_amp}, got {modified_gx.amplitude}'
    )

    # Check area scaling (area should scale with amplitude)
    assert abs(modified_gx.area - 2 * original_gx_area) < 1e-10, (
        f'Expected area {2 * original_gx_area}, got {modified_gx.area}'
    )

    # Check timing unchanged
    modified_timing = (modified_gx.rise_time, modified_gx.flat_time, modified_gx.fall_time, modified_gx.delay)
    for orig, mod in zip(original_gx_timing, modified_timing, strict=False):
        assert abs(orig - mod) < 1e-12, f'Timing should not change: {original_gx_timing} vs {modified_timing}'

    # Check other axes unchanged
    assert abs(block.gy.amplitude - gy.amplitude) < 1e-12, 'Y gradient should be unchanged'
    assert abs(block.gz.amplitude - gz.amplitude) < 1e-12, 'Z gradient should be unchanged'


def test_mod_grad_axis_arbitrary_gradient_scaling():
    """Test that arbitrary gradient amplitude scaling works correctly."""
    seq = pp.Sequence()

    # Create arbitrary gradient using make_extended_trapezoid (simpler than make_arbitrary_grad)
    gx = pp.make_extended_trapezoid('x', amplitudes=[0, 1000, 1000, 0], times=[0, 1e-3, 4e-3, 5e-3])

    seq.add_block(gx)

    # Store original values
    original_amplitude = gx.waveform.max()
    original_area = np.trapezoid(gx.waveform, gx.tt)
    original_first = gx.first
    original_last = gx.last

    # Scale x-axis by factor of 3
    seq.mod_grad_axis('x', 3.0)

    # Get the modified gradient
    block = seq.get_block(1)
    modified_gx = block.gx

    # Check amplitude scaling
    assert abs(modified_gx.waveform.max() - 3 * original_amplitude) < 1e-10, 'Waveform amplitude should scale by factor'

    # Check area scaling
    modified_area = np.trapezoid(modified_gx.waveform, modified_gx.tt)
    assert abs(modified_area - 3 * original_area) < 1e-10, 'Waveform area should scale by factor'

    # Check first and last values scaling
    assert abs(modified_gx.first - 3 * original_first) < 1e-10, 'First value should scale by factor'
    assert abs(modified_gx.last - 3 * original_last) < 1e-10, 'Last value should scale by factor'

    # Check timing unchanged
    np.testing.assert_array_almost_equal(modified_gx.tt, gx.tt, decimal=12, err_msg='Time array should not change')


def test_mod_grad_axis_zero_scaling():
    """Test setting gradients to zero."""
    seq = pp.Sequence()

    # Create gradients on all axes
    gx = pp.make_trapezoid('x', amplitude=1000, flat_time=5e-3)
    gy = pp.make_trapezoid('y', amplitude=2000, flat_time=3e-3)
    gz = pp.make_trapezoid('z', amplitude=1500, flat_time=4e-3)

    seq.add_block(gx, gy, gz)

    # Zero out y-axis
    seq.mod_grad_axis('y', 0)

    # Get the modified block
    block = seq.get_block(1)

    # Check y-gradient is zeroed
    assert abs(block.gy.amplitude) < 1e-12, 'Y gradient amplitude should be zero'
    assert abs(block.gy.area) < 1e-12, 'Y gradient area should be zero'

    # Check other axes unchanged
    assert abs(block.gx.amplitude - gx.amplitude) < 1e-12, 'X gradient should be unchanged'
    assert abs(block.gz.amplitude - gz.amplitude) < 1e-12, 'Z gradient should be unchanged'

    # Check timing preserved
    assert abs(block.gy.rise_time - gy.rise_time) < 1e-12, 'Rise time should be preserved'
    assert abs(block.gy.flat_time - gy.flat_time) < 1e-12, 'Flat time should be preserved'
    assert abs(block.gy.fall_time - gy.fall_time) < 1e-12, 'Fall time should be preserved'


@pytest.mark.parametrize('factor', [2.5, -1.7, 0.3, 10.0])
def test_mod_grad_axis_consistency_scaling(factor):
    """Test consistency: scale by factor 'a' then by '1/a' should return to original."""
    seq = pp.Sequence()

    # Create gradients with different types
    gx_trap = pp.make_trapezoid('x', amplitude=1234.5, flat_time=5e-3)

    # Create arbitrary gradient using make_extended_trapezoid
    gy_arb = pp.make_extended_trapezoid('y', amplitudes=[0, 987.6, 500, 0], times=[0, 1e-3, 3e-3, 4e-3])

    gz_trap = pp.make_trapezoid('z', amplitude=-567.8, flat_time=3e-3)

    seq.add_block(gx_trap, gy_arb, gz_trap)

    # Store original values
    block_orig = seq.get_block(1)
    orig_gx_amp = block_orig.gx.amplitude
    orig_gx_area = block_orig.gx.area
    orig_gy_waveform = block_orig.gy.waveform.copy()
    orig_gy_area = np.trapezoid(block_orig.gy.waveform, block_orig.gy.tt)
    orig_gz_amp = block_orig.gz.amplitude
    orig_gz_area = block_orig.gz.area

    # Scale by factor
    seq.mod_grad_axis('x', factor)
    seq.mod_grad_axis('y', factor)
    seq.mod_grad_axis('z', factor)

    # Scale back by 1/factor
    seq.mod_grad_axis('x', 1.0 / factor)
    seq.mod_grad_axis('y', 1.0 / factor)
    seq.mod_grad_axis('z', 1.0 / factor)

    # Check we're back to original values
    block_final = seq.get_block(1)

    # Check trapezoid gradients
    assert abs(block_final.gx.amplitude - orig_gx_amp) < 1e-10, f'X amplitude not restored after factor {factor}'
    assert abs(block_final.gx.area - orig_gx_area) < 1e-10, f'X area not restored after factor {factor}'

    assert abs(block_final.gz.amplitude - orig_gz_amp) < 1e-10, f'Z amplitude not restored after factor {factor}'
    assert abs(block_final.gz.area - orig_gz_area) < 1e-10, f'Z area not restored after factor {factor}'

    # Check arbitrary gradient
    np.testing.assert_array_almost_equal(
        block_final.gy.waveform,
        orig_gy_waveform,
        decimal=10,
        err_msg=f'Y waveform not restored after factor {factor}',
    )

    final_gy_area = np.trapezoid(block_final.gy.waveform, block_final.gy.tt)
    assert abs(final_gy_area - orig_gy_area) < 1e-10, f'Y area not restored after factor {factor}'


def test_mod_grad_axis_negative_scaling():
    """Test gradient inversion with negative scaling."""
    seq = pp.Sequence()

    gx = pp.make_trapezoid('x', amplitude=1000, flat_time=5e-3)
    seq.add_block(gx)

    # Store original values
    original_amp = gx.amplitude
    original_area = gx.area

    # Invert gradient
    seq.mod_grad_axis('x', -1)

    # Check inversion
    block = seq.get_block(1)
    assert abs(block.gx.amplitude - (-original_amp)) < 1e-12, 'Amplitude should be inverted'
    assert abs(block.gx.area - (-original_area)) < 1e-12, 'Area should be inverted'


def test_mod_grad_axis_multiple_blocks():
    """Test scaling gradients across multiple blocks."""
    seq = pp.Sequence()

    # Create multiple blocks with y-gradients
    gy1 = pp.make_trapezoid('y', amplitude=1000, flat_time=2e-3)
    gy2 = pp.make_trapezoid('y', amplitude=1500, flat_time=3e-3)
    gy3 = pp.make_trapezoid('y', amplitude=-800, flat_time=1e-3)

    seq.add_block(gy1)
    seq.add_block(gy2)
    seq.add_block(gy3)

    # Scale all y-gradients by 2.5
    seq.mod_grad_axis('y', 2.5)

    # Check all blocks are scaled
    block1 = seq.get_block(1)
    block2 = seq.get_block(2)
    block3 = seq.get_block(3)

    assert abs(block1.gy.amplitude - 2.5 * 1000) < 1e-10, 'Block 1 not scaled'
    assert abs(block2.gy.amplitude - 2.5 * 1500) < 1e-10, 'Block 2 not scaled'
    assert abs(block3.gy.amplitude - 2.5 * (-800)) < 1e-10, 'Block 3 not scaled'


def test_mod_grad_axis_invalid_axis():
    """Test error handling for invalid axis."""
    seq = pp.Sequence()

    with pytest.raises(ValueError, match='Invalid axis'):
        seq.mod_grad_axis('w', 1.0)

    with pytest.raises(ValueError, match='Invalid axis'):
        seq.mod_grad_axis('xy', 1.0)


def test_mod_grad_axis_shared_gradient_error():
    """Test error when same gradient is used on multiple axes."""
    seq = pp.Sequence()

    # Create a gradient
    gx = pp.make_trapezoid('x', amplitude=1000, flat_time=5e-3)
    seq.add_block(gx)

    block_id = next(iter(seq.block_events.keys()))
    original_block = next(iter(seq.block_events.values()))

    # Modify the block to have the same gradient ID on both x and y axes
    # Block structure: [rf_id, gx_id, gy_id, gz_id, adc_id, delay_id, ...]
    modified_block = list(original_block)
    modified_block[3] = modified_block[2]  # Set gy_id = gx_id (same gradient on both axes)
    seq.block_events[block_id] = tuple(modified_block)

    # Now trying to modify x-axis should raise RuntimeError
    with pytest.raises(
        RuntimeError, match='mod_grad_axis does not yet support the same gradient event used on multiple axes'
    ):
        seq.mod_grad_axis('x', 2.0)


def test_mod_grad_axis_empty_sequence():
    """Test mod_grad_axis on empty sequence (should do nothing)."""
    seq = pp.Sequence()

    # Should not raise error
    seq.mod_grad_axis('x', 2.0)

    # Sequence should still be empty
    assert len(seq.block_events) == 0, 'Empty sequence should remain empty'


def test_mod_grad_axis_no_gradients_on_axis():
    """Test mod_grad_axis when no gradients exist on specified axis."""
    seq = pp.Sequence()

    # Add gradients only on x and z
    gx = pp.make_trapezoid('x', amplitude=1000, flat_time=5e-3)
    gz = pp.make_trapezoid('z', amplitude=1500, flat_time=3e-3)
    seq.add_block(gx, gz)

    # Try to modify y-axis (no gradients there)
    seq.mod_grad_axis('y', 2.0)  # Should do nothing

    # Check x and z are unchanged
    block = seq.get_block(1)
    assert abs(block.gx.amplitude - 1000) < 1e-12, 'X gradient should be unchanged'
    assert abs(block.gz.amplitude - 1500) < 1e-12, 'Z gradient should be unchanged'
    assert not hasattr(block, 'gy') or block.gy is None, 'Y gradient should not exist'


def test_mod_grad_axis_keymap_integrity():
    """Test that EventLibrary keymap integrity is maintained after modification."""
    seq = pp.Sequence()

    # Create a gradient and add it to the sequence
    gx = pp.make_trapezoid('x', amplitude=1000, flat_time=5e-3)
    seq.add_block(gx)

    # Store original library state
    original_keymap_size = len(seq.grad_library.keymap)
    original_data_size = len(seq.grad_library.data)

    # Modify the gradient
    seq.mod_grad_axis('x', -1.0)

    # Check that library sizes are maintained (no duplicates created)
    assert len(seq.grad_library.keymap) == original_keymap_size, 'Keymap size should be unchanged'
    assert len(seq.grad_library.data) == original_data_size, 'Data size should be unchanged'

    # Verify that a modification does not create a duplicate gradient event
    gx_inverted = pp.make_trapezoid('x', amplitude=-1000, flat_time=5e-3)
    seq.add_block(gx_inverted)

    # Library should still have the same number of unique gradients
    assert len(seq.grad_library.data) == original_data_size, 'No duplicate gradient should be created'

    # Verify that both gradients reference the same gradient event
    block1_events = seq.block_events[1]
    block2_events = seq.block_events[2]

    # The gradient event ID is at index 2
    gx_event_id_block1 = block1_events[2]
    gx_event_id_block2 = block2_events[2]

    assert gx_event_id_block1 == gx_event_id_block2, 'Both gradients should reference the same gradient event'
