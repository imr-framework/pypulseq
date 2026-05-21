"""
Tests for soft delay functionality including numID auto-assignment and validation.
"""

import pypulseq as pp
import pytest


def test_soft_delay_numid_auto_assignment():
    """Test automatic numID assignment for soft delays."""
    seq = pp.Sequence()

    # Test 1: Auto-assignment for different hints
    te_delay = pp.make_soft_delay('TE', default_duration=5e-3)
    tr_delay = pp.make_soft_delay('TR', default_duration=100e-3)

    seq.add_block(te_delay)
    seq.add_block(tr_delay)

    # TE should get numID 0, TR should get numID 1
    assert te_delay.numID == 0, f'Expected TE numID=0, got {te_delay.numID}'
    assert tr_delay.numID == 1, f'Expected TR numID=1, got {tr_delay.numID}'

    # Test 2: Reuse numID for same hint
    te_delay2 = pp.make_soft_delay('TE', default_duration=5e-3)
    seq.add_block(te_delay2)

    # Should reuse numID 0
    assert te_delay2.numID == 0, f'Expected TE reuse numID=0, got {te_delay2.numID}'

    # Test 3: Manual numID assignment
    ti_delay = pp.make_soft_delay('TI', numID=5, default_duration=1e-3)
    seq.add_block(ti_delay)

    assert ti_delay.numID == 5, f'Expected TI numID=5, got {ti_delay.numID}'

    # Test 4: Next auto-assignment should skip used numID
    td_delay = pp.make_soft_delay('TD', default_duration=2e-3)
    seq.add_block(td_delay)

    assert td_delay.numID == 6, f'Expected TD numID=6 (next after 5), got {td_delay.numID}'


def test_soft_delay_numid_conflicts():
    """Test error handling for numID conflicts."""
    seq = pp.Sequence()

    # Create first soft delay
    te_delay1 = pp.make_soft_delay('TE', default_duration=5e-3)
    seq.add_block(te_delay1)  # Gets numID 0

    # Test 1: Try to use different numID for same hint
    with pytest.raises(ValueError, match="Soft delay hint 'TE' is already assigned to numID 0"):
        te_delay2 = pp.make_soft_delay('TE', numID=1, default_duration=5e-3)
        seq.add_block(te_delay2)

    # Test 2: Try to reuse numID for different hint
    with pytest.raises(ValueError, match="numID 0 is already used by soft delay 'TE'"):
        tr_delay = pp.make_soft_delay('TR', numID=0, default_duration=100e-3)
        seq.add_block(tr_delay)


def test_soft_delay_apply_with_auto_numid():
    """Test applying soft delays with auto-assigned numIDs."""
    seq = pp.Sequence()

    # Create soft delays with auto-assigned numIDs
    te_delay = pp.make_soft_delay('TE', default_duration=5e-3)
    tr_delay = pp.make_soft_delay('TR', default_duration=100e-3)

    seq.add_block(te_delay)
    seq.add_block(tr_delay)

    # Apply soft delays using hints (not numIDs)
    seq.apply_soft_delay(TE=8e-3, TR=500e-3)

    # Check that block durations were updated
    assert seq.block_durations[1] == 8e-3, f'Expected TE block duration 8ms, got {seq.block_durations[1] * 1e3}ms'
    assert seq.block_durations[2] == 500e-3, f'Expected TR block duration 500ms, got {seq.block_durations[2] * 1e3}ms'


def test_soft_delay_validation():
    """Test parameter validation in make_soft_delay."""

    # Test empty hint
    with pytest.raises(ValueError, match="Parameter 'hint' cannot be empty"):
        pp.make_soft_delay('', default_duration=5e-3)

    # Test whitespace in hint
    with pytest.raises(ValueError, match="Parameter 'hint' may not contain white space characters"):
        pp.make_soft_delay('T E', default_duration=5e-3)

    # Test non-string hint
    with pytest.raises(TypeError, match="'int' object is not iterable"):
        pp.make_soft_delay(123, default_duration=5e-3)

    # Test zero factor
    with pytest.raises(ValueError, match="Parameter 'factor' cannot be zero"):
        pp.make_soft_delay('TE', factor=0, default_duration=5e-3)

    # Test negative default_duration
    with pytest.raises(ValueError, match='Default duration must be greater than 0'):
        pp.make_soft_delay('TE', default_duration=-1e-3)

    # Test invalid numID
    with pytest.raises(ValueError, match="Parameter 'numID' must be a non-negative integer or None"):
        pp.make_soft_delay('TE', numID=-1, default_duration=5e-3)


def test_soft_delay_hint_consistency():
    """Test that soft delays with same hint have consistent parameters."""
    import warnings

    seq = pp.Sequence()

    # Create two soft delays with same hint but different parameters
    te_delay1 = pp.make_soft_delay('TE', offset=1e-3, factor=1.0, default_duration=5e-3)
    te_delay2 = pp.make_soft_delay('TE', offset=2e-3, factor=1.5, default_duration=8e-3)

    seq.add_block(te_delay1)
    seq.add_block(te_delay2)

    # Both should get the same numID despite different parameters
    assert te_delay1.numID == te_delay2.numID, 'Soft delays with same hint should have same numID'

    # Apply soft delay should work with either - suppress expected rounding warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        seq.apply_soft_delay(TE=10e-3)

    # Both blocks should be updated
    assert seq.block_durations[1] > 0, 'First TE block should have positive duration'
    assert seq.block_durations[2] > 0, 'Second TE block should have positive duration'


def test_soft_delay_error_messages():
    """Test that error messages are helpful and informative."""
    seq = pp.Sequence()

    # Create a soft delay
    te_delay = pp.make_soft_delay('TE', default_duration=5e-3)
    seq.add_block(te_delay)

    # Test missing soft delay error shows available options
    with pytest.raises(ValueError, match=r"Available soft delays: \['TE'\]"):
        seq.apply_soft_delay(TR=100e-3)  # TR doesn't exist

    # Test negative duration error shows helpful info
    seq_neg = pp.Sequence()
    neg_delay = pp.make_soft_delay('NEG', offset=-1e-3, factor=1.0, default_duration=5e-3)
    seq_neg.add_block(neg_delay)

    with pytest.raises(ValueError, match='Calculated duration is negative'):
        seq_neg.apply_soft_delay(NEG=0.5e-3)  # Will result in negative duration


def test_soft_delay_automatic_duration():
    """Test that default_duration automatically becomes block duration."""
    seq = pp.Sequence()

    # Create soft delay with specific default_duration
    te_delay = pp.make_soft_delay('TE', default_duration=7.5e-3)

    # Add to sequence without specifying duration
    seq.add_block(te_delay)

    # Block duration should automatically be the default_duration
    assert seq.block_durations[1] == 7.5e-3, f'Expected block duration 7.5ms, got {seq.block_durations[1] * 1e3}ms'

    # Test with different duration
    tr_delay = pp.make_soft_delay('TR', default_duration=150e-3)
    seq.add_block(tr_delay)

    assert seq.block_durations[2] == 150e-3, f'Expected block duration 150ms, got {seq.block_durations[2] * 1e3}ms'


def test_raw_float_rejection():
    """Test that raw float durations are rejected with helpful error message."""
    seq = pp.Sequence()

    # Test that raw floats are rejected
    with pytest.raises(
        ValueError,
        match=r'Raw float values are not allowed in add_block\(\)\. Use pp.make_delay\(0.001\) for delays\.',
    ):
        seq.add_block(1e-3)

    # Test that the proper way still works
    seq.add_block(pp.make_delay(1e-3))
    assert seq.block_durations[1] == 1e-3, 'make_delay should still work'


def test_soft_delay_edge_cases():
    """Test edge cases and boundary conditions for soft delays."""
    seq = pp.Sequence()

    # Test very small default_duration
    tiny_delay = pp.make_soft_delay('TINY', default_duration=1e-9)
    seq.add_block(tiny_delay)
    assert seq.block_durations[1] == 1e-9, 'Very small durations should work'

    # Test very large default_duration
    large_delay = pp.make_soft_delay('LARGE', default_duration=10.0)
    seq.add_block(large_delay)
    assert seq.block_durations[2] == 10.0, 'Large durations should work'

    # Test zero factor (should be rejected)
    with pytest.raises(ValueError, match="Parameter 'factor' cannot be zero"):
        pp.make_soft_delay('ZERO_FACTOR', factor=0.0, default_duration=1e-3)

    # Test negative factor (should work)
    neg_factor_delay = pp.make_soft_delay('NEG_FACTOR', factor=-1.0, default_duration=1e-3)
    seq.add_block(neg_factor_delay)

    # Test very large positive offset
    large_offset_delay = pp.make_soft_delay('LARGE_OFFSET', offset=1.0, default_duration=1e-3)
    seq.add_block(large_offset_delay)

    # Test very large negative offset
    neg_offset_delay = pp.make_soft_delay('NEG_OFFSET', offset=-0.5, default_duration=1e-3)
    seq.add_block(neg_offset_delay)


def test_soft_delay_apply_edge_cases():
    """Test edge cases when applying soft delays."""
    seq = pp.Sequence()

    # Create delay with negative factor and positive offset
    # Formula: duration = (input / factor) + offset
    # With factor=-2, offset=0.1, input=0.04: duration = 0.04/(-2) + 0.1 = -0.02 + 0.1 = 0.08
    tricky_delay = pp.make_soft_delay('TRICKY', factor=-2.0, offset=0.1, default_duration=0.05)
    seq.add_block(tricky_delay)

    # Apply a value that results in positive duration
    seq.apply_soft_delay(TRICKY=0.04)
    expected_duration = 0.04 / (-2.0) + 0.1  # = -0.02 + 0.1 = 0.08
    assert abs(seq.block_durations[1] - expected_duration) < 1e-10, (
        f'Expected {expected_duration}, got {seq.block_durations[1]}'
    )

    # Test applying very small values
    seq2 = pp.Sequence()
    small_delay = pp.make_soft_delay('SMALL', factor=1000.0, offset=0, default_duration=1e-3)
    seq2.add_block(small_delay)
    seq2.apply_soft_delay(SMALL=1e-6)  # Very small input
    expected_small = 1e-6 / 1000.0  # = 1e-9
    # Use looser tolerance due to rounding to block duration raster
    assert abs(seq2.block_durations[1] - expected_small) < 1e-9, 'Very small applied values should work'

    # Test applying very large values
    seq3 = pp.Sequence()
    large_delay = pp.make_soft_delay('BIG', factor=0.1, offset=0, default_duration=1e-3)
    seq3.add_block(large_delay)
    seq3.apply_soft_delay(BIG=100.0)  # Large input
    expected_large = 100.0 / 0.1  # = 1000.0
    assert abs(seq3.block_durations[1] - expected_large) < 1e-10, 'Large applied values should work'


def test_soft_delay_multiple_sequences():
    """Test soft delays across multiple sequence instances."""
    # Test that numID assignment is independent across sequences
    seq1 = pp.Sequence()
    seq2 = pp.Sequence()

    # Both sequences should start numID assignment from 0
    te1 = pp.make_soft_delay('TE', default_duration=5e-3)
    te2 = pp.make_soft_delay('TE', default_duration=5e-3)

    seq1.add_block(te1)
    seq2.add_block(te2)

    # Both should get numID 0 in their respective sequences
    assert te1.numID == 0, 'First sequence should start numID from 0'
    assert te2.numID == 0, 'Second sequence should also start numID from 0'

    # Add more delays to each sequence
    tr1 = pp.make_soft_delay('TR', default_duration=100e-3)
    tr2 = pp.make_soft_delay('TR', default_duration=100e-3)

    seq1.add_block(tr1)
    seq2.add_block(tr2)

    # Both should get numID 1
    assert tr1.numID == 1, 'TR in first sequence should get numID 1'
    assert tr2.numID == 1, 'TR in second sequence should get numID 1'


def test_soft_delay_hint_edge_cases():
    """Test edge cases for hint parameter validation."""
    # Test single character hint
    single_char = pp.make_soft_delay('T', default_duration=1e-3)
    assert single_char.hint == 'T', 'Single character hints should work'

    # Test long hint
    long_hint = pp.make_soft_delay('VERY_LONG_HINT_NAME_WITH_UNDERSCORES', default_duration=1e-3)
    assert long_hint.hint == 'VERY_LONG_HINT_NAME_WITH_UNDERSCORES', 'Long hints should work'

    # Test hint with numbers
    numeric_hint = pp.make_soft_delay('TE123', default_duration=1e-3)
    assert numeric_hint.hint == 'TE123', 'Hints with numbers should work'

    # Test hint with special characters (except whitespace)
    special_hint = pp.make_soft_delay('TE-TR_123', default_duration=1e-3)
    assert special_hint.hint == 'TE-TR_123', 'Hints with hyphens and underscores should work'

    # Test that whitespace is still rejected
    with pytest.raises(ValueError, match="Parameter 'hint' may not contain white space characters"):
        pp.make_soft_delay('TE TR', default_duration=1e-3)

    # Test tab character
    with pytest.raises(ValueError, match="Parameter 'hint' may not contain white space characters"):
        pp.make_soft_delay('TE\tTR', default_duration=1e-3)

    # Test newline character
    with pytest.raises(ValueError, match="Parameter 'hint' may not contain white space characters"):
        pp.make_soft_delay('TE\nTR', default_duration=1e-3)


def test_soft_delay_numid_edge_cases():
    """Test edge cases for numID parameter."""
    seq = pp.Sequence()

    # Test numID = 0 (should work)
    delay_zero = pp.make_soft_delay('ZERO', numID=0, default_duration=1e-3)
    seq.add_block(delay_zero)
    assert delay_zero.numID == 0, 'numID=0 should work'

    # Test very large numID
    delay_large = pp.make_soft_delay('LARGE', numID=999999, default_duration=1e-3)
    seq.add_block(delay_large)
    assert delay_large.numID == 999999, 'Large numID should work'

    # Test that next auto-assignment skips the large numID
    delay_auto = pp.make_soft_delay('AUTO', default_duration=1e-3)
    seq.add_block(delay_auto)
    assert delay_auto.numID == 1000000, f'Auto-assignment should skip to {1000000}, got {delay_auto.numID}'

    # Test negative numID (should be rejected)
    with pytest.raises(ValueError, match="Parameter 'numID' must be a non-negative integer or None"):
        pp.make_soft_delay('NEG', numID=-1, default_duration=1e-3)

    # Test float numID (should be rejected)
    with pytest.raises(ValueError, match="Parameter 'numID' must be a non-negative integer or None"):
        pp.make_soft_delay('FLOAT', numID=1.5, default_duration=1e-3)


def test_soft_delay_sequence_integration():
    """Test soft delays integrated with other sequence events."""
    seq = pp.Sequence()

    # Create a sequence with mixed events
    rf_pulse = pp.make_block_pulse(flip_angle=1.57, duration=1e-3)  # π/2 pulse
    grad_x = pp.make_trapezoid('x', area=1000)
    adc_event = pp.make_adc(num_samples=100, duration=5e-3)
    te_delay = pp.make_soft_delay('TE', default_duration=10e-3)

    # Add events in sequence
    seq.add_block(rf_pulse)
    seq.add_block(grad_x)
    seq.add_block(te_delay)
    seq.add_block(adc_event)

    # Check that soft delay block has correct duration
    assert seq.block_durations[3] == 10e-3, 'Soft delay should have correct duration in mixed sequence'

    # store current durations for comparison
    dur1 = seq.block_durations[1]
    dur2 = seq.block_durations[2]
    dur4 = seq.block_durations[4]

    # Apply soft delay and check duration changes
    seq.apply_soft_delay(TE=20e-3)
    assert seq.block_durations[3] == 20e-3, 'Applied soft delay should update duration'

    # Check that other blocks are unaffected
    assert seq.block_durations[1] == dur1, 'RF block should be unaffected'
    assert seq.block_durations[2] == dur2, 'Gradient block should be unaffected'
    assert seq.block_durations[4] == dur4, 'ADC block should be unaffected'
