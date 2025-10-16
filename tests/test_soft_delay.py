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
    with pytest.raises(TypeError, match="argument of type 'int' is not iterable"):
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
