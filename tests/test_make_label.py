"""
Tests for make_label functionality and supported labels.
"""

from types import SimpleNamespace

import pypulseq as pp
import pytest
from pypulseq.make_label import make_label
from pypulseq.supported_labels_rf_use import (
    get_flag_labels,
    get_label_categories,
    get_labels_by_category,
    get_supported_labels,
    get_supported_rf_uses,
    is_valid_label,
    is_valid_rf_use,
)


def test_get_supported_labels_returns_tuple():
    """Test that get_supported_labels returns a tuple with expected labels."""
    labels = get_supported_labels()

    assert isinstance(labels, tuple)
    assert len(labels) > 0

    # Check that essential labels are present
    expected_labels = ['SLC', 'REP', 'NAV', 'PMC', 'TRID']
    for label in expected_labels:
        assert label in labels


def test_get_supported_labels_order_consistency():
    """Test that get_supported_labels returns labels in consistent order."""
    labels1 = get_supported_labels()
    labels2 = get_supported_labels()

    assert labels1 == labels2

    # Verify the expected order: counters, then data flags, then control flags
    labels_list = list(labels1)

    # First 10 should be counters
    counter_labels = ['SLC', 'SEG', 'REP', 'AVG', 'SET', 'ECO', 'PHS', 'LIN', 'PAR', 'ACQ']
    for i, expected_counter in enumerate(counter_labels):
        assert labels_list[i] == expected_counter


def test_get_supported_rf_uses_returns_expected_values():
    """Test that get_supported_rf_uses returns expected RF use cases."""
    rf_uses = get_supported_rf_uses()

    assert isinstance(rf_uses, tuple)
    expected_uses = ['excitation', 'refocusing', 'inversion', 'saturation', 'preparation']

    assert len(rf_uses) == len(expected_uses)
    for use in expected_uses:
        assert use in rf_uses


def test_get_labels_by_category_data_counters():
    """Test getting data counter labels by category."""
    counters = get_labels_by_category('data_counters')

    assert isinstance(counters, list)
    expected_counters = ['SLC', 'SEG', 'REP', 'AVG', 'SET', 'ECO', 'PHS', 'LIN', 'PAR', 'ACQ']

    assert len(counters) == len(expected_counters)
    for counter in expected_counters:
        assert counter in counters


def test_get_labels_by_category_data_flags():
    """Test getting data flag labels by category."""
    flags = get_labels_by_category('data_flags')

    assert isinstance(flags, list)
    expected_flags = ['NAV', 'REV', 'SMS', 'REF', 'IMA', 'OFF', 'NOISE']

    for flag in expected_flags:
        assert flag in flags


def test_get_labels_by_category_control_flags():
    """Test getting control flag labels by category."""
    flags = get_labels_by_category('control_flags')

    assert isinstance(flags, list)
    expected_flags = ['PMC', 'NOROT', 'NOPOS', 'NOSCL', 'ONCE', 'TRID']

    for flag in expected_flags:
        assert flag in flags


def test_get_labels_by_category_invalid_category():
    """Test that invalid category raises KeyError."""
    with pytest.raises(KeyError) as exc_info:
        get_labels_by_category('invalid_category')

    assert 'Unknown category' in str(exc_info.value)
    assert 'invalid_category' in str(exc_info.value)


def test_get_label_categories():
    """Test getting label categories with descriptions."""
    categories = get_label_categories()

    assert isinstance(categories, dict)
    expected_categories = ['data_counters', 'data_flags', 'control_flags']

    for category in expected_categories:
        assert category in categories
        assert isinstance(categories[category], str)
        assert len(categories[category]) > 0

    # Check that SET/INC information is included in descriptions
    assert 'SET and INC' in categories['data_counters']
    assert 'SET operation' in categories['data_flags']
    assert 'SET operation' in categories['control_flags']


def test_get_flag_labels():
    """Test getting all flag labels (data + control flags)."""
    flags = get_flag_labels()

    assert isinstance(flags, list)

    # Should include both data flags and control flags
    data_flags = get_labels_by_category('data_flags')
    control_flags = get_labels_by_category('control_flags')
    expected_total = len(data_flags) + len(control_flags)

    assert len(flags) == expected_total

    # All data flags should be in the result
    for flag in data_flags:
        assert flag in flags

    # All control flags should be in the result
    for flag in control_flags:
        assert flag in flags


def test_is_valid_label():
    """Test label validation function."""
    # Valid labels
    assert is_valid_label('REP') is True
    assert is_valid_label('NAV') is True
    assert is_valid_label('PMC') is True

    # Invalid labels
    assert is_valid_label('INVALID') is False
    assert is_valid_label('') is False
    assert is_valid_label('rep') is False


def test_is_valid_rf_use():
    """Test RF use validation function."""
    # Valid RF uses
    assert is_valid_rf_use('excitation') is True
    assert is_valid_rf_use('refocusing') is True
    assert is_valid_rf_use('inversion') is True

    # Invalid RF uses
    assert is_valid_rf_use('invalid') is False
    assert is_valid_rf_use('') is False
    assert is_valid_rf_use('EXCITATION') is False


def test_make_label_counter_set_operation():
    """Test creating a counter label with SET operation."""
    label = make_label('REP', 'SET', 5)

    assert isinstance(label, SimpleNamespace)
    assert label.label == 'REP'
    assert label.type == 'labelset'
    assert label.value == 5


def test_make_label_counter_inc_operation():
    """Test creating a counter label with INC operation."""
    label = make_label('REP', 'INC', 2)

    assert isinstance(label, SimpleNamespace)
    assert label.label == 'REP'
    assert label.type == 'labelinc'
    assert label.value == 2


def test_make_label_flag_set_operation():
    """Test creating a flag label with SET operation."""
    label = make_label('NAV', 'SET', 1)

    assert isinstance(label, SimpleNamespace)
    assert label.label == 'NAV'
    assert label.type == 'labelset'
    assert label.value == 1


def test_make_label_invalid_label():
    """Test that invalid label names raise ValueError."""
    with pytest.raises(ValueError) as exc_info:
        make_label('INVALID_LABEL', 'SET', 1)

    assert 'Invalid label' in str(exc_info.value)


def test_make_label_invalid_type():
    """Test that invalid operation types raise ValueError."""
    with pytest.raises(ValueError) as exc_info:
        make_label('REP', 'INVALID_TYPE', 1)

    assert "Invalid type. Must be one of 'SET' or 'INC'" in str(exc_info.value)


def test_make_label_invalid_value_type():
    """Test that invalid value types raise ValueError."""
    invalid_values = ['string', None, [1, 2, 3], {'key': 'value'}]

    for invalid_value in invalid_values:
        with pytest.raises(ValueError) as exc_info:
            make_label('REP', 'SET', invalid_value)

        assert 'Must supply a valid numerical or logical value' in str(exc_info.value)


def test_make_label_all_counter_labels():
    """Test that all counter labels work with both SET and INC."""
    counters = get_labels_by_category('data_counters')

    for counter in counters:
        # Test SET operation
        label_set = make_label(counter, 'SET', 5)
        assert label_set.label == counter
        assert label_set.type == 'labelset'
        assert label_set.value == 5

        # Test INC operation
        label_inc = make_label(counter, 'INC', 1)
        assert label_inc.label == counter
        assert label_inc.type == 'labelinc'
        assert label_inc.value == 1


def test_make_label_all_flag_labels_set_only():
    """Test that all flag labels work with SET but not INC."""
    flags = get_flag_labels()

    for flag in flags:
        # Test SET operation (should work)
        label_set = make_label(flag, 'SET', 1)
        assert label_set.label == flag
        assert label_set.type == 'labelset'
        assert label_set.value == 1

        # Test INC operation (should fail)
        with pytest.raises(ValueError):
            make_label(flag, 'INC', 1)


def test_label_integration_with_sequence():
    """Test that labels can be integrated with sequence blocks."""
    seq = pp.Sequence()

    # Create some labels
    rep_label = make_label('REP', 'SET', 1)
    nav_label = make_label('NAV', 'SET', 1)

    # Create a simple block with ADC
    adc = pp.make_adc(num_samples=64, duration=1e-3)

    # Add block with labels
    seq.add_block(adc, rep_label, nav_label)

    # Verify the sequence has the block
    assert len(seq.block_events) == 1

    # Get the block and verify it has labels
    block = seq.get_block(1)
    assert hasattr(block, 'label')

    # Labels are stored as a dictionary with indices as keys
    if hasattr(block, 'label') and block.label is not None:
        labels = block.label
        assert isinstance(labels, dict)
        assert len(labels) >= 1  # At least one label should be present

        # Verify that labels have the expected structure
        for _, label in labels.items():
            assert hasattr(label, 'label')  # Label name
            assert hasattr(label, 'type')  # Label type (labelset/labelinc)
            assert hasattr(label, 'value')  # Label value

            # Check that it's one of our expected labels
            assert label.label in ['REP', 'NAV']
            assert label.type in ['labelset', 'labelinc']

        # Verify we have the expected labels
        label_names = [label.label for label in labels.values()]
        assert 'REP' in label_names
        assert 'NAV' in label_names


def test_comprehensive_label_coverage():
    """Test that all supported labels are properly categorized."""
    all_labels = get_supported_labels()
    counters = get_labels_by_category('data_counters')
    data_flags = get_labels_by_category('data_flags')
    control_flags = get_labels_by_category('control_flags')

    # All labels should be in exactly one category
    categorized_labels = set(counters + data_flags + control_flags)
    all_labels_set = set(all_labels)

    assert categorized_labels == all_labels_set

    # No label should appear in multiple categories
    assert len(categorized_labels) == len(counters) + len(data_flags) + len(control_flags)


@pytest.mark.parametrize(
    'label_type,expected_type',
    [
        ('SET', 'labelset'),
        ('INC', 'labelinc'),
    ],
)
def test_make_label_type_mapping(label_type, expected_type):
    """Test that label types are correctly mapped to internal types."""
    if label_type == 'INC':
        # Use a counter for INC test
        label = make_label('REP', label_type, 1)
    else:
        # Use any label for SET test
        label = make_label('NAV', label_type, 1)

    assert label.type == expected_type


@pytest.mark.parametrize(
    'value,expected',
    [
        (0, 0),
        (1, 1),
        (10, 10),
        (-5, -5),
        (3.14, 3),
        (2.9, 2),
        (True, 1),
        (False, 0),
    ],
)
def test_make_label_value_conversion(value, expected):
    """Test various value conversions to integer."""
    label = make_label('REP', 'SET', value)
    assert label.value == expected
    assert isinstance(label.value, int)
