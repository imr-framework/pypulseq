from pathlib import Path

import pypulseq as pp
import pytest

data_path = Path(__file__).parent / 'expected_output'

old_seq_files = [
    'simple_mprage120.seq',
    'simple_mprage130.seq',
    'simple_mprage131.seq',
    'simple_mprage140.seq',
    'simple_mprage141.seq',
    'simple_mprage142.seq',
]


@pytest.mark.parametrize('seq_file', old_seq_files)
def test_sequence_backwards_compatibility(seq_file):
    path = data_path / seq_file

    # Get version of seq file to be tested
    version = int(seq_file.split('.')[0][-3:])

    # Read v1.5.0 seq fileas reference
    seq150 = pp.Sequence()
    seq150.read(data_path / 'simple_mprage150.seq')

    # Read seq file to be tested
    seq = pp.Sequence()
    if version < 141:
        with pytest.warns(UserWarning):
            seq.read(path)
    else:
        seq.read(path)

    # Check that the number of blocks is the same
    assert len(seq150.block_events) == len(seq.block_events)

    # Check that the number of RF events is the same
    assert len(seq150.rf_library.data) == len(seq.rf_library.data)

    # Check that the number of arbitrary gradients is the same
    n_grad_ref = sum('g' in v for v in seq150.grad_library.type.values())
    n_grad_cur = sum('g' in v for v in seq.grad_library.type.values())
    assert n_grad_ref == n_grad_cur

    # Check that the number of trapezoid gradients is the same
    n_trap_ref = sum('t' in v for v in seq150.grad_library.type.values())
    n_trap_cur = sum('t' in v for v in seq.grad_library.type.values())
    assert n_trap_ref == n_trap_cur

    # Check that the number of ADC events is the same
    assert len(seq150.adc_library.data) == len(seq.adc_library.data)

    # Create test report for v1.5.0 seq file
    rep150 = seq150.test_report()

    # Create test report for all seq files with version >= 1.4.0
    if version >= 140:
        rep = seq.test_report()
        assert rep == rep150
