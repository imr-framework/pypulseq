from pathlib import Path

import pypulseq as pp
import pytest

data_path = Path(__file__).parent / 'expected_output'

seq_compat = [
    'trufi_v120.seq',
    'trufi_v130.seq',
    'trufi_v131.seq',
    'trufi_v140.seq',
    'trufi_v141.seq',
    'trufi_v142.seq',
    'trufi_v150.seq',
]


@pytest.mark.parametrize('seq_file', seq_compat)
def test_backwards_compat(seq_file):
    path = data_path / seq_file

    # Get version
    version = int(seq_file.split('.')[0][-3:])

    # Read written sequence file
    seq = pp.Sequence()
    if version < 141:
        with pytest.warns(UserWarning):
            seq.read(path)
    else:
        seq.read(path)

    # For now, only verify check timing
    # TODO: find a more comprehensive way to assert read is correct
    assert seq.check_timing()[0]
