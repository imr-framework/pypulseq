from pathlib import Path

import pypulseq as pp
import pytest

data_path = Path(__file__).parent / 'expected_output'

seq_compat = [
    'simple_mprage140.seq',
    'simple_mprage141.seq',
    'simple_mprage142.seq',
]


@pytest.mark.parametrize('seq_file', seq_compat)
def test_backwards_compat(seq_file):
    path = data_path / seq_file

    # Get version
    version = int(seq_file.split('.')[0][-3:])
    
    # Read v1.5.0 as reference
    seq150 = pp.Sequence()
    seq150.read(data_path / 'simple_mprage150.seq')
    rep150 = seq150.test_report()

    # Read written sequence file
    seq = pp.Sequence()
    if version < 141:
        with pytest.warns(UserWarning):
            seq.read(path)
    else:
        seq.read(path)
    rep = seq.test_report()

    assert rep == rep150
