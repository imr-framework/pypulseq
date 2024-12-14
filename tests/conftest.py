from pathlib import Path

import numpy as np
import pytest


# this is currently not used, but might be useful in the future
@pytest.fixture
def main(script: callable, matlab_seq_filename: str, pypulseq_seq_filename: str):
    path_here = Path(__file__)
    pypulseq_seq_filename = path_here.parent / pypulseq_seq_filename
    matlab_seq_filename = path_here.parent / 'matlab_seqs' / matlab_seq_filename

    # Run PyPulseq script and write seq file
    script.main(plot=False, write_seq=True, seq_filename=str(pypulseq_seq_filename))

    # Read MATLAB and PyPulseq seq files, discard header and signature
    seq_matlab = matlab_seq_filename.read_text().splitlines()[4:-7]
    seq_pypulseq = pypulseq_seq_filename.read_text().splitlines()[4:-7]

    pypulseq_seq_filename.unlink()

    diff_lines = np.setdiff1d(seq_matlab, seq_pypulseq)
    percentage_diff = len(diff_lines) / len(seq_matlab)
    assert percentage_diff < 1e-3  # tolerate upto 0.1% difference.


@pytest.fixture
def compare_seq_file():
    def compare(file1, file2):
        """
        Compare two sequence files for exact equality
        """
        contents1 = Path(file1).read_text().splitlines()[7:-2]
        contents2 = Path(file2).read_text().splitlines()[7:-2]

        for line1, line2 in zip(contents1, contents2):
            assert line1 == line2

    return compare
