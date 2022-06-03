from pathlib import Path

import numpy as np


def main(script: callable, matlab_seq_filename: str, pypulseq_seq_filename: str):
    path_here = Path(__file__)  # Path of this file
    pypulseq_seq_filename = (
        path_here.parent / pypulseq_seq_filename
    )  # Path to PyPulseq seq
    matlab_seq_filename = (
        path_here.parent / "matlab_seqs" / matlab_seq_filename
    )  # Path to MATLAB seq

    # Run PyPulseq script and write seq file
    script.main(plot=False, write_seq=True, seq_filename=str(pypulseq_seq_filename))

    # Read MATLAB and PyPulseq seq files, discard header and signature
    seq_matlab = matlab_seq_filename.read_text().splitlines()[4:-7]
    seq_pypulseq = pypulseq_seq_filename.read_text().splitlines()[4:-7]

    pypulseq_seq_filename.unlink()  # Delete PyPulseq seq

    diff_lines = np.setdiff1d(seq_matlab, seq_pypulseq)  # Mismatching lines
    percentage_diff = len(diff_lines) / len(
        seq_matlab
    )  # % of lines that are mismatching; we tolerate upto 0.1%
    assert percentage_diff < 1e-3  # Unit test
