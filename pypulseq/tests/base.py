from types import SimpleNamespace

from pathlib import Path
import numpy as np
import pytest
from _pytest.python_api import ApproxBase


def main(script: callable, matlab_seq_filename: str, pypulseq_seq_filename: str):
    path_here = Path(__file__)  # Path of this file
    pypulseq_seq_filename = path_here.parent / pypulseq_seq_filename  # Path to PyPulseq seq
    matlab_seq_filename = path_here.parent / "matlab_seqs" / matlab_seq_filename  # Path to MATLAB seq

    # Run PyPulseq script and write seq file
    script.main(plot=False, write_seq=True, seq_filename=str(pypulseq_seq_filename))

    # Read MATLAB and PyPulseq seq files, discard header and signature
    seq_matlab = matlab_seq_filename.read_text().splitlines()[4:-7]
    seq_pypulseq = pypulseq_seq_filename.read_text().splitlines()[4:-7]

    pypulseq_seq_filename.unlink()  # Delete PyPulseq seq

    diff_lines = np.setdiff1d(seq_matlab, seq_pypulseq)  # Mismatching lines
    percentage_diff = len(diff_lines) / len(seq_matlab)  # % of lines that are mismatching; we tolerate upto 0.1%
    assert percentage_diff < 1e-3  # Unit test


class Approx(ApproxBase):
    """
    Extension of pytest.approx that also handles approximate equality
    recursively within dicts, lists, tuples, and SimpleNamespace
    """

    def __repr__(self):
        return str(self.expected)

    def __eq__(self, actual):
        # if type(actual) != type(self.expected):
        #     return False
        if isinstance(self.expected, dict):
            if set(self.expected.keys()) != set(actual.keys()):
                return False

            for k in self.expected:
                if actual[k] != Approx(self.expected[k], rel=self.rel, abs=self.abs, nan_ok=self.nan_ok):
                    return False
            return True
        elif isinstance(self.expected, (list, tuple)):
            if len(self.expected) != len(actual):
                return False

            for e, a in zip(self.expected, actual):
                if a != Approx(e, rel=self.rel, abs=self.abs, nan_ok=self.nan_ok):
                    return False
            return True
        elif isinstance(self.expected, SimpleNamespace):
            return actual.__dict__ == Approx(self.expected.__dict__, rel=self.rel, abs=self.abs, nan_ok=self.nan_ok)
        else:
            return actual == pytest.approx(self.expected, rel=self.rel, abs=self.abs, nan_ok=self.nan_ok)

    def _repr_compare(self, actual):
        # if type(actual) != type(self.expected):
        #     return [f'Actual and expected types do not match: {type(actual)} != {type(self.expected)}']
        if isinstance(self.expected, dict):
            if set(self.expected.keys()) != set(actual.keys()):
                return [f"Actual and expected keys do not match: {set(actual.keys())} != {set(self.expected.keys())}"]

            r = []
            for k in self.expected:
                approx_obj = Approx(self.expected[k], rel=self.rel, abs=self.abs, nan_ok=self.nan_ok)
                if actual[k] != approx_obj:
                    r += [f"{k} does not match:"]
                    r += [f"  {x}" for x in approx_obj._repr_compare(actual[k])]
            return r
        elif isinstance(self.expected, (list, tuple)):
            if len(self.expected) != len(actual):
                return [f"Actual and expected lengths do not match: {len(actual)} != {len(self.expected)}"]
            r = []
            for i, (e, a) in enumerate(zip(self.expected, actual)):
                approx_obj = Approx(e, rel=self.rel, abs=self.abs, nan_ok=self.nan_ok)
                if a != approx_obj:
                    r += [f"Index {i} does not match:"]
                    r += [f"  {x}" for x in approx_obj._repr_compare(a)]
            return r
        elif isinstance(self.expected, SimpleNamespace):
            return Approx(self.expected.__dict__, rel=self.rel, abs=self.abs, nan_ok=self.nan_ok)._repr_compare(
                actual.__dict__
            )
        else:
            return pytest.approx(self.expected, rel=self.rel, abs=self.abs, nan_ok=self.nan_ok)._repr_compare(actual)


def compare_seq_file(file1, file2):
    """
    Compare two sequence files for exact equality
    """
    contents1 = Path(file1).read_text().splitlines()[7:-2]
    contents2 = Path(file2).read_text().splitlines()[7:-2]

    for line1, line2 in zip(contents1, contents2):
        assert line1 == line2
