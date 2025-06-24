import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import numpy.testing as npt
import pytest
from _pytest.python_api import ApproxBase
from scipy.spatial.transform import Rotation as R


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


class Approx(ApproxBase):
    """
    Fast approximate equality for nested dicts, lists, tuples, SimpleNamespace, and numpy arrays,
    derived from pytest's ApproxBase for seamless pytest integration.
    """

    def __init__(self, expected, *, rel=1e-6, abs=1e-12, nan_ok=False):  # noqa: A002
        super().__init__(expected, rel=rel, abs=abs, nan_ok=nan_ok)
        self._errors = []

    def __eq__(self, actual):
        # reset errors
        self._errors.clear()
        # stack: (path, expected, actual)
        stack = [((), self.expected, actual)]
        rel_tol, abs_tol, nan_ok, errs = self.rel, self.abs, self.nan_ok, self._errors
        isclose = math.isclose

        while stack:
            path, exp, act = stack.pop()
            if isinstance(exp, R):
                exp = exp.as_matrix()
            if isinstance(act, R):
                act = act.as_matrix()

            # dict
            if isinstance(exp, dict):
                if not isinstance(act, dict) or set(exp) != set(act):
                    errs.append(
                        f'{".".join(path) or "<root>"}: key-sets differ; expected {set(exp)}, got {set(getattr(act, "keys", lambda: act)())}'  # noqa: B023
                    )
                    return False
                for k in exp:
                    stack.append((path + (str(k),), exp[k], act[k]))  # noqa: RUF005
                continue

            # list/tuple
            if isinstance(exp, (list, tuple)):
                if not isinstance(act, type(exp)) or len(exp) != len(act):
                    errs.append(
                        f'{".".join(path) or "<root>"}: length/type mismatch; expected {type(exp).__name__}[{len(exp)}], got {type(act).__name__}[{len(act)}]'
                    )
                    return False
                for idx, (e, a) in enumerate(zip(exp, act)):
                    stack.append((path + (str(idx),), e, a))  # noqa: RUF005
                continue

            # SimpleNamespace
            if isinstance(exp, SimpleNamespace):
                if not isinstance(act, SimpleNamespace):
                    errs.append(f'{".".join(path)}: expected SimpleNamespace, got {type(act).__name__}')
                    return False
                stack.append((path, exp.__dict__, act.__dict__))
                continue

            # numpy arrays
            if isinstance(exp, np.ndarray) or isinstance(act, np.ndarray):
                try:
                    npt.assert_allclose(act, exp, rtol=rel_tol, atol=abs_tol, equal_nan=nan_ok)
                except AssertionError as e:
                    errs.append(f'{".".join(path) or "<array>"}: {e}')
                    return False
                continue

            # scalar or fallback
            try:
                if not (
                    isclose(act, exp, rel_tol=rel_tol, abs_tol=abs_tol)
                    or (nan_ok and math.isnan(act) and math.isnan(exp))
                ):
                    errs.append(
                        f'{".".join(path) or "<value>"}: {act!r} != {exp!r} within (rel={rel_tol}, abs={abs_tol})'
                    )
                    return False
            except TypeError:
                approx = pytest.approx(exp, rel=rel_tol, abs=abs_tol, nan_ok=nan_ok)
                if act != approx:
                    msgs = approx._repr_compare(act)
                    errs.extend(msgs)
                    return False

        return True

    def __repr__(self):
        return str(self.expected)

    def _repr_compare(self, actual):
        # populate errors
        _ = actual == self
        return self._errors


# Rotation Matrix creation routine
def get_rotation_matrix(channel, angle):
    return R.from_euler(channel, angle, degrees=False).as_matrix()
