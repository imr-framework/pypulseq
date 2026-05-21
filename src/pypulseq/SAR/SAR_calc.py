from pathlib import Path
from typing import Union

from pypulseq.Sequence.sequence import Sequence


def calc_SAR(file: Union[str, Path, Sequence]) -> None:  # noqa: ARG001
    """Temporary placeholder for DeprecationError."""
    raise RuntimeError(
        'Built-in SAR computation has been removed; use PySar4seq (https://github.com/imr-framework/sar4seq/tree/PySar4seq) instead.'
    )
