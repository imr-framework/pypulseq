"""Python translation of the SAR4seq MATLAB toolbox.

Modules mirror the MATLAB files where possible:
- q_mat_gen.py      -> Q_mat_gen.m
- utils/*           -> utils/*.m
- sar4seq.py        -> SAR4seq.m
"""

from .sar4seq import SAR4seq

# Provide both naming conventions for compatibility
sar4seq = SAR4seq

__all__ = ["sar4seq", "SAR4seq"]



