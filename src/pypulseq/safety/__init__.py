"""Safety-related modules for pypulseq.

This package contains tools for peripheral‑nerve stimulation (PNS) prediction
and SAR calculations.

The structure mirrors the original MATLAB/stand‑alone packages that were
ported into Python:

* :mod:`pypulseq.safety.pns` - SAFE PNS model and helpers
* :mod:`pypulseq.safety.sar4seq` - SAR4seq toolbox bindings
"""

from __future__ import annotations

from . import pns, sar4seq

__all__ = ["pns", "sar4seq"]
