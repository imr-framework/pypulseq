"""Peripheral nerve stimulation utils based on the SAFE prediction model.

The original code is a Python translation of the SAFE-Model implementation by
Szczepankiewicz & Witzel.  Functions here are used by
:mod:`pypulseq.Sequence.calc_pns` and also available as standalone helpers for
experimentation and plotting.
"""

from __future__ import annotations

from .safe_pns import (
    safe_example_hw,
    safe_example_gwf,
    safe_hw_check,
    safe_longest_time_const,
    safe_pns_model,
    safe_tau_lowpass,
    safe_gwf_to_pns,
    safe_plot,
    safe_example,
)

__all__ = [
    "safe_example_hw",
    "safe_example_gwf",
    "safe_hw_check",
    "safe_longest_time_const",
    "safe_pns_model",
    "safe_tau_lowpass",
    "safe_gwf_to_pns",
    "safe_plot",
    "safe_example",
]
