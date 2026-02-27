from types import SimpleNamespace

import numpy as np


def make_rf_shim(shim_vec: np.ndarray) -> SimpleNamespace:
    """
    Create an event describing RF shimming in the current block.

    It is only valid if the block contains an RF pulse

    See also `pypulseq.Sequence.sequence.Sequence.add_block()`.

    Parameters
    ----------
    shim_vec : np.ndarray
        A 1D array of complex numbers describing the relative amplitude
        and phase of each transmit channel.
    """
    return SimpleNamespace(type='rf_shim', shim_vector=shim_vec)
