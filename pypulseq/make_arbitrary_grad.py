from types import SimpleNamespace

import numpy as np

from pypulseq.opts import Opts


def make_arbitrary_grad(
    channel: str,
    waveform: np.ndarray,
    delay: float = 0,
    max_grad: float = 0,
    max_slew: float = 0,
    system: Opts = Opts(),
) -> SimpleNamespace:
    """
    Creates a gradient event with arbitrary waveform.

    See also `pypulseq.Sequence.sequence.Sequence.add_block()`.

    Parameters
    ----------
    channel : str
        Orientation of gradient event of arbitrary shape. Must be one of `x`, `y` or `z`.
    waveform : numpy.ndarray
        Arbitrary waveform.
    system : Opts, default=Opts()
        System limits.
    max_grad : float, default=0
        Maximum gradient strength.
    max_slew : float, default=0
        Maximum slew rate.
    delay : float, default=0
        Delay in seconds (s).

    Returns
    -------
    grad : SimpleNamespace
        Gradient event with arbitrary waveform.

    Raises
    ------
    ValueError
        If invalid `channel` is passed. Must be one of x, y or z.
        If slew rate is violated.
        If gradient amplitude is violated.
    """
    if channel not in ["x", "y", "z"]:
        raise ValueError(
            f"Invalid channel. Must be one of x, y or z. Passed: {channel}"
        )

    if max_grad <= 0:
        max_grad = system.max_grad

    if max_slew <= 0:
        max_slew = system.max_slew

    g = waveform
    slew = np.squeeze(np.subtract(g[1:], g[:-1]) / system.grad_raster_time)
    if max(abs(slew)) >= max_slew:
        raise ValueError(f"Slew rate violation {max(abs(slew)) / max_slew * 100}")
    if max(abs(g)) >= max_grad:
        raise ValueError(f"Gradient amplitude violation {max(abs(g)) / max_grad * 100}")

    grad = SimpleNamespace()
    grad.type = "grad"
    grad.channel = channel
    grad.waveform = g
    grad.delay = delay
    # True timing and aux shape data
    grad.tt = (np.arange(1, len(g) + 1) - 0.5) * system.grad_raster_time
    grad.shape_dur = len(g) * system.grad_raster_time
    grad.first = (3 * g[0] - g[1]) * 0.5  # Extrapolate by 1/2 gradient raster
    grad.last = (g[-1] * 3 - g[-2]) * 0.5  # Extrapolate by 1/2 gradient raster

    return grad
