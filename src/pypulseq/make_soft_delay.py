from types import SimpleNamespace
from typing import Union

from pypulseq.utils.tracing import trace, trace_enabled


def make_soft_delay(
    hint: str,
    numID: Union[int, None] = None,
    offset: float = 0.0,
    factor: float = 1.0,
    default_duration: float = 10e-6,
) -> SimpleNamespace:
    """
    Create a soft delay extension event for dynamic timing adjustment.

    Soft delays allow runtime modification of block durations through the scanner interface.
    They must be used in empty blocks (blocks containing no RF, gradient, or ADC events).
    The final block duration is calculated as: duration = (user_input / factor) + offset.

    This feature enables dynamic adjustment of timing parameters like TE, TR, or other delays
    without recompiling the sequence, making it useful for parameter optimization and
    real-time sequence adjustments.

    Parameters
    ----------
    hint : str
        Human-readable identifier for the soft delay shown in the scanner interface.
        Must not contain whitespace characters. Examples: 'TE', 'TR', 'TI'.
    numID : int or None, optional
        Numeric identifier for the soft delay. If None (recommended), will be
        auto-assigned based on the hint. Each unique hint gets its own numID.
        Rarely needed - only specify if you need explicit control over scanner
        interface ordering.
    offset : float, optional
        Time offset in seconds added to the calculated duration. Can be positive
        or negative. Default is 0.0.
    factor : float, optional
        Scaling factor for user input. Determines how the user input maps to
        actual duration. Can be positive or negative. Default is 1.0.
    default_duration : float, optional
        Default duration in seconds used as the initial block duration and
        fallback value. Must be greater than 0. Default is 10e-6 (10 μs).

    Returns
    -------
    soft_delay : SimpleNamespace
        Soft delay event object with the following attributes:
        - type : str = 'soft_delay'
        - hint : str
        - numID : int or None
        - offset : float
        - factor : float
        - default_duration : float

    Raises
    ------
    ValueError
        If hint contains whitespace characters.
    ValueError
        If default_duration is not greater than 0.

    Examples
    --------
    Create a basic TE soft delay (numID and duration handled automatically):

    >>> te_delay = pp.make_soft_delay('TE', default_duration=5e-3)
    >>> seq.add_block(te_delay)  # Block duration automatically becomes 5ms

    Create a TR delay with scaling and offset:

    >>> tr_delay = pp.make_soft_delay('TR', offset=-10e-3, factor=1.0, default_duration=100e-3)
    >>> seq.add_block(tr_delay)  # Block duration automatically becomes 100ms

    Multiple delays with same hint reuse the same numID:

    >>> te1 = pp.make_soft_delay('TE', default_duration=5e-3)  # Gets numID 0
    >>> te2 = pp.make_soft_delay('TE', default_duration=5e-3)  # Reuses numID 0

    Apply soft delays in the sequence:

    >>> seq.apply_soft_delay(TE=8e-3, TR=500e-3)

    See Also
    --------
    pypulseq.Sequence.sequence.Sequence.add_block : Add blocks to sequence
    pypulseq.Sequence.sequence.Sequence.apply_soft_delay : Apply soft delay values

    Notes
    -----
    - Soft delays require file format version 1.5.0 or higher
    - Each soft delay must be in its own empty block
    - The default_duration automatically becomes the block duration when added to sequence
    - The block duration equation is: duration = (user_input / factor) + offset
    - Soft delays with identical hints automatically share the same numID
    - The scanner interface displays delays ordered by numID (auto-assigned by hint order)
    - For most use cases, omit numID and let the system auto-assign based on hints
    """
    soft_delay = SimpleNamespace()

    # Validate hint parameter
    if not hint:
        raise ValueError("Parameter 'hint' cannot be empty.")
    if any(c.isspace() for c in hint):
        raise ValueError("Parameter 'hint' may not contain white space characters.")
    if not isinstance(hint, str):
        raise TypeError("Parameter 'hint' must be a string.")

    # Validate numeric parameters
    if default_duration <= 0:
        raise ValueError('Default duration must be greater than 0.')
    if factor == 0:
        raise ValueError("Parameter 'factor' cannot be zero (would make duration calculation undefined).")
    if numID is not None and (not isinstance(numID, int) or numID < 0):
        raise ValueError("Parameter 'numID' must be a non-negative integer or None.")

    soft_delay.type = 'soft_delay'
    soft_delay.numID = numID
    soft_delay.hint = hint
    soft_delay.offset = offset
    soft_delay.factor = factor
    soft_delay.default_duration = default_duration

    if trace_enabled():
        soft_delay.trace = trace()

    return soft_delay
