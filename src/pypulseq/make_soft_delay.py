from types import SimpleNamespace

from pypulseq.utils.tracing import trace, trace_enabled


def make_soft_delay(
    hint: str, numID: int | None = None, offset: float = 0.0, factor: float = 1.0, default_duration: float = 10e-6
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
        Numeric identifier for the soft delay. If None, will be auto-assigned.
        Multiple soft delays with the same numID must have identical hints.
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
    Create a basic TE soft delay:

    >>> te_delay = pp.make_soft_delay('TE', default_duration=5e-3)
    >>> seq.add_block(te_delay)

    Create a TR delay with scaling and offset:

    >>> tr_delay = pp.make_soft_delay('TR', offset=-10e-3, factor=1.0, default_duration=100e-3)
    >>> seq.add_block(tr_delay)

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
    - The block duration equation is: duration = (user_input / factor) + offset
    - Soft delays with identical numID must have identical hint strings
    - The scanner interface will display delays ordered by numID
    """
    soft_delay = SimpleNamespace()

    if ' ' in hint:
        raise ValueError("Parameter 'hint' may not contain white space characters.")

    if default_duration <= 0:
        raise ValueError('Default duration must be greater than 0.')

    soft_delay.type = 'soft_delay'
    soft_delay.numID = numID
    soft_delay.hint = hint
    soft_delay.offset = offset
    soft_delay.factor = factor
    soft_delay.default_duration = default_duration

    if trace_enabled():
        soft_delay.trace = trace()

    return soft_delay
