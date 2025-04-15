from types import SimpleNamespace


def make_soft_delay(numID: int, hint: str, offset: float = 0.0, factor: float = 1.0) -> SimpleNamespace:
    """
    Creates a soft delay extension event.

    Create a soft delay event, that can be used in combination with an empty (pure delay) block
    e.g. to adjust TE, TR or other delays. The soft delay extension acts by rewriting the block duration
    based on the user input (to the interpreter) according to the equation dur=input/factor+offset.
    Required parameters are 'numeric ID' and 'string hint'. Optional parameter 'factor' can be either
    positive and negative. Optional parameter 'offset' given in seconds can also be either positive and
    negative. The 'hint' parameter is expected to be for identical 'numID'.

    See Also
    --------
    - `pypulseq.Sequence.sequence.Sequence.add_block()`
    - `pypulseq.Sequence.sequence.Sequence.apply_soft_delay()`

    Parameters
    ----------
    numID : int
        Numeric ID of the soft delay event.
    hint : str
        Human readable text hint for the soft delay event to be shown at the interpreter.
    offset : float, optional
        Offset in seconds, default is 0.
    factor : float, optional
        Delay factor determines how fast the delay will scale, default is 1.0.

    Returns
    -------
    soft_delay : SimpleNamespace
        Soft delay event.

    Raises
    ------
    ValueError
        If the 'hint' parameter contains white space characters.
    """
    soft_delay = SimpleNamespace()

    if ' ' in hint:
        raise ValueError("Parameter 'hint' may not contain white space characters.")

    soft_delay.type = 'soft_delay'
    soft_delay.numID = numID
    soft_delay.hint = hint
    soft_delay.offset = offset
    soft_delay.factor = factor
    return soft_delay
