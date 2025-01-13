from types import SimpleNamespace
from typing import Union

from pypulseq.supported_labels_rf_use import get_supported_labels


def make_label(label: str, type: str, value: Union[bool, float, int]) -> SimpleNamespace:  # noqa: A002
    """
    Create an ADC Label.

    Parameters
    ----------
    type : str
        Label type. Must be one of 'SET' or 'INC'.
    label : str
        Must be one of 'SLC', 'SEG', 'REP', 'AVG', 'SET', 'ECO', 'PHS', 'LIN', 'PAR', 'NAV', 'REV', or 'SMS'.
    value : bool, float or int
        Label value.

    Returns
    -------
    out : SimpleNamespace
        Label object.

    Raises
    ------
    ValueError
        If a valid `label` was not passed. Must be one of 'SLC', 'SEG', 'REP', 'AVG', 'SET', 'ECO', 'PHS', 'LIN', 'PAR',
                                                                                                NAV', 'REV', or 'SMS'.
        If a valid `type` was not passed. Must be one of 'SET' or 'INC'.
        If `value` was not a valid numerical or logical value.
    """
    arr_supported_labels = get_supported_labels()

    if label not in arr_supported_labels:
        raise ValueError(
            "Invalid label. Must be one of 'SLC', 'SEG', 'REP', 'AVG', 'SET', 'ECO', 'PHS', 'LIN', 'PAR', "
            "NAV', 'REV', or 'SMS'."
        )
    if type not in ['SET', 'INC']:
        raise ValueError("Invalid type. Must be one of 'SET' or 'INC'.")
    if not isinstance(value, (bool, float, int)):
        raise ValueError('Must supply a valid numerical or logical value.')

    out = SimpleNamespace()
    if type == 'SET':
        out.type = 'labelset'
    elif type == 'INC':
        out.type = 'labelinc'

    out.label = label
    # Force value to an integer, because that is how it will be written to the sequence file
    out.value = int(value)

    return out
