from types import SimpleNamespace
from typing import Union

from pypulseq import supported_labels


def make_label(type: str, label: str, value: Union[bool, float, int]) -> SimpleNamespace:
    """
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
    arr_supported_labels = supported_labels.get_supported_labels()

    if label not in arr_supported_labels:
        raise ValueError("Invalid label. Must be one of 'SLC', 'SEG', 'REP', 'AVG', 'SET', 'ECO', 'PHS', 'LIN', 'PAR', "
                         "NAV', 'REV', or 'SMS'.")
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
    out.value = value

    return out
