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
        Must be one of `pypulseq.get_supported_labels()`.
    value : bool, float or int
        Label value.

    Returns
    -------
    out : SimpleNamespace
        Label object.

    Raises
    ------
    ValueError
        If a valid `label` was not passed. Must be one of `pypulseq.get_supported_labels()`.
        If a valid `type` was not passed. Must be one of 'SET' or 'INC'.
        If `value` was not a valid numerical or logical value.
    """
    arr_supported_labels = get_supported_labels()
    arr_flags = arr_supported_labels[10:-1]

    if label not in arr_supported_labels:
        raise ValueError(f'Invalid label. Must be one of {arr_supported_labels}.')
    if type not in ['SET', 'INC']:
        raise ValueError("Invalid type. Must be one of 'SET' or 'INC'.")
    if not isinstance(value, (bool, float, int)):
        raise ValueError('Must supply a valid numerical or logical value.')

    out = SimpleNamespace()
    if type == 'SET':
        out.type = 'labelset'
    elif type == 'INC':
        if label in arr_flags:
            raise ValueError(f'As per Pulseq specification, labelinc is not compatible with flags: {arr_flags}.')
        out.type = 'labelinc'

    out.label = label
    # Force value to an integer, because that is how it will be written to the sequence file
    out.value = int(value)

    return out
