from types import SimpleNamespace
from typing import Union

from pypulseq.supported_labels_rf_use import get_supported_labels


def make_label(label: str, type: str, value: Union[bool, float, int]) -> SimpleNamespace:  # noqa: A002
    """
    Create an ADC Label.

    Parameters
    ----------
    label : str
        Must be one of the following:

            - 'SLC' (counter): slice counter (or slab counter for 3D multi-slab sequences).
            - 'SEG' (counter): segment counter e.g. for segmented FLASH or EPI.
            - 'REP' (counter): repetition counter.
            - 'AVG' (counter): averaging counter.
            - 'SET' (counter): flexible counter without firm assignment.
            - 'ECO' (counter): echo counter in multi-echo sequences.
            - 'PHS' (counter): cardiac phase counter.
            - 'LIN' (counter): line counter in 2D and 3D acquisitions.
            - 'PAR' (counter): partition counter; itt counts phase encoding steps in the 2nd (through-slab) phase encoding direction in 3D sequences.
            - 'ACQ' (counter): spectroscopic acquisition counter.
            - 'NAV' (flag): navigator data flag.
            - 'REV' (flag): flag indicating that the readout direction is reversed.
            - 'SMS' (flag): simultaneous multi-slice (SMS) acquisition.
            _ 'REF' (flag): parallel imaging flag indicating reference / auto-calibration data.
            - 'IMA' (flag): parallel imaging flag indicating imaging data within the ACS region.
            - 'NOISE' (flag): noise adjust scan, for iPAT acceleration.
            - 'PMC' (flag): for MoCo/PMC Pulseq version to recognize blocks that can be prospectively corrected for motion.
            - 'NOROT' (flag): instruct the interpreter to ignore the rotation of the FOV specified on the UI.
            - 'NOPOS' (flag): instruct the interpreter to ignore the position of the FOV specified on the UI.
            - 'NOSCL' (flag): instruct the interpreter to ignore the scaling of the FOV specified on the UI.
            - 'ONCE' (flag): a 3-state flag that instructs the interpreter as follows:

                * `ONCE == 0` blocks are executed on every repetition;
                * `ONCE == 1`: only the first repetition of the block is executed;
                * `ONCE == 2`: only the last repetition of the block is executed.

            -'TRID' (counter): marks the beginning of a repeatable module in the sequence (e.g. TR).

        Label type. Must be one of 'SET' or 'INC' (not compatible with flags).
     value : bool, float or int
        Label value.

    Returns
    -------
    out : SimpleNamespace
        Label object.

    Raises
    ------
    ValueError
        If a valid `label` was not passed. Must be one of 'pypulseq.get_supported_labels()'.
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
