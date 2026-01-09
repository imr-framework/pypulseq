from types import SimpleNamespace
from typing import Union

from pypulseq.supported_labels_rf_use import get_flag_labels, get_supported_labels


def make_label(label: str, type: str, value: Union[bool, float, int]) -> SimpleNamespace:  # noqa: A002
    """
    Create an ADC Label.

    Labels are used to mark specific data acquisitions and control sequence behavior.
    There are two types of labels with different SET/INC compatibility:

    - **Counters** (data_counters): Support both SET and INC operations
    - **Flags** (data_flags, control_flags): Only support SET operations (INC not allowed per Pulseq specification)

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
            - 'REF' (flag): parallel imaging flag indicating reference / auto-calibration data.
            - 'OFF' (flag): Offline flag that labels the data, that should not be used for the online-reconstruction (on Siemens it negates the ONLINE MDH flag).
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

            - 'TRID' (counter): marks the beginning of a repeatable module in the sequence (e.g. TR).

    type : str
        Label operation type. Must be one of:

        - 'SET': Assigns an absolute value to the label (compatible with all labels)
        - 'INC': Increments the label value (only compatible with counters, not flags)

    value : bool, float or int
        Label value. For SET operations, this is the absolute value to assign.
        For INC operations, this is the increment amount.

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
        If 'INC' type is used with flag labels (not allowed per Pulseq specification).

    Notes
    -----
    The Pulseq specification defines different behavior for counters and flags:

    **Counters** (SLC, SEG, REP, AVG, SET, ECO, PHS, LIN, PAR, ACQ, TRID):
    - Support both SET and INC operations
    - Used for tracking acquisition parameters and loop indices
    - Values are typically copied to MDH fields on Siemens scanners

    **Flags** (NAV, REV, SMS, REF, IMA, OFF, NOISE, PMC, NOROT, NOPOS, NOSCL, ONCE):
    - Only support SET operations (INC is prohibited)
    - Used for boolean-like control and data marking
    - Control acquisition behavior or sequence execution

    **Operation Types:**
    - SET: Sets the label to an absolute value
    - INC: Increments the current label value (counters only)

    Examples
    --------
    >>> # Counter with SET operation
    >>> rep_label = make_label('REP', 'SET', 5)
    >>> rep_label.type
    'labelset'

    >>> # Counter with INC operation
    >>> rep_inc = make_label('REP', 'INC', 1)
    >>> rep_inc.type
    'labelinc'

    >>> # Flag with SET operation
    >>> nav_label = make_label('NAV', 'SET', 1)
    >>> nav_label.type
    'labelset'
    """
    all_labels = get_supported_labels()
    flag_labels = get_flag_labels()

    if label not in all_labels:
        raise ValueError(f'Invalid label. Must be one of {all_labels}.')
    if type not in ['SET', 'INC']:
        raise ValueError("Invalid type. Must be one of 'SET' or 'INC'.")
    if not isinstance(value, (bool, float, int)):
        raise ValueError('Must supply a valid numerical or logical value.')

    out = SimpleNamespace()
    if type == 'SET':
        out.type = 'labelset'
    elif type == 'INC':
        if label in flag_labels:
            raise ValueError(f'As per Pulseq specification, labelinc is not compatible with flags: {flag_labels}.')
        out.type = 'labelinc'

    out.label = label
    # Force value to an integer, because that is how it will be written to the sequence file
    out.value = int(value)

    return out
