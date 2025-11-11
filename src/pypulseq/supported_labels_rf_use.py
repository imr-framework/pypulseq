from typing import Dict, List, Tuple

# Structured label definitions with metadata (descriptions from make_label.py)
_LABEL_DEFINITIONS = {
    'data_counters': {
        'description': 'Data counters copied to corresponding MDH fields on Siemens. Compatible with both SET and INC operations.',
        'labels': {
            'SLC': 'Slice counter (or slab counter for 3D multi-slab sequences)',
            'SEG': 'Segment counter e.g. for segmented FLASH or EPI',
            'REP': 'Repetition counter',
            'AVG': 'Averaging counter',
            'SET': 'Flexible counter without firm assignment',
            'ECO': 'Echo counter in multi-echo sequences',
            'PHS': 'Cardiac phase counter',
            'LIN': 'Line counter in 2D and 3D acquisitions',
            'PAR': 'Partition counter; counts phase encoding steps in the 2nd (through-slab) phase encoding direction in 3D sequences',
            'ACQ': 'Spectroscopic acquisition counter',
        },
    },
    'data_flags': {
        'description': 'Data flags for acquisition control and parallel imaging. Only compatible with SET operation.',
        'labels': {
            'NAV': 'Navigator data flag',
            'REV': 'Flag indicating that the readout direction is reversed',
            'SMS': 'Simultaneous multi-slice (SMS) acquisition',
            'REF': 'Parallel imaging flag indicating reference / auto-calibration data',
            'IMA': 'Parallel imaging flag indicating imaging data within the ACS region',
            'OFF': 'Offline flag that labels the data, that should not be used for the online-reconstruction (on Siemens it negates the ONLINE MDH flag)',
            'NOISE': 'Noise adjust scan, for iPAT acceleration',
        },
    },
    'control_flags': {
        'description': 'Control flags affecting sequence behavior, not data. Only compatible with SET operation.',
        'labels': {
            'PMC': 'For MoCo/PMC Pulseq version to recognize blocks that can be prospectively corrected for motion',
            'NOROT': 'Instruct the interpreter to ignore the rotation of the FOV specified on the UI',
            'NOPOS': 'Instruct the interpreter to ignore the position of the FOV specified on the UI',
            'NOSCL': 'Instruct the interpreter to ignore the scaling of the FOV specified on the UI',
            'ONCE': 'A 3-state flag that instructs the interpreter as follows: ONCE == 0 blocks are executed on every repetition; ONCE == 1: only the first repetition; ONCE == 2: only the last repetition',
            'TRID': 'Marks the beginning of a repeatable module in the sequence (e.g. TR)',
        },
    },
}

# RF use definitions
_RF_USE_DEFINITIONS = {
    'description': 'Supported RF pulse use cases',
    'uses': [
        'excitation',  # Excitation pulse
        'refocusing',  # Refocusing pulse (e.g., in spin echo)
        'inversion',  # Inversion pulse (e.g., in IR sequences)
        'saturation',  # Saturation pulse (e.g., fat sat)
        'preparation',  # Preparation pulse (e.g., T2 prep)
    ],
}


def get_supported_labels() -> Tuple[str, ...]:
    """
    Get all supported sequence labels.

    Returns a tuple of all supported label strings in the order they were
    originally defined, maintaining backward compatibility.

    Returns
    -------
    Tuple[str, ...]
        Tuple containing all supported label strings. Labels are organized
        by category: data counters, data flags, and control flags.

    Notes
    -----
    Data counters (SLC, SEG, REP, etc.) are copied to corresponding MDH
    fields on Siemens scanners and support both SET and INC operations.
    Data flags and control flags only support SET operations - INC is not
    allowed per Pulseq specification. Data flags control acquisition behavior
    and parallel imaging. Control flags affect sequence execution but
    not the acquired data itself.

    Label Operation Types:
    - SET: Assigns an absolute value to the label
    - INC: Increments the label value (only valid for counters, not flags)
    """
    # Flatten all labels while preserving order
    all_labels = []
    for category in ['data_counters', 'data_flags', 'control_flags']:
        all_labels.extend(_LABEL_DEFINITIONS[category]['labels'].keys())
    return tuple(all_labels)


def get_supported_rf_uses() -> Tuple[str, ...]:
    """
    Get all supported RF pulse use cases.

    Returns
    -------
    Tuple[str, ...]
        Tuple containing all supported RF use case strings.

    Notes
    -----
    RF use cases define the purpose of RF pulses in the sequence:
    - excitation: Tip spins into transverse plane
    - refocusing: Refocus spins (e.g., 180° in spin echo)
    - inversion: Invert magnetization (e.g., in IR sequences)
    - saturation: Saturate specific tissue (e.g., fat saturation)
    - preparation: Prepare magnetization (e.g., T2 preparation)
    """
    return tuple(_RF_USE_DEFINITIONS['uses'])


def get_labels_by_category(category: str) -> List[str]:
    """
    Get labels filtered by category.

    Parameters
    ----------
    category : str
        Category name: 'data_counters', 'data_flags', or 'control_flags'

    Returns
    -------
    List[str]
        List of labels in the specified category

    Raises
    ------
    KeyError
        If category is not recognized

    Examples
    --------
    >>> get_labels_by_category('data_counters')
    ['SLC', 'SEG', 'REP', 'AVG', 'SET', 'ECO', 'PHS', 'LIN', 'PAR', 'ACQ']
    """
    if category not in _LABEL_DEFINITIONS:
        valid_categories = list(_LABEL_DEFINITIONS.keys())
        raise KeyError(f"Unknown category '{category}'. Valid categories: {valid_categories}")

    return list(_LABEL_DEFINITIONS[category]['labels'].keys())


def get_flag_labels() -> List[str]:
    """
    Get all flag labels (data_flags and control_flags combined).

    This function replaces the hard-coded index slicing used in make_label.py
    to identify flag labels that are incompatible with 'INC' type.

    Returns
    -------
    List[str]
        List of all flag labels (both data flags and control flags)

    Examples
    --------
    >>> flags = get_flag_labels()
    >>> 'NAV' in flags
    True
    >>> 'REP' in flags  # REP is a counter, not a flag
    False
    """
    flag_labels = []
    flag_labels.extend(_LABEL_DEFINITIONS['data_flags']['labels'].keys())
    flag_labels.extend(_LABEL_DEFINITIONS['control_flags']['labels'].keys())
    return flag_labels


def get_label_categories() -> Dict[str, str]:
    """
    Get all label categories with their descriptions.

    Returns
    -------
    Dict[str, str]
        Dictionary mapping category names to descriptions

    Examples
    --------
    >>> categories = get_label_categories()
    >>> categories['data_counters']
    'Data counters copied to corresponding MDH fields on Siemens'
    """
    return {category: info['description'] for category, info in _LABEL_DEFINITIONS.items()}


def is_valid_label(label: str) -> bool:
    """
    Check if a label is supported.

    Parameters
    ----------
    label : str
        Label to check

    Returns
    -------
    bool
        True if label is supported, False otherwise

    Examples
    --------
    >>> is_valid_label('REP')
    True
    >>> is_valid_label('INVALID')
    False
    """
    return label in get_supported_labels()


def is_valid_rf_use(rf_use: str) -> bool:
    """
    Check if an RF use case is supported.

    Parameters
    ----------
    rf_use : str
        RF use case to check

    Returns
    -------
    bool
        True if RF use is supported, False otherwise

    Examples
    --------
    >>> is_valid_rf_use('excitation')
    True
    >>> is_valid_rf_use('invalid')
    False
    """
    return rf_use in get_supported_rf_uses()
