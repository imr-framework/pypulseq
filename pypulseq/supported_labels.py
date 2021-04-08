from typing import Tuple


def get_supported_labels() -> Tuple[str, str, str, str, str, str, str, str, str, str, str, str]:
    """
    Returns
    -------
    tuple
        Tuple of supported labels.
    """
    return 'SLC', 'SEG', 'REP', 'AVG', 'SET', 'ECO', 'PHS', 'LIN', 'PAR', 'NAV', 'REV', 'SMS'
