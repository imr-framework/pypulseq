from typing import Tuple


def get_supported_labels() -> Tuple[
    str, str, str, str, str, str, str, str, str, str, str, str, str
]:
    """
    Returns
    -------
    tuple
        Supported labels.
    """
    return (
        "SLC",
        "SEG",
        "REP",
        "AVG",
        "SET",
        "ECO",
        "PHS",
        "LIN",
        "PAR",
        "NAV",
        "REV",
        "SMS",
        "PMC",
    )


def get_supported_rf_uses() -> Tuple[str, str, str, str, str]:
    """
    Returns
    -------
    tuple
        Supported RF use labels.
    """
    return "excitation", "refocusing", "inversion", "saturation", "preparation"
