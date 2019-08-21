from math import pi


def convert(from_value: float, from_unit: str, to_unit: str = None, gamma: float = 42.576e6) -> float:
    """"
    Converts gradient amplitude or slew rate from unit `from_unit` to unit `to_unit` with gyromagnetic ratio `gamma`.

    Parameters
    ----------
    from_value : float
        Gradient amplitude or slew rate to convert from.
    from_unit : str
        Unit of gradient amplitude or slew rate to convert from.
    to_unit : str, optional
        Unit of gradient amplitude or slew rate to convert to. Default is `None`.
    gamma : float, optional
        Gyromagnetic ratio. Default is 42.576e6, for Hydrogen.

    Returns
    -------
    out : float
        Converted gradient amplitude or slew rate.
    """
    valid_grad_units = ['Hz/m', 'mT/m', 'rad/ms/mm']
    valid_slew_units = ['Hz/m/s', 'mT/m/ms', 'T/m/s', 'rad/ms/mm/ms']
    valid_units = valid_grad_units
    valid_units.extend(valid_slew_units)

    if from_unit not in valid_grad_units:
        raise ValueError()

    if to_unit is not None and to_unit not in valid_units:
        raise ValueError()

    if to_unit is None:
        if from_unit in valid_grad_units:
            to_unit = valid_grad_units[0]
        elif from_unit in valid_slew_units:
            to_unit = valid_slew_units[0]

    # Convert to standard units
    # Grad units
    if from_unit == 'Hz/m':
        standard = from_value
    elif from_unit == 'mT/m':
        standard = from_value * 1e-3 * gamma
    elif from_unit == 'rad/ms/mm':
        standard = from_value * 1e6 / (2 * pi)
    # Slew units
    elif from_unit == 'Hz/m/s':
        standard = from_value
    elif from_unit == 'mT/m/ms' or from_unit == 'T/m/s':
        standard = from_value * gamma
    elif from_unit == 'rad/ms/mm/ms':
        standard = from_value * 1e9 / (2 * pi)

    # Convert from standard units
    # Grad units
    if to_unit == 'Hz/m':
        out = standard
    elif to_unit == 'mT/m':
        out = 1e3 * standard / gamma
    elif to_unit == 'rad/ms/mm':
        out = standard * 2 * pi * 1e-6
    # Slew units
    elif to_unit == 'Hz/m/s':
        out = standard
    elif to_unit == 'mT/m/ms' or to_unit == 'T/m/s':
        out = standard / gamma
    elif to_unit == 'rad/ms/mm/ms':
        out = standard * 2 * pi * 1e-9

    return out
