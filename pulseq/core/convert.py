from math import pi


def convert_from_to(from_value, from_unit):
    if isinstance(from_value, float) and isinstance(from_unit, str):
        gamma, standard = 42.576e6, 0
        # Converting gradient
        if from_unit == 'Hz/m':
            standard = from_value
        elif from_unit == 'mT/m':
            standard = from_value * 1e-3 * gamma
        elif from_unit == 'rad/ms/mm':
            standard = from_value * 1e6 / (2 * pi)
        # Converting slew rate
        elif from_unit == 'Hz/m/s':
            standard = from_value
        elif from_unit == 'mT/m/ms' or 'T/m/s':
            standard = from_value * gamma
        elif from_unit == 'rad/ms/mm/ms':
            standard = from_value * 1e9 / (2 * pi)
        return standard
    else:
        raise TypeError("input parameters_params should be: from_value, from_unit")
