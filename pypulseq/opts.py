from pypulseq.convert import convert
from types import SimpleNamespace


class Opts():
    """This class contains the gradient limits of the MR system."""

    def __init__(self, grad_unit=None, slew_unit=None, max_grad=None, max_slew=None, rise_time=None, rf_dead_time=0,
                 rf_ringdown_time=0, adc_dead_time=0, rf_raster_time=1e-6, grad_raster_time=10e-6, gamma=42.576e6):
        valid_grad_units = ['Hz/m', 'mT/m', 'rad/ms/mm']
        valid_slew_units = ['Hz/m/s', 'mT/m/ms', 'T/m/s', 'rad/ms/mm/ms']

        if grad_unit is not None and grad_unit not in valid_grad_units:
            raise ValueError()

        if slew_unit is not None and slew_unit not in valid_slew_units:
            raise ValueError()

        if max_grad is None:
            max_grad = convert(from_value=40, from_unit='mT/m', gamma=gamma)
        else:
            max_grad = convert(from_value=max_grad, from_unit=grad_unit, to_unit='Hz/m', gamma=gamma)

        if max_slew is None:
            max_slew = convert(from_value=170, from_unit='T/m/s', gamma=gamma)
        else:
            max_slew = convert(from_value=max_slew, from_unit=slew_unit, to_unit='Hz/m', gamma=gamma)

        if rise_time is not None:
            max_slew = None

        self.max_grad = max_grad
        self.max_slew = max_slew
        self.rise_time = rise_time
        self.rf_dead_time = rf_dead_time
        self.rf_ringdown_time = rf_ringdown_time
        self.adc_dead_time = adc_dead_time
        self.rf_raster_time = rf_raster_time
        self.grad_raster_time = grad_raster_time
        self.gamma = gamma

    def __str__(self):
        s = "System limits:"
        s += "\nmax_grad: " + str(self.max_grad) + str(self.grad_unit)
        s += "\nmax_slew: " + str(self.max_slew) + str(self.slew_unit)
        s += "\nte: " + str(self.te)
        s += "\ntr: " + str(self.tr)
        s += "\nflip: " + str(self.flip)
        s += "\nfov: " + str(self.fov)
        s += "\nNx: " + str(self.Nx)
        s += "\nNy: " + str(self.Ny)
        s += "\nrise_time: " + str(self.rise_time)
        s += "\nrf_dead_time: " + str(self.rf_dead_time)
        s += "\nadc_dead_time: " + str(self.adc_dead_time)
        s += "\nrf_raster_time: " + str(self.rf_raster_time)
        s += "\ngrad_raster_time: " + str(self.grad_raster_time)
        return s
