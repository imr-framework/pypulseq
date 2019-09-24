from pypulseq.convert import convert


class Opts:
    """
    System limits of an MR scanner.

    Attributes
    ----------
    grad_unit : str
        Unit of maximum gradient amplitude. Must be one of Hz/m, mT/m or rad/ms/mm.
    slew_unit : str
        Unit of maximum slew rate. Must be one of Hz/m/s, mT/m/ms, T/m/s or rad/ms/mm/ms.
    max_grad : float
        Maximum gradient amplitude.
    max_slew : float
        Maximum slew rate.
    rise_time : float
        Rise time for gradients.
    rf_dead_time : float
        Dead time for radio-frequency pulses.
    rf_ringdown_time : float
        Ringdown time for radio-frequency pulses.
    adc_dead_time : float
        Dead time for ADC readout pulses.
    rf_raster_time : float
        Raster time for radio-frequency pulses.
    grad_raster_time : float
        Raster time for gradient waveforms.
    gamma : float
        Gyromagnetic ratio. Default is 42.576 MHz for Hydrogen.
    """
    def __init__(self, grad_unit: str = None, slew_unit: str = None, max_grad: float = None, max_slew: float = None,
                 rise_time: float = None, rf_dead_time: float = 0, rf_ringdown_time: float = 0,
                 adc_dead_time: float = 0, rf_raster_time: float = 1e-6, grad_raster_time: float = 10e-6,
                 gamma: float = 42.576e6):
        valid_grad_units = ['Hz/m', 'mT/m', 'rad/ms/mm']
        valid_slew_units = ['Hz/m/s', 'mT/m/ms', 'T/m/s', 'rad/ms/mm/ms']

        if grad_unit is not None and grad_unit not in valid_grad_units:
            raise ValueError(f'Invalid gradient unit, must be one of Hz/m, mT/m or rad/ms/mm. You passed: {grad_unit}')

        if slew_unit is not None and slew_unit not in valid_slew_units:
            raise ValueError(f'Invalid slew rate unit, must be one of Hz/m/s, mT/m/ms, T/m/s or rad/ms/mm/ms. You '
                             f'passed: {slew_unit}')

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
        s += "\nrise_time: " + str(self.rise_time)
        s += "\nrf_dead_time: " + str(self.rf_dead_time)
        s += "\nrf_ring_time: " + str(self.rf_ringdown_time)
        s += "\nadc_dead_time: " + str(self.adc_dead_time)
        s += "\nrf_raster_time: " + str(self.rf_raster_time)
        s += "\ngrad_raster_time: " + str(self.grad_raster_time)
        s += "\ngamma: " + str(self.gamma)
        return s
