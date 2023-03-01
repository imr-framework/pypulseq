from pypulseq.convert import convert


class Opts:
    """
    System limits of an MR scanner.

    Attributes
    ----------
    adc_dead_time : float, default=0
        Dead time for ADC readout pulses.
    gamma : float, default=42.576e6
        Gyromagnetic ratio. Default gamma is specified for Hydrogen.
    grad_raster_time : float, default=10e-6
        Raster time for gradient waveforms.
    grad_unit : str, default='Hz/m'
        Unit of maximum gradient amplitude. Must be one of 'Hz/m', 'mT/m' or 'rad/ms/mm'.
    max_grad : float, default=0
        Maximum gradient amplitude.
    max_slew : float, default=0
        Maximum slew rate.
    rf_dead_time : float, default=0
        Dead time for radio-frequency pulses.
    rf_raster_time : float, default=1e-6
        Raster time for radio-frequency pulses.
    rf_ringdown_time : float, default=0
        Ringdown time for radio-frequency pulses.
    rise_time : float, default=0
        Rise time for gradients.
    slew_unit : str, default='Hz/m/s'
        Unit of maximum slew rate. Must be one of 'Hz/m/s', 'mT/m/ms', 'T/m/s' or 'rad/ms/mm/ms'.

    Raises
    ------
    ValueError
        If invalid `grad_unit` is passed. Must be one of 'Hz/m', 'mT/m' or 'rad/ms/mm'.
        If invalid `slew_unit` is passed. Must be one of 'Hz/m/s', 'mT/m/ms', 'T/m/s' or 'rad/ms/mm/ms'.
    """

    def __init__(
        self,
        adc_dead_time: float = 0,
        adc_raster_time: float = 100e-9,
        block_duration_raster: float = 10e-6,
        gamma: float = 42.576e6,
        grad_raster_time: float = 10e-6,
        grad_unit: str = "Hz/m",
        max_grad: float = 0,
        max_slew: float = 0,
        rf_dead_time: float = 0,
        rf_raster_time: float = 1e-6,
        rf_ringdown_time: float = 0,
        rise_time: float = 0,
        slew_unit: str = "Hz/m/s",
        B0: float = 1.5,
    ):
        valid_grad_units = ["Hz/m", "mT/m", "rad/ms/mm"]
        valid_slew_units = ["Hz/m/s", "mT/m/ms", "T/m/s", "rad/ms/mm/ms"]

        if grad_unit not in valid_grad_units:
            raise ValueError(
                f"Invalid gradient unit. Must be one of 'Hz/m', 'mT/m' or 'rad/ms/mm'. "
                f"Passed: {grad_unit}"
            )

        if slew_unit not in valid_slew_units:
            raise ValueError(
                f"Invalid slew rate unit. Must be one of 'Hz/m/s', 'mT/m/ms', 'T/m/s' or 'rad/ms/mm/ms'. "
                f"Passed: {slew_unit}"
            )

        if max_grad == 0:
            max_grad = convert(from_value=40, from_unit="mT/m", gamma=gamma)
        else:
            max_grad = convert(
                from_value=max_grad, from_unit=grad_unit, to_unit="Hz/m", gamma=gamma
            )

        if max_slew == 0:
            max_slew = convert(from_value=170, from_unit="T/m/s", gamma=gamma)
        else:
            max_slew = convert(
                from_value=max_slew, from_unit=slew_unit, to_unit="Hz/m", gamma=gamma
            )

        if rise_time != 0:
            max_slew = max_grad / rise_time

        self.max_grad = max_grad
        self.max_slew = max_slew
        self.rise_time = rise_time
        self.rf_dead_time = rf_dead_time
        self.rf_ringdown_time = rf_ringdown_time
        self.adc_dead_time = adc_dead_time
        self.adc_raster_time = adc_raster_time
        self.rf_raster_time = rf_raster_time
        self.grad_raster_time = grad_raster_time
        self.block_duration_raster = block_duration_raster
        self.gamma = gamma
        self.B0 = B0

    def __str__(self) -> str:
        """
        Print a string representation of the system limits objects.
        """
        variables = vars(self)
        s = [f"{key}: {value}" for key, value in variables.items()]
        s = "\n".join(s)
        s = "System limits:\n" + s
        return s
