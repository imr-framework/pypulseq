from pypulseq.convert import convert


class Opts:
    """
    System limits of an MR scanner.
    
    Note: Default values can be overwritten by creating an Opts object and
    calling `set_as_default`.

    Attributes
    ----------
    adc_dead_time : float, default=0
        Dead time for ADC readout pulses.
    adc_raster_time : float, default=100e-9
        Raster time for ADC readout pulses.
    block_duration_raster : float, default=10e-6
        Raster time for block durations.
    gamma : float, default=42.576e6
        Gyromagnetic ratio. Default gamma is specified for Hydrogen.
    grad_raster_time : float, default=10e-6
        Raster time for gradient waveforms.
    grad_unit : str, default='Hz/m'
        Unit of maximum gradient amplitude. Must be one of 'Hz/m', 'mT/m' or 'rad/ms/mm'.
    max_grad : float, default=40 mT/m
        Maximum gradient amplitude.
    max_slew : float, default=170 T/m/s
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
    B0 : float, default=1.5
        Main magnetic field strength (in tesla)

    Raises
    ------
    ValueError
        If invalid `grad_unit` is passed. Must be one of 'Hz/m', 'mT/m' or 'rad/ms/mm'.
        If invalid `slew_unit` is passed. Must be one of 'Hz/m/s', 'mT/m/ms', 'T/m/s' or 'rad/ms/mm/ms'.
    """
    def __init__(
        self,
        adc_dead_time: float = None,
        adc_raster_time: float = None,
        block_duration_raster: float = None,
        gamma: float = None,
        grad_raster_time: float = None,
        grad_unit: str = "Hz/m",
        max_grad: float = None,
        max_slew: float = None,
        rf_dead_time: float = None,
        rf_raster_time: float = None,
        rf_ringdown_time: float = None,
        rise_time: float = None,
        slew_unit: str = "Hz/m/s",
        B0: float = None,
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

        if gamma == None:
            gamma = Opts.default.gamma

        if max_grad != None:
            max_grad = convert(
                from_value=max_grad, from_unit=grad_unit, to_unit="Hz/m", gamma=abs(gamma)
            )
        else:
            max_grad = Opts.default.max_grad

        if max_slew != None:
            max_slew = convert(
                from_value=max_slew, from_unit=slew_unit, to_unit="Hz/m", gamma=abs(gamma)
            )
        else:
            max_slew = Opts.default.max_slew

        if rise_time != None:
            max_slew = max_grad / rise_time
        
        if adc_dead_time == None:
            adc_dead_time = Opts.default.adc_dead_time
        if adc_raster_time == None:
            adc_raster_time = Opts.default.adc_raster_time
        if block_duration_raster == None:
            block_duration_raster = Opts.default.block_duration_raster
        
        if rf_dead_time == None:
            rf_dead_time = Opts.default.rf_dead_time
        if rf_raster_time == None:
            rf_raster_time = Opts.default.rf_raster_time
        if grad_raster_time == None:
            grad_raster_time = Opts.default.grad_raster_time 
        if rf_ringdown_time == None:
            rf_ringdown_time = Opts.default.rf_ringdown_time
        if B0 == None:
            B0 = Opts.default.B0

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
    
    def set_as_default(self):
        Opts.default = self
    
    @classmethod
    def reset_default(cls):
        cls.default = Opts(max_grad=convert(from_value=40, from_unit="mT/m"),
                           max_slew=convert(from_value=170, from_unit="T/m/s"),
                           rf_dead_time=0,
                           rf_ringdown_time=0,
                           adc_dead_time=0,
                           adc_raster_time=100e-9,
                           rf_raster_time=1e-6,
                           grad_raster_time=10e-6,
                           block_duration_raster=10e-6,
                           gamma=42576000,
                           B0=1.5)

    def __str__(self) -> str:
        """
        Print a string representation of the system limits objects.
        """
        variables = vars(self)
        s = [f"{key}: {value}" for key, value in variables.items()]
        s = "\n".join(s)
        s = "System limits:\n" + s
        return s

Opts.reset_default()
