import pypulseq.convert as convert


class Opts():
    """This class contains the gradient limits of the MR system."""

    def __init__(self, kwargs=dict()):
        valid_grad_units = ['Hz/m', 'mT/m', 'rad/ms/mm']
        valid_slew_units = ['Hz/m/s', 'mT/m/ms', 'T/m/s', 'rad/ms/mm/ms']
        self.max_grad = kwargs.get("max_grad", 40)
        self.max_slew = kwargs.get("max_slew", 170)
        self.grad_unit = kwargs.get("grad_unit", valid_grad_units[1])
        self.slew_unit = kwargs.get("slew_unit", valid_slew_units[1])

        # Convert input values if not provided in standard units
        self.max_grad = convert.convert_from_to(float(self.max_grad), self.grad_unit)
        self.max_slew = convert.convert_from_to(float(self.max_slew), self.slew_unit)

        self.te = kwargs.get("te", 0)
        self.tr = kwargs.get("tr", 0)
        self.flip = kwargs.get("flip", 0)
        self.fov = kwargs.get("fov", 0)
        self.Nx = kwargs.get("Nx", 0)
        self.Ny = kwargs.get("Ny", 0)
        self.rise_time = kwargs.get("rise_time", 0)
        self.rf_dead_time = kwargs.get("rf_dead_time", 0)
        self.rf_raster_time = kwargs.get("rf_raster_time", 1e-6)
        self.rf_ring_down_time = kwargs.get("rf_ring_down_time", 0)
        self.adc_dead_time = kwargs.get("adc_dead_time", 0)
        self.grad_raster_time = kwargs.get("grad_raster_time", 10e-6)

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
