class pulse_opts:
    """
    System limits of an MR scanner.

    Attributes¬øˆ
    ----------
    pulse_type: str, default='slr'
        Type of rf pulse.
    ptype : str, default='st'
        # Gyromagnetic ratio. Default gamma is specified for Hydrogen.
    ftype : str, default='ls'
        #Raster time for gradient waveforms.
    d1 : float, default=0.01
        #Unit of maximum gradient amplitude. Must be one of 'Hz/m', 'mT/m' or 'rad/ms/mm'.
    d2 : float, default=0.01
        #Maximum gradient amplitude.
    cancel_alpha_phs: str, default='None'
        #Maximum slew rate.
    n_bands : int, default=3
        Number of SMS slices to excite
    band_sep : int, default=20
        #Raster time for radio-frequency pulses.
    phs_0_pht : str, default='None'
        #Unit of maximum slew rate. Must be one of 'Hz/m/s', 'mT/m/ms', 'T/m/s' or 'rad/ms/mm/ms'.

    Raises
    ------
    ValueError
        If invalid `grad_unit` is passed. Must be one of 'Hz/m', 'mT/m' or 'rad/ms/mm'.
        If invalid `slew_unit` is passed. Must be one of 'Hz/m/s', 'mT/m/ms', 'T/m/s' or 'rad/ms/mm/ms'.
    """

    def __init__(self, pulse_type: str = 'slr', ptype: str = 'st', ftype: str = 'ls', d1: float = 0.01,
                 d2: float = 0.01,
                 cancel_alpha_phs: bool = False, n_bands: int = 3, band_sep: int = 20, phs_0_pt: str = 'None'):
        self.pulse_type = pulse_type
        if pulse_type == 'slr':
            self.ptype = ptype
            self.ftype = ftype
            self.d1 = d1
            self.d2 = d2
            self.cancel_alpha_phs = cancel_alpha_phs

        if pulse_type == 'sms':
            self.ptype = ptype
            self.ftype = ftype
            self.d1 = d1
            self.d2 = d2
            self.cancel_alpha_phs = cancel_alpha_phs
            self.n_bands = n_bands
            self.band_sep = band_sep
            self.phs_0_pt = phs_0_pt

    def __str__(self):
        s = "Pulse options:"
        s += "\nptype: " + str(self.ptype)
        s += "\nftype: " + str(self.ftype)
        s += "\nd1: " + str(self.d1)
        s += "\nd2: " + str(self.d2)
        s += "\ncancel_alpha_phs: " + str(self.cancel_alpha_phs)

        return s
