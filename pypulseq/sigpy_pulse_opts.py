class SigpyPulseOpts:
    def __init__(
        self,
        pulse_type: str = 'slr',
        ptype: str = 'st',
        ftype: str = 'ls',
        d1: float = 0.01,
        d2: float = 0.01,
        cancel_alpha_phs: bool = False,
        n_bands: int = 3,
        band_sep: int = 20,
        phs_0_pt: str = 'None',
    ):
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

    def __str__(self) -> str:
        s = 'Pulse options:'
        s += '\nptype: ' + str(self.ptype)
        s += '\nftype: ' + str(self.ftype)
        s += '\nd1: ' + str(self.d1)
        s += '\nd2: ' + str(self.d2)
        s += '\ncancel_alpha_phs: ' + str(self.cancel_alpha_phs)

        return s
