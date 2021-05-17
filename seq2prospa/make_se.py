import math

import numpy as np

from pypulseq.Sequence.sequence import Sequence
from pypulseq.make_adc import make_adc
from pypulseq.make_delay import make_delay
from pypulseq.make_sinc_pulse import make_sinc_pulse
from pypulseq.make_trap_pulse import make_trapezoid
from pypulseq.opts import Opts


def main():
    seq = Sequence()
    fov = 250e-3
    Nx = 128
    Ny = 128
    alpha = 10
    alpha180 = 180
    slice_thickness = 3e-3

    """
    n7 = 128
    gradRamp = 250 us
    acqTime = 128 * 50 us
    pulseLength = 100 us
    pulseAmplitude = -18
    """
    acqTime = Nx * 1e-6
    nrPnts = Nx
    bandWidth = acqTime / (nrPnts * 1000)
    gradAmpDelay = 43 * 1e-6
    gradRamp = 250e-6
    eddyCurrentCorr = (bandWidth / 10000 - 0.5) * acqTime * 1000 / nrPnts
    d1 = 100e-6
    d2 = acqTime * 500 - (2 * gradRamp)
    d3 = d2 + (gradRamp / 2)
    d4 = d3  # echoTime / 2 - d1 - d2 - d3 - 4 * gradRamp - pgo
    d5 = 2e-3 - (
            250 * 1e-6)  # echoTime / 2 - d1 / 2 + rxLat - acqTime * 500 - gradRamp - gradAmpDelay - eddyCurrentCorr
    d6 = gradAmpDelay + eddyCurrentCorr

    adc_dwell = 50 * 1e-6
    delta_k = 1 / fov

    sys = Opts(max_grad=28, grad_unit='mT/m', max_slew=150, slew_unit='T/m/s', rf_ringdown_time=20e-6,
               rf_dead_time=0, adc_dead_time=10e-6)

    rf = make_sinc_pulse(flip_angle=alpha * math.pi / 180, duration=d1, slice_thickness=slice_thickness,
                         apodization=0.5, time_bw_product=4, system=sys, return_gz=False)
    rf2 = make_sinc_pulse(flip_angle=alpha180 * math.pi / 180, duration=d1, slice_thickness=slice_thickness,
                          apodization=0.5, time_bw_product=4, system=sys, return_gz=False)

    phase_areas = (np.arange(Ny) - Ny / 2) * delta_k
    gx1 = make_trapezoid(channel='x', flat_area=Nx * delta_k, flat_time=d3, system=sys, rise_time=0.25e-3)
    gy_pre = make_trapezoid(channel='y', area=phase_areas[0], duration=d2, system=sys, rise_time=0.25e-3)

    gx2 = make_trapezoid(channel='x', flat_area=Nx * delta_k, flat_time=d6, rise_time=0.25e-3, system=sys)
    adc = make_adc(num_samples=Nx, duration=Nx * adc_dwell, system=sys)

    seq.add_block(rf)
    seq.add_block(gx1, gy_pre)
    seq.add_block(make_delay(d4))
    seq.add_block(rf2)
    seq.add_block(make_delay(d5))
    seq.add_block(gx2, adc)

    # seq.plot()

    return seq


if __name__ == '__main__':
    main()
