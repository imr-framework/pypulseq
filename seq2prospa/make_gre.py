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
    Nx = 64
    Ny = 64
    alpha = 10
    slice_thickness = 3e-3

    d1 = 4e-3  # RF duration
    d2 = 2e-3 - (250 * 1e-6)  # Gy duration
    d3 = 2e-3 - (250 * 1e-6)  # Subtract grad ramp time from Prospa GUI
    adc_dwell = 50 * 1e-6
    delta_k = 1 / fov

    sys = Opts(max_grad=28, grad_unit='mT/m', max_slew=150, slew_unit='T/m/s', rf_ringdown_time=20e-6,
               rf_dead_time=0, adc_dead_time=10e-6)

    rf = make_sinc_pulse(flip_angle=alpha * math.pi / 180, duration=d1, slice_thickness=slice_thickness,
                         apodization=0.5, time_bw_product=4, system=sys, return_gz=False)

    phase_areas = (np.arange(Ny) - Ny / 2) * delta_k
    gx1 = make_trapezoid(channel='x', flat_area=Nx * delta_k, flat_time=d2, system=sys, rise_time=0.25e-3)
    gy_pre = make_trapezoid(channel='y', area=phase_areas[0], duration=d2, system=sys, rise_time=0.25e-3)

    gx2 = make_trapezoid(channel='x', flat_area=Nx * delta_k, flat_time=6.4e-3, rise_time=0.25e-3, system=sys)
    d4 = (d2 + (2 * 250 * 1e-6)) - (0.5 * Ny * adc_dwell) - (0.5 * gx2.rise_time)  # ADC delay
    adc = make_adc(num_samples=Nx, duration=gx2.flat_time - d3, delay=d4, system=sys)

    seq.add_block(rf)
    seq.add_block(gy_pre, gx1)
    seq.add_block(make_delay(d3))
    seq.add_block(gx2, adc)

    # seq.plot()

    return seq


if __name__ == '__main__':
    main()
