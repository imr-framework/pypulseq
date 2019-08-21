import math

import numpy as np

from pypulseq.Sequence.sequence import Sequence
from pypulseq.calc_duration import calc_duration
from pypulseq.make_adc import make_adc
from pypulseq.make_sinc_pulse import make_sinc_pulse
from pypulseq.make_trap_pulse import make_trapezoid

seq = Sequence()
fov = 220e-3
Nx, Ny = 256, 256

blip_amp = 0.2

rf, gz, _ = make_sinc_pulse(flip_angle=20 * math.pi / 180, duration=4e-3, slice_thickness=5e-3, apodization=0.5,
                            time_bw_product=4)

delta_k = 1 / fov
gx = make_trapezoid(channel='x', flat_area=Nx * delta_k, flat_time=6.4e-3)
adc = make_adc(num_samples=Nx, duration=gx.flat_time, delay=gx.rise_time)
gx_pre = make_trapezoid(channel='x', area=-gx.area / 2, duration=2e-3)
gz_reph = make_trapezoid(channel='z', area=-gz.area / 2, duration=2e-3)
phase_areas = (np.arange(Ny) - Ny / 2) * delta_k

delay_TE = 10e-3 - calc_duration(gx_pre) - calc_duration(rf) / 2 - calc_duration(gx) / 2
delay_TR = 40e-3 - calc_duration(gx_pre) - calc_duration(rf) - calc_duration(gx) - delay_TE

g1 = make_trapezoid()
# TODO Bug in Matlab
