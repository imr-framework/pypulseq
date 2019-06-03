"""
This is starter code to demonstrate a working example of HASTE on PyPulseq.
Author: Sneha Potdar
Date: January 01, 2018.
"""

from math import pi

import numpy as np

from pypulseq.Sequence.sequence import Sequence
from pypulseq.calc_duration import calc_duration
from pypulseq.make_adc import make_adc
from pypulseq.make_delay import make_delay
from pypulseq.make_sinc_pulse import make_sinc_pulse
from pypulseq.make_trap_pulse import make_trapezoid
from pypulseq.opts import Opts

kwargs_for_opts = {"max_grad": 33, "grad_unit": "mT/m", "max_slew": 100, "slew_unit": "T/m/s", "rf_dead_time": 1e-6,
                   "adc_dead_time": 10e-6}
system = Opts(kwargs_for_opts)
seq = Sequence(system)

fov = 256e-3
Nx = 256
Ny = 256  # Define FOV and resolution
count = 0  # counter for updatng phase encode lines
Nex_fact = round((5 * Ny) / 8)  # A little more than half k-space; HASTE acquisition
TE = 100e-3  # in s
TR = 180e-3  # in s
BW = 50e3  # Hz
Slicethickness = 3e-3
dt = 4e-6
GradRasterTime = 1 / (2 * BW)  # s

flip = 90 * pi / 180
kwargs_for_sinc = {"flip_angle": flip, "system": system, "duration": 3e-3, "slice_thickness": Slicethickness,
                   "apodization": 0.5, "time_bw_product": 4}
rf, gz = make_sinc_pulse(kwargs_for_sinc, 2)

delta_k = 1 / fov
kWidth = Nx * delta_k
readoutTime = GradRasterTime * Nx
kwargs_for_gx = {"channel": 'x', "system": system, "flat_area": kWidth, "flat_time": readoutTime}
gx = make_trapezoid(kwargs_for_gx)

# kwargs_for_adc = {"num_samples": Nx, "system": system, "duration": gx.flat_time,
#                   "delay": gx.rise_time}  # adc = mr.makeAdc(Nx,lims,'Duration',gx.flatTime,'Delay',gx.riseTime)
# CHANGE: Matched ADC parameters with Matlab code
kwargs_for_adc = {"num_samples": Nx, "dwell": 4e-6}
adc = make_adc(kwargs_for_adc)
# kwargs_for_adc = {"num_samples": Nx, "Dwell": dt}
# adc = makeadc(kwargs_for_adc)


count = count + 1
# kwargs_for_gxPre = {"channel": 'x', "system": system, "area": -gx.area/2, "Duration": readoutTime/2}
# gxPre = maketrapezoid(kwargs_for_gxPre)


kwargs_for_gxpre = {"channel": 'x', "system": system, "area": -gx.area / 2, "duration": readoutTime / 2}
gxPre = make_trapezoid(kwargs_for_gxpre)

# kwargs_for_gzReph = {"channel": 'z', "system": system, "area": -gz.area/2, "Duration": 3e-3}
# gzReph = maketrapezoid(kwargs_for_gzReph)

# CHANGE: Duration changed from 2e-3 to 3e-3 to match Matlab code
kwargs_for_gz_reph = {"channel": 'z', "system": system, "area": -gz.area / 2, "duration": 3e-3}
gzReph = make_trapezoid(kwargs_for_gz_reph)

# kwargs_for_gy_pre = {"channel": 'y', "system": system, "area": -(Ny/2-count)*delta_k, "Duration": readoutTime/2}
# gyPre = maketrapezoid(kwargs_for_gy_pre)

kwargs_for_gy_pre = {"channel": 'y', "system": system, "area": -(Ny / 2 - count) * delta_k, "duration": readoutTime / 2}
gyPre = make_trapezoid(kwargs_for_gy_pre)

flip = 180 * pi / 180
kwargs_for_sinc = {"flip_angle": flip, "system": system, "duration": 3e-3, "slice_thickness": Slicethickness,
                   "apodization": 0.5, "time_bw_product": 4}
rf180, gz180 = make_sinc_pulse(kwargs_for_sinc, 2)

delayTE1 = TE / 2 - calc_duration(gzReph) - calc_duration(rf) - calc_duration(rf180) / 2
a = calc_duration(gx) / 2
b = calc_duration(rf180) / 2
delayTE2 = TE / 2 - calc_duration(gx) / 2 - calc_duration(rf180) / 2
delayTE3 = TR - TE - calc_duration(gx)
delay1 = make_delay(delayTE1)
delay2 = make_delay(delayTE2)
delay3 = make_delay(delayTE3)

# phaseAreas = ((0:Ny-1)-Ny/2)*deltak
# phaseAreas = round((0 in (Ny-1) - Ny/2) * delta_k)
# phase_areas = np.array(([x for x in range(0, Ny - 1)]))
phase_areas = np.arange(Ny)
phase_areas = (phase_areas - (Ny / 2)) * delta_k

seq.add_block(rf, gz)
seq.add_block(delay1)

count = 1
for j in range(Nex_fact):
    seq.add_block(rf180, gz180)
    seq.add_block(delay2)

    kwargs_for_gyPre_1 = {"channel": 'y', "area": -phase_areas[j], "duration": 2e-3}
    gyPre_1 = make_trapezoid(kwargs_for_gyPre_1)

    seq.add_block(gxPre, gyPre_1)
    seq.add_block(gx, adc)

    # CHANGE: removed 'system' keyword
    kwargs_for_gyPre = {"channel": 'y', "area": phase_areas[j], "duration": 2e-3}
    gyPre = make_trapezoid(kwargs_for_gyPre)

    seq.add_block(gxPre, gyPre)
    seq.add_block(delay3)

# seq.plot(time_range=(0, TR))
# seq.plot()
seq.write("/Users/sravan953/Desktop/haste_py.seq")
