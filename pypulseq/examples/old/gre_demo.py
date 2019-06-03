"""
This is starter code to demonstrate a working example of the Gradient Recalled Echo on PyPulseq.
Author: Keerthi Sravan Ravi
Date: December 15, 2017.
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

kwargs_for_opts = {"rf_ring_down_time": 30e-6, "rf_dead_time": 100e-6}
system = Opts(kwargs_for_opts)
seq = Sequence(system)

fov = 220e-3
Nx = 256
Ny = 256
slice_thickness = 5e-3

flip = 15 * pi / 180
kwargs_for_sinc = {"flip_angle": flip, "system": system, "duration": 4e-3, "slice_thickness": slice_thickness,
                   "apodization": 0.5, "time_bw_product": 4}
rf, gz = make_sinc_pulse(kwargs_for_sinc, 2)
# plt.plot(rf.t[0], rf.signal[0])
# plt.show()

delta_k = 1 / fov
kWidth = Nx * delta_k
readoutTime = 6.4e-3
kwargs_for_gx = {"channel": 'x', "system": system, "flat_area": kWidth, "flat_time": readoutTime}
gx = make_trapezoid(kwargs_for_gx)
kwargs_for_adc = {"num_samples": Nx, "duration": gx.flat_time, "delay": gx.rise_time}
adc = make_adc(kwargs_for_adc)

kwargs_for_gxpre = {"channel": 'x', "system": system, "area": -gx.area / 2, "duration": 2e-3}
gx_pre = make_trapezoid(kwargs_for_gxpre)
kwargs_for_gz_reph = {"channel": 'z', "system": system, "area": -gz.area / 2, "duration": 2e-3}
gz_reph = make_trapezoid(kwargs_for_gz_reph)
phase_areas = (np.arange(Ny) - (Ny / 2)) * delta_k

TE, TR = 10e-3, 200e-3
delayTE = TE - calc_duration(gx_pre) - calc_duration(gz) / 2 - calc_duration(gx) / 2
delayTR = TR - calc_duration(gx_pre) - calc_duration(gz) - calc_duration(gx) - delayTE
delay1 = make_delay(delayTE)
delay2 = make_delay(delayTR)

for i in range(Ny):
    seq.add_block(rf, gz)
    kwargsForGyPre = {"channel": 'y', "system": system, "area": phase_areas[i], "duration": 2e-3}
    gyPre = make_trapezoid(kwargsForGyPre)
    seq.add_block(gx_pre, gyPre, gz_reph)
    seq.add_block(delay1)
    seq.add_block(gx, adc)
    seq.add_block(delay2)

# Display 1 TR
seq.plot(time_range=(0, TR))

# Display entire plot
# seq.plot()

# The .seq file will be available inside the /gpi/<user>/imr_framework folder
# seq.write("gre_python.seq")
seq.write("/Users/sravan953/Desktop/gre_python.seq")
