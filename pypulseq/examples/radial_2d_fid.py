"""
This is starter code to demonstrate a working example of a 2D Radial FID on PyPulseq.
Author: Keerthi Sravan Ravi
Date: April 05, 2018.
"""
from math import pi, ceil

import numpy as np

from pypulseq.Sequence.sequence import Sequence
from pypulseq.calc_duration import calc_duration
from pypulseq.make_adc import makeadc
from pypulseq.make_delay import make_delay
from pypulseq.make_sinc import make_sinc_pulse
from pypulseq.make_trap import make_trapezoid
from pypulseq.opts import Opts

kwargs_for_opts = {"max_grad": 32, "grad_unit": "mT/m", "max_slew": 130, "slew_unit": "T/m/s"}
system = Opts(kwargs_for_opts)
seq = Sequence(system)

fov = 256e-3
Nx = 256
Ny = 256
slice_thickness = 5e-3
dx = fov / Nx
TR = 20e-3
TE = 5e-3

flip = 15 * pi / 180
kwargs_for_sinc = {"flip_angle": flip, "system": system, "duration": 1.5e-3, "slice_thickness": slice_thickness,
                   "apodization": 0.5, "time_bw_product": 4}
rf, gz = make_sinc_pulse(kwargs_for_sinc, 2)

delta_k = 1 / fov
k_width = Nx * delta_k
readout_time = 6.4e-3
kwargs_for_gx = {"channel": 'x', "system": system, "flat_area": k_width, "flat_time": readout_time}
gx = make_trapezoid(kwargs_for_gx)
kwargs_for_adc = {"num_samples": Nx, "duration": gx.flat_time, "delay": gx.rise_time}
adc = makeadc(kwargs_for_adc)

n_slices = 3
deltaz = n_slices * slice_thickness
z = np.linspace(-(deltaz / 2), (deltaz / 2), deltaz / slice_thickness + 1)

pre_time = 8e-4
kwargs_for_gz_reph = {"channel": 'z', "system": system, "area": -gz.area / 2, "duration": 1e-3}
gz_reph = make_trapezoid(kwargs_for_gz_reph)

kwargs_for_gz_spoil = {"channel": 'z', "system": system, "area": gz.area * 2, "duration": 3 * pre_time}
gz_spoil = make_trapezoid(kwargs_for_gz_spoil)

delayTE = TE - calc_duration(gz_reph) - calc_duration(rf) / 2
delayTR = TR - calc_duration(gz_reph) - calc_duration(rf) - calc_duration(gx) - calc_duration(gz_spoil)
delay1 = make_delay(delayTE)
delay2 = make_delay(delayTR)

N_p = 256
N_s = ceil(pi * N_p)
dtheta = 360 / N_s
theta = np.arange(0, N_s * dtheta, dtheta)
ktraj = np.zeros((N_s, adc.num_samples))

for s in range(n_slices):
    freq_offset = gz.amplitude * z[s]
    rf.freq_offset = freq_offset

    for N_p in range(N_s):
        seq.add_block(rf, gz)
        seq.add_block(gz_reph)

        kWidth_projx = np.multiply(k_width, np.cos(theta[N_p] * pi / 180))
        kWidth_projy = np.multiply(k_width, np.sin(theta[N_p] * pi / 180))

        kwargs_for_gx = {"channel": 'x', 'system': system, 'flat_area': kWidth_projx, 'flat_time': readout_time}
        gx = make_trapezoid(kwargs_for_gx)
        kwargs_for_gy = {"channel": 'y', 'system': system, 'flat_area': kWidth_projy, 'flat_time': readout_time}
        gy = make_trapezoid(kwargs_for_gy)

        seq.add_block(delay1)
        seq.add_block(gx, gy, adc)
        seq.add_block(gz_spoil)
        seq.add_block(delay2)

# Display entire plot
seq.plot()

# The .seq file will be available inside the /gpi/<user>/imr_framework folder
seq.write('radial_2d_256_3_16_python.seq')
