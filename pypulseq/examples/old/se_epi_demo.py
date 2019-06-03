"""
This is starter code to demonstrate a working example of a single-shot EPI Spin Echo on PyPulseq.
Author: Keerthi Sravan Ravi
Date: June 13, 2017.
"""
from math import pi, sqrt, ceil

from pypulseq.Sequence.sequence import Sequence
from pypulseq.calc_duration import calc_duration
from pypulseq.make_adc import make_adc
from pypulseq.make_block_pulse import make_block_pulse
from pypulseq.make_delay import make_delay
from pypulseq.make_sinc_pulse import make_sinc_pulse
from pypulseq.make_trap_pulse import make_trapezoid
from pypulseq.opts import Opts

kwargs_for_opts = {"max_grad": 33, "grad_unit": "mT/m", "max_slew": 110, "slew_unit": "T/m/s", "rf_dead_time": 10e-6,
                   "adc_dead_time": 10e-6}
system = Opts(kwargs_for_opts)
seq = Sequence(system)

fov = 220e-3
Nx = 128
Ny = 128
slice_thickness = 3e-3

flip = 90 * pi / 180
kwargs_for_sinc = {"flip_angle": flip, "system": system, "duration": 2.5e-3, "slice_thickness": slice_thickness,
                   "apodization": 0.5, "time_bw_product": 4}
rf, gz = make_sinc_pulse(kwargs_for_sinc, 2)
# plt.plot(rf.t[0], rf.signal[0])
# plt.show()

delta_k = 1 / fov
kWidth = Nx * delta_k
readoutTime = Nx * 4e-6
kwargs_for_gx = {"channel": 'x', "system": system, "flat_area": kWidth, "flat_time": readoutTime}
gx = make_trapezoid(kwargs_for_gx)
kwargs_for_adc = {"num_samples": Nx, "system": system, "duration": gx.flat_time, "delay": gx.rise_time}
adc = make_adc(kwargs_for_adc)

pre_time = 8e-4
kwargs_for_gxpre = {"channel": 'x', "system": system, "area": -gx.area / 2, "duration": pre_time}
gx_pre = make_trapezoid(kwargs_for_gxpre)
kwargs_for_gz_reph = {"channel": 'z', "system": system, "area": -gz.area / 2, "duration": pre_time}
gz_reph = make_trapezoid(kwargs_for_gz_reph)
kwargs_for_gy_pre = {"channel": 'y', "system": system, "area": -Ny / 2 * delta_k, "duration": pre_time}
gy_pre = make_trapezoid(kwargs_for_gy_pre)

dur = ceil(2 * sqrt(delta_k / system.max_slew) / 10e-6) * 10e-6
kwargs_for_gy = {"channel": 'y', "system": system, "area": delta_k, "duration": dur}
gy = make_trapezoid(kwargs_for_gy)

flip = 180 * pi / 180
kwargs_for_sinc = {"flip_angle": flip, "system": system, "duration": 2.5e-3}
rf180 = make_block_pulse(kwargs_for_sinc)
kwargs_for_gz_spoil = {"channel": 'z', "system": system, "area": gz.area * 2, "duration": 3 * pre_time}
gz_spoil = make_trapezoid(kwargs_for_gz_spoil)

TE, TR = 200e-3, 1000e-3
duration_to_center = (Nx / 2 + 0.5) * calc_duration(gx) + Ny / 2 * calc_duration(gy)
delayTE1 = TE / 2 - calc_duration(gz) / 2 - pre_time - calc_duration(gz_spoil) - calc_duration(rf180) / 2
delayTE2 = TE / 2 - calc_duration(rf180) / 2 - calc_duration(gz_spoil) - duration_to_center
delay1 = make_delay(delayTE1)
delay2 = make_delay(delayTE2)

seq.add_block(rf, gz)
seq.add_block(gx_pre, gy_pre, gz_reph)
seq.add_block(delay1)
seq.add_block(gz_spoil)
seq.add_block(rf180)
seq.add_block(gz_spoil)
seq.add_block(delay2)
for i in range(Ny):
    seq.add_block(gx, adc)
    seq.add_block(gy)
    gx.amplitude = -gx.amplitude
seq.add_block(make_delay(1))

# Display 1 TR
seq.plot(time_range=(0, TR))

# Display entire plot
# seq.plot()

# The .seq file will be available inside the /gpi/<user>/imr_framework folder
# seq.write("se_epi_python.seq")
