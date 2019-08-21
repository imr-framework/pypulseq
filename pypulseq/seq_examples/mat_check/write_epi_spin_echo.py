import math

import matplotlib.pyplot as plt
import numpy as np

from pypulseq.Sequence.sequence import Sequence
from pypulseq.calc_duration import calc_duration
from pypulseq.calc_rf_center import calc_rf_center
from pypulseq.make_adc import make_adc
from pypulseq.make_block_pulse import make_block_pulse
from pypulseq.make_delay import make_delay
from pypulseq.make_sinc_pulse import make_sinc_pulse
from pypulseq.make_trap_pulse import make_trapezoid
from pypulseq.opts import Opts

seq = Sequence()
fov = 256e-3
Nx = 64
Ny = 64
slice_thickness = 3e-3

system = Opts(max_grad=32, grad_unit='mT/m', max_slew=130, slew_unit='T/m/s', rf_ringdown_time=30e-6,
              rf_dead_time=100e-6, adc_dead_time=20e-6)

rf, gz, _ = make_sinc_pulse(flip_angle=math.pi / 2, system=system, duration=3e-3, slice_thickness=slice_thickness,
                            apodization=0.5, time_bw_product=4)

delta_k = 1 / fov
k_width = Nx * delta_k
readout_time = 3.2e-4
gx = make_trapezoid(channel='x', system=system, flat_area=k_width, flat_time=readout_time)
adc = make_adc(num_samples=Nx, system=system, duration=gx.flat_time, delay=gx.rise_time)

pre_time = 8e-4
gz_reph = make_trapezoid(channel='z', system=system, area=-gz.area / 2, duration=pre_time)
gx_pre = make_trapezoid(channel='x', system=system, area=gx.area / 2 - delta_k / 2, duration=pre_time)
gy_pre = make_trapezoid(channel='y', system=system, area=Ny / 2 * delta_k, duration=pre_time)

dur = math.ceil(2 * math.sqrt(delta_k / system.max_slew) / 10e-6) * 10e-6
gy = make_trapezoid(channel='y', system=system, area=delta_k, duration=dur)

rf180, _ = make_block_pulse(flip_angle=math.pi, system=system, duration=500e-6, use='refocusing')
gz_spoil = make_trapezoid(channel='z', system=system, area=gz.area * 2, duration=3 * pre_time)

TE = 60e-3
duration_to_center = (Nx / 2 + 0.5) * calc_duration(gx) + Ny / 2 * calc_duration(gy)
rf_center_incl_delay = rf.delay + calc_rf_center(rf)[0]
rf180_center_incl_delay = rf180.delay + calc_rf_center(rf180)[0]
delay_TE1 = TE / 2 - calc_duration(gz) + rf_center_incl_delay - pre_time - calc_duration(
    gz_spoil) - rf180_center_incl_delay
delay_TE2 = TE / 2 - calc_duration(rf180) + rf180_center_incl_delay - calc_duration(gz_spoil) - duration_to_center

seq.add_block(rf, gz)
seq.add_block(gx_pre, gy_pre, gz_reph)
seq.add_block(make_delay(delay_TE1))
seq.add_block(gz_spoil)
seq.add_block(rf180)
seq.add_block(gz_spoil)
seq.add_block(make_delay(delay_TE2))
for i in range(Ny):
    seq.add_block(gx, adc)
    seq.add_block(gy)
    gx.amplitude = -gx.amplitude
seq.add_block(make_delay(1e-4))

seq.write('epi_se_pypulseq.seq')
# seq.plot()
#
# ktraj_adc, ktraj, t_excitation, t_refocusing, t_adc = seq.calculate_kspace()
#
# time_axis = np.arange(1, ktraj.shape[1] + 1) * system.grad_raster_time
# plt.plot(time_axis, ktraj.T)
# plt.plot(t_adc, ktraj_adc[0, :], '.')
# plt.figure()
# plt.plot(ktraj[0, :], ktraj[1, :], 'b', ktraj_adc[0, :], ktraj_adc[1, :], 'r.')
# plt.axis('equal')
# plt.show()
#
# TE_check = (t_refocusing[0] - t_excitation[0]) * 2
# print(f'Intend TE: {TE * 1e3:.3f} ms, actual spin echo TE: {TE:.3f} ms')
