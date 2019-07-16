import math

import numpy as np

from pypulseq.Sequence.sequence import Sequence
from pypulseq.make_adc import make_adc
from pypulseq.make_sinc_pulse import make_sinc_pulse
from pypulseq.make_trap_pulse import make_trapezoid
from pypulseq.opts import Opts
import matplotlib.pyplot as plt

seq = Sequence()
fov = 220e-3
Nx = 64
Ny = 64
slice_thickness = 3e-3
n_slices = 1

system = Opts(max_grad=32, grad_unit='mT/m', max_slew=130, slew_unit='T/m/s', rf_ringdown_time=30e-6,
              rf_dead_time=100e-6)

rf, gz, _ = make_sinc_pulse(flip_angle=math.pi / 2, system=system, duration=3e-3, slice_thickness=slice_thickness,
                            apodization=0.5, time_bw_product=4)

delta_k = 1 / fov
k_width = Nx * delta_k
dwell_time = 4e-6
readout_time = Nx * dwell_time
flat_time = math.ceil(readout_time * 1e5) * 1e-5
gx = make_trapezoid(channel='x', system=system, amplitude=k_width / readout_time, flat_time=flat_time)
adc = make_adc(num_samples=Nx, duration=readout_time,
               delay=gx.rise_time + flat_time / 2 - (readout_time - dwell_time) / 2)

pre_time = 8e-4
gx_pre = make_trapezoid(channel='x', system=system, area=-gx.area / 2, duration=pre_time)
gz_reph = make_trapezoid(channel='z', system=system, area=-gz.area / 2, duration=pre_time)
gy_pre = make_trapezoid(channel='y', system=system, area=-Ny / 2 * delta_k, duration=pre_time)

dur = math.ceil(2 * math.sqrt(delta_k / system.max_slew) / 10e-6) * 10e-6
gy = make_trapezoid(channel='y', system=system, area=delta_k, duration=dur)

for s in range(n_slices):
    rf.freq_offset = gz.amplitude * slice_thickness * (s - (n_slices - 1) / 2)
    seq.add_block(rf, gz)
    seq.add_block(gx_pre, gy_pre, gz_reph)
    for i in range(Ny):
        seq.add_block(gx, adc)
        seq.add_block(gy)
        gx.amplitude = -gx.amplitude

seq.plot()

ktraj_adc, ktraj, t_excitation, t_refocusing, t_adc = seq.calculate_kspace()

time_axis = np.arange(1, ktraj.shape[1] + 1) * system.grad_raster_time
plt.plot(time_axis, ktraj.T)
plt.plot(t_adc, ktraj_adc[0, :], '.')
plt.figure()
plt.plot(ktraj[0, :], ktraj[1, :], 'b')
plt.axis('equal')
plt.plot(ktraj_adc[0, :], ktraj_adc[1, :], 'r.')
plt.show()

seq.write('epi_pypulseq.seq')
