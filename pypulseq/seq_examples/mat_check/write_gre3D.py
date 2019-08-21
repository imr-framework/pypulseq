import math
from types import SimpleNamespace

import numpy as np

from pypulseq.Sequence.sequence import Sequence
from pypulseq.calc_duration import calc_duration
from pypulseq.calc_rf_center import calc_rf_center
from pypulseq.make_adc import make_adc
from pypulseq.make_block_pulse import make_block_pulse
from pypulseq.make_delay import make_delay
from pypulseq.make_trap_pulse import make_trapezoid
from pypulseq.opts import Opts

seq = Sequence()
fov = [190e-3, 190e-3, 190e-3]
Nx = 64
Ny = 64
Nz = 64
T_read = 3.2e-3
T_pre = 3e-3
rise_time = 400e-6
N_dummy = 50

system = Opts(max_grad=20, grad_unit='mT/m', rise_time=rise_time, rf_ringdown_time=30e-6, rf_dead_time=100e-6)

rf, _ = make_block_pulse(flip_angle=8 * math.pi / 180, system=system, duration=0.2e-3)

delta_k = np.divide(1, fov)
gx = make_trapezoid(channel='x', system=system, flat_area=Nx * delta_k[0], flat_time=T_read)
adc = make_adc(num_samples=Nx, duration=gx.flat_time, delay=gx.rise_time)
gx_pre = make_trapezoid(channel='x', system=system, area=-gx.area / 2, duration=T_pre)
gx_spoil = make_trapezoid(channel='x', system=system, area=gx.area, duration=T_pre)
area_y = np.subtract(list(range(0, Ny)), Ny / 2) * delta_k[1]
area_z = np.subtract(list(range(0, Nz)), Nz / 2) * delta_k[2]

TE = 10e-3
TR = 40e-3
delay_TE = math.ceil((TE - calc_duration(rf) + calc_rf_center(rf)[0] + rf.delay - calc_duration(gx_pre) - calc_duration(
    gx) / 2) / seq.grad_raster_time) * seq.grad_raster_time
delay_TR = math.ceil((TR - calc_duration(rf) - calc_duration(gx_pre) - calc_duration(gx) - calc_duration(
    gx_spoil) - delay_TE) / seq.grad_raster_time) * seq.grad_raster_time

for i in range(1, N_dummy + 1):
    rf.phase_offset = (117 * (i ** 2 + i + 2) * math.pi / 180) % (2 * math.pi)
    seq.add_block(rf)

    gy_pre = make_trapezoid(channel='y', area=area_y[math.floor(Ny / 2) - 1], duration=T_pre)
    gy_reph = make_trapezoid(channel='y', area=-area_y[math.floor(Ny / 2) - 1], duration=T_pre)
    seq.add_block(gx_pre, gy_pre)
    seq.add_block(make_delay(delay_TE))
    seq.add_block(gx)
    seq.add_block(gy_reph, gx_spoil)
    seq.add_block(make_delay(delay_TR))

gy_pre = np.empty(Ny, dtype=SimpleNamespace)
gy_reph = np.empty(Ny, dtype=SimpleNamespace)
for i in range(Ny):
    gy_pre[i] = make_trapezoid(channel='y', area=area_y[i], duration=T_pre)
    gy_reph[i] = make_trapezoid(channel='y', area=-area_y[i], duration=T_pre)

for i in range(Nz):
    gz_pre = make_trapezoid(channel='z', area=area_z[i], duration=T_pre)
    gz_reph = make_trapezoid(channel='z', area=-area_z[i], duration=T_pre)
    for j in range(1, Ny + 1):
        rf.phase_offset = (117 * (j ** 2 + j + 2) * math.pi / 180) % (2 * math.pi)
        adc.phase_offset = rf.phase_offset

        seq.add_block(rf)

        seq.add_block(gx_pre, gy_pre[j - 1], gz_pre)
        seq.add_block(make_delay(delay_TE))
        seq.add_block(gx, adc)
        seq.add_block(gy_reph[j - 1], gz_reph, gx_spoil)
        seq.add_block(make_delay(delay_TR))

# time_range = np.multiply([N_dummy + 1, N_dummy + 3], TR)
# seq.plot(time_range=time_range)
seq.write('gre3d_python.seq')
