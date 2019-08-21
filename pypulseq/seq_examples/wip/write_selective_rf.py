import math

import numpy as np

from pypulseq.Sequence.sequence import Sequence
from pypulseq.add_ramps import add_ramps
from pypulseq.calc_duration import calc_duration
from pypulseq.make_adc import make_adc
from pypulseq.make_arbitrary_grad import make_arbitrary_grad
from pypulseq.make_arbitrary_rf import make_arbitrary_rf
from pypulseq.make_delay import make_delay
from pypulseq.make_sinc_pulse import make_sinc_pulse
from pypulseq.make_trap_pulse import make_trapezoid
from pypulseq.opts import Opts
from pypulseq.traj_to_grad import traj_to_grad

seq = Sequence()
fov = 220e-3
Nx, Ny = 256, 256

foe = 200e-3
target_width = 22.5e-3
n = 8
T = 8e-3

system = Opts(max_grad=32, grad_unit='mT/m', max_slew=130, slew_unit='T/m/s', rf_ringdown_time=30e-6,
              rf_dead_time=100e-6)

k_max = (2 * n) / foe / 2
tk = np.linspace(0, T - seq.grad_raster_time, num=T / seq.grad_raster_time)
kx = k_max * (1 - tk / T) * np.cos(2 * np.pi * n * tk / T)
ky = k_max * (1 - tk / T) * np.sin(2 * np.pi * n * tk / T)

tr = np.linspace(0, T - seq.rf_raster_time, num=T / seq.rf_raster_time)
kx_rf = np.interp(x=tr, xp=tk, fp=kx)
ky_rf = np.interp(x=tr, xp=tk, fp=ky)
beta = 2 * math.pi * k_max * target_width / 2 / math.sqrt(2)
signal0 = np.exp(-beta ** 2 * (1 - tr / T) ** 2) * np.sqrt((2 * np.pi * n * (1 - tr / T)) ** 2 + 1)
signal = signal0 * (1 + np.exp(-1j * 2 * np.pi * 5e-2 * (kx_rf + ky_rf)))

kx, ky, signal = add_ramps(k=[kx, ky], rf=signal)

rf, _ = make_arbitrary_rf(signal=signal, flip_angle=20 * math.pi / 180, system=system)
gx_rf = make_arbitrary_grad(channel='x', waveform=traj_to_grad(k=kx)[0])
gy_rf = make_arbitrary_grad(channel='y', waveform=traj_to_grad(k=ky)[0])

delta_k = 1 / fov
gx = make_trapezoid(channel='x', flat_area=Nx * delta_k, flat_time=6.4e-3)
adc = make_adc(num_samples=Nx, duration=gx.flat_time, delay=gx.rise_time)
gx_pre = make_trapezoid(channel='x', area=-gx.area / 2, duration=2e-3)
phase_areas = (np.arange(Ny) - Ny / 2) * delta_k

rf180, gz, _ = make_sinc_pulse(flip_angle=math.pi, system=system, duration=3e-3, slice_thickness=5e-3, apodization=0.5,
                               time_bw_product=4)
gz_spoil = make_trapezoid(channel='z', area=gx.area, duration=2e-3)

delay_TE1 = math.ceil(
    (20e-3 / 2 - calc_duration(gz_spoil) - calc_duration(rf180) / 2) / seq.grad_raster_time) * seq.grad_raster_time
delay_TE2 = delay_TE1 - calc_duration(gx_pre) - calc_duration(gx) / 2
delay_TR = 500e-3 - 20e-3 - calc_duration(rf) - calc_duration(gx) / 2

for i in range(Ny):
    seq.add_block(rf, gx_rf, gy_rf)
    seq.add_block(make_delay(delay_TE1))
    seq.add_block(gz_spoil)
    seq.add_block(rf180, gz)
    seq.add_block(gz_spoil)
    seq.add_block(make_delay(delay_TE2))
    gy_pre = make_trapezoid(channel='y', area=phase_areas[i], duration=2e-3)
    seq.add_block(gx_pre, gy_pre)
    seq.add_block(gx, adc)
    seq.add_block(make_delay(delay_TR))

seq.write('selective_rf_pypulseq.seq')
# seq.plot()
