import math

import numpy as np

from pypulseq.Sequence.sequence import Sequence
from pypulseq.calc_duration import calc_duration
from pypulseq.make_adc import make_adc
from pypulseq.make_delay import make_delay
from pypulseq.make_sinc_pulse import make_sinc_pulse
from pypulseq.make_trap_pulse import make_trapezoid
from pypulseq.opts import Opts

seq = Sequence()
fov = 250e-3
Nx = 256
Ny = 256
alpha = 10
slice_thickness = 3e-3
TE = np.array([7.38, 9.84]) * 1e-3
TR = 100e-3

rf_spoiling_inc = 117

sys = Opts(max_grad=28, grad_unit='mT/m', max_slew=150, slew_unit='T/m/s', rf_ringdown_time=20e-6, rf_dead_time=100e-6,
           adc_dead_time=10e-6)

rf, gz, gzr = make_sinc_pulse(flip_angle=alpha * math.pi / 180, duration=4e-3, slice_thickness=slice_thickness,
                              apodization=0.5, time_bw_product=4, system=sys)

delta_k = 1 / fov
gx = make_trapezoid(channel='x', flat_area=Nx * delta_k, flat_time=6.4e-3, system=sys)
adc = make_adc(num_samples=Nx, duration=gx.flat_time, delay=gx.rise_time, system=sys)
gx_pre = make_trapezoid(channel='x', area=-gx.area / 2, duration=2e-3, system=sys)
gz_reph = make_trapezoid(channel='z', area=-gz.area / 2, duration=2e-3, system=sys)
phase_areas = (np.arange(Ny) - Ny / 2) * delta_k

gx_spoil = make_trapezoid(channel='x', area=2 * Nx * delta_k, system=sys)
gz_spoil = make_trapezoid(channel='z', area=4 / slice_thickness, system=sys)

delay_TE = np.ceil((TE - calc_duration(gx_pre) - gz.fall_time - gz.flat_time / 2 - calc_duration(
    gx) / 2) / seq.grad_raster_time) * seq.grad_raster_time
delay_TR = np.ceil((TR - calc_duration(gx_pre) - calc_duration(gz) - calc_duration(
    gx) - delay_TE) / seq.grad_raster_time) * seq.grad_raster_time

assert np.all(delay_TR >= calc_duration(gx_spoil, gz_spoil))

rf_phase = 0
rf_inc = 0

for i in range(Ny):
    for j in range(len(TE)):
        rf.phase_offset = rf_phase / 180 * np.pi
        adc.phase_offset = rf_phase / 180 * np.pi
        rf_inc = divmod(rf_inc + rf_spoiling_inc, 360.0)[1]
        rf_phase = divmod(rf_phase + rf_inc, 360.0)[1]

        seq.add_block(rf, gz)
        gy_pre = make_trapezoid(channel='y', area=phase_areas[i], duration=2e-3, system=sys)
        seq.add_block(gx_pre, gy_pre, gz_reph)
        seq.add_block(make_delay(delay_TE[j]))
        seq.add_block(gx, adc)
        gy_pre.amplitude = -gy_pre.amplitude
        seq.add_block(make_delay(delay_TR[j]), gx_spoil, gy_pre, gz_spoil)

report = seq.test_report()
print(report)
seq.calculate_kspace()
seq.plot()
seq.write('gre_pypulseq.seq')
