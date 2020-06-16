import math
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np

from pypulseq.Sequence.sequence import Sequence
from pypulseq.add_gradients import add_gradients
from pypulseq.align import align
from pypulseq.calc_duration import calc_duration
from pypulseq.make_adc import make_adc
from pypulseq.make_gauss_pulse import make_gauss_pulse
from pypulseq.make_sinc_pulse import make_sinc_pulse
from pypulseq.make_trap_pulse import make_trapezoid
from pypulseq.opts import Opts
from pypulseq.split_gradient_at import split_gradient_at

seq = Sequence()
fov = 250e-3
Nx = 64
Ny = 64
slice_thickness = 3e-3
n_slices = 1

pe_enable = 1

system = Opts(max_grad=32, grad_unit='mT/m', max_slew=130, slew_unit='T/m/s', rf_ringdown_time=30e-6,
              rf_dead_time=100e-6)

b0 = 2.89
sat_ppm = -3.45
sat_freq = sat_ppm * 1e-6 * b0 * system.gamma
rf_fs, _, _ = make_gauss_pulse(flip_angle=110 * math.pi / 180, system=system, duration=8e-3, bandwidth=abs(sat_freq),
                               freq_offset=sat_freq)
gz_fs = make_trapezoid(channel='z', system=system, delay=calc_duration(rf_fs), area=1 / 1e-4)

rf, gz, gz_reph = make_sinc_pulse(flip_angle=math.pi / 2, system=system, duration=3e-3, slice_thickness=slice_thickness,
                                  apodization=0.5, time_bw_product=4)

delta_k = 1 / fov
k_width = Nx * delta_k
readout_time = 4.2e-4

blip_dur = math.ceil(2 * math.sqrt(delta_k / system.max_slew) / 10e-6 / 2) * 10e-6 * 2
gy = make_trapezoid(channel='y', system=system, area=delta_k, duration=blip_dur)

extra_area = blip_dur / 2 * blip_dur / 2 * system.max_slew
gx = make_trapezoid(channel='x', system=system, area=k_width + extra_area, duration=readout_time + blip_dur)
actual_area = gx.area - gx.amplitude / gx.rise_time * blip_dur / 2 * blip_dur / 2 / 2 - gx.amplitude / gx.fall_time * blip_dur / 2 * blip_dur / 2 / 2
gx.amplitude = gx.amplitude / actual_area * k_width
gx.area = gx.amplitude * (gx.flat_time + gx.rise_time / 2 + gx.fall_time / 2)
gx.flat_area = gx.amplitude * gx.flat_time

adc_dwell_nyquist = delta_k / gx.amplitude
adc_dwell = math.floor(adc_dwell_nyquist * 1e7) * 1e-7
adc_samples = math.floor(readout_time / adc_dwell / 4) * 4

adc = make_adc(num_samples=adc_samples, dwell=adc_dwell, delay=blip_dur / 2)

time_to_center = adc.dwell * (adc_samples - 1) / 2
adc.delay = round((gx.rise_time + gx.flat_time / 2 - time_to_center) * 1e6) * 1e-6

gy_parts = split_gradient_at(grad=gy, time_point=blip_dur / 2, system=system)
gy_blipup, gy_blipdown, _ = align('right', gy_parts[0], 'left', gy_parts[1], gx)
gy_blipdownup = add_gradients([gy_blipdown, gy_blipup], system=system)

gy_blipup.waveform = gy_blipup.waveform * pe_enable
gy_blipdown.waveform = gy_blipdown.waveform * pe_enable
gy_blipdownup.waveform = gy_blipdownup.waveform * pe_enable

gx_pre = make_trapezoid(channel='x', system=system, area=-gx.area / 2)
gy_pre = make_trapezoid(channel='y', system=system, area=-Ny / 2 * delta_k)
gx_pre, gy_pre, gz_reph = align('right', gx_pre, 'left', gy_pre, gz_reph)
gy_pre = make_trapezoid(channel='y', system=system, area=gy_pre.area, duration=calc_duration(gx_pre, gy_pre, gz_reph))
gy_pre.amplitude = gy_pre.amplitude * pe_enable

for s in range(n_slices):
    seq.add_block(rf_fs, gz_fs)
    rf.freq_offset = gz.amplitude * slice_thickness * (s - (n_slices - 1) / 2)
    seq.add_block(rf, gz)
    seq.add_block(gx_pre, gy_pre, gz_reph)
    for i in range(1, Ny + 1):
        if i == 1:
            seq.add_block(gx, gy_blipup, adc)
        elif i == Ny:
            seq.add_block(gx, gy_blipdown, adc)
        else:
            seq.add_block(gx, gy_blipdownup, adc)
        gx.amplitude = -gx.amplitude

ok, error_report = seq.check_timing()

if ok:
    print('Timing check passed succesfully.')
else:
    warn('Timing check failed! Error listing follows:\n')
    print(error_report)
    print('\n')

seq.plot()

ktraj_adc, ktraj, t_excitation, t_refocusing, t_adc = seq.calculate_kspace()

time_axis = np.arange(1, ktraj.shape[1] + 1) * system.grad_raster_time
plt.plot(time_axis, ktraj.T)
plt.plot(t_adc, ktraj_adc[0], '.')
plt.figure()
plt.plot(ktraj_adc[0], ktraj_adc[1], 'b')
plt.axis('equal')
plt.plot(ktraj_adc[0], ktraj_adc[1], 'r.')
plt.show()

seq.set_definition('FOV', np.array([fov, fov, slice_thickness]) * 1e3)
seq.set_definition('Name', 'epi')

seq.write('epi_rs_pypulseq.seq')
