import math
import warnings

import matplotlib.pyplot as plt
import numpy as np

from pypulseq.Sequence.sequence import Sequence
from pypulseq.calc_rf_center import calc_rf_center
from pypulseq.make_adc import make_adc
from pypulseq.make_delay import make_delay
from pypulseq.make_extended_trapezoid import make_extended_trapezoid
from pypulseq.make_sinc_pulse import make_sinc_pulse
from pypulseq.make_trap_pulse import make_trapezoid
from pypulseq.opts import Opts

dG = 250e-6
system = Opts(max_grad=30, grad_unit='mT/m', max_slew=170, slew_unit='T/m/s', rf_ringdown_time=100e-6,
              rf_dead_time=100e-6, adc_dead_time=10e-6)

seq = Sequence(system=system)

fov = 256e-3
Ny_pre = 8
Nx, Ny = 128, 128
n_echo = int(Ny / 2 + Ny_pre)
n_slices = 1
rf_flip = 180
if isinstance(rf_flip, int):
    rf_flip = np.zeros(n_echo) + rf_flip
slice_thickness = 5e-3
TE = 12e-3
TR = 2000e-3
TE_eff = 60e-3
k0 = round(TE_eff / TE)
PE_type = 'linear'

readout_time = 6.4e-3 + 2 * system.adc_dead_time
t_ex = 2.5e-3
t_ex_wd = t_ex + system.rf_ringdown_time + system.rf_dead_time
t_ref = 2e-3
tf_ref_wd = t_ref + system.rf_ringdown_time + system.rf_dead_time
t_sp = 0.5 * (TE - readout_time - tf_ref_wd)
t_sp_ex = 0.5 * (TE - t_ex_wd - tf_ref_wd)
fspR = 1.0
fspS = 0.5

rfex_phase = math.pi / 2
rfref_phase = 0

flipex = 90 * math.pi / 180
rfex, gz, _ = make_sinc_pulse(flip_angle=flipex, system=system, duration=t_ex, slice_thickness=slice_thickness,
                              apodization=0.5, time_bw_product=4, phase_offset=rfex_phase)
GS_ex = make_trapezoid(channel='z', system=system, amplitude=gz.amplitude, flat_time=t_ex_wd, rise_time=dG)

flipref = rf_flip[0] * math.pi / 180
rfref, gz, _ = make_sinc_pulse(flip_angle=flipref, system=system, duration=t_ref, slice_thickness=slice_thickness,
                               apodization=0.5, time_bw_product=4, phase_offset=rfref_phase, use='refocusing')
GS_ref = make_trapezoid(channel='z', system=system, amplitude=GS_ex.amplitude, flat_time=tf_ref_wd, rise_time=dG)

AGS_ex = GS_ex.area / 2
GS_spr = make_trapezoid(channel='z', system=system, area=AGS_ex * (1 + fspS), duration=t_sp, rise_time=dG)
GS_spex = make_trapezoid(channel='z', system=system, area=AGS_ex * fspS, duration=t_sp_ex, rise_time=dG)

delta_k = 1 / fov
k_width = Nx * delta_k

GR_acq = make_trapezoid(channel='x', system=system, flat_area=k_width, flat_time=readout_time, rise_time=dG)
adc = make_adc(num_samples=Nx, duration=GR_acq.flat_time - 40e-6, delay=20e-6)
GR_spr = make_trapezoid(channel='x', system=system, area=GR_acq.area * fspR, duration=t_sp, rise_time=dG)
GR_spex = make_trapezoid(channel='x', system=system, area=GR_acq.area * (1 + fspR), duration=t_sp_ex, rise_time=dG)

AGR_spr = GR_spr.area
AGR_preph = GR_acq.area / 2 + AGR_spr
GR_preph = make_trapezoid(channel='x', system=system, area=AGR_preph, duration=t_sp_ex, rise_time=dG)

n_ex = 1
PE_order = np.arange(-Ny_pre, Ny + 1).T
phase_areas = PE_order * delta_k

GS1_times = [0, GS_ex.rise_time]
GS1_amp = [0, GS_ex.amplitude]
GS1 = make_extended_trapezoid(channel='z', times=GS1_times, amplitudes=GS1_amp)

GS2_times = [0, GS_ex.flat_time]
GS2_amp = [GS_ex.amplitude, GS_ex.amplitude]
GS2 = make_extended_trapezoid(channel='z', times=GS2_times, amplitudes=GS2_amp)

GS3_times = [0, GS_spex.rise_time, GS_spex.rise_time + GS_spex.flat_time,
             GS_spex.rise_time + GS_spex.flat_time + GS_spex.fall_time]
GS3_amp = [GS_ex.amplitude, GS_spex.amplitude, GS_spex.amplitude, GS_ref.amplitude]
GS3 = make_extended_trapezoid(channel='z', times=GS3_times, amplitudes=GS3_amp)

GS4_times = [0, GS_ref.flat_time]
GS4_amp = [GS_ref.amplitude, GS_ref.amplitude]
GS4 = make_extended_trapezoid(channel='z', times=GS4_times, amplitudes=GS4_amp)

GS5_times = [0, GS_spr.rise_time, GS_spr.rise_time + GS_spr.flat_time,
             GS_spr.rise_time + GS_spr.flat_time + GS_spr.fall_time]
GS5_amp = [GS_ref.amplitude, GS_spr.amplitude, GS_spr.amplitude, 0]
GS5 = make_extended_trapezoid(channel='z', times=GS5_times, amplitudes=GS5_amp)

GS7_times = [0, GS_spr.rise_time, GS_spr.rise_time + GS_spr.flat_time,
             GS_spr.rise_time + GS_spr.flat_time + GS_spr.fall_time]
GS7_amp = [0, GS_spr.amplitude, GS_spr.amplitude, GS_ref.amplitude]
GS7 = make_extended_trapezoid(channel='z', times=GS7_times, amplitudes=GS7_amp)

GR3 = GR_preph

GR5_times = [0, GR_spr.rise_time, GR_spr.rise_time + GR_spr.flat_time,
             GR_spr.rise_time + GR_spr.flat_time + GR_spr.fall_time]
GR5_amp = [0, GR_spr.amplitude, GR_spr.amplitude, GR_acq.amplitude]
GR5 = make_extended_trapezoid(channel='x', times=GR5_times, amplitudes=GR5_amp)

GR6_times = [0, readout_time]
GR6_amp = [GR_acq.amplitude, GR_acq.amplitude]
GR6 = make_extended_trapezoid(channel='x', times=GR6_times, amplitudes=GR6_amp)

GR7_times = [0, GR_spr.rise_time, GR_spr.rise_time + GR_spr.flat_time,
             GR_spr.rise_time + GR_spr.flat_time + GR_spr.fall_time]
GR7_amp = [GR_acq.amplitude, GR_spr.amplitude, GR_spr.amplitude, 0]
GR7 = make_extended_trapezoid(channel='x', times=GR7_times, amplitudes=GR7_amp)

tex = GS1.t[-1] + GS2.t[-1] + GS3.t[-1]
tref = GS4.t[-1] + GS5.t[-1] + GS7.t[-1] + readout_time
tend = GS4.t[-1] + GS5.t[-1]
TE_train = tex + n_echo * tref + tend
TR_fill = (TR - n_slices * TE_train) / n_slices

TR_fill = system.grad_raster_time * round(TR_fill / system.grad_raster_time)
if TR_fill < 0:
    TR_fill = 1e-3
    warnings.warn(f'TR too short, adapted to include all slices to: {1000 * n_slices * (TE_train + TR_fill)} ms')
else:
    print(f'TR fill: {1000 * TR_fill} ms')
delay_TR = make_delay(TR_fill)
delay_end = make_delay(5)

for k_ex in range(n_ex):
    for s in range(n_slices):
        rfex.freq_offset = GS_ex.amplitude * slice_thickness * (s - (n_slices - 1) / 2)
        rfref.freq_offset = GS_ref.amplitude * slice_thickness * (s - (n_slices - 1) / 2)
        rfex.phase_offset = rfex_phase - 2 * math.pi * rfex.freq_offset * calc_rf_center(rfex)[0]
        rfref.phase_offset = rfref_phase - 2 * math.pi * rfref.freq_offset * calc_rf_center(rfref)[0]

        seq.add_block(GS1)
        seq.add_block(GS2, rfex)
        seq.add_block(GS3, GR3)

        for k_ech in range(n_echo):
            if k_ex >= 0:
                phase_area = phase_areas[k_ech]
            else:
                phase_area = 0

            GP_pre = make_trapezoid(channel='y', system=system, area=phase_area, duration=t_sp, rise_time=dG)
            GP_rew = make_trapezoid(channel='y', system=system, area=-phase_area, duration=t_sp, rise_time=dG)

            seq.add_block(GS4, rfref)
            seq.add_block(GS5, GR5, GP_pre)

            if k_ex >= 0:
                seq.add_block(GR6, adc)
            else:
                seq.add_block(GR6)

            seq.add_block(GS7, GR7, GP_rew)

        seq.add_block(GS4)
        seq.add_block(GS5)
        seq.add_block(delay_TR)

seq.add_block(delay_end)

ktraj_adc, ktraj, t_excitation, t_refocusing, _ = seq.calculate_kspace()
plt.plot(ktraj.T)
plt.figure()
plt.plot(ktraj[0], ktraj[1], 'b', ktraj_adc[0], ktraj_adc[1], 'r.')
plt.show()

seq.write('haste_pypulseq.seq')
