from math import pi

import numpy as np

import pypulseq as pp

Nx = 128
Ny = 128
n_slices = 3

system = pp.Opts(max_grad=32, grad_unit='mT/m', max_slew=130, slew_unit='T/m/s', grad_raster_time=10e-6,
                 rf_ringdown_time=10e-6, rf_dead_time=100e-6)
seq = pp.Sequence(system)

fov = 220e-3
slice_thickness = 5e-3
slice_gap = 15e-3

delta_z = n_slices * slice_gap
rf_offset = 0
z = np.linspace((-delta_z / 2), (delta_z / 2), n_slices) + rf_offset

# =========
# RF90, RF180
# =========
flip = 12 * pi / 180
rf, gz, _ = pp.make_sinc_pulse(flip_angle=flip, system=system, duration=2e-3, slice_thickness=slice_thickness,
                               apodization=0.5, time_bw_product=4, return_gz=True)

flip90 = 90 * pi / 180
rf90 = pp.make_block_pulse(flip_angle=flip90, system=system, duration=500e-6, slice_thickness=slice_thickness,
                           time_bw_product=4)

# =========
# Readout
# =========
delta_k = 1 / fov
k_width = Nx * delta_k
readout_time = 6.4e-3
gx = pp.make_trapezoid(channel='x', system=system, flat_area=k_width, flat_time=readout_time)
adc = pp.make_adc(num_samples=Nx, duration=gx.flat_time, delay=gx.rise_time)

# =========
# Prephase and Rephase
# =========
phase_areas = (np.arange(Ny) - (Ny / 2)) * delta_k
gy_pre = pp.make_trapezoid(channel='y', system=system, area=phase_areas[-1], duration=2e-3)

gx_pre = pp.make_trapezoid(channel='x', system=system, area=-gx.area / 2, duration=2e-3)

gz_reph = pp.make_trapezoid(channel='z', system=system, area=-gz.area / 2, duration=2e-3)

# =========
# Spoilers
# =========
pre_time = 8e-4
gx_spoil = pp.make_trapezoid(channel='x', system=system, area=gz.area * 4, duration=pre_time * 4)
gy_spoil = pp.make_trapezoid(channel='y', system=system, area=gz.area * 4, duration=pre_time * 4)
gz_spoil = pp.make_trapezoid(channel='z', system=system, area=gz.area * 4, duration=pre_time * 4)

# =========
# Delays
# =========
TE, TI, TR = 13e-3, 140e-3, 65e-3
delay_TE = TE - pp.calc_duration(rf) / 2 - pp.calc_duration(gy_pre) - pp.calc_duration(gx) / 2
delay_TE = pp.make_delay(delay_TE)
delay_TI = TI - pp.calc_duration(rf90) / 2 - pp.calc_duration(gx_spoil)
delay_TI = pp.make_delay(delay_TI)
delay_TR = TR - pp.calc_duration(rf) / 2 - pp.calc_duration(gx) / 2 - pp.calc_duration(gy_pre) - TE
delay_TR = pp.make_delay(delay_TR)

for j in range(n_slices):
    freq_offset = gz.amplitude * z[j]
    rf.freq_offset = freq_offset

    for i in range(Ny):
        seq.add_block(rf90)
        seq.add_block(gx_spoil, gy_spoil, gz_spoil)
        seq.add_block(delay_TI)
        seq.add_block(rf, gz)
        gy_pre = pp.make_trapezoid(channel='y', system=system, area=phase_areas[i], duration=2e-3)
        seq.add_block(gx_pre, gy_pre, gz_reph)
        seq.add_block(delay_TE)
        seq.add_block(gx, adc)
        gy_pre = pp.make_trapezoid(channel='y', system=system, area=-phase_areas[i], duration=2e-3)
        seq.add_block(gx_spoil, gy_pre)
        seq.add_block(delay_TR)

seq.set_definition(key='Name', val='2D T1 MPRAGE')
seq.write('2d_mprage_pypulseq.seq')
