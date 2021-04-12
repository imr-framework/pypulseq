import math

import numpy as np
from matplotlib import pyplot as plt

import pypulseq as pp

# ======
# SETUP
seq = pp.Sequence()  # Create a new sequence object
# Define FOV and resolution
fov = 224e-3
Nx = 256
Ny = Nx
alpha = 10  # Flip angle
slice_thickness = 3e-3  # Slice thickness
n_slices = 1
TE = 4.3e-3
TR = 10e-3

rf_spoiling_inc = 117  # RF spoiling increment
ro_duration = 3.2e-3  # ADC duration

# Set system limits
system = pp.Opts(max_grad=28, grad_unit='mT/m', max_slew=150, slew_unit='T/m/s', rf_ringdown_time=20e-6,
                 rf_dead_time=100e-6, adc_dead_time=10e-6)

# ======
# CREATE EVENTS
# ======
# Create alpha-degree slice selection pulse and gradient
rf, gz, _ = pp.make_sinc_pulse(flip_angle=alpha * np.pi / 180, duration=3e-3, slice_thickness=slice_thickness,
                               apodization=0.5, time_bw_product=4, system=system, return_gz=True)

# Define other gradients and ADC events
delta_k = 1 / fov
gx = pp.make_trapezoid(channel='x', flat_area=Nx * delta_k, flat_time=ro_duration, system=system)
adc = pp.make_adc(num_samples=Nx, duration=gx.flat_time, delay=gx.rise_time, system=system)
gx_pre = pp.make_trapezoid(channel='x', area=-gx.area / 2, duration=1e-3, system=system)
gz_reph = pp.make_trapezoid(channel='z', area=-gz.area / 2, duration=1e-3, system=system)
phase_areas = -(np.arange(Ny) - Ny / 2) * delta_k

# Gradient spoiling
gx_spoil = pp.make_trapezoid(channel='x', area=2 * Nx * delta_k, system=system)
gz_spoil = pp.make_trapezoid(channel='z', area=4 / slice_thickness, system=system)

# Calculate timing
delay_TE = math.ceil((TE - pp.calc_duration(gx_pre) - gz.fall_time - gz.flat_time / 2 - pp.calc_duration(
    gx) / 2) / seq.grad_raster_time) * seq.grad_raster_time
delay_TR = math.ceil((TR - pp.calc_duration(gz) - pp.calc_duration(gx_pre) - pp.calc_duration(
    gx) - delay_TE) / seq.grad_raster_time) * seq.grad_raster_time
assert np.all(delay_TE >= 0)
assert np.all(delay_TR >= pp.calc_duration(gx_spoil, gz_spoil))

rf_phase = 0
rf_inc = 0

# ======
# CONSTRUCT SEQUENCE
# ======
# Loop over slices
for s in range(n_slices):
    rf.freq_offset = gz.amplitude * slice_thickness * (s - (n_slices - 1) / 2)
    # Loop over phase encodes and define sequence blocks
    for i in range(Ny):
        rf.phase_offset = rf_phase / 180 * np.pi
        adc.phase_offset = rf_phase / 180 * np.pi
        rf_inc = divmod(rf_inc + rf_spoiling_inc, 360.0)[1]
        rf_phase = divmod(rf_phase + rf_inc, 360.0)[1]

        seq.add_block(rf, gz)
        gy_pre = pp.make_trapezoid(channel='y', area=phase_areas[i], duration=pp.calc_duration(gx_pre), system=system)
        seq.add_block(gx_pre, gy_pre, gz_reph)
        seq.add_block(pp.make_delay(delay_TE))
        seq.add_block(gx, adc)
        gy_pre.amplitude = -gy_pre.amplitude
        spoil_block_contents = [pp.make_delay(delay_TR), gx_spoil, gy_pre, gz_spoil]
        if i != Ny - 1:
            spoil_block_contents.append(pp.make_label(type='INC', label='LIN', value=1))
        else:
            spoil_block_contents.extend([pp.make_label(type='SET', label='LIN', value=0),
                                         pp.make_label(type='INC', label='SLC', value=1)])
        seq.add_block(*spoil_block_contents)

ok, error_report = seq.check_timing()

if ok:
    print('Timing check passed successfully')
else:
    print('Timing check failed. Error listing follows:')
    [print(e) for e in error_report]

# ======
# VISUALIZATION
# ======
seq.plot(label='lin', time_range=np.array([0, 32]) * TR, time_disp='ms')

# Trajectory calculation and plotting
ktraj_adc, ktraj, t_excitation, t_refocusing, t_adc = seq.calculate_kspace()
time_axis = np.arange(1, ktraj.shape[1] + 1) * system.grad_raster_time
plt.figure()
plt.plot(time_axis, ktraj.T)  # Plot the entire k-space trajectory
plt.plot(t_adc, ktraj_adc[0], '.')  # Plot sampling points on the kx-axis
plt.figure()
plt.plot(ktraj[0], ktraj[1], 'b')  # 2D plot
plt.axis('equal')  # Enforce aspect ratio for the correct trajectory display
plt.plot(ktraj_adc[0], ktraj_adc[1], 'r.')  # Plot  sampling points
plt.show()

seq.set_definition(key='FOV', val=[fov, fov, slice_thickness * n_slices])
seq.set_definition(key='Name', val='gre_label')

seq.write('gre_label_pypulseq.seq')
