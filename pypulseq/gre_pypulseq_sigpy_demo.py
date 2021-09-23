import math
import numpy as np
from matplotlib import pyplot as plt
from pulse_opts import pulse_opts
import make_sigpy_pulse as sp
import pypulseq as pp

# ======
# SETUP
# ======
# Create a new sequence object
seq = pp.Sequence()

# Define FOV and resolution
fov = 256e-3
Nx = 256
Ny = 256
alpha = 10  # flip angle
slice_thickness = 3e-3  # slice
TE = np.array([4.3e-3])
TR = 10e-3

ext_pulse_library = 1  # 0 - pypulseq, 1 - sigpy
rf_spoiling_inc = 117  # RF spoiling increment

system = pp.Opts(max_grad=28, grad_unit='mT/m', max_slew=150, slew_unit='T/m/s', rf_ringdown_time=20e-6,
                 rf_dead_time=100e-6, adc_dead_time=10e-6)

# ======
# CREATE EVENTS
# ======
if (ext_pulse_library):
    pulse_cfg = pulse_opts(pulse_type='sms', ptype='st', ftype='ls', d1=0.01, d2=0.01, cancel_alpha_phs=False,
                           n_bands=3, band_sep=20, phs_0_pt='None')
    rf, gz, gzr,_ = sp.sigpy_n_seq(flip_angle=np.pi / 2, system=system, duration=3e-3, slice_thickness=slice_thickness,
                               time_bw_product=4, return_gz=True, pulse_cfg=pulse_cfg)
else:
    rf, gz, gzr = pp.make_sinc_pulse(flip_angle=alpha * math.pi / 180, duration=3e-3, slice_thickness=slice_thickness,
                                 apodization=0.5, time_bw_product=4, system=system, return_gz=True)
# Define other gradients and ADC events
delta_k = 1 / fov
gx = pp.make_trapezoid(channel='x', flat_area=Nx * delta_k, flat_time=3.2e-3, system=system)
adc = pp.make_adc(num_samples=Nx, duration=gx.flat_time, delay=gx.rise_time, system=system)
gx_pre = pp.make_trapezoid(channel='x', area=-gx.area / 2, duration=1e-3, system=system)
gz_reph = pp.make_trapezoid(channel='z', area=-gz.area / 2, duration=1e-3, system=system)
phase_areas = (np.arange(Ny) - Ny / 2) * delta_k

# gradient spoiling
gx_spoil = pp.make_trapezoid(channel='x', area=2 * Nx * delta_k, system=system)
gz_spoil = pp.make_trapezoid(channel='z', area=4 / slice_thickness, system=system)

# Calculate timing
delay_TE = np.ceil((TE - pp.calc_duration(gx_pre) - gz.fall_time - gz.flat_time / 2 - pp.calc_duration(
    gx) / 2) / seq.grad_raster_time) * seq.grad_raster_time
delay_TR = np.ceil((TR - pp.calc_duration(gz) - pp.calc_duration(gx_pre) - pp.calc_duration(
    gx) - delay_TE) / seq.grad_raster_time) * seq.grad_raster_time

assert np.all(delay_TE >= 0)
assert np.all(delay_TR >= pp.calc_duration(gx_spoil, gz_spoil))

rf_phase = 0
rf_inc = 0

# ======
# CONSTRUCT SEQUENCE
# ======
# Loop over phase encodes and define sequence blocks
for i in range(Ny):
    for j in range(len(TE)):
        rf.phase_offset = rf_phase / 180 * np.pi
        adc.phase_offset = rf_phase / 180 * np.pi
        rf_inc = divmod(rf_inc + rf_spoiling_inc, 360.0)[1]
        rf_phase = divmod(rf_phase + rf_inc, 360.0)[1]

        seq.add_block(rf, gz)
        gy_pre = pp.make_trapezoid(channel='y', area=phase_areas[i], duration=pp.calc_duration(gx_pre), system=system)
        seq.add_block(gx_pre, gy_pre, gz_reph)
        seq.add_block(pp.make_delay(delay_TE[j]))
        seq.add_block(gx, adc)
        gy_pre.amplitude = -gy_pre.amplitude
        seq.add_block(pp.make_delay(delay_TR[j]), gx_spoil, gy_pre, gz_spoil)

ok, error_report = seq.check_timing()  # Check whether the timing of the sequence is correct
if ok:
    print('Timing check passed successfully')
else:
    print('Timing check failed. Error listing follows:')
    [print(e) for e in error_report]

# ======
# VISUALIZATION
# ======
seq.plot()

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

# Prepare the sequence output for the scanner
seq.set_definition('FOV', [fov, fov, slice_thickness])
seq.set_definition('Name', 'gre')

seq.write('gre_pypulseq.seq')

# Very optional slow step, but useful for testing during development e.g. for the real TE, TR or for staying within
# slew-rate limits
rep = seq.test_report()
print(rep)