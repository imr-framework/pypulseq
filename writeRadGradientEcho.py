import math

import numpy as np
from matplotlib import pyplot as plt

import pypulseq as pp
seq=pp.Sequence() #Create a new sequence object
fov=250e-3 #Define FOV
Nx=256 #Define resolution
alpha=10 #flip angle
slice_thickness = 3e-3  # slice
TE = np.array([8e-3])  # TE; give a vector here to have multiple TEs (e.g. for field mapping)
TR = 100e-3  # only a single value for now
Nr=128 #number of radial spokes
Ndummy = 20  # number of dummy scans
delta = np.pi / Nr  # angular increment; try golden angle pi*(3-5^0.5) or 0.5 of it

# more in-depth parameters
rfSpoilingInc = 117  # RF spoiling increment
system = pp.Opts(max_grad=28, grad_unit='mT/m', max_slew=80, slew_unit='T/m/s', rf_ringdown_time=20e-6,
                 rf_dead_time=100e-6, adc_dead_time=10e-6)
rf, gz, gzr = pp.make_sinc_pulse(flip_angle=alpha * np.pi / 180, duration=4e-3, slice_thickness=slice_thickness,
                                 apodization=0.5, time_bw_product=4, system=system, return_gz=True)

# Define other gradients and ADC events
delta_k = 1 / fov
gx = pp.make_trapezoid(channel='x', flat_area=Nx * delta_k, flat_time=6.4e-3, system=system)
adc = pp.make_adc(num_samples=Nx, duration=gx.flat_time, delay=gx.rise_time, system=system)
adc.delay = adc.delay - 0.5 * adc.dwell  # compensate for the 0.5 samples shift
gx_pre = pp.make_trapezoid(channel='x', area=-gx.area / 2, duration=2e-3, system=system)
gz_reph = pp.make_trapezoid(channel='z', area=-gz.area / 2, duration=2e-3, system=system)

# Gradient spoiling
gx_spoil = pp.make_trapezoid(channel='x', area=0.5 * Nx * delta_k, system=system)
gz_spoil = pp.make_trapezoid(channel='z', area=4 / slice_thickness, system=system)

# Calculate timing
delay_TE = np.ceil((TE - pp.calc_duration(gx_pre) - gz.fall_time - gz.flat_time / 2 - pp.calc_duration(
    gx) / 2) / seq.grad_raster_time) * seq.grad_raster_time
delay_TR = np.ceil((TR - pp.calc_duration(gx_pre) - pp.calc_duration(gz) - pp.calc_duration(
    gx) - delay_TE) / seq.grad_raster_time) * seq.grad_raster_time

assert np.all(delay_TR >= pp.calc_duration(gx_spoil, gz_spoil))

rf_phase = 0
rf_inc = 0

for i in range(-Ndummy, Nr, 1):
    for j in range(1, len(TE)):
        rf.phase_offset = rf_phase / 180 * np.pi
        adc.phase_offset = rf_phase / 180 * np.pi
        rf_inc = divmod(rf_inc + rf_spoiling_inc, 360.0)[1]
        rf_phase = divmod(rf_phase + rf_inc, 360.0)[1]

        seq.add_block(rf, gz)
        phi = delta * (i - 1)
        #seq.add_block(
