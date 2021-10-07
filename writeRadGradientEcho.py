import math
import numpy as np
from matplotlib import pyplot as plt

import pypulseq as pp
seq=pp.Sequence() #Create a new sequence object
fov=250e-3 #Define FOV
Nx=256 #Define resolution
alpha=10 #flip angle
slice_thickness=3e-3 # slice
TE=8e-3 # TE; give a vector here to have multiple TEs (e.g. for field mapping)
TR=100e-3 # only a single value for now
Nr=128 #number of radial spokes
Ndummy=20 # number of dummy scans
delta=math.pi / Nr #angular increment; try golden angle pi*(3-5^0.5) or 0.5 of it

# more in-depth parameters
rfSpoilingInc=117 # RF spoiling increment
system = pp.Opts(max_grad=28, grad_unit='mT/m', max_slew=80, slew_unit='T/m/s', rf_ringdown_time=20e-6,
                 rf_dead_time=100e-6, adc_dead_time=10e-6)
rf, gz, gzr = pp.make_sinc_pulse(flip_angle=alpha * math.pi / 180, duration=4e-3, slice_thickness=slice_thickness,
                                 apodization=0.5, time_bw_product=4, system=system, return_gz=True)
# Define other gradients and ADC events
delta_k = 1 / fov

# Define other gradients and ADC events
delta_k = 1 / fov
gx = pp.make_trapezoid(channel='x', flat_area=Nx * delta_k, flat_time=3.2e-3, system=system)
adc = pp.make_adc(num_samples=Nx, duration=gx.flat_time, delay=gx.rise_time, system=system)
gx_pre = pp.make_trapezoid(channel='x', area=-gx.area / 2, duration=1e-3, system=system)
gz_reph = pp.make_trapezoid(channel='z', area=-gz.area / 2, duration=1e-3, system=system)
