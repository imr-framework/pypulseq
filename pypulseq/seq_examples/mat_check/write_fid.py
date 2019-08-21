import math

from pypulseq.Sequence.sequence import Sequence
from pypulseq.calc_duration import calc_duration
from pypulseq.make_adc import make_adc
from pypulseq.make_block_pulse import make_block_pulse
from pypulseq.make_delay import make_delay
from pypulseq.opts import Opts

system = Opts(rf_ringdown_time=30e-6, rf_dead_time=100e-6, adc_dead_time=20e-6)
seq = Sequence(system=system)
Nx = 256
N_rep = 16

rf, _ = make_block_pulse(flip_angle=math.pi / 2, duration=0.1e-3, system=system)

adc = make_adc(num_samples=Nx, duration=3.2e-3, system=system, delay=system.adc_dead_time)
delay_TE = 20e-3
delay_TR = 1000e-3

for i in range(N_rep):
    seq.add_block(rf)
    seq.add_block(make_delay(delay_TE))
    seq.add_block(adc, make_delay(calc_duration(adc)))
    seq.add_block(make_delay(delay_TR))

seq.write('fid_pypulseq.seq')
