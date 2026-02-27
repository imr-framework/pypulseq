import pulserver.core.Sequence as pseq

import pypulseq as pp

do_plot = True

# Set system limits
system = pp.Opts(
    max_grad=32,
    grad_unit='mT/m',
    max_slew=130,
    slew_unit='T/m/s',
    rf_ringdown_time=0.0,
    rf_dead_time=0.0,
    adc_dead_time=0.0,
    grad_raster_time=4e-6,
    rf_raster_time=2e-6,
    adc_raster_time=2e-6,
    block_duration_raster=4e-6,
)

# %% bSSFP (TR: 7ms)
seq0 = pp.Sequence(system)
seq0.read('./bssfp.seq')
seq = pseq.PulserverSequence(seq0)
pns = pseq.get_pns(seq, chronaxie_us=360.0, rheobase=20.0, alpha=0.333, do_plot=True)

# %% GRE (TR: 12ms)
seq0 = pp.Sequence(system)
seq0.read('./gre.seq')
seq = pseq.PulserverSequence(seq0)
pns = pseq.get_pns(seq, chronaxie_us=360.0, rheobase=20.0, alpha=0.333, do_plot=True)

# %% HASTE (16ms esp)
seq0 = pp.Sequence(system)
seq0.read('./haste.seq')
seq = pseq.PulserverSequence(seq0)
pns = pseq.get_pns(seq, chronaxie_us=360.0, rheobase=20.0, alpha=0.333, do_plot=True)

# %% EPI (640us esp)
seq0 = pp.Sequence(system)
seq0.read('./epi.seq')
seq = pseq.PulserverSequence(seq0)
pns = pseq.get_pns(seq, chronaxie_us=360.0, rheobase=20.0, alpha=0.333, do_plot=True)

# %% MPRAGE (10s TR)
seq0 = pp.Sequence(system)
seq0.read('./mprage.seq')
seq = pseq.PulserverSequence(seq0)
pns = pseq.get_pns(seq, chronaxie_us=360.0, rheobase=20.0, alpha=0.333, do_plot=True)
