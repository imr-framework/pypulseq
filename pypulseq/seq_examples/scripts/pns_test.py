import pypulseq as pp
from pypulseq.utils.safe_pns_prediction import safe_example_hw
import numpy as np

from copy import copy

# Set system limits
sys = pp.Opts(max_grad=32, grad_unit='mT/m',
              max_slew=130, slew_unit='T/m/s',
              rf_ringdown_time=20e-6,
              rf_dead_time=100e-6,
              adc_dead_time=20e-6,
              B0=2.89)

seq = pp.Sequence()      # Create a new sequence object

## prepare test objects
# pns is induced by the ramps, so we use long gradients to isolate the
# effects of the ramps
gpt = 10e-3
delay = 30e-3
rt_min = sys.grad_raster_time
rt_test = np.floor(sys.max_grad/sys.max_slew/sys.grad_raster_time)*sys.grad_raster_time
ga_min = sys.max_slew*rt_min
ga_test = sys.max_slew*rt_test

gx_min = pp.make_trapezoid(channel='x',system=sys,amplitude=ga_min,rise_time=rt_min,fall_time=2*rt_min,flat_time=gpt)
gy_min = copy(gx_min)
gy_min.channel = 'y'
gz_min = copy(gx_min)
gz_min.channel = 'z'

gx_test = pp.make_trapezoid(channel='x',system=sys,amplitude=ga_test,rise_time=rt_test,fall_time=2*rt_test,flat_time=gpt)
gy_test = copy(gx_test)
gy_test.channel = 'y'
gz_test = copy(gx_test)
gz_test.channel = 'z'

g_min = [gx_min,gy_min,gz_min]
g_test = [gx_test,gy_test,gz_test]

# dummy FID sequence
# Create non-selective pulse 
rf = pp.make_block_pulse(np.pi/2,duration=0.1e-3, system=sys)
# Define delays and ADC events
adc = pp.make_adc(512,duration=6.4e-3, system=sys)


## Define sequence blocks
seq.add_block(pp.make_delay(delay))
for a in range(3):
    seq.add_block(g_min[a])
    seq.add_block(pp.make_delay(delay))
    seq.add_block(g_test[a])
    seq.add_block(pp.make_delay(delay))
    for b in range(a+1, 3):
        seq.add_block(g_min[a],g_min[b])
        seq.add_block(pp.make_delay(delay))
        seq.add_block(g_test[a],g_test[b])
        seq.add_block(pp.make_delay(delay))

seq.add_block(*g_min)
seq.add_block(pp.make_delay(delay))
seq.add_block(*g_test)
seq.add_block(pp.make_delay(delay))
seq.add_block(*g_min)
seq.add_block(rf)
seq.add_block(adc)

## check whether the timing of the sequence is correct
ok, error_report = seq.check_timing()

if (ok):
    print('Timing check passed successfully')
else:
    print('Timing check failed! Error listing follows:')
    print(error_report)
    print('\n')


## do some visualizations
seq.plot()             # Plot all sequence waveforms

## 'install' to the IDEA simulator
# seq.write('idea/external.seq')

## PNS calc
pns_ok, pns_n, pns_c, tpns = seq.calculate_pns(safe_example_hw(), do_plots=True) # Safe example HW

# pns_ok, pns_n, pns_c, tpns = seq.calculate_pns('idea/asc/MP_GPA_K2309_2250V_951A_AS82.asc', do_plots=True) # prisma
# pns_ok, pns_n, pns_c, tpns = seq.calculate_pns('idea/asc/MP_GPA_K2309_2250V_951A_GC98SQ.asc', do_plots=True) # aera-xq

# ## load simulation results 

# #[sll,~,~,vscale]=dsv_read('idea/dsv/prisma_pulseq_SLL.dsv')
# #[sll,~,~,vscale]=dsv_read('idea/dsv/aera_pulseq_SLL.dsv')
# [sll,~,~,vscale]=dsv_read('idea/dsv/terra_pulseq_SLL.dsv')
# sll=cumsum(sll/vscale)

# ## plot
# figureplot(sll(104:end)) # why 104? good question
# hold on
# plot(tpns*1e5-0.5,pns_n)
# title('comparing internal and IDEA PNS predictions')

# ## manual time alignment to calculate relative differences
# ssl_s=104+tpns(1)*1e5-1.5
# ssl_e=ssl_s+length(pns_n)-1
# #figureplot(sll(ssl_s:ssl_e))hold on plot(pns_n)
# figureplot((sll(ssl_s:ssl_e)-pns_n)./pns_n*100)
# title('relative difference in #')