import pypulseq as pp 
import copy 



def combine_blocks(block1, block2):
    
    pass


def get_event_updates(base_block, block, indices):
    """
    This function will return the event updates to the base block
    """
    
    update_block = copy.copy(base_block)
    
    if block.block_duration  > base_block.block_duration: # need to check this condition
        update_block.block_duration = block.block_duration - base_block.block_duration
        
    for i in indices:
        if i == 1: # RF update
            update_block.rf = update_RF(base_block.rf, block.rf)
        elif i == 2: # Gradient x update
            update_block.gx = update_gradient(base_block.gx, block.gx)
        elif i == 3: # Gradient y update
            update_block.gy = update_gradient(base_block.gy, block.gy)
        elif i == 4: # Gradient z update
            update_block.gz = update_gradient(base_block.gz, block.gz)
        elif i == 5: # ADC update
            update_block.adc = update_ADC(base_block.adc, block.adc)
              
    return update_block
        
    
    
    
def update_RF(base_rf, rf):
    update_rf = copy.copy(base_rf)
    
    update_rf.dead_time = 0
    update_rf.delay = 0
    update_rf.freq_offset = 0
    update_rf.phase_offset = 0
    update_rf.ringdown_time = 0
    update_rf.signal = 0
    update_rf.t = 0
    update_rf.shape_dur = 0

    if base_rf.freq_offset != rf.freq_offset:
        update_rf.freq_offset = rf.freq_offset - base_rf.freq_offset

    if base_rf.phase_offset != rf.phase_offset:
        update_rf.phase_offset = rf.phase_offset - base_rf.phase_offset
        
    if base_rf.dead_time != rf.dead_time:
        update_rf.dead_time = rf.dead_time - base_rf.dead_time
        
    if base_rf.ringdown_time != rf.ringdown_time:
        update_rf.ringdown_time = rf.ringdown_time - base_rf.ringdown_time
        
    if base_rf.delay != rf.delay:
        update_rf.delay = rf.delay - base_rf.delay
    
    s = base_rf.t != rf.t
    if True in s:
        update_rf.t = rf.t 
    
    if base_rf.shape_dur != rf.shape_dur :
        update_rf.shape_dur = rf.shape_dur 
        
    s = base_rf.signal != rf.signal
    if True in s:
        update_rf.signal = rf.signal - base_rf.signal
        
    return update_rf
        
        
def update_gradient(base_grad, grad):
    update_gradient = copy.copy(base_grad)
    
    if grad is not None:
        if grad.type == 'trap': # trapezoidal gradient waveform case
            
            update_gradient.amplitude = 0
            update_gradient.area = 0
            update_gradient.delay = 0
            update_gradient.fall_time = 0
            update_gradient.flat_time = 0
            update_gradient.flat_area = 0
            update_gradient.rise_time = 0
            
            if base_grad.amplitude != grad.amplitude:
                update_gradient.amplitude = grad.amplitude - base_grad.amplitude
                
            if base_grad.area != grad.area:
                update_gradient.area = grad.area - base_grad.area
                
            if base_grad.delay != grad.delay:
                update_gradient.delay = grad.delay - base_grad.delay
                
            if base_grad.fall_time != grad.fall_time:
                update_gradient.fall_time = grad.fall_time - base_grad.fall_time
                
            if base_grad.flat_area != grad.flat_area:
                update_gradient.flat_area = grad.flat_area - base_grad.flat_area
                
            if base_grad.flat_time != grad.flat_time:
                update_gradient.flat_time = grad.flat_time - base_grad.flat_time
                
            if base_grad.rise_time != grad.rise_time:
                update_gradient.rise_time = grad.rise_time - base_grad.rise_time
                
        elif grad.type == 'grad': # arbitrary gradient waveform case - need to think this more
            print('Arbitrary gradient waveform case')
            update_gradient = grad
    else:
        update_gradient = None
            
    return update_gradient

def update_ADC(base_adc, adc):
    if adc is not None:
        update_ADC = copy.copy(base_adc)
        update_ADC.delay = 0
        update_ADC.dwell = 0
        update_ADC.freq_offset = 0
        update_ADC.num_samples = 0
        update_ADC.phase_offset = 0
        
        if base_adc.delay != adc.delay:
            update_ADC.delay = adc.delay - base_adc.delay
        if base_adc.dwell != adc.dwell:
            update_ADC.dwell = adc.dwell - base_adc.dwell
        if base_adc.freq_offset != adc.freq_offset:
            update_ADC.freq_offset = adc.freq_offset - base_adc.freq_offset
        if base_adc.num_samples != adc.num_samples:
            update_ADC.num_samples = adc.num_samples - base_adc.num_samples
        if base_adc.phase_offset != adc.phase_offset:
            update_ADC.phase_offset = adc.phase_offset - base_adc.phase_offset
    else:
        update_ADC = None # some dummy scan blocks do not have ADCs  
    pass