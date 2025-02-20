import pypulseq as pp 
import copy 
import numpy as np



def combine_blocks(block1, block2):
    
    pass


def get_event_updates(base_block, block, indices, trid, base_events, num_shots):
    """
    This function will return the event updates to the base block
    """
    
    # Create empty dict to store the event updates
    event_update_list = []
    
    event_id = []
    event_type = []
    event_attribute_name =[] 
    event_attribute_unit = []
    event_strength =[] 
    event_step =[]
    event_factor =[]

    update_block = copy.copy(base_block)
    
    if block.block_duration  > base_block.block_duration: # need to check this condition
        update_block.block_duration = block.block_duration - base_block.block_duration
        
    for i in indices:
        if i == 1: # RF update
            event_id.append(base_events[1])
            event_type.append('RF')
            name, unit, strength, step = update_RF(base_block.rf, block.rf, trid, 
                                                                       method='loop')

        elif i == 2: # Gradient x update
            event_type.append('GX')
            event_id.append(base_events[2])
            event_factor.append(num_shots)
            name, unit, strength, step =  update_gradient(base_block.gx, block.gx, trid, method='loop')
            

            
        elif i == 3: # Gradient y update
            name, unit, strength, step =  update_gradient(base_block.gy, block.gy, trid, method='loop')
            event_type.append('GY')
            event_id.append(base_events[3])
            
        elif i == 4: # Gradient z update
            name, unit, strength, step =  update_gradient(base_block.gz, block.gz, trid, method='loop')
            event_type.append('GZ')
            event_id.append(base_events[4])

            
        elif i == 5: # ADC update
            name, unit, strength, step =  update_ADC(base_block.adc, block.adc, trid, method='loop')
            event_type.append('ADC')
            event_id.append(base_events[5])
            
        event_factor.append(num_shots)
        event_attribute_name.append(name)
        event_attribute_unit.append(unit)
        event_strength.append(strength)
        event_step.append(step)   
              
    return  event_type, event_id, event_attribute_name, event_attribute_unit, event_strength, event_step, event_factor
        
    
    
    
def update_RF(base_rf, rf, trid, method='loop'):
    update_rf = copy.copy(base_rf)
    
    update_rf.dead_time = 0
    update_rf.delay = 0
    update_rf.freq_offset = 0
    update_rf.phase_offset = 0
    update_rf.ringdown_time = 0
    update_rf.signal = 0
    update_rf.t = 0
    update_rf.shape_dur = 0
    
    name = []
    unit = []
    strength = []
    step = []

    if base_rf.freq_offset != rf.freq_offset:
        name.append('freq_offset')
        unit.append('Hz')
        strength.append(base_rf.freq_offset)
        if method == 'loop':
            step_size = (rf.freq_offset - base_rf.freq_offset) / trid
            step.append(step_size)
        else:
             update_rf.freq_offset = rf.freq_offset - base_rf.freq_offset

    if base_rf.phase_offset != rf.phase_offset:
        name.append('phase_offset')
        unit.append('rad')
        strength.append(base_rf.phase_offset)
        if method == 'loop':
            if trid == 1:
                phase_offset = (rf.phase_offset - base_rf.phase_offset) 
                phase_offset = np.mod(phase_offset, 2 * np.pi)
                step.append(phase_offset)
            else:
                step.append(np.nan)
        else:
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
        
    return name, unit, strength, step
        
        
def update_gradient(base_grad, grad, trid, method='loop'):
    update_gradient = copy.copy(base_grad)
    gammabar = 42.576e6
    name = []
    unit = []
    strength = []
    step = []
    
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
                if method == 'loop':
                    name.append('amplitude')
                    unit.append('mT/m')  # pp default is Hz/m - but need to convert here
                    strength.append(np.round(base_grad.amplitude * 1000 / gammabar, decimals=3))
                    step_size = (grad.amplitude - base_grad.amplitude) / trid
                    step.append(np.round(step_size * 1000 / gammabar, decimals=3))
                else:
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
            
    return name, unit, strength, step

def update_ADC(base_adc, adc, trid, method='loop'):
    if adc is not None:
        update_ADC = copy.copy(base_adc)
        update_ADC.delay = 0
        update_ADC.dwell = 0
        update_ADC.freq_offset = 0
        update_ADC.num_samples = 0
        update_ADC.phase_offset = 0
        
        name = []
        unit = []
        strength = []
        step = []
        
        if base_adc.delay != adc.delay:
            update_ADC.delay = adc.delay - base_adc.delay
        if base_adc.dwell != adc.dwell:
            update_ADC.dwell = adc.dwell - base_adc.dwell
        if base_adc.freq_offset != adc.freq_offset:
            update_ADC.freq_offset = adc.freq_offset - base_adc.freq_offset
        if base_adc.num_samples != adc.num_samples:
            update_ADC.num_samples = adc.num_samples - base_adc.num_samples
        if base_adc.phase_offset != adc.phase_offset:
            if trid == 1:
                strength.append(base_adc.phase_offset)
                name.append('phase_offset')
                unit.append('rad')
                phase_offset = (adc.phase_offset - base_adc.phase_offset) 
                phase_offset = np.mod(phase_offset, 2 * np.pi)
                step.append(phase_offset)
            else:
                step.append(np.nan)
            # update_ADC.phase_offset = adc.phase_offset - base_adc.phase_offset
    else:
        update_ADC = None # some dummy scan blocks do not have ADCs  
    return name, unit, strength, step