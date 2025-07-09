# import the libraries
import numpy as np
import pypulseq as pp
from util_seq2lite import *
from colorama import Fore, Style
from pathlib import Path
np.set_printoptions(legacy='1.25')

# ======
# TODO section here
# 1. Include defintions for repetitions: Shots, Slices, Averages, Echoes, Dynamics, Cardiac Phases, Respiratory Phases
# 2. Include definitions for the RF, GR, ADC events


# Read the sequence file
class seqlite():
    
# ======
# READ SEQ FILE 
# ======
    def __init__(self, seq_file):
        
        
        
        # Version information
        self.version_major = 0.0
        self.version_minor = 0.0
        self.version_revision = 0.0
        
        # Definitions
        self.definitions = {}
        # Sequence information
        self.seq = pp.Sequence()
        self.seq.read(seq_file, detect_rf_use=True)
        self.definitions = self.seq.definitions
        self.system = self.seq.system
        self.num_blocks = len(self.seq.block_events)
        print(Fore.GREEN + 'Sequence read successfully' + Style.RESET_ALL)
        print(Fore.YELLOW + 'The sequence has a total of {} blocks'.format(self.num_blocks) + Style.RESET_ALL)
        # Fix the delay blocks 
        self.group_blocks_into_TRs()
        self.identify_base_TR() # self.base_tr, self.base_tr_blocks, self.base_tr_events
        self.SQ_TR_base, self.SOR_TR_base, self.SQ_base_event_list = self.TR2SQ(self.base_tr_blocks, self.base_tr_events) # self.SQ, self.SOR
        self.SQ_label = self.label_SQ(self.SOR_TR_base, self.SQ_TR_base) # self.SQ_label
        # self.generate_tr_waveforms() # Generate waveforms of SQ base [and xbase]
        self.get_TR_updates() # Compare each TR and figure out the differences between the SQ objects - SOR remains the same
 
       
        
# ======
# GROUP BLOCKS INTO TRs
# ======

    def group_blocks_into_TRs(self):
        # Group blocks into TRs based on TRID label
        tr_blocks = {}
        tr_block_events = {}
        trid = -1
        for i in range(1, self.num_blocks +1):
            block = self.seq.get_block(i)
            block_event = self.seq.block_events[i]
            if block.label is not None:
                if 'SQID' in block.label[0].label:      # Think if this is a hardcoding issue to access the first label
                    trid += 1
                    if trid not in tr_blocks:
                        tr_blocks[trid] = []
                        tr_block_events[trid] = []
                    tr_blocks[trid].append(block)
                    tr_block_events[trid].append(block_event)
            else:
                if trid is None:
                    print('Block {} does not have a TRID label'.format(i))
                    print('Please add a TRID label to the first block of each TR')
                tr_blocks[trid].append(block)
                tr_block_events[trid].append(block_event)
                
        print(Fore.YELLOW + 'Identified a total of {} TRs'.format(len(tr_blocks)) + Style.RESET_ALL)
        if len(tr_blocks) != (trid +1):
            print(Fore.RED + 'The total number of {} TRs does not match the TRID labels {}, aborting'.format(len(tr_blocks), trid + 1) + Style.RESET_ALL)

        self.tr_blocks = tr_blocks
        self.tr_block_events = tr_block_events

# ======
# IDENTIFY THE BASE TR - SQ OBJECT
# ======

    def identify_base_TR(self):
        # Identify the base TR
        num_type_events_TR_max = 0
        num_unique_events_max = 0 # needs to be the maximum number of unique events, not just events
        
        base_tr = None
        for trid, blocks in self.tr_blocks.items():
            num_type_events = [0, 0, 0, 0, 0, 0]
            num_shape_events = [0, 0, 0, 0, 0, 0]
            
            block_durations = []
            rf_ids =[]
            gx_ids = []
            gy_ids = []
            gz_ids = []
            adc_ids = []
            block_ind = -1
            for block in blocks: 
                block_ind += 1
                
                if block.block_duration is not None:
                    num_type_events[0] = 1
                    block_duration = block.block_duration
                    
                    if block_durations is None:
                        block_durations.append(block_duration)
                        num_shape_events[0] += 1
                    else:
                        if block_duration not in block_durations:
                            num_shape_events[0] += 1
                    
                if block.rf is not None:
                    num_type_events[1] = 1
                    rf_id = self.tr_block_events[trid][block_ind][1]
                    if rf_id is None:
                        rf_ids.append(rf_id)
                        num_shape_events[1] += 1
                    else:
                        if rf_id not in rf_ids:
                            rf_ids.append(rf_id)
                            num_shape_events[1] += 1
                            
                            
                if block.gx is not None:
                    num_type_events[2] = 1
                    gx_id = self.tr_block_events[trid][block_ind][2]
                    if gx_id is None:
                        gx_ids.append(gx_id)
                        num_shape_events[2] += 1
                    else:
                        if gx_id not in gx_ids:
                            gx_ids.append(gx_id)
                            num_shape_events[2] += 1
                            
                if block.gy is not None:
                    num_type_events[3] = 1
                    gy_id = self.tr_block_events[trid][block_ind][3]
                    if gy_id is None:
                        gy_ids.append(gy_id)
                        num_shape_events[3] += 1
                    else:
                        if gy_id not in gy_ids:
                            gy_ids.append(gy_id)
                            num_shape_events[3] += 1
                            
                if block.gz is not None:
                    num_type_events[4] = 1
                    gz_id = self.tr_block_events[trid][block_ind][4]
                    if gz_id is None:
                        gz_ids.append(gz_id)
                        num_shape_events[4] += 1
                    else:
                        if gz_id not in gz_ids:
                            gz_ids.append(gz_id)
                            num_shape_events[4] += 1
                            
                if block.adc is not None:
                    num_type_events[5] = 1
                    adc_id = self.tr_block_events[trid][block_ind][5]
                    if adc_id is None:
                        adc_ids.append(adc_id)
                        num_shape_events[5] += 1
                    else:
                        if adc_id not in adc_ids:
                            adc_ids.append(adc_id)
                            num_shape_events[5] += 1
                            
            num_type_events_TR = np.sum(num_type_events)
            num_unique_events_TR = np.sum(num_shape_events)
            
            if num_type_events_TR  > num_type_events_TR_max:
                num_type_events_TR_max = num_type_events_TR
                base_tr = trid # because TRID starts from 1 - This might get confusing - TRID = trid - 1
                if num_unique_events_max == 0:
                    num_unique_events_max = num_unique_events_TR
                    print(Fore.YELLOW + 'Using first TR as base with unique events {}'.format(num_unique_events_max) + Style.RESET_ALL) # to line up with the .seq text value
                    
                    
            elif num_type_events_TR == num_type_events_TR_max:
                if num_unique_events_TR > num_unique_events_max:
                    num_unique_events_max = num_unique_events_TR
                    base_tr = trid
                    print(Fore.YELLOW + 'Identified a new base TR with more unique events {}'.format(num_unique_events_max) + Style.RESET_ALL) # to line up with the .seq text value
                    
        if base_tr is None:
            print(Fore.RED + 'Could not identify the base TR, aborting' + Style.RESET_ALL)
        else:
            print(Fore.YELLOW + 'Identified the base TR as TRID {}'.format(base_tr + 1) + Style.RESET_ALL) # to line up with the .seq text value
            self.base_tr = base_tr
            self.base_tr_blocks = self.tr_blocks[self.base_tr]
            self.base_tr_events = self.tr_block_events[self.base_tr]
            self.base_tr_blocks, self.base_tr_events = self.adjust_delay_blocks(self.base_tr_blocks, self.base_tr_events)
            

            
# ======
# View Base TR
# ======

    def view_base_TR(self, mode='blocks'):
        if mode == 'blocks':
            if self.base_tr_blocks is None:
                print(Fore.RED + 'Base TR not identified, aborting' + Style.RESET_ALL)
            else:
                print(Fore.YELLOW + 'Displaying Base TR blocks...' + Style.RESET_ALL)
                concatenated_blocks = pp.Sequence()
                for block in self.base_tr_blocks:
                    concatenated_blocks.add_block(block)
                concatenated_blocks.plot()

        elif mode == 'SOR':
            if self.SOR_TR is None:
                print(Fore.RED + 'SOR not identified, aborting' + Style.RESET_ALL)
            else:
                concatenated_blocks = pp.Sequence()
                for block_num in range(len(self.SOR_TR)):
                    block_num_SQ = self.SOR_TR[block_num]
                    block = self.SQ_TR_base.get_block(block_num_SQ + 1) # count starts from 1 in the .seq file
                    concatenated_blocks.add_block(block)
                concatenated_blocks.plot()
                    
                    
            pass

    def adjust_delay_blocks(self, tr_blocks = None, tr_events = None):  # This is something that is best handled in the .seq file
        
        # Adjust the delays to be encapsulated in the next block
        
        print(Fore.YELLOW + 'Adjusting delays...' + Style.RESET_ALL)
        print(Fore.YELLOW  + 'Number of blocks in base TR before adjustment: {}'.format(len(tr_blocks)) + Style.RESET_ALL)
        
        for block in tr_blocks:
            if block.block_duration is not None and block.rf is None and block.gx is None and block.gy is None and block.gz is None and block.adc is None:
                print(Fore.BLUE + 'Block with only block duration found: {}'.format(block.block_duration) + Style.RESET_ALL)
                delay = block.block_duration
                previous_block = tr_blocks[tr_blocks.index(block) - 1]
                    
                previous_block.block_duration += delay
                
                tr_blocks[tr_blocks.index(block) -1] = previous_block
                tr_events.pop(tr_blocks.index(block))
                tr_blocks.remove(block)
                
                    
        print(Fore.YELLOW + 'Number of blocks in base TR after adjustment: {}'.format(len(tr_blocks)) + Style.RESET_ALL)


        return tr_blocks, tr_events

        
    def TR2SQ(self, tr_blocks = None, tr_events = None):
        
        print(Fore.YELLOW + 'Converting the base TR to a SQ object...' + Style.RESET_ALL)
        print(Fore.YELLOW + 'Number of blocks in base TR: {}'.format(len(tr_blocks)) + Style.RESET_ALL)
        # The base TR might also have repeated blocks - represents a multi-echo sequence
        
        SQ = pp.Sequence()
        SQ_event_list = []
        SOR_TR = []
        SQ_index = -1
        for block_num in range(len(tr_events)):
            event_list = tr_events[block_num]
            # is_in_list = np.any(np.all(event_list == event_list_SQ, axis=1))
            if list(event_list) not in SQ_event_list:
                SQ_index += 1
                SQ_event_list.append(list(event_list))
                SQ.add_block(tr_blocks[block_num])
                SOR_TR.append(SQ_index)
            else:
                repeating_index = SQ_event_list.index(list(event_list))
                SOR_TR.append(repeating_index)
                
        print(Fore.YELLOW + 'Number of blocks in SQ object: {}'.format(len(SQ.block_events)) + Style.RESET_ALL)
        print(Fore.YELLOW + 'SOR list for base TR is: {}'.format((SOR_TR)) + Style.RESET_ALL)
        print(Fore.GREEN + 'SQ objects created' + Style.RESET_ALL)
        print(Fore.GREEN + 'SOR list created' + Style.RESET_ALL)

        
        return SQ, SOR_TR, SQ_event_list
        
    
    def label_SQ(self, SOR_TR = None, SQ = None):
        # adc_blocks = [block for block in self.SQ.blocks if block.adc is not None]
        num_adc_blocks = 0
        tr_blocks = pp.Sequence()
        SQ_label = []
        for block_num in range(len(SOR_TR)):
            block_num_SQ = SOR_TR[block_num]
            block = SQ.get_block(block_num_SQ + 1) # count starts from 1 in the .seq file
            tr_blocks.add_block(block)
        
        num_blocks = len(tr_blocks.block_events)
        for block_num in range(num_blocks):
            block = tr_blocks.get_block(block_num + 1)
            if block.adc is not None:
                num_adc_blocks += 1
                
        if num_adc_blocks <= 1:
            print(Fore.YELLOW + 'Single ADC block found in SQ, setting SQ_base' + Style.RESET_ALL)
            SQ_base = tr_blocks
            print(Fore.YELLOW + 'Number of blocks in SQ_base: {}'.format(len(SQ_base.block_events)) + Style.RESET_ALL)
        elif num_adc_blocks > 1:
            print(Fore.YELLOW + 'Multiple ADC blocks found in SQ, multi-echo sequence, setting SQ_base and SQ_xbase' + Style.RESET_ALL)

        xbase_flag = False
        adc_events = 0
        for block_num in range(num_blocks):
            block = tr_blocks.get_block(block_num + 1)
            if block.adc is not None:
                adc_events += 1
                if block_num < num_blocks - 1:
                    next_block = tr_blocks.get_block(block_num + 2)
                    if next_block.adc is not None: # if the next block is also an adc block then we are in the xbase
                        adc_events += 1
                        xbase_flag = True
                        
            if block_num > 1:
                previous_block = tr_blocks.get_block(block_num - 1)  
                if previous_block.adc is not None: # ADC, gradient and then the next block
                    if previous_block.gx is not None or previous_block.gy is not None or previous_block.gz is not None:
                        if (previous_block.gx is not None and previous_block.gx.type == 'grad') or \
                            (previous_block.gy is not None and previous_block.gy.type == 'grad') or \
                            (previous_block.gz is not None and previous_block.gz.type == 'grad'):
                            xbase_flag = False
                        else:
                            xbase_flag = True
                        
            if xbase_flag:
                SQ_label.append('XBASE')      
            else:
                SQ_label.append('BASE')
                
        self.SQ_label = SQ_label
        if len(SQ_label) != len(SOR_TR):
            print(Fore.RED + 'SQ labels are not aligned with SOR - this is an issue!' + Style.RESET_ALL)

        base_count = SQ_label.count('BASE')
        xbase_count = SQ_label.count('XBASE')
        print(Fore.YELLOW + 'Number of BASE instances: {}'.format(base_count) + Style.RESET_ALL)
        print(Fore.YELLOW + 'Number of XBASE instances: {}'.format(xbase_count) + Style.RESET_ALL)

    
        return SQ_label
            

    
# ======
# DESCRIBE THE OTHER TR BLOCKS IN RELATION TO BASE BLOCK
# ======

    def get_TR_updates(self, debug = False):
        # for each TR, check what is different from the base TR
        # self.SQ_TR_base, self.SOR_TR_base, self.SQ_base_event_list
        tr_updates = []
        num_shots = len(self.tr_blocks)
        for trid, tr_blocks in self.tr_blocks.items(): # for each TR indexed by trid
            if trid != self.base_tr:

                tr_blocks, tr_block_events = self.adjust_delay_blocks(tr_blocks, self.tr_block_events[trid])
                SQ, SOR_TR, SQ_event_list = self.TR2SQ(tr_blocks, tr_block_events)
                
                # Compare SORs first
                if len(SOR_TR) != len(self.SOR_TR_base):
                    print(Fore.RED + 'SOR TRs are not aligned, aborting' + Style.RESET_ALL)
                
                # Compare SQ_event_list s next to identify event changes
                for events in SQ_event_list:
                    base_events = self.SQ_base_event_list[SQ_event_list.index(events)]
                    if debug is True:
                         print(events)
                         print(base_events)
 
                    events_different = [event != base_event for event, base_event in zip(events, base_events)]
                    events_different_indices = [index for index, value in enumerate(events_different) if value]
                    
                    if debug is True:
                        print(Fore.YELLOW + 'Differences found at indices: {}'.format(events_different_indices) + Style.RESET_ALL)
                
                    SQ_base_block = self.SQ_TR_base.get_block(SQ_event_list.index(events) + 1)
                    SQ_block = SQ.get_block(SQ_event_list.index(events) + 1)
                    if trid == 1: # to avoid duplicates filtering later on - MRF?
                        # get event wise update and store it in a new set of blocks in SQ_updates_blocks
                        type, id, name, unit, strength, step, factor = get_event_updates(SQ_base_block, SQ_block, events_different_indices, 
                                                    trid, base_events, num_shots)
                        tr_updates.append([type, id, name, unit, strength, step, factor])
                        print(Fore.YELLOW + 'TR {} updates: {}'.format(trid + 1, [id, type, name, unit, strength, step, factor]) + Style.RESET_ALL)
                        
                
            else:
                print(Fore.GREEN + 'No differences found between TR {} and the base TR'.format(trid + 1) + Style.RESET_ALL)
                tr_updates.append(0)
        
        self.tr_updates = tr_updates
        print(Fore.GREEN + 'TR updates generated' + Style.RESET_ALL)
        

# ======
# WRITE THE lite DESCRIPTION FILE - JSON?
# ======
    def write_seqlite(self, file_name):
        
        file_name = Path(file_name)
        if file_name.suffix != '.seq-lite':
        # Append .lite-seq suffix
            file_name = file_name.with_suffix(file_name.suffix + '.seq-lite')
            
        # If removing duplicates, make a copy of the sequence with the duplicate
        # # events removed.
        # if remove_duplicates:
        #     self = self.remove_duplicates()
        
        # Define headers including computed values from .seq file and other definitions
        with open(file_name, 'w') as output_file:
            output_file.write('# Pulseq lite sequence file\n')
            output_file.write('# Created by PyPulseq\n\n')

            output_file.write('[VERSION]\n')
            output_file.write(f'major {self.version_major}\n')
            output_file.write(f'minor {self.version_minor}\n')
            output_file.write(f'revision {self.version_revision}\n')
            output_file.write('\n')

            if len(self.definitions) != 0:
                output_file.write('[DEFINITIONS]\n')
                keys = sorted(self.definitions.keys())
                values = [self.definitions[k] for k in keys]
                for block_counter in range(len(keys)):
                    output_file.write(f'{keys[block_counter]} ')
                    if isinstance(values[block_counter], str):
                        output_file.write(values[block_counter] + ' ')
                    elif isinstance(values[block_counter], (int, float)):
                        output_file.write(f'{values[block_counter]:0.9g} ')
                    elif isinstance(values[block_counter], (list, tuple, np.ndarray)):  # e.g. [FOV_x, FOV_y, FOV_z]
                        for i in range(len(values[block_counter])):
                            if isinstance(values[block_counter][i], (int, float)):
                                output_file.write(f'{values[block_counter][i]:0.9g} ')
                            else:
                                output_file.write(f'{values[block_counter][i]} ')
                    else:
                        raise RuntimeError('Unsupported definition')
                    output_file.write('\n')
                output_file.write('\n')
                
            # Zero/First level combined : SOR/SQ
            # Each line represents one SQ object in order of the SOR
            # #SQ, #RF, #GR, #AQ, RF_FE, AQ_FE
            output_file.write('# Format of first level hierarchy - BLOCKS:\n')
            output_file.write('SQ-NAME id RF RF_FE GX GY GZ AQ AQ_FE\n') # need to reclarify with Ganji on FE
            output_file.write('[BLOCKS]\n')
            
            rf_id = 0
            gr_id = 0
            aq_id = 0
            
            gr_wf = []
            rf_wf = []
            adc_wf = []
            
            for block in self.base_tr_blocks:
                # Each line represents one block
                # #, Set Attributes - ID, Update attributes - ID, Read attributes ID
                id = self.base_tr_blocks.index(block) 
                sq_name = self.SQ_label[id]
                output_file.write(f'{sq_name}\t {id + 1}\t') # seq block count starts from 1
            
                if block.rf is not None:
                    rf_id  += 1
                    rf_wf.append(block.rf)
                    output_file.write(f'{rf_id}\t')
                    output_file.write(f'{1}\t') # turn on RF_FE because we have an RF event
                else:
                    output_file.write(f'{0}\t')
                    output_file.write(f'{0}\t') # turn off RF_FE because we DONT have an RF event
                    
                if block.gx is not None or block.gy is not None or block.gz is not None:
                    if block.gx is not None:
                        gr_id += 1
                        gr_wf.append(block.gx)
                        output_file.write(f'{gr_id}\t')
                    else:
                        output_file.write(f'{0}\t')
                        
                    if block.gy is not None:
                        gr_id += 1
                        gr_wf.append(block.gy)
                        output_file.write(f'{gr_id}\t')
                    else:
                        output_file.write(f'{0}\t')
                        
                    if block.gz is not None:
                        gr_id += 1
                        gr_wf.append(block.gz)
                        output_file.write(f'{gr_id}\t')
                    else:
                        output_file.write(f'{0}\t')
                        
                    
                    
                if block.adc is not None:
                    aq_id += 1
                    adc_wf.append(block.adc)
                    output_file.write(f'{aq_id}\t')
                    output_file.write(f'{1}\t') # turn on AQ_FE because we have an ADC event
                else:
                    output_file.write(f'{0}\t')
                    output_file.write(f'{0}\t') # turn off AQ_FE because we DONT have an ADC event
                
                output_file.write('\n')
            
            
            output_file.write('\n# Format of first level hierarchy - TR UPDATES:\n')
            output_file.write('# Value in a step = strength + (factor * step)\n')
            output_file.write('# Event_type event_id attribute_name attribute_unit strength step factor\n')
            output_file.write('[TR_UPDATES]\n')
            
            for update in range(1, len(self.tr_updates)): # trid starts from 1, so 0 is empty
                update_info = self.tr_updates[update]
                update_info_flat = [item for sublist in update_info for item in sublist]
                output_file.write('\t'.join(map(str, update_info_flat)) + '\n')
                # output_file.write(f'{str(update_info[0][0])}\t{update_info[1][0]}\t{str(update_info[2][0])}\t {update_info[3][0]}\t {str(update_info[4][0])}\t {str(update_info[5][0])}\t {update_info[6][0]}\n')

            
            
            # Second level -  RF
            # Each line represents one RF object in order of the RF objects in the SQ
            # #, Set Attributes - ID, Update attributes - ID, Read attributes ID
            output_file.write('\n# Format of second level hierarchy - RF:\n')
            output_file.write('# Format of RF events:\n')
            output_file.write('# id  flip delay freq phase use\n')
            output_file.write('# ..   deg    us   Hz   rad  excitation | refocusing | inversion\n')
            output_file.write('[RF]\n')
           
            for rf_event in rf_wf:
                rf_event = self.pp2lite_rf(rf_event)
                rf_event_id = rf_wf.index(rf_event) + 1
                output_file.write(f'{rf_event_id}\t{rf_event.FA}\t {rf_event.delay}\t{rf_event.freq_offset}\t{rf_event.phase_offset}\t {rf_event.use}\n')
        
        # Second level -  GR
        # Each line represents one GR object in order of the GR objects in the SQ
        # #, Set Attributes - ID, Update attributes - ID, Read attributes ID
            output_file.write('\n# Format of second level hierarchy - GR:\n')
            output_file.write('# Format of trapezoid gradients:\n')
            output_file.write('# id axis amplitude rise flat fall delay\n')
            output_file.write('# ..      mT   us   us   us    us\n')
            output_file.write('[TRAP]\n')
            
            for gr_event in gr_wf:
                gr_event = self.pp2lite_gr(gr_event)
                gr_event_id = gr_wf.index(gr_event) + 1
                output_file.write(f'{gr_event_id}\t{gr_event.channel}\t{gr_event.amplitude}\t{gr_event.rise_time}\t{gr_event.flat_time}\t{gr_event.fall_time}\t {gr_event.delay}\n')
            
        # Second level -  AQ
        # Each line represents one AQ object in order of the AQ objects in the SQ
        # #, Set Attributes - ID, Update attributes - ID, Read attributes ID
            output_file.write('\n# Format of second level hierarchy - AQ:\n')
            output_file.write('# Format of ADC events:\n')
            output_file.write('# id num dwell delay freq phase\n')
            output_file.write('# ..  ..    ns    us   Hz   rad\n')
            output_file.write('[ADC]\n')
            
            for adc_event in adc_wf:
                adc_event_id = adc_wf.index(adc_event) + 1
                adc_event = self.pp2lite_adc(adc_event)
                output_file.write(f'{adc_event_id}\t{adc_event.num_samples}\t{adc_event.dwell}\t{adc_event.delay}\t{adc_event.freq_offset}\t{adc_event.phase_offset}\n')
         
            # Third level -  RF - arbitrary waveforms 
            output_file.write('\n# Format of third level hierarchy - RF shapes: arbitrary:\n')  
            output_file.write(f'[SHAPES]\n')
            
            for rf_event in rf_wf:
                rf_event_id = rf_wf.index(rf_event) + 1
                output_file.write(f'\nshape id \t{rf_event_id}\n')
                output_file.write(f'num_samples\t{len(rf_event.t)}\n')
                output_file.write(f'time\tvalue\n')
                output_file.write(f'us\t a.u\n')
                
                for instant in range(len(rf_event.signal)):
                    output_file.write(f'{rf_event.t[instant]}\t {rf_event.signal_rescaled[instant]}\n')
         
            # Third level -  GR - arbitrary waveforms 
            # output_file.write('\n# Format of third level hierarchy - GR shapes: arbitrary:\n')  
            # output_file.write(f'[SHAPES]\n')
            
        
        output_file.close()
        pass
    
    def pp2lite_gr(self, gr):
        gammabar = 42.576e6
        # Convert a PyPulseq trapezoid gradient object to a lite gradient object
        gr.amplitude = np.round(gr.amplitude * 1e3 / gammabar, decimals=3) # convert to mT/m
        gr.rise_time = np.round(gr.rise_time * 1e6, decimals=0)
        gr.flat_time = np.round(gr.flat_time * 1e6, decimals=0)
        gr.fall_time = np.round(gr.fall_time * 1e6, decimals=0)
        gr.delay = np.round(gr.delay * 1e6, decimals=0)

        return gr
    
    def pp2lite_rf(self, rf):
        gammabar = 42.576e6
        # Convert a PyPulseq sinc pulse object to a lite RF object
        rf.delay = np.round(rf.delay * 1e6, decimals=0)
        rf.amplitude = np.abs(np.max(rf.signal))
        rf.signal_rescaled = rf.signal / rf.amplitude # we know that waveforms were designed for unit amplitude max.
        if np.imag(rf.signal_rescaled).all() == 0:
            rf.signal_rescaled = np.real(rf.signal_rescaled)
        flip = np.abs(np.sum(rf.signal_rescaled) * self.definitions['RadiofrequencyRasterTime'] * 2 * np.pi)
        rf.FA = np.int16(np.round(np.rad2deg(rf.amplitude * flip), decimals=0))
        rf.t = np.round(rf.t * 1e6, decimals=1) # us
        return rf
    
    def pp2lite_adc(self, adc):
        # Convert a PyPulseq ADC object to a lite ADC object
        adc.dwell = np.round(adc.dwell * 1e9, decimals=0)
        adc.delay = np.round(adc.delay * 1e6, decimals=0)
        return adc
    
    