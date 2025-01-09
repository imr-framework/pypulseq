# import the libraries
import numpy as np
import pypulseq as pp
from util_seq2philips import *
from colorama import Fore, Style


# Read the sequence file
class PhilipsTranslator():
    
# ======
# READ SEQ FILE 
# ======
    def __init__(self, seq_file):
        self.seq = pp.Sequence()
        self.seq.read(seq_file)
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
                if 'TRID' in block.label[0].label:      # Think if this is a hardcoding issue to access the first label
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
                    
                    # get event wise update and store it in a new set of blocks in SQ_updates_blocks
                    SQ_updates = get_event_updates(SQ_base_block, SQ_block, events_different_indices)
                    tr_updates.append(SQ_updates)
                
            else:
                print(Fore.GREEN + 'No differences found between TR {} and the base TR'.format(trid + 1) + Style.RESET_ALL)
                tr_updates.append(0)
        
        self.tr_updates = tr_updates
        print(Fore.GREEN + 'TR updates generated' + Style.RESET_ALL)
# ======
# WRITE THE PHILIPS DESCRIPTION FILE - JSON?
# ======