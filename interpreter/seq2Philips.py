# import the libraries
import numpy as np
import pypulseq as pp
from colorama import Fore, Style


# Read the sequence file
class PhilipsInterpreter():
# ======
# READ SEQ FILE 
# ======
    def __init__(self, seq_file):
        self.seq = pp.Sequence()
        self.seq.read(seq_file)
        self.num_blocks = len(self.seq.block_events)
        print(Fore.GREEN + 'Sequence read successfully' + Style.RESET_ALL)
        print(Fore.YELLOW + 'The sequence has a total of {} blocks'.format(self.num_blocks) + Style.RESET_ALL)
        self.group_blocks_into_TRs()
        self.identify_base_TR()
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
# IDENTIFY THE BASE TR 
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
            
# ======
# View Base TR
# ======

    def view_base_TR(self):
        if self.base_tr is None:
            print(Fore.RED + 'Base TR not identified, aborting' + Style.RESET_ALL)
        else:
            print(Fore.YELLOW + 'Displaying Base TR blocks...' + Style.RESET_ALL)
            base_tr_blocks = self.tr_blocks[self.base_tr]
            concatenated_blocks = pp.Sequence()
            for block in base_tr_blocks:
                concatenated_blocks.add_block(block)
            concatenated_blocks.plot()

# ======
# CONVERT THE BASE TR BLOCKS TO PHILIPS VARIABLES
# ======


# ======
# DESCRIBE THE OTHER TR BLOCKS IN RELATION TO BASE BLOCK
# ======