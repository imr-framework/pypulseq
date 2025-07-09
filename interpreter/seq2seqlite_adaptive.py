# This file converts a .seq file to a .seq-lite file using the seq2seqlite library.
# The .seq-lite file is a simplified version of the .seq file that can be used for adaptive sequence design.
# Rule-set:
# 1. For SQ membership, only a new series of events initiate a new SQ object.
# 2. For SQ membership, some unique rules may apply such as those for EPI readout or TSE readout
# Available SQ objects: 10
# Available GR objects: 24
# Available RF objects: 24
# Available ADC objects: 10

# Logic of implementation:
# Keep adding events to a SQ object until it repeats.
# Identify RF, GR and ADC object attributes that changes over loop counters
# Write the seqlite file as a hierarchical structure from SOR, SQ, GR, ADC, RF objects

import numpy as np
import pypulseq as pp
from colorama import Fore, Style

class seqlite:
    def __init__(self, seq_file):
        # Version information
        self.version_major = 0.0
        self.version_minor = 0.0
        self.version_revision = 0.0
        self.verbose = True  # Set to True for verbose output
        
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
        
        # Implement the logic in init itself
        self.blocks_to_SQs()
        # self.TRs2SQs()
        # self.SQs2SOR()
        # self.identify_base_TR() # self.base_tr, self.base_tr_blocks, self.base_tr_events
        # self.SQ_TR_base, self.SOR_TR_base, self.SQ_base_event_list = self.TR2SQ(self.base_tr_blocks, self.base_tr_events) # self.SQ, self.SOR
        # self.SQ_label = self.label_SQ(self.SOR_TR_base, self.SQ_TR_base) # self.SQ_label
        # # self.generate_tr_waveforms() # Generate waveforms of SQ base [and xbase]
        # self.get_TR_updates() # Compare each TR and figure out the differences between the SQ objects - SOR remains the same
 

    def blocks_to_SQs(self):
        # Group blocks into TRs and assign memberships based on SQ label - One TR can have multiple SQ objects
       
        self.SQs_blocks = []  # List to hold SQ objects
        self.SQ_repeats = []  # List to hold SQ repeats
        self.SQ_block_events = []  # List to hold block events for SQs
        self.SOR = []  # List to hold SOR objects
        repeats=[]
        for i in range(1, self.num_blocks +1):
            block = self.seq.get_block(i)
            block_event = self.seq.block_events[i]
            if block.label is not None:
                if 'SQID' in block.label[0].label:  
                    SQ_ind = int(block.label[0].value) - 1 # Extract SQID from the label
                    self.SOR.append(SQ_ind)  # Append SQ index to SOR list
                    if SQ_ind > len(self.SQs_blocks) - 1:
                        self.SQs_blocks.append([[]])  # Create a new SQ as a 2D list (for repeats and blocks)
                        self.SQ_repeats.append([])  # Initialize repeat count for the new SQ
                        self.SQ_block_events.append([[]])  # Initialize block events for the new SQ as a 2D list
                        self.SQ_repeats[SQ_ind] = 0
                    else:
                       self.SQ_repeats[SQ_ind] += 1
                    repeats = self.SQ_repeats[SQ_ind]

                    # Check if the second dimension (repeat) exists, if not, add a new repeat list
                    while len(self.SQs_blocks[SQ_ind]) <= repeats:
                        self.SQs_blocks[SQ_ind].append([])
                        self.SQ_block_events[SQ_ind].append([])
                    self.SQs_blocks[SQ_ind][repeats].append(block)
                    self.SQ_block_events[SQ_ind][repeats].append(block_event)
            else:
                if repeats is None:
                    print('Block {} does not have a TRID label'.format(i))
                    print('Please add a TRID label to the first block in and out of each for loop')
                self.SQs_blocks[SQ_ind][repeats].append(block)
                self.SQ_block_events[SQ_ind][repeats].append(block_event)

        if self.verbose:
            print(Fore.GREEN + 'Blocks grouped into SQs successfully' + Style.RESET_ALL)
            print(Fore.YELLOW + 'Total SQs found: {}'.format(len(self.SQs_blocks)) + Style.RESET_ALL)
            for i, sq in enumerate(self.SQs_blocks):
                num_repeats = len(sq)
                if num_repeats > 0:
                    num_blocks_per_repeat = len(sq[0])
                else:
                    num_blocks_per_repeat = 0
                print(Fore.CYAN + f'SQ {i + 1}: {num_blocks_per_repeat} blocks per repeat, {num_repeats} repeats' + Style.RESET_ALL)
                
            print(Fore.MAGENTA + 'First 20 SOR items: {}'.format(self.SOR[:20]) + Style.RESET_ALL)

            # Check total number of blocks by summing all SQ blocks * repeats
            total_blocks_counted = 0
            for sq in self.SQs_blocks:
                for repeat in sq:
                    total_blocks_counted += len(repeat)
            print(Fore.BLUE + f'Counted total blocks in all SQs: {total_blocks_counted}' + Style.RESET_ALL)
            if total_blocks_counted == self.num_blocks:
                print(Fore.GREEN + 'Block count matches self.num_blocks.' + Style.RESET_ALL)
            else:
                print(Fore.RED + f'Block count mismatch! self.num_blocks={self.num_blocks}, counted={total_blocks_counted}' + Style.RESET_ALL)


        


    def TRs2SQs(self):
        # Implementation for converting TRs to SQs goes here
        pass

    def SQs2SOR(self):
        # Implementation for converting SQs to SOR goes here
        pass

    def adjust_delay_blocks(self):
        # Implementation for adjusting delay blocks goes here
        pass
    
    def get_updates_SQ(self):
        # Implementation for getting updates to SQs goes here
        pass

    def get_GR_attributes(self):
        # Implementation for getting GR attributes goes here
        pass

    def get_RF_attributes(self):
        # Implementation for getting RF attributes goes here
        pass

    def get_ADC_attributes(self):
        # Implementation for getting ADC attributes goes here
        pass

    def write_seqlite(self, output_file):
        # Implementation for writing the seqlite file goes here
        pass