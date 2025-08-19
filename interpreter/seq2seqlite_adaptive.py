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

# block_num starts from 1 in the .seq file, so we need to adjust it accordingly - it is 0 indexed in python and here

import numpy as np
import pypulseq as pp
from colorama import Fore, Style
from pathlib import Path
import types

class seqlite:
    def __init__(self, seq_file):
        # Version information
        self.version_major = 0.0
        self.version_minor = 0.0
        self.version_revision = 0.0
        self.verbose = True  # Set to True for verbose output
        self.visualize = False  # Set to True for visualization

        self.gammabar = 42.577478518e6  # Hz/T, default value for proton
        self.precision_digits = 2
        
        # Definitions
        self.definitions = {}
        # Sequence information
        self.seq = pp.Sequence()
        self.seq.read(seq_file, detect_rf_use=True)
        self.definitions = self.seq.definitions
        self.system = self.seq.system
        self.num_blocks = len(self.seq.block_events)
        self.SQ_repeat_attributes = []  # List to hold SQ attributes
        print(Fore.GREEN + 'Sequence read successfully' + Style.RESET_ALL)
        print(Fore.YELLOW + 'The sequence has a total of {} blocks'.format(self.num_blocks) + Style.RESET_ALL)
        
        # Implement the logic in init itself
        self.blocks_to_SQs()   # Group blocks into SQs and assign memberships based on SQ label
        self.get_SOR_kernel()  # Get SOR kernel after grouping blocks into SQs
        self.apply_special_rules()  # Apply special rules to SQs such as EPI or TSE readout
        self.get_SQ_details()  # Provides base SQ_attributes and SQ_repeat_attributes
        self.get_SQ_rt_attributes()  # Get real-time attributes for SQs - strength, step, factor - self.SQ_repeat_rt_attributes, self.SQ_rt_attributes

        if self.verbose:
            # Print summary of number of changes in SQs per SQ
            print(Fore.YELLOW + "Summary of number of RT updates in SQs per SQ:" + Style.RESET_ALL)
            for sq_idx, sq_attr_list in enumerate(self.SQ_attributes):
                if not sq_attr_list or not isinstance(sq_attr_list, list):
                    continue
                total_changes = 0
                for kernel_repeat in sq_attr_list:
                    if not kernel_repeat or not isinstance(kernel_repeat, list):
                        continue
                    for attr in kernel_repeat:
                        if isinstance(attr, dict) and 'changes' in attr:
                            total_changes += len(attr['changes'])
                print(Fore.CYAN + f"SQ {sq_idx}: {total_changes} RT updates" + Style.RESET_ALL)
       

    def blocks_to_SQs(self):
        # Group blocks into TRs and assign memberships based on SQ label - One TR can have multiple SQ objects
       
        self.SQs_blocks = []  # List to hold SQs objects
        self.SQ_repeats = []  # List to hold SQ repeats
        self.SQs_block_events = []  # List to hold block events for SQs
        self.SOR = []  # List to hold SOR objects
        self.SOR_SQ_repeats = []  # List to hold SOR repeats for each SQ
        repeats=[]
        SQ_ind_prev = []
        for i in range(1, self.num_blocks +1):
            block = self.seq.get_block(i)
            block_event = self.seq.block_events[i]
            block_event[0] = int(block.block_duration/self.seq.block_duration_raster)
            if block.label is not None:
                if 'SQID' in block.label[0].label:  
                    SQ_ind = int(block.label[0].value) - 1 # Extract SQID from the label

                    if (SQ_ind > len(self.SQs_blocks) - 1):
                        self.SQ_repeats.append([])  # Initialize repeat count for the new SQ
                        self.SQ_repeats[SQ_ind] = 0
                        self.SQs_blocks.append([[]])  # Create a new SQ as a 2D list (for repeats and blocks)
                        self.SQs_block_events.append([[]])  # Initialize block events for the new SQ as a 2D list
                    else:
                       self.SQ_repeats[SQ_ind] += 1
                    repeats = self.SQ_repeats[SQ_ind]
                    
                    # or (SQ_ind == SQ_ind_prev)
                    # Update SOR and SOR_SQ_repeats
                    if len(self.SOR) > 0 and self.SOR[-1] == SQ_ind:
                        self.SOR_SQ_repeats[-1] += 1
                    else:
                        self.SOR.append(SQ_ind)
                        self.SOR_SQ_repeats.append(1)


                    # Check if the second dimension (repeat) exists, if not, add a new repeat list
                    while len(self.SQs_blocks[SQ_ind]) <= repeats:
                        self.SQs_blocks[SQ_ind].append([])
                        self.SQs_block_events[SQ_ind].append([])
                    self.SQs_blocks[SQ_ind][repeats].append(block)
                    self.SQs_block_events[SQ_ind][repeats].append(block_event)
            else:
                if repeats is None:
                    print('Block {} does not have a SQID label'.format(i))
                    print('Please add a SQID label to the first block in and out of each for loop')
                self.SQs_blocks[SQ_ind][repeats].append(block)
                self.SQs_block_events[SQ_ind][repeats].append(block_event)

        if self.verbose:
            print(Fore.GREEN + 'Blocks grouped into SQs successfully' + Style.RESET_ALL)
            print(Fore.YELLOW + 'Total SQs found: {}'.format(len(self.SQs_blocks)) + Style.RESET_ALL)
            for i, sq in enumerate(self.SQs_blocks):
                num_repeats = len(sq)
                if num_repeats > 0:
                    num_blocks_per_repeat = len(sq[0])
                else:
                    num_blocks_per_repeat = 0
                print(Fore.CYAN + f'SQ {i}: {num_blocks_per_repeat} blocks per repeat, {num_repeats} repeats' + Style.RESET_ALL)
                
            print(Fore.MAGENTA + 'SOR (all): {}'.format(self.SOR) + Style.RESET_ALL)
            print(Fore.MAGENTA + 'SOR_SQ_repeats (all): {}'.format(self.SOR_SQ_repeats) + Style.RESET_ALL)

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
        
        # # Get SOR kernel before applying special rules
        # self.get_SOR_kernel()


        # # Apply special rules to SQs
        # self.apply_special_rules()


        
    def apply_special_rules(self): #SQs_blocks, SQ_block_events, SOR
    # Placeholder for the method that applies special rules to SQs, such as EPI or TSE readout
    # Rule 1: All delay blocks need to be adjusted to the next block

        print(Fore.YELLOW + 'Applying special rules to SQs...' + Style.RESET_ALL)
        self.SQs_blocks_compact = self.SQs_blocks.copy()  # Create a copy of SQs blocks for compacting
        self.SQs_block_events_compact = self.SQs_block_events.copy()  # Create a copy of SQs block events for compacting
        
        # Apply Rule 1: Adjust delay blocks
        # self.SQs_blocks_compact, self.SQs_block_events_compact = self.apply_special_rule1()
        # self.apply_special_rule2()  # Apply Rule 2: Merge identical repeating TRs
        # Apply Rule 2: Multi-echo rule :: example for EPI readout
        


    def apply_special_rule1(self):
    # Rule 1 implementation only impacts blocks and events, not SOR
        print(Fore.YELLOW + 'Rule 1: Encapsulating delay blocks in SQs...' + Style.RESET_ALL)
        SQs_blocks_compact = []
        SQs_block_events_compact = []

        # Iterate through each SQ and its repeats
        for sq_blocks, sq_events in zip(self.SQs_blocks, self.SQs_block_events):
            sq_blocks_compact = []
            sq_events_compact = []
            for repeat_blocks, repeat_events in zip(sq_blocks, sq_events):
                # Adjust delay blocks for each repeat
                new_blocks, new_events = self.adjust_delay_blocks(repeat_blocks.copy(), repeat_events.copy())
                sq_blocks_compact.append(new_blocks)
                sq_events_compact.append(new_events)
            SQs_blocks_compact.append(sq_blocks_compact)
            SQs_block_events_compact.append(sq_events_compact)


        print(Fore.GREEN + 'Rule 1 applied: Delay blocks adjusted.' + Style.RESET_ALL)
        # Print the number of blocks per SQ for all SQ objects after delay encapsulation
        print(Fore.CYAN + 'Number of blocks per SQ after delay encapsulation:' + Style.RESET_ALL)
        for i, sq in enumerate(SQs_blocks_compact):
            num_repeats = len(sq)
            if num_repeats > 0:
                num_blocks_per_repeat = len(sq[0])
            else:
                num_blocks_per_repeat = 0
            print(Fore.CYAN + f'SQ {i}: {num_blocks_per_repeat} blocks per repeat, {num_repeats} repeats' + Style.RESET_ALL)
        
        return SQs_blocks_compact, SQs_block_events_compact

    def apply_special_rule2(self):

        def merge_repeats(blocks, events, repeats_to_merge):
            # Merge the specified number of repeats into a single SQ object
            merged_blocks = []
            merged_events = []
            for num_events in range(len(events)):
                if len(blocks) < repeats_to_merge:
                    print(Fore.RED + f'Not enough repeats to merge: {len(blocks)} < {repeats_to_merge}' + Style.RESET_ALL)
            
            num_merges_needed = len(blocks) // repeats_to_merge
            if num_merges_needed == 0:
                return blocks, events
            
            # Merge the blocks and events
            for num_merge in range(num_merges_needed):
                start_idx = num_merge * repeats_to_merge
                end_idx = start_idx + repeats_to_merge
                if end_idx > len(blocks):
                    break

                merged_block = blocks[start_idx:end_idx]
                merged_event = events[start_idx:end_idx]
                merged_blocks.append(merged_block)
                merged_events.append(merged_event)


            return merged_blocks, merged_events

        # if there are identical repeating TRs within a SQ, merge those into a single SQ object
        # Rule 1 implementation only impacts blocks and events, not SOR
        print(Fore.YELLOW + 'Rule 2: Merging repeats of a particular SQ object...' + Style.RESET_ALL)


        # If a particular SQ repeats consecutively and contains RF or AQ objects, merge them into a single SQ and update the 
        # self.SOR_kernel_SQ_repeats, self.SQ_blocks_compact, self.SQ_block_events_compact
        if len(self.SOR_kernel) > 1:
            for SQ_idx in range(len(self.SOR_kernel_SQ_repeats)):
                if self.SOR_kernel_SQ_repeats[SQ_idx] > 1: # Check if the SQ repeats
                    repeats_to_merge = self.SOR_kernel_SQ_repeats[SQ_idx]
                    # merge the SQ blocks and events
                    merged_blocks, merged_events = merge_repeats(self.SQs_blocks_compact[SQ_idx], self.SQs_block_events_compact[SQ_idx], repeats_to_merge)
                    self.SQs_blocks_compact[SQ_idx] = merged_blocks
                    self.SQs_block_events_compact[SQ_idx] = merged_events
                    # Update the self.SOR_kernel_SQ_repeats
                    self.SOR_kernel_SQ_repeats[SQ_idx] = 1  # Set the repeat count to 1 for the merged SQ

                

        print(Fore.GREEN + 'Rule 1 applied: Delay blocks adjusted.' + Style.RESET_ALL)
        # Print the number of blocks per SQ for all SQ objects after merging repeats
        print(Fore.CYAN + 'Number of blocks per SQ after merging repeats:' + Style.RESET_ALL)
        for i, sq in enumerate(self.SQs_blocks_compact):
            num_repeats = len(sq)
            if num_repeats > 0:
                num_blocks_per_repeat = len(sq[0])
            else:
                num_blocks_per_repeat = 0
            print(Fore.CYAN + f'SQ {i}: {num_blocks_per_repeat} blocks per repeat, {num_repeats} repeats' + Style.RESET_ALL)

        

    def get_SOR_kernel(self):

        # Implementation for getting SOR kernel goes here
        # This will return the SOR object with all SQs and their repeats
        # Identify the shortest repeating pattern in self.SOR
        
        def find_repeating_pattern(SOR):
            pattern = []
            repeat_count = 1

            for i in range(len(SOR)):
                    if not pattern:
                        pattern.append(SOR[i])
                    elif SOR[i] == pattern[0]:
                        # Check if the next len(pattern) elements match the pattern
                        match = True
                        for j in range(len(pattern)):
                            if i + j >= len(SOR) or SOR[i + j] != pattern[j]:
                                if i + j >= len(self.SOR_SQ_repeats) or self.SOR_SQ_repeats[i + j] != self.SOR_SQ_repeats[i]:
                                    match = False
                                    break
                        if match:
                            # Count how many times the pattern repeats
                            repeat_count = 1
                            idx = i
                            pattern_SQ_repeats = self.SOR_SQ_repeats[:len(pattern)]
                            while idx + len(pattern) <= len(SOR) and SOR[idx:idx+len(pattern)] == pattern:
                                repeat_count += 1
                                idx += len(pattern)
                            return pattern, repeat_count, pattern_SQ_repeats
                        else:
                            pattern.append(SOR[i])
                    else:
                        pattern.append(SOR[i])
                # If no repeating pattern found, return the whole SOR as pattern
            pattern_SQ_repeats = self.SOR_SQ_repeats[:len(pattern)]
            return pattern, repeat_count, pattern_SQ_repeats

        self.SOR_kernel, self.SOR_kernel_repeats, self.SOR_kernel_SQ_repeats = find_repeating_pattern(self.SOR)


        if self.verbose:
            print(Fore.GREEN + f'SOR_kernel: {self.SOR_kernel}' + Style.RESET_ALL)
            print(Fore.GREEN + f'SOR_kernel_repeats: {self.SOR_kernel_repeats}' + Style.RESET_ALL)
            print(Fore.GREEN + f'SOR_kernel_SQ_repeats: {self.SOR_kernel_SQ_repeats}' + Style.RESET_ALL)

    def compare_attributes(self,attr1, attr2):
            # Compare two attributes and return a dictionary with differences
            if (attr1 == None or attr1.type == 'rf' or None) and attr2.type == 'rf':
                if attr1.type == None:
                    attr1.freq_offset = 0
                    attr1.flip_angle = 0
                    attr1.duration = 0
                    attr1.phase_offset = 0

                freq_offset = attr2.freq_offset - attr1.freq_offset
                phase_offset = attr2.phase_offset - attr1.phase_offset
                result = {}
                if freq_offset != 0:
                    result['freq_offset'] = freq_offset
                if phase_offset != 0:
                    result['phase_offset'] = phase_offset
                return result

            elif attr1.type == 'grad' and attr2.type == 'grad':
                diff = attr2.waveform - attr1.waveform
                if np.any(diff != 0):
                    return {'waveform': diff}
                else:
                    return None
            
            elif attr1.type == 'trap' and attr2.type == 'trap':
                result = {}
                if attr2.amplitude - attr1.amplitude != 0:
                    result['amplitude'] = attr2.amplitude - attr1.amplitude
                # if attr2.area - attr1.area != 0:
                #     result['area'] = attr2.area - attr1.area
                if attr2.delay - attr1.delay != 0:
                    result['delay'] = attr2.delay - attr1.delay
                return result if result else None

            elif (attr1.type == 'adc' or None or attr1 == None) and (attr2.type == 'adc' or attr2 == None):
                if attr2 == None:
                    # If attr2 is None, create a dummy object with required properties
                    class DummyADC:
                        def __init__(self):
                            self.type = 'adc'
                            self.num_samples = 0
                            self.phase_offset = 0
                    if attr1 is None:
                        attr1 = DummyADC()
                    if attr2 is None:
                        attr2 = DummyADC()
                    attr2 = DummyADC()
                if attr2.type == None:
                    attr2.num_samples = 0
                    attr2.phase_offset = 0
                elif attr1.type == None:
                    attr1.num_samples = 0
                    attr1.phase_offset = 0

                result = {}
                if attr2.num_samples - attr1.num_samples != 0:
                    result['num_samples'] = attr2.num_samples - attr1.num_samples
                if attr2.phase_offset - attr1.phase_offset != 0:
                    result['phase_offset'] = attr2.phase_offset - attr1.phase_offset
                return result if result else None

            elif attr1.type == 'label' and attr2.type == 'label':
                result = {}
                if attr1.text != attr2.text:
                    result['label'] = attr2.text - attr1.text
                return result if result else None
            elif attr1.type == 'ext' and attr2.type == 'ext':
                result = {}
                if attr1.text != attr2.text:
                    result['text'] = attr2.text - attr1.text
                return result if result else None
            else:
                return None

    def get_SQ_repeat_attributes(self, tr_blocks, tr_events, SQ_base, SQ_idx, repeat_idx):
        event_names = ['', 'RF',  'GX',  'GY',  'GZ',  'ADC',  'LABEL']
        attributes_changed = {}
        # Ensure the outer list is long enough
        while len(self.SQ_repeat_attributes) <= SQ_idx:
            self.SQ_repeat_attributes.append([])
        # Ensure the inner list is long enough
        while len(self.SQ_repeat_attributes[SQ_idx]) <= repeat_idx:
            self.SQ_repeat_attributes[SQ_idx].append({})
        self.SQ_repeat_attributes[SQ_idx][repeat_idx] = {'changes': [], 'repeat_idx': repeat_idx}
        SQ_base_events = SQ_base['events']
        SQ_base_blocks = SQ_base['blocks']
        for block_num in range(len(tr_events)):
            if not np.array_equal(tr_events[block_num], SQ_base_events[block_num]):
                events_different = [event != base_event for event, base_event in zip(tr_events[block_num], SQ_base_events[block_num])]
                events_different_indices = [
                    index
                    for index, value in enumerate(events_different)
                    if (np.any(value) if isinstance(value, np.ndarray) else value)
                ]
                # Record the block number, event type and the indices of the changed events
                for idx in events_different_indices:
                    events_change_name = event_names[idx]
                    events_change_id = int(tr_events[block_num][idx])

                    if idx == 1:  # RF event
                        events_change_attributes = self.compare_attributes(tr_blocks[block_num].rf, SQ_base_blocks[block_num].rf)
                    elif idx == 2:  # GX event
                        events_change_attributes = self.compare_attributes(tr_blocks[block_num].gx, SQ_base_blocks[block_num].gx)
                    elif idx == 3:  # GY event
                        events_change_attributes = self.compare_attributes(tr_blocks[block_num].gy, SQ_base_blocks[block_num].gy)
                    elif idx == 4:  # GZ event
                        events_change_attributes = self.compare_attributes(tr_blocks[block_num].gz, SQ_base_blocks[block_num].gz)
                    elif idx == 5:  # ADC event
                        class DummyADC_type:
                            def __init__(self):
                                self.type = 'adc'
                                self.num_samples = 0
                                self.phase_offset = 0
                        if tr_blocks[block_num].adc is None:
                            tr_blocks[block_num].adc = DummyADC_type()
                        if SQ_base_blocks[block_num].adc is None:
                            SQ_base_blocks[block_num].adc = DummyADC_type()
                        events_change_attributes = self.compare_attributes(tr_blocks[block_num].adc, SQ_base_blocks[block_num].adc)
                    elif idx == 6:  # LABEL event
                        events_change_attributes = self.compare_attributes(tr_blocks[block_num].label, SQ_base_blocks[block_num].label)
                    else:
                        events_change_attributes = None

                    if events_change_attributes is not None:
                        if events_change_name not in attributes_changed:
                            attributes_changed[events_change_name] = []
                        attributes_changed[events_change_name].append(events_change_attributes)
                    
                    self.SQ_repeat_attributes[SQ_idx][repeat_idx]['changes'].append({
                        'real_time': True,
                        'block_num': block_num,
                        'event_name': events_change_name,
                        'event_change_id': events_change_id,
                        'events_change_attributes': events_change_attributes,
                        'repeat_idx': repeat_idx
                    })
                        

    def get_SQ_details(self):
        # Implementation for getting details per SQ goes here - [blocks, events], # real time attributes,
        # Second level description of event_type, event_id, attribute_name, attribute_unit, strength, step, factor
        
        def get_SQ_attributes(TRs_blocks, TRs_events, SQ_base):
            # Implementation for getting SQ attributes goes here
            # This will return a dictionary with attributes for each SQ
            identical_prev = True  # Assume the previous TRs were identical
            SQ_attributes = {}
            identical = True  # Assume the current TRs are identical
            # first check if the events are identical, if they are then real-time attributes = 0
            # if they are part of a SQ, either all events are identical or something changes every block
            for TR_events in TRs_events:
                
                for block_num, (e1, e2) in enumerate(zip(TR_events, SQ_base['events'])):
                    if not np.array_equal(e1, e2):
                        identical = False
                    else:
                        identical = True
                    identical = identical and identical_prev  # Check if the current TR is identical to the previous one
                    identical_prev = identical  # Update the previous identical status

            if identical:
                SQ_attributes = {
                    'real_time': False,
                    'loop': len(TRs_events)
                }
            else:
                SQ_attributes = get_SQ_updates(TRs_blocks, TRs_events, SQ_base)
           
            return SQ_attributes
             # there are changes in the events, so we need to check for attributes
                

            return SQ_attributes

        def get_SQ_updates(TRs_blocks, TRs_events, SQ_base):
        # This function checks for attributes that have changed in the blocks and events
        # It returns a dictionary with attribute names as keys and their values as lists of changes
            attributes_changed = {}
            event_names = ['DUR', 'RF',  'GX',  'GY',  'GZ',  'ADC',  'LABEL']
            SQ_event_changes = {}

          
            for tr_idx, tr_events in enumerate(TRs_events):
                # Check if the events are identical to the base events
                if tr_idx == 0 and len(TRs_events) == 1:
                    TR_ref_blocks = SQ_base['blocks']
                    TR_ref_events = SQ_base['events']
                elif tr_idx == 0:
                    TR_ref_blocks = TRs_blocks[tr_idx]
                    TR_ref_events = TRs_events[tr_idx]
                elif tr_idx > 0: # tr_idx == 0 is SQ_base
                    TR_ref_blocks = TRs_blocks[tr_idx - 1]
                    TR_ref_events = TRs_events[tr_idx - 1]

                for block_num in range(len(tr_events)):
                    if not np.array_equal(tr_events[block_num], TR_ref_events[block_num]):
                        events_different = [event != base_event for event, base_event in zip(tr_events[block_num], TR_ref_events[block_num])]
                        events_different_indices = [index for index, value in enumerate(events_different) if bool(value)]
                        # Record the block number, event type and the indices of the changed events
                        for idx in events_different_indices:
                            events_change_name = event_names[idx]
                            events_change_id = int(tr_events[block_num][idx])

                            if idx == 1:  # RF event
                                events_change_attributes = self.compare_attributes(TR_ref_blocks[block_num].rf, TRs_blocks[tr_idx][block_num].rf)
                            elif idx == 2:  # GX event
                                events_change_attributes = self.compare_attributes(TR_ref_blocks[block_num].gx, TRs_blocks[tr_idx][block_num].gx)
                            elif idx == 3:  # GY event
                                events_change_attributes = self.compare_attributes(TR_ref_blocks[block_num].gy, TRs_blocks[tr_idx][block_num].gy)
                            elif idx == 4:  # GZ event
                                events_change_attributes = self.compare_attributes(TR_ref_blocks[block_num].gz, TRs_blocks[tr_idx][block_num].gz)
                            elif idx == 5:  # ADC event
                                    class DummyADC_type:
                                        def __init__(self):
                                            self.type = 'adc'
                                            self.num_samples = 0
                                            self.phase_offset = 0
                                    if TR_ref_blocks[block_num].adc is None:
                                        TR_ref_blocks[block_num].adc = DummyADC_type()
                                    if TRs_blocks[tr_idx][block_num].adc is None:
                                        TRs_blocks[tr_idx][block_num].adc = DummyADC_type()
                                    events_change_attributes = self.compare_attributes(TR_ref_blocks[block_num].adc, TRs_blocks[tr_idx][block_num].adc)
                            elif idx == 6:  # LABEL event
                                events_change_attributes = self.compare_attributes(TR_ref_blocks[block_num].label, TRs_blocks[tr_idx][block_num].label)
                            else:
                                events_change_attributes = None

                            if events_change_attributes is not None:
                                if events_change_name not in attributes_changed:
                                    attributes_changed[events_change_name] = []
                                attributes_changed[events_change_name].append(events_change_attributes)
            
                            # events_change_attributes = compare(SQ_base['blocks'][block_num][idx], tr_events[block_num][idx])
                            SQ_event_changes.setdefault('changes', []).append({
                                'real_time': True,
                                'block_num': block_num,
                                'event_name': events_change_name,
                                'event_change_id': events_change_id,
                                'events_change_attributes': events_change_attributes
                            })
        


            return SQ_event_changes
        # There are three levels of changes:
        # Per TR - update SQ attributes 
        # Per SQ repeat - update SQ repeat attributes
        # Per SQ - update SQ base attributes

        self.SQ_attributes = [[]]
        self.SQ_repeat_attributes = [[]]  # List to hold SQ repeat attributes
        # Initialize SQ_base as a nested dictionary to include only SQ_idx and kernel_repeat
        self.SQ_base = {
            SQ_idx: {
            kernel_repeat: {}
            for kernel_repeat in range(self.SOR_kernel_repeats)
            }
            for SQ_idx in range(len(self.SOR_kernel))
        }

        for kernel_repeat in range(self.SOR_kernel_repeats):
            for SQ_idx, SQ in enumerate(self.SOR_kernel):

              
                tr_repeat_start = kernel_repeat * self.SOR_kernel_SQ_repeats[SQ]
                tr_repeat_end = tr_repeat_start + self.SOR_kernel_SQ_repeats[SQ]

                TRs_blocks = self.SQs_blocks_compact[SQ][tr_repeat_start:tr_repeat_end]
                TRs_events = self.SQs_block_events_compact[SQ][tr_repeat_start:tr_repeat_end]
                
                # Check if there is a possibility of merging multiple TRs into a single SQ object to help save RF, AQ delays
                if len(self.SOR_kernel) > 1: # it is a multi-SQ kernel
                    if tr_repeat_end > 1: # multiple repeats of the same SQ consecutively
                        # Merge the repeats into a single SQ object
                        TRs_blocks, TRs_events = self.merge_blocks_events(TRs_blocks, TRs_events)
                        
                    
                # Store the base SQ for each SQ and kernel repeat

                self.SQ_base[SQ_idx][kernel_repeat] = {
                    'blocks': TRs_blocks[0] if TRs_blocks else [],
                    'events': TRs_events[0] if TRs_events else []
                }
                TRs_prev = {
                    'blocks': TRs_blocks[0] if TRs_blocks else [],
                    'events': TRs_events[0] if TRs_events else []
                }

                # Ensure self.SQ_attributes has enough SQs (first dimension)
                while len(self.SQ_attributes) <= SQ_idx:
                    self.SQ_attributes.append([])
                # Ensure self.SQ_attributes[SQ] has enough kernel repeats (second dimension)
                while len(self.SQ_attributes[SQ_idx]) <= kernel_repeat:
                    self.SQ_attributes[SQ_idx].append([])
                self.SQ_attributes[SQ_idx][kernel_repeat].append(get_SQ_attributes(TRs_blocks, TRs_events, TRs_prev))
                
                # Now we have SQ_base and SQ_attributes populated for each SQ and kernel repeat
                if (kernel_repeat > 0):
                    TRs_blocks_repeat = [[]]
                    TRs_events_repeat = [[]]
                    SQ_prev_repeat = {
                        'blocks': self.SQ_base[SQ_idx][kernel_repeat-1]['blocks'],
                        'events': self.SQ_base[SQ_idx][kernel_repeat-1]['events'] 
                    }
                    TRs_blocks_repeat[0] = self.SQ_base[SQ_idx][kernel_repeat]['blocks']
                    TRs_events_repeat[0] = self.SQ_base[SQ_idx][kernel_repeat]['events']
                    # Ensure the inner list is long enough before appending
                    # Ensure self.SQ_repeat_attributes has enough SQs (first dimension)
                    while len(self.SQ_repeat_attributes) <= SQ_idx:
                        self.SQ_repeat_attributes.append([])
                    while len(self.SQ_repeat_attributes[SQ_idx]) <= kernel_repeat:
                        self.SQ_repeat_attributes[SQ_idx].append([])
                    # Get the SQ repeat attributes for each SQ and kernel repeat
                    self.SQ_repeat_attributes[SQ_idx][kernel_repeat].append(get_SQ_attributes(TRs_blocks_repeat, TRs_events_repeat, SQ_prev_repeat))
        
        for SQ_idx, SQ in enumerate(self.SOR_kernel):
            self.SOR_kernel_SQ_repeats[SQ] = 1  # Set the repeat count to 1 for the merged SQ
        self.SOR_SQ_repeats = self.SOR_kernel_SQ_repeats * self.SOR_kernel_repeats


    def get_GR_attributes(self, gr):
        # Implementation for getting GR attributes goes here
        gammabar = self.gammabar
        # Convert a PyPulseq trapezoid gradient object to a lite gradient object
        gr_seqlite_form = types.SimpleNamespace()
        gr_seqlite_form.amplitude = np.round(gr.amplitude * 1e3 / gammabar, decimals=3) # convert to mT/m
        gr_seqlite_form.rise_time = np.round(gr.rise_time * 1e6, decimals=0)
        gr_seqlite_form.flat_time = np.round(gr.flat_time * 1e6, decimals=0)
        gr_seqlite_form.fall_time = np.round(gr.fall_time * 1e6, decimals=0)
        gr_seqlite_form.delay = np.round(gr.delay * 1e6, decimals=0)
        return gr_seqlite_form

    def get_RF_attributes(self, rf):
        gammabar = self.gammabar
        rf.delay = np.round(rf.delay * 1e6, decimals=0)
        rf.amplitude = np.abs(np.max(rf.signal))
        rf.signal_rescaled = rf.signal / rf.amplitude # we know that waveforms were designed for unit amplitude max.
        if np.imag(rf.signal_rescaled).all() == 0:
            rf.signal_rescaled = np.real(rf.signal_rescaled)
        flip = np.abs(np.sum(rf.signal_rescaled) * self.definitions['RadiofrequencyRasterTime'] * 2 * np.pi)
        rf.FA = np.int16(np.round(np.rad2deg(rf.amplitude * flip), decimals=0))
        rf.t = np.round(rf.t * 1e6, decimals=1) # us
        return rf

    def get_ADC_attributes(self, adc):
        # Convert a PyPulseq ADC object to a lite ADC object
        adc.dwell = np.round(adc.dwell * 1e6, decimals=2)
        adc.delay = np.round(adc.delay * 1e6, decimals=2)
        adc.num_samples = adc.num_samples
        adc.phase_offset = np.round(np.rad2deg(adc.phase_offset), decimals=3)
        adc.freq_offset = np.round(adc.freq_offset * 1e3, decimals=3)  # convert to kHz
        return adc


    def adjust_delay_blocks(self, tr_blocks = None, tr_events = None, verbose_local=False):  # This is something that is best handled in the .seq file
        
        # Adjust the delays to be encapsulated in the next block
        if verbose_local:
            print(Fore.YELLOW + 'Adjusting delays...' + Style.RESET_ALL)
            print(Fore.YELLOW  + 'Number of blocks in base TR before adjustment: {}'.format(len(tr_blocks)) + Style.RESET_ALL)
        
        for block in tr_blocks:
            if block.block_duration is not None and block.rf is None and block.gx is None and block.gy is None and block.gz is None and block.adc is None:
                if verbose_local:
                    print(Fore.BLUE + 'Block with only block duration found: {}'.format(block.block_duration) + Style.RESET_ALL)
                delay = block.block_duration
                previous_block = tr_blocks[tr_blocks.index(block) - 1]
                    
                previous_block.block_duration += delay
                
                tr_blocks[tr_blocks.index(block) -1] = previous_block
                tr_events.pop(tr_blocks.index(block))
                tr_blocks.remove(block)
                
        if verbose_local:           
             print(Fore.YELLOW + 'Number of blocks in base TR after adjustment: {}'.format(len(tr_blocks)) + Style.RESET_ALL)


        return tr_blocks, tr_events

    def merge_blocks_events(self, tr_blocks, tr_events):
            # initialize tr_blocks_new and tr_events_new as lists of lists
            # Ensure tr_blocks_new and tr_events_new have the same data type as tr_blocks and tr_events
            tr_blocks_new = []
            tr_events_new = []
            block_num_new = 0
            block_num_total = len(tr_blocks) * len(tr_blocks[0]) if tr_blocks and tr_blocks[0] else 0

            # Flatten all blocks and events into single lists
            for tr_num in range(len(tr_blocks)):
                for block_num in range(len(tr_blocks[tr_num])):
                    block = tr_blocks[tr_num][block_num]
                    event = tr_events[tr_num][block_num]

                    tr_blocks_new.append(block)
                    tr_events_new.append(event)


            # Wrap in a list to match the original structure (list of lists)
            tr_blocks_new = [tr_blocks_new]
            tr_events_new = [tr_events_new]

            return tr_blocks_new, tr_events_new

    def merge_repeats(self, tr_blocks, tr_events):

        def add_blocks_events(tr_blocks, tr_events, num_event, tr_blocks_new, tr_events_new, merge=False):
            for block, event in zip(tr_blocks[num_event], tr_events[num_event]):
                # Extend the length of tr_blocks_new and tr_events_new if needed
                while len(tr_blocks_new) <= num_event:
                    tr_blocks_new.append([])
                while len(tr_events_new) <= num_event:
                    tr_events_new.append([])
                if merge:
                    # Ensure the label in the event is set to 0
                    event[-1] = 0
                tr_blocks_new[num_event].append(block)
                tr_events_new[num_event].append(event)
        
            return tr_blocks_new, tr_events_new

        tr_events_new = [[]]
        tr_blocks_new = [[]]
        num_events_new = 0
        for num_event, event in enumerate(tr_events):
            if num_event == 0:
                tr_blocks_new, tr_events_new = add_blocks_events(tr_blocks, tr_events, num_events_new, tr_blocks_new, tr_events_new)
            elif np.array_equal(tr_events[num_event], tr_events[num_event - 1]):
                tr_blocks_new, tr_events_new = add_blocks_events(tr_blocks, tr_events, num_events_new, tr_blocks_new, tr_events_new, merge=True)
            elif not np.array_equal(tr_events[num_event], tr_events[num_event - 1]):
                num_events_new += 1
                tr_blocks_new, tr_events_new = add_blocks_events(tr_blocks, tr_events, num_events_new, tr_blocks_new, tr_events_new)

        return tr_blocks_new, tr_events_new

    def get_SQ_rt_attributes(self):

        def get_blocks_rt_attributes(blocks_set, SQ_idx, kernel_repeat):
            # expect changes every TR, so can obtain step by subtracting the first block from the second block and considering the first block as strength and the number of blocks as factor
            strength = []
            step = []
            factor = []
            attribute = []
            event_name_store = []
            block_num_store = []
            units = []
            gamma = 42.576e3 # to convert it into mT/m
            for block_num in blocks_set:
                # Access the base event attributes for the given block_num and event_type
                # event_type is an integer index (e.g., 1=RF, 2=GX, etc.)
                event_names = list(blocks_set[block_num].keys())
                for event_name in event_names:
                    factor.append(len(blocks_set[block_num][event_name]) + 1)  # Number of changes in this block_num + 1 for the base block
                    base_block = self.SQ_base[SQ_idx][kernel_repeat]['blocks'][block_num]
                    event_name_store.append( event_name)
                    block_num_store.append(block_num)
                    if event_name == 'RF':
                        base_attr = base_block.rf
                        attr_changes = list(blocks_set[block_num][event_name][0]['events_change_attributes'].keys())
                        for attr_change in attr_changes:
                            if attr_change == 'freq_offset':
                                attribute.append(attr_change)
                                strength.append(np.round(base_attr.freq_offset, self.precision_digits))
                                step.append(np.round(blocks_set[block_num][event_name][0]['events_change_attributes']['freq_offset'], self.precision_digits))
                                units.append('Hz')
                            elif attr_change == 'phase_offset':
                                attribute.append(attr_change)
                                strength.append(np.round(np.rad2deg(base_attr.phase_offset), self.precision_digits))
                                step.append(np.round(np.rad2deg(blocks_set[block_num][event_name][0]['events_change_attributes']['phase_offset']), self.precision_digits))
                                units.append('degrees')
                            else:
                                strength.append(0)
                                step.append(0)

                    elif event_name == 'GX':
                        base_attr = base_block.gx
                        attr_changes = list(blocks_set[block_num][event_name][0]['events_change_attributes'].keys())
                        for attr_change in attr_changes:
                            if attr_change == 'amplitude':
                                attribute.append(attr_change)
                                strength.append(np.round(base_attr.amplitude/gamma, self.precision_digits))
                                step.append(np.round(blocks_set[block_num][event_name][0]['events_change_attributes']['amplitude']/gamma, self.precision_digits))
                                units.append('mT/m')
                            else:
                                strength.append(0)
                                step.append(0)

                    elif event_name == 'GY':
                        base_attr = base_block.gy
                        attr_changes = list(blocks_set[block_num][event_name][0]['events_change_attributes'].keys())
                        for attr_change in attr_changes:
                            if attr_change == 'amplitude':
                                attribute.append(attr_change)
                                strength.append(np.round(base_attr.amplitude/gamma, self.precision_digits))
                                step.append(np.round(blocks_set[block_num][event_name][0]['events_change_attributes']['amplitude']/gamma, self.precision_digits))
                                units.append('mT/m')
                            else:
                                strength.append(0)
                                step.append(0)

                    elif event_name == 'GZ':
                        base_attr = base_block.gz
                        attr_changes = list(blocks_set[block_num][event_name][0]['events_change_attributes'].keys())
                        for attr_change in attr_changes:
                            if attr_change == 'amplitude':
                                attribute.append(attr_change)
                                strength.append(np.round(base_attr.amplitude/gamma, self.precision_digits))
                                step.append(np.round(blocks_set[block_num][event_name][0]['events_change_attributes']['amplitude']/gamma, self.precision_digits))
                                units.append('mT/m')
                            else:
                                strength.append(0)
                                step.append(0)
                                
                    elif event_name == 'ADC':
                        base_attr = base_block.adc
                        if base_attr is None:
                            class DummyADC:
                                def __init__(self):
                                    self.type = 'adc'
                                    self.num_samples = 0
                                    self.phase_offset = 0
                                    self.delay = 0
                                    self.dwell = 0
                                    self.dead_time = 0
                                    self.freq_offset = 0
                                    self.phase = 0
                                    self.trigger = 0
                            base_attr = DummyADC()
                        attr_changes = list(blocks_set[block_num][event_name][0]['events_change_attributes'])
                        for attr_change in attr_changes:
                            if attr_change == 'num_samples':
                                attribute.append(attr_change)
                                strength.append(np.round(base_attr.num_samples, self.precision_digits))
                                step.append(np.round(blocks_set[block_num][event_name][0]['events_change_attributes']['num_samples'], self.precision_digits))
                                units.append('samples')
                            elif attr_change == 'phase_offset':
                                attribute.append(attr_change)
                                strength.append(np.round(np.rad2deg(base_attr.phase_offset), self.precision_digits))
                                step.append(np.round(np.rad2deg(blocks_set[block_num][event_name][0]['events_change_attributes']['phase_offset']), self.precision_digits))
                                units.append('degrees')
                            else:
                                strength.append(0)
                                step.append(0)
                # Now you can use base_attr for further processing
            # if SQ_repeat:
            #     factor = int(self.SOR_kernel_repeats / (len(self.SOR_kernel) * len(self.SOR_kernel_SQ_repeats)))  # TODO: Think about this to be more general

            rt_attributes = {
                'block_num': block_num_store,
                'block_event_name': event_name_store,
                'attribute': attribute,
                'units': units,
                'strength': strength,
                'step': step,
                'factor': factor
            }

            return rt_attributes
            


        self.SQ_rt_attributes = {
            SQ_idx: {kernel_repeat: {} for kernel_repeat in range(self.SOR_kernel_repeats)}
            for SQ_idx in range(len(self.SOR_kernel))
        }
        self.SQ_repeat_rt_attributes = {
            SQ_idx: {kernel_repeat: {} for kernel_repeat in range(self.SOR_kernel_repeats)}
            for SQ_idx in range(len(self.SOR_kernel))
        }

        for SQ_idx in range(len(self.SOR_kernel)):
            for kernel_repeat in range(self.SOR_kernel_repeats):
                # Check if 'changes' key exists before proceeding
                if (
                    not self.SQ_attributes[SQ_idx][kernel_repeat][0].get('changes')
                    or self.SQ_attributes[SQ_idx][kernel_repeat][0].get('real_time') is False
                ):
                    self.SQ_rt_attributes[SQ_idx][kernel_repeat] = False
                else:
                    blocks_changes = self.SQ_attributes[SQ_idx][kernel_repeat][0]['changes']
                    # from blocks_changes sort to create a set of blocks that have changes in a particular block_num and particular event type
                    blocks_set = {}
                    for change in blocks_changes:
                        block_num = change['block_num']
                        event_name = change['event_name']
                        if block_num not in blocks_set:
                            blocks_set[block_num] = {}
                        if event_name not in blocks_set[block_num]:
                            blocks_set[block_num][event_name] = []
                        blocks_set[block_num][event_name].append(change)

                    self.SQ_rt_attributes[SQ_idx][kernel_repeat] = get_blocks_rt_attributes(blocks_set, SQ_idx, kernel_repeat)
                
                # TODO: Now identify changes in SQ across repeats
                # Check for existence and validity of required keys and values before accessing
                if (
                    (self.SQ_repeat_attributes is None)
                    or (SQ_idx not in range(len(self.SQ_repeat_attributes)))
                    or (kernel_repeat not in range(len(self.SQ_repeat_attributes[SQ_idx])))
                    or (not isinstance(self.SQ_repeat_attributes[SQ_idx][kernel_repeat], list))
                    or (len(self.SQ_repeat_attributes[SQ_idx][kernel_repeat]) == 0)
                    or ('real_time' in self.SQ_repeat_attributes[SQ_idx][kernel_repeat][0])
                    or ('changes' not in self.SQ_repeat_attributes[SQ_idx][kernel_repeat][0])
                    or (self.SQ_repeat_attributes[SQ_idx][kernel_repeat][0]['changes'] is None)
                ):
                    self.SQ_repeat_rt_attributes[SQ_idx][kernel_repeat] = False
                else:
                    blocks_changes = self.SQ_repeat_attributes[SQ_idx][kernel_repeat][0]['changes']
                    # from blocks_changes sort to create a set of blocks that have changes in a particular block_num and particular event type
                    blocks_set = {}
                    for change in blocks_changes:
                        block_num = change['block_num']
                        event_name = change['event_name']
                        if block_num not in blocks_set:
                            blocks_set[block_num] = {}
                        if event_name not in blocks_set[block_num]:
                            blocks_set[block_num][event_name] = []
                        blocks_set[block_num][event_name].append(change)

                    self.SQ_repeat_rt_attributes[SQ_idx][kernel_repeat] = get_blocks_rt_attributes(blocks_set, SQ_idx, kernel_repeat)

        # self.SQ_repeat_rt_attributes[SQ_idx][kernel_repeat] must be compressed to reveal kernel repeat changes

        # Clean up the SQ_rt_attributes and SQ_repeat_rt_attributes to remove empty entries
        # Remove entries with the value of False from SQ_rt_attributes (for each SQ_idx and kernel_repeat)
        
        
        self.SQ_rt_attributes = {
            sq_idx: {k: v for k, v in kernel_repeats.items() if v is not False}
            for sq_idx, kernel_repeats in self.SQ_rt_attributes.items()
        }

        # SQ_rt_attributes contains the real-time changes over TRs within each kernel repeat - can compress this further by comparing changes


        # Remove entries with the value of False from SQ_repeat_rt_attributes
        self.SQ_repeat_rt_attributes = {
            sq_idx: {k: v for k, v in kernel_repeats.items() if v is not False}
            for sq_idx, kernel_repeats in self.SQ_repeat_rt_attributes.items()
        }

        # SQ_repeat_rt_attributes contains the real-time changes over kernel repeats within each SQ
        # This can be further compressed by comparing changes across repeats

        # Print the real-time attributes for each SQ repeat
        if self.verbose and self.SQ_repeat_rt_attributes:
            print(Fore.GREEN + 'Real-time attributes for SQ repeats obtained successfully' + Style.RESET_ALL)
            print(Fore.YELLOW + 'SQ repeat real-time attributes:' + Style.RESET_ALL)
            for sq_idx, rt_attr in self.SQ_repeat_rt_attributes.items():
                print(Fore.CYAN + f"SQ {sq_idx}:" + Style.RESET_ALL)
                if isinstance(rt_attr, dict):
                    for key, value in rt_attr.items():
                        if isinstance(value, float):
                            print(Fore.CYAN + f"  {key}: {value:.2f}" + Style.RESET_ALL)
                        elif isinstance(value, list):
                            formatted = [
                                f"{v:.2f}" if isinstance(v, float) else v
                                for v in value
                            ]
                            print(Fore.CYAN + f"  {key}: {formatted}" + Style.RESET_ALL)
                        else:
                            print(Fore.CYAN + f"  {key}: {value}" + Style.RESET_ALL)
                else:
                    if isinstance(rt_attr, float):
                        print(Fore.CYAN + f"  {rt_attr:.2f}" + Style.RESET_ALL)
                    else:
                        print(Fore.CYAN + f"  {rt_attr}" + Style.RESET_ALL)

        # Print the real-time attributes for each SQ
        if self.verbose and self.SQ_rt_attributes is not False:
            print(Fore.GREEN + 'Real-time attributes for SQs obtained successfully' + Style.RESET_ALL)
            print(Fore.YELLOW + 'SQ real-time attributes:' + Style.RESET_ALL)
            for sq_idx, rt_attr in self.SQ_rt_attributes.items():
                print(Fore.CYAN + f"SQ {sq_idx}:" + Style.RESET_ALL)
                if isinstance(rt_attr, dict):
                    for key, value in rt_attr.items():
                        if isinstance(value, float):
                            print(Fore.MAGENTA + f"  {key}: {value:.2f}" + Style.RESET_ALL)
                        elif isinstance(value, list):
                            formatted = [
                                f"{v:.2f}" if isinstance(v, float) else v
                                for v in value
                            ]
                            print(Fore.MAGENTA + f"  {key}: {formatted}" + Style.RESET_ALL)
                        else:
                            print(Fore.MAGENTA + f"  {key}: {value}" + Style.RESET_ALL)
                else:
                    if isinstance(rt_attr, float):
                        print(Fore.MAGENTA + f"  {rt_attr:.2f}" + Style.RESET_ALL)
                    else:
                        print(Fore.MAGENTA + f"  {rt_attr}" + Style.RESET_ALL)

    def write(self, file_name = 'test.seqlite'):
        file_name = Path(file_name)
        if file_name.suffix != '.seqlite':
        # Append .seqlite suffix
            file_name = file_name.with_suffix(file_name.suffix + '.seqlite')

        
        # Define headers including computed values from .seq file and other definitions
        with open(file_name, 'w') as output_file:
            output_file.write('# Pulseq-lite sequence file\n')
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

            # Zero: SOR kernel
            output_file.write('# Format of zero level hierarchy - SOR kernel:\n')
            output_file.write('[SOR]\n')
            output_file.write(f'kernel {self.SOR_kernel}\n')
            output_file.write(f'SQs_repeats {self.SOR_kernel_repeats}\n')
            output_file.write('\n')


            # One: SQ blocks
            output_file.write('# Format of first level hierarchy - SQs:\n')
            for SQ_idx, SQ in enumerate(self.SOR_kernel):
                output_file.write(f'SQ#  {SQ_idx}\n')
                output_file.write(f'[BASE BLOCKS]\n')
                output_file.write('# DUR RF GX GY GZ AQ LABEL\n')
                for event_num, block in enumerate(self.SQ_base[SQ_idx][0]['events']):
                    output_file.write(f'{event_num} {block[0]}   {block[1]}   {block[2]}   {block[3]}   {block[4]}   {block[5]}   {block[6]}\n')
                output_file.write('\n')

                # Two: SQ attributes
                output_file.write('# Format of second level hierarchy - SQ repeat attributes:\n')
                output_file.write('[SQ REPEAT ATTRIBUTES]\n')
                output_file.write('#  Kernel_repeat  Block_num  Event_name  Event_change_id  Events_change_attributes\n')
                # check if there is a SQ_repeat_attributes for this SQ, if not print "No SQ repeat attributes for this SQ"
                if len(self.SQ_repeat_attributes) > 0 and SQ_idx < len(self.SQ_repeat_attributes) and len(self.SQ_repeat_attributes[SQ_idx]) > 0:
                    if 'changes' in self.SQ_repeat_attributes[SQ_idx][1][0]:
                        loop_repeat = 1
                        for kernel_repeat in range(1, len(self.SQ_repeat_attributes[SQ_idx])):

                            SQ_repeat_attributes = self.SQ_repeat_attributes[SQ_idx][kernel_repeat]
                            # Compare the reported changes with the changes of the previous kernel_repeat
                            for block_num in range(len(SQ_repeat_attributes)):
                                SQ_repeat_changes = self.SQ_repeat_attributes[SQ_idx][kernel_repeat][0]['changes'] if self.SQ_repeat_attributes[SQ_idx][kernel_repeat] else []
                                if kernel_repeat == 1:
                                    SQ_repeat_changes_prev = SQ_repeat_changes

                                # Compare SQ_repeat_changes with SQ_repeat_attributes_prev
                                if SQ_repeat_changes:
                                    if SQ_repeat_changes != SQ_repeat_changes_prev or kernel_repeat == 1:
                                        for change, change_prev in zip(SQ_repeat_changes, SQ_repeat_changes_prev):
                                            block_num = change['block_num']
                                            event_name = change['event_name']
                                            event_change_id = change['event_change_id']
                                            events_change_attributes = change['events_change_attributes']
                                            # check if it is the same event ID as the previous kernel_repeat and if the event changes attributes are the same

                                            block_num_prev = change_prev['block_num']
                                            event_name_prev = change_prev['event_name']
                                            event_change_id_prev = change_prev['event_change_id']
                                            events_change_attributes_prev = change_prev['events_change_attributes']
                                            
                                            if kernel_repeat > 1:
                                                if block_num != block_num_prev or event_name != event_name_prev or event_change_id != event_change_id_prev or events_change_attributes != events_change_attributes_prev:
                                                    output_file.write(f'{kernel_repeat}  {block_num}  {event_name}  {event_change_id}  {events_change_attributes}\n')
                                            else:
                                                output_file.write(f'{kernel_repeat}  {block_num}  {event_name}  {event_change_id}  {events_change_attributes}\n')

                                        SQ_repeat_changes_prev = SQ_repeat_changes
                                    
                                    else:
                                        loop_repeat += 1
                    if loop_repeat >= 1:
                       output_file.write(f'Loop count = {loop_repeat}\n\n')    
                else:
                    output_file.write('None\n\n') # No SQ repeat attributes for this SQ

                # Three: SQ real-time attributes
                output_file.write('# Format of third level hierarchy - SQ real-time attributes:\n')
                output_file.write('[RT UPDATES PER TR]\n')
                output_file.write('#  Block_num  Event_name  Attribute  Strength  Step  Factor\n')
                for kernel_repeat in  range(len(self.SQ_rt_attributes[SQ_idx])):
                    SQ_rt_changes = self.SQ_rt_attributes[SQ_idx][kernel_repeat]
                    for rt_changes_num in range(len(SQ_rt_changes['block_event_name'])):
                        block_num = SQ_rt_changes['block_num'][rt_changes_num]
                        event_name = SQ_rt_changes['block_event_name'][rt_changes_num]
                        attribute_name = SQ_rt_changes['attribute'][rt_changes_num]
                        strength = SQ_rt_changes['strength'][rt_changes_num]
                        step = SQ_rt_changes['step'][rt_changes_num]
                        factor = SQ_rt_changes['factor'][rt_changes_num]
                        output_file.write(f'{kernel_repeat}\t {block_num}\t {event_name}\t {attribute_name}\t  {strength}\t {step}\t {factor}\n')
                output_file.write(f'End of SQ{SQ_idx} description\n\n')


            # Store the RF, GR and ADC attributes to describe the waveforms later
            self.RF_ids = []
            self.RF_events = []
            self.GR_ids = []
            self.GR_events = []
            self.ADC_ids = []
            self.ADC_events = []
            for SQ_idx in range(len(self.SQ_base)):
                for kernel_repeat in range(len(self.SQ_base[SQ_idx])):
                    # Check if the event has occured before
                    blocks_tr= self.SQ_base[SQ_idx][kernel_repeat]['events']

                    for block_num, block in enumerate(blocks_tr):
                        rf_id = block[1]
                        gx_id = block[2]
                        gy_id = block[3]
                        gz_id = block[4]
                        adc_id = block[5]
                        if rf_id != 0: # RF event
                            if rf_id not in self.RF_ids:
                                self.RF_ids.append(rf_id)
                                self.RF_events.append(self.SQ_base[SQ_idx][kernel_repeat]['blocks'][block_num].rf)

                        if gx_id != 0: # GX event
                            if gx_id not in self.GR_ids:
                                self.GR_ids.append(gx_id)
                                self.GR_events.append(self.SQ_base[SQ_idx][kernel_repeat]['blocks'][block_num].gx)
                        
                        if gy_id != 0: # GY event
                            if gy_id not in self.GR_ids:
                                self.GR_ids.append(gy_id)
                                self.GR_events.append(self.SQ_base[SQ_idx][kernel_repeat]['blocks'][block_num].gy)
                        
                        if gz_id != 0: # GZ event
                            if gz_id not in self.GR_ids:
                                self.GR_ids.append(gz_id)
                                self.GR_events.append(self.SQ_base[SQ_idx][kernel_repeat]['blocks'][block_num].gz)

                        if adc_id != 0: # ADC event
                            if adc_id not in self.ADC_ids:
                                self.ADC_ids.append(adc_id)
                                self.ADC_events.append(self.SQ_base[SQ_idx][kernel_repeat]['blocks'][block_num].adc)

            # Start writing the GR, ADC and RF events to the seqlite file
            if self.GR_ids:
                output_file.write('[GR EVENTS]\n')
                output_file.write('ID amplitude(mT/m) rise_time(us) flat_time(us) fall_time(us) delay(us)\n')
                for gr_id, gr_event in zip(self.GR_ids, self.GR_events):
                    gr_seqlite_form = self.get_GR_attributes(gr_event)
                    # Write the gradient attributes in the seqlite format
                    output_file.write(f'{gr_id} {gr_seqlite_form.amplitude} {gr_seqlite_form.rise_time} {gr_seqlite_form.flat_time} {gr_seqlite_form.fall_time} {gr_seqlite_form.delay}\n')
                output_file.write('\n')

            if self.ADC_ids:
                output_file.write('[ADC EVENTS]\n')
                output_file.write('ID num_samples dwell(us) delay(us) freq_offset(Hz) phase_offset(deg)\n')
                for adc_id, adc_event in zip(self.ADC_ids, self.ADC_events):
                    adc_seqlite_form = self.get_ADC_attributes(adc_event)
                    output_file.write(f'{adc_id} {adc_seqlite_form.num_samples} {adc_seqlite_form.dwell} {adc_seqlite_form.delay} {adc_seqlite_form.freq_offset} {adc_seqlite_form.phase_offset}\n')
                output_file.write('\n')

            if self.RF_ids:
                output_file.write('[RF EVENTS]\n')
                for rf_id, rf_event in zip(self.RF_ids, self.RF_events):
                    output_file.write('ID FA delay(us) freq_offset(Hz) phase_offset(deg)\n')
                    rf_seqlite_form = self.get_RF_attributes(rf_event)
                    # Write t and signal_rescaled as two columns side by side
                    t_str = ' '.join([f'{v:.1f}' for v in rf_seqlite_form.t])
                    signal_str = ' '.join([f'{v:.6f}' for v in rf_seqlite_form.signal_rescaled])
                    output_file.write(f'{rf_id} {rf_seqlite_form.FA} {rf_seqlite_form.delay} {rf_seqlite_form.freq_offset} {rf_seqlite_form.phase_offset}\n')
                    output_file.write('t(us)\t signal_rescaled\n')
                    for instant in range(len(rf_seqlite_form.t)):
                        output_file.write(f'{rf_seqlite_form.t[instant]} {rf_seqlite_form.signal_rescaled[instant]}\n')
                output_file.write('\n')

    def convert_seqlite_to_seq(self, seqlite_file = 'test.seqlite', seq_file = 'test.seq', time_range = [0, 1]):
        """
        Write the sequence represented in the .seqlite file to a .seq file.
        """
        seqlite_file = Path(seqlite_file)
        seq_file = Path(seq_file)
        if seqlite_file.suffix != '.seqlite':
            # Append .seqlite suffix
            seqlite_file = seqlite_file.with_suffix(seqlite_file.suffix + '.seqlite')
        
        if seq_file.suffix != '.seq':
            # Append .seq suffix
            seq_file = seq_file.with_suffix(seq_file.suffix + '.seq')

        # Read the .seqlite file
        with open(seqlite_file, 'r') as input_file:
            lines = input_file.readlines()




        seqlite_file_list = []
        # Write the .seq file
        with open(seq_file, 'w') as output_file:
            for line in lines:
                # store all the lines in a dictionary
                seqlite_file_list.append(line.strip())
        # close the file


        # Definitions and SOR extraction
        for line in seqlite_file_list:
                # First get system definitions
                if '[DEFINITIONS]' in line:
                    print(Fore.GREEN + "Found [DEFINITIONS] section" + Style.RESET_ALL)
                    # Within the SOR loop, use the pypulseq seq.add_block() method to add the blocks to the sequence
                
                if '[SOR]' in line:
                    print(Fore.GREEN + "Found [SOR] section" + Style.RESET_ALL)
                
                
                tokens = line.split()
                if not tokens:
                    continue
                key = tokens[0]
                value = tokens[-1]
                if key == 'ADC_deadtime':
                    adc_dead_time = float(value)
                if key == 'AdcRasterTime':
                    adc_raster_time = float(value)
                if key == 'BlockDurationRaster':
                    block_duration_raster = float(value)
                if key == 'GradientRasterTime':
                    gradient_raster_time = float(value)
                if key == 'RF_deadtime':
                    rf_dead_time = float(value)
                if key == 'RadiofrequencyRasterTime':
                    rf_raster_time = float(value)
                if key == 'RF_ringdown':
                    rf_ringdown_time = float(value)
                if key == 'FOV':
                    fov = float(value)
                if key == 'Name':
                    name = value
                if key == 'TotalDuration':
                    total_duration = float(value)
                if key == 'kernel':
                    # Strip brackets and read the int numbers within
                    kernel_str = value.strip('[]')
                    kernel = [int(x) for x in kernel_str.split() if x.strip()]
                if key == 'SQs_repeats':
                    sqs_repeats = int(value)

                    # Now we have the SOR kernel and repeats
                   

        # Now we have the system definitions, we can create a pypulseq sequence object
        # Initiaite an empty pypulseq sequence - this will be updated from the DEFINTIONS section of the .seqlite file
        system = pp.Opts(
        max_grad=28,
        grad_unit='mT/m',
        max_slew=150,
        slew_unit='T/m/s',
        rf_ringdown_time=rf_ringdown_time, # note down values from Sandeep - 
        rf_dead_time=rf_dead_time,
        adc_dead_time=adc_dead_time,
        rf_raster_time=rf_raster_time,
        grad_raster_time=gradient_raster_time,
        adc_raster_time=adc_raster_time,
        block_duration_raster=block_duration_raster,
        )
        
        seq = pp.Sequence(system=system)
        if 'fov' in locals():
            seq.set_definition('FOV', fov)
        if 'name' in locals():
            seq.set_definition('Name', name)
        if 'total_duration' in locals():
            seq.set_definition('TotalDuration', total_duration)


        # SECTION EXTRACTION OF SQs and EVENTS
        # in seqlite_file_dict, find the SQs and their attributes and store them in SQs_blocks, SQ_base, SQ_attributes, SQ_repeat_attributes, SQ_rt_attributes
        # This correspondds to the lines betweeen '[SQ#' and [GR EVENTS]
        # Find the indices for 'SQ#' and '[GR EVENTS]'
        sq_indices = [i for i, line in enumerate(seqlite_file_list) if line.startswith('SQ#')]
        gr_events_index = next((i for i, line in enumerate(seqlite_file_list) if '[GR EVENTS]' in line), None)

        # SQ EXTRACTION
        # Extract the blocks for each SQ
        SQs_sections = []
        for idx, sq_idx in enumerate(sq_indices):
            next_sq = sq_indices[idx + 1] if idx + 1 < len(sq_indices) else gr_events_index
            if next_sq is not None:
                SQs_sections.append(seqlite_file_list[sq_idx:next_sq])

        # Parse each SQ section to extract ID, BLOCKS, SQ_repeat_attributes, and SQ_rt_updates
        SQs_parsed = []
        for sq_section in SQs_sections:
            sq_info = {}
            # Get SQ ID
            for line in sq_section:
                if line.startswith('SQ#'):
                    parts = line.split()
                    if len(parts) > 1:
                        sq_info['ID'] = int(parts[1])
                    break

            # Get BLOCKS
            blocks = []
            blocks_start = None
            blocks_end = None
            for i, line in enumerate(sq_section):
                if '# DUR RF GX GY GZ AQ LABEL' in line:
                    blocks_start = i + 1
                    continue
                if blocks_start is not None and line.strip() == '':
                    blocks_end = i
                    break
            if blocks_start is not None and blocks_end is not None:
                for line in sq_section[blocks_start:blocks_end]:
                    if line.strip():
                        # Convert the line to a list of integers (skip the first column if it's an index)
                        parts = line.strip().split()
                        # If the first column is an index (e.g., event_num), skip it
                        if parts and parts[0].isdigit():
                            parts = parts[1:]
                        int_parts = [int(x) for x in parts]
                        blocks.append(int_parts)
            sq_info['BLOCKS'] = blocks

            # Get SQ_repeat_attributes
            sq_repeat_attributes = []
            repeat_attr_start = None
            repeat_attr_end = None
            for i, line in enumerate(sq_section):
                if '#  Kernel_repeat  Block_num  Event_name  Event_change_id  Events_change_attributes' in line:
                    repeat_attr_start = i + 1
                    continue
                if repeat_attr_start is not None and line.strip() == '':
                    repeat_attr_end = i
                    break
            if repeat_attr_start is not None and repeat_attr_end is not None:
                for line in sq_section[repeat_attr_start:repeat_attr_end]:
                    if line.strip() and not line.startswith('Loop count'):
                        sq_repeat_attributes.append(line.strip())
            sq_info['SQ_repeat_attributes'] = sq_repeat_attributes

            # Get SQ_rt_updates
            sq_rt_updates = []
            rt_updates_start = None
            rt_updates_end = None
            for i, line in enumerate(sq_section):
                if '#  Block_num  Event_name  Attribute  Strength  Step  Factor' in line:
                    rt_updates_start = i + 1
                    continue
                if rt_updates_start is not None and line.strip() == '':
                    rt_updates_end = i
                    break
            if rt_updates_start is not None and rt_updates_end is not None:
                for line in sq_section[rt_updates_start:rt_updates_end]:
                    if line.strip() and not line.startswith('End of SQ'):
                        # Split the line into words, remove tabs, and store as a dict
                        parts = line.replace('\t', ' ').split()
                        if len(parts) >= 6:
                            sq_rt_updates.append({
                                'Block_num': parts[1],
                                'Event_name': parts[2],
                                'Attribute': parts[3],
                                'Strength': parts[4],
                                'Step': parts[5],
                                'Factor': parts[6] if len(parts) > 6 else None
                            })
            sq_info['SQ_rt_updates'] = sq_rt_updates

            SQs_parsed.append(sq_info)

        # Now SQs_sections is a list, each element is the lines for one SQ (from 'SQ#' up to the next 'SQ#' or '[GR EVENTS]')

        # GR EXTRACTION
        # Find the index for '[GR EVENTS]' and the next blank line after it
        gr_events_start = next((i for i, line in enumerate(seqlite_file_list) if '[GR EVENTS]' in line), None)
        if gr_events_start is not None:
            # Find the next blank line after '[GR EVENTS]'
            gr_events_end = next(
            (i for i in range(gr_events_start + 1, len(seqlite_file_list)) if seqlite_file_list[i] == ''),
            len(seqlite_file_list)
            )
            # Extract GR section lines (excluding the header and blank line)
            GR_section = seqlite_file_list[gr_events_start + 2:gr_events_end]
            # Now GR_section contains the gradient event lines

            GR_list = []
            for gr_line in GR_section:
                if gr_line.strip():
                    # Split the line by spaces and convert to appropriate types
                    parts = gr_line.split()
                    gr_id = int(parts[0])
                    amplitude = float(parts[1])
                    rise_time = float(parts[2]) * 1e-6  # Convert microseconds to seconds
                    flat_time = float(parts[3]) * 1e-6  # Convert microseconds to seconds
                    fall_time = float(parts[4]) * 1e-6  # Convert microseconds to seconds
                    delay = float(parts[5]) * 1e-6  # Convert microseconds to seconds

                    # Create a pp Gradient object with the extracted values
                    grad_event = pp.make_trapezoid(
                        channel='x',  # Assuming 'x' channel for simplicity; adjust as needed during the blocks reading
                        amplitude=amplitude,
                        rise_time=rise_time,
                        flat_time=flat_time,
                        fall_time=fall_time,
                        delay=delay
                    )
                    # Append the gradient event to the GR_list
                    GR_list.append((gr_id, grad_event))



        # ADC EXTRACTION
        # Find the index for '[ADC EVENTS]' and the next blank line after it
        adc_events_start = next((i for i, line in enumerate(seqlite_file_list) if '[ADC EVENTS]' in line), None)
        if adc_events_start is not None:
            # Find the next blank line after '[ADC EVENTS]'
            adc_events_end = next(
                (i for i in range(adc_events_start + 1, len(seqlite_file_list)) if seqlite_file_list[i] == ''),
                len(seqlite_file_list)
            )
            # Extract ADC section lines (excluding the header and blank line)
            ADC_section = seqlite_file_list[adc_events_start + 2:adc_events_end]
            # Now ADC_section contains the ADC event lines
            ADC_list = []
            for adc_line in ADC_section:
                if adc_line.strip():
                    # Split the line by spaces and convert to appropriate types
                    parts = adc_line.split()
                    adc_id = int(parts[0])
                    num_samples = int(parts[1])
                    dwell = float(parts[2]) * 1e-6  # Convert microseconds to seconds
                    delay = float(parts[3]) * 1e-6  # Convert microseconds to seconds
                    freq_offset = float(parts[4])
                    phase_offset = float(parts[5])

                    # Create a pp ADC object with the extracted values
                    adc_event = pp.make_adc(
                        num_samples=num_samples,
                        dwell=dwell,
                        delay=delay,
                        freq_offset=freq_offset,
                        phase_offset=phase_offset
                    )
                    # Append the ADC event to the ADC_list
                    ADC_list.append((adc_id, adc_event))



        # RF EXTRACTION
        # Find the index for '[RF EVENTS]' and the next blank line after it
        rf_events_start = next((i for i, line in enumerate(seqlite_file_list) if '[RF EVENTS]' in line), None)
        if rf_events_start is not None:       
            # Find the next blank line after '[RF EVENTS]'
            rf_events_end = next(
                (i for i in range(rf_events_start + 1, len(seqlite_file_list)) if seqlite_file_list[i] == ''),
                len(seqlite_file_list)
            )
            # Extract RF section lines (excluding the header and blank line)
            RF_section = seqlite_file_list[rf_events_start + 2:rf_events_end]
            # Now RF_section contains the RF event lines

            RF_list = []
            idx = 0
            while idx < len(RF_section):
                rf_line = RF_section[idx]
                if rf_line.strip() and not rf_line.startswith('ID') and not rf_line.startswith('t(us)'):
                    # Split the line by spaces and convert to appropriate types
                    parts = rf_line.split()
                    rf_id = int(parts[0])
                    FA = float(parts[1])
                    delay = float(parts[2])
                    freq_offset = float(parts[3])
                    phase_offset = float(parts[4])

                    # Find the index of the 't(us)\t signal_rescaled' header
                    t_signal_idx = None
                    for j in range(idx + 1, len(RF_section)):
                        if RF_section[j].startswith('t(us)'):
                            t_signal_idx = j
                            break
                    # Find the next 'ID' line or end of RF_section
                    next_id_idx = None
                    for j in range((t_signal_idx or idx) + 1, len(RF_section)):
                        if RF_section[j].startswith('ID'):
                            next_id_idx = j
                            break
                    if next_id_idx is None:
                        next_id_idx = len(RF_section)
                    # Extract waveform data
                    t_vals = []
                    signal_vals = []
                    if t_signal_idx is not None:
                        for k in range(t_signal_idx + 1, next_id_idx):
                            line = RF_section[k]
                            if line.strip():
                                parts = line.split()
                                if len(parts) >= 2:
                                    t_vals.append(float(parts[0]))  
                                    signal_vals.append(float(parts[1]))
                    # Create a pp RF object with the extracted values
                    rf_event = pp.make_arbitrary_rf(
                        flip_angle=np.deg2rad(FA),  # Convert flip angle to radians
                        delay=delay * 1e-6,  # Convert microseconds to seconds
                        freq_offset=freq_offset,
                        phase_offset=phase_offset,
                        signal=signal_vals
                    )
                    RF_list.append((rf_id, rf_event))
                    idx = next_id_idx
                else:
                    idx += 1

        # Ensure sqs_repeats is a list/array of int, not a single int
        if isinstance(sqs_repeats, int):
            sqs_repeats = [sqs_repeats for _ in range(len(kernel))]
        elif not isinstance(sqs_repeats, (list, tuple, np.ndarray)):
            sqs_repeats = [int(sqs_repeats) for _ in range(len(kernel))]
        else:
            sqs_repeats = [int(x) for x in sqs_repeats]


        # Now we have all the SQs, GR, ADC and RF events extracted from the .seqlite file
        # We can now create the sequence from the SQs and their attributes
        for sq_id in kernel: # Iterate over the SOR kernel
            for sq_repeat in range(sqs_repeats[sq_id]): # Iterate over the repeats for each SQ
                sq = SQs_parsed[sq_id]
                # Check if the sq has any repeat attributes 
                if sq['SQ_repeat_attributes'] != ['None']:
                    # If it has, we need to apply the repeat attributes to the SQ
                    sq['SQ_repeat_attributes'] = [
                        attr for attr in sq['SQ_repeat_attributes'] if int(attr.split()[0]) == sq_repeat + 1
                    ]
                else:
                    sq['SQ_repeat_attributes'] = []


                # Check if the sq has any real-time updates
                if sq['SQ_rt_updates']:
                    # Check that all factors are the same for this SQ's RT updates
                    factors = [float(rt_update['Factor']) if rt_update['Factor'] else 1.0 for rt_update in sq['SQ_rt_updates']]
                    if factors:
                        if not all(f == factors[0] for f in factors):
                            raise ValueError(f"Not all factors are the same in SQ {sq['ID']} RT updates: {factors}")
                        num_TRs = int(factors[0])
                    else:
                        num_TRs = 1.0

                TR_duration = 0.0 # for plotting one or two TRs
                for factor in range(num_TRs):
                    for block_num, block in enumerate(sq['BLOCKS']):
                        # Get the block attributes
                        dur, rf_id, gx_id, gy_id, gz_id, adc_id, sq_label = block
                        if factor == 0:
                            TR_duration += float(dur) * block_duration_raster 

                        # Create a new block for this SQ
                        class CustomBlock:
                            def __init__(self):
                                self.rf = None
                                self.gx = None
                                self.gy = None
                                self.gz = None
                                self.adc = None
                                self.label = None
                                self.sq_label = None
                                self.delay = None
                        new_block = CustomBlock()

                        # Add RF event if it exists
                        if rf_id != 0 and rf_id <= len(RF_list): # IDs start from 1
                            rf_event = RF_list[rf_id - 1][1]
                            new_block.rf = rf_event

                        # Add gradient events if they exist
                        if gx_id != 0 and gx_id <= len(GR_list):
                            gr_event = GR_list[gx_id - 1][1]
                            gr_event.channel = 'x'
                            new_block.gx = gr_event
                        if gy_id != 0 and gy_id <= len(GR_list):
                            gr_event = GR_list[gy_id - 1][1]
                            gr_event.channel = 'y'
                            new_block.gy = gr_event
                        if gz_id != 0 and gz_id <= len(GR_list):
                            gr_event = GR_list[gz_id - 1][1]
                            gr_event.channel = 'z'
                            new_block.gz = gr_event

                        # Add ADC event if it exists
                        if adc_id != 0 and adc_id <= len(ADC_list):
                            adc_event = ADC_list[adc_id - 1][1]
                            new_block.adc = adc_event

                        # Add LABEL event if it exists
                        if sq_label != 0:
                            # Check if the label is a string or an integer
                            if isinstance(sq_label, int):
                                sq_label_event = pp.make_label(label='SQID', type='SET', value=sq_label)
                            else:
                                raise ValueError(f"Invalid label type for SQ {sq_id}: {type(sq_label)}")
                            new_block.sq_label = sq_label_event


                        # if there are no events in the block and the duration is not zero, then it is a delay event
                        if (new_block.rf is None and new_block.gx is None and new_block.gy is None and new_block.gz is None and new_block.adc is None and new_block.sq_label is None) and dur != 0:
                            # If there are no events in the block and the duration is not zero, then it is a delay event
                            new_block.delay = pp.make_delay(float(dur * block_duration_raster))


                        # Check for the rt updates for this block and compute value based on  - strength + step * factor
                        # Find RT updates for this block and event
                        for rt_update in sq['SQ_rt_updates']:
                            if int(rt_update['Block_num']) == block_num:
                                event_name = rt_update['Event_name']
                                attribute = rt_update['Attribute']
                                strength = float(rt_update['Strength'])
                                step = float(rt_update['Step'])
                                value = strength + (step * factor)

                        # Update the corresponding event attribute in the block
                                if event_name == 'RF' and rf_id != 0 and rf_id <= len(RF_list):
                                    rf_event = RF_list[rf_id - 1][1]
                                    if hasattr(rf_event, attribute):
                                        # Convert degrees to radians for phase_offset or flip_angle
                                        if attribute in ['phase_offset', 'flip_angle']:
                                            setattr(rf_event, attribute, np.deg2rad(value))
                                        else:
                                            setattr(rf_event, attribute, value)
                                    new_block.rf = rf_event
                                elif event_name == 'GX' and gx_id != 0 and gx_id <= len(GR_list):
                                    gr_event = GR_list[gx_id - 1][1]
                                    if hasattr(gr_event, attribute):
                                        setattr(gr_event, attribute, value)
                                    new_block.gx = gr_event
                                elif event_name == 'GY' and gy_id != 0 and gy_id <= len(GR_list):
                                    gr_event = GR_list[gy_id - 1][1]
                                    if hasattr(gr_event, attribute):
                                        setattr(gr_event, attribute, value)
                                    new_block.gy = gr_event
                                elif event_name == 'GZ' and gz_id != 0 and gz_id <= len(GR_list):
                                    gr_event = GR_list[gz_id - 1][1]
                                    if hasattr(gr_event, attribute):
                                        setattr(gr_event, attribute, value)
                                    new_block.gz = gr_event
                                elif event_name == 'ADC' and adc_id != 0 and adc_id <= len(ADC_list):
                                    adc_event = ADC_list[adc_id - 1][1]
                                    if hasattr(adc_event, attribute):
                                        # Convert degrees to radians for phase_offset
                                        if attribute == 'phase_offset':
                                            setattr(adc_event, attribute, np.deg2rad(value))
                                        else:
                                            setattr(adc_event, attribute, value)
                                    new_block.adc = adc_event    
                                elif event_name == 'DELAY':
                                    # If the event is a delay, we can set the duration directly
                                    if dur != 0:
                                        new_block.delay = pp.make_delay(float(dur * block_duration_raster))




                    


                        # Add the block to the sequence
                        # seq.add_block(new_block.rf, new_block.gx, new_block.gy, new_block.gz, new_block.adc, new_block.aq_label)
                        # Prepare arguments for seq.add_block, only include non-None events

                        block_events = [None] * 7
                        block_duration = 0
                        if new_block.delay is not None:
                            block_events[0] = new_block.delay
                            block_duration = np.max([block_duration, pp.calc_duration(new_block.delay)])
                        if new_block.rf is not None:
                            block_events[1] = new_block.rf
                            block_duration = np.max([block_duration, pp.calc_duration(new_block.rf)])
                        if new_block.gx is not None:
                            block_events[2] = new_block.gx
                            block_duration = np.max([block_duration, pp.calc_duration(new_block.gx)])
                        if new_block.gy is not None:
                            block_events[3] = new_block.gy
                            block_duration = np.max([block_duration, pp.calc_duration(new_block.gy)])
                        if new_block.gz is not None:
                            block_events[4] = new_block.gz
                            block_duration = np.max([block_duration, pp.calc_duration(new_block.gz)])
                        if new_block.adc is not None:
                            block_events[5] = new_block.adc
                            block_duration = np.max([block_duration, pp.calc_duration(new_block.adc)])
                        if new_block.sq_label is not None:
                            block_events[6] = new_block.sq_label

                        # Check for block duration 
                        if ((float(dur) * block_duration_raster) - block_duration) > 1e-6:
                            # suspected delay block present along with other events
                            new_block.delay = pp.make_delay((float(dur) * block_duration_raster))
                            block_events[0] = new_block.delay

                        if block_events:
                            # seq.add_block(**block_args)
                            # Only include non-None events in the correct order
                            non_none_events = [event for event in block_events if event is not None]
                            if non_none_events:
                                seq.add_block(*non_none_events)
                
        # Now we have the sequence created from the SQs and their attributes
        # We can now write the sequence to a .seq file
        print(Fore.GREEN + f"Total TR duration: {TR_duration:.2f} seconds" + Style.RESET_ALL)
        if self.visualize:
            # Visualize the sequence

            seq.plot(time_range=[0, TR_duration * 2])  # Plot two TRs for better visualization
            print(Fore.GREEN + "Sequence visualized." + Style.RESET_ALL)

        # Only write the sequence if it contains at least one block
        if hasattr(seq, "block_events") and len(seq.block_events) > 0:
            seq.write(seq_file)
        else:
            print(Fore.RED + "No blocks found in the sequence. File not written." + Style.RESET_ALL)

        print(Fore.GREEN + f"Sequence written to {seq_file}" + Style.RESET_ALL)

                       
                       
 



               

           


