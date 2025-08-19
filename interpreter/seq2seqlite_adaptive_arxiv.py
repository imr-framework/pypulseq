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
        self.SQ_repeat_attributes = []  # List to hold SQ attributes
        print(Fore.GREEN + 'Sequence read successfully' + Style.RESET_ALL)
        print(Fore.YELLOW + 'The sequence has a total of {} blocks'.format(self.num_blocks) + Style.RESET_ALL)
        
        # Implement the logic in init itself
        self.blocks_to_SQs()   # Group blocks into SQs and assign memberships based on SQ label
        self.get_SOR_kernel()  # Get SOR kernel after grouping blocks into SQs
        self.get_SQ_details()  # Provides base SQs and SQ attribute changes
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
        for i in range(1, self.num_blocks +1):
            block = self.seq.get_block(i)
            block_event = self.seq.block_events[i]
            if block.label is not None:
                if 'SQID' in block.label[0].label:  
                    SQ_ind = int(block.label[0].value) - 1 # Extract SQID from the label

                    
                    if SQ_ind > len(self.SQs_blocks) - 1:
                        self.SQs_blocks.append([[]])  # Create a new SQ as a 2D list (for repeats and blocks)
                        self.SQ_repeats.append([])  # Initialize repeat count for the new SQ
                        self.SQs_block_events.append([[]])  # Initialize block events for the new SQ as a 2D list
                        self.SQ_repeats[SQ_ind] = 0
                    else:
                       self.SQ_repeats[SQ_ind] += 1
                    repeats = self.SQ_repeats[SQ_ind]

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
        # Apply special rules to SQs
        self.apply_special_rules()
        
    def apply_special_rules(self): #SQs_blocks, SQ_block_events, SOR
    # Placeholder for the method that applies special rules to SQs, such as EPI or TSE readout
    # Rule 1: All delay blocks need to be adjusted to the next block

        print(Fore.YELLOW + 'Applying special rules to SQs...' + Style.RESET_ALL)
        # Apply Rule 1: Adjust delay blocks
        self.SQs_blocks_compact, self.SQs_block_events_compact = self.apply_special_rule1()
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
            if (attr1.type == 'rf' or None) and attr2.type == 'rf':
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

            elif (attr1.type == 'adc' or None) and attr2.type == 'adc':
                if attr2.type ==None:
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
            if any(tr_events[block_num] != SQ_base_events[block_num]):
                events_different = [event != base_event for event, base_event in zip(tr_events[block_num], SQ_base_events[block_num])]
                events_different_indices = [index for index, value in enumerate(events_different) if value]
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
                        if tr_blocks[block_num].adc is None:
                            events_change_attributes = SQ_base_blocks[block_num].adc

                        if SQ_base_blocks[block_num].adc is None:
                            # Create a dummy ADC event with zeroed attributes
                            class DummyADC:
                                def __init__(self):
                                    self.type = 'adc'
                                    self.num_samples = 0
                                    self.phase_offset = 0

                            SQ_base_blocks[block_num].adc = DummyADC()
    
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
                return SQ_attributes
            else: # there are changes in the events, so we need to check for attributes
                SQ_attributes = get_SQ_updates(TRs_blocks, TRs_events, SQ_base)

            return SQ_attributes

        def get_SQ_updates(TRs_blocks, TRs_events, SQ_base):
        # This function checks for attributes that have changed in the blocks and events
        # It returns a dictionary with attribute names as keys and their values as lists of changes
            attributes_changed = {}
            event_names = ['', 'RF',  'GX',  'GY',  'GZ',  'ADC',  'LABEL']
            SQ_event_changes = {}

          
            for tr_idx, tr_events in enumerate(TRs_events):
                # Check if the events are identical to the base events
                if tr_idx > 0: # tr_idx == 0 is SQ_base
                    TR_ref_blocks = TRs_blocks[tr_idx - 1]
                    TR_ref_events = TRs_events[tr_idx - 1]
                    for block_num in range(len(tr_events)):
                        if any(tr_events[block_num] != TR_ref_events[block_num]):
                            events_different = [event != base_event for event, base_event in zip(tr_events[block_num], TR_ref_events[block_num])]
                            events_different_indices = [index for index, value in enumerate(events_different) if value]
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
                                    if TR_ref_blocks[block_num].adc is None:
                                        events_change_attributes = TRs_blocks[tr_idx][block_num].adc
                                    else:
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
                        # else:
                        #     print(Fore.REDs
                # Set the current TR's events and blocks as the reference for the next iteration


            return SQ_event_changes

        self.SQ_attributes = [[]]
        self.SQ_base = [{} for _ in range(len(self.SOR_kernel))]  # Dictionary to hold the base SQ for each SQ
        SQ_prev_repeat = [{} for _ in range(len(self.SOR_kernel))]  # Dictionary to hold the previous repeat attributes for each SQ
        self.SQ_repeat_num = np.zeros(len(self.SOR_kernel), dtype=int)  # Array to hold the repeat number for each SQ
        for kernel_repeat in range(self.SOR_kernel_repeats):
            for SQ_idx, SQ in enumerate(self.SOR_kernel):
                SQ_repeat_start = kernel_repeat * self.SOR_kernel_SQ_repeats[SQ]
                SQ_repeat_end = SQ_repeat_start + self.SOR_kernel_SQ_repeats[SQ]
                TRs_blocks = self.SQs_blocks_compact[SQ][SQ_repeat_start:SQ_repeat_end]
                TRs_events = self.SQs_block_events_compact[SQ][SQ_repeat_start:SQ_repeat_end]

                # Store the base SQ for each SQ
                if kernel_repeat == 0:
                    self.SQ_base[SQ_idx] = {
                        'blocks': TRs_blocks[0] if TRs_blocks else [],
                        'events': TRs_events[0] if TRs_events else []
                    }
                    SQ_prev_repeat[SQ_idx] = {
                        'blocks': TRs_blocks[0] if TRs_blocks else [],
                        'events': TRs_events[0] if TRs_events else []
                    }
                    self.SQ_repeat_num[SQ_idx] = 1
                else:
                    # Check if the SQ has changed from the previous repeat
                    events_equal = len(SQ_prev_repeat[SQ_idx]['events']) == len(TRs_events[0]) and all(
                        np.array_equal(e1, e2) for e1, e2 in zip(SQ_prev_repeat[SQ_idx]['events'], TRs_events[0])
                    )
                    
                    if not (events_equal):
                        self.get_SQ_repeat_attributes(TRs_blocks[0], TRs_events[0], SQ_prev_repeat[SQ_idx], SQ_idx, kernel_repeat) # produces self.SQ_repeat_attributes[SQ_idx] which is important for capturing changes between SQ repeats
                    else:
                        # the new repeat is identical to the previous one, so we can use the previous attributes
                        self.SQ_repeat_num[SQ_idx] += 1
                    SQ_prev_repeat[SQ_idx] = {
                        'blocks': TRs_blocks[0] if TRs_blocks else [],
                        'events': TRs_events[0] if TRs_events else []
                    }
                # TODO: The case of changes in SQ attributes between repeats need to be handled


                if len(self.SQ_attributes) <= SQ:
                    self.SQ_attributes.append([])
                if len(self.SQ_attributes[SQ]) <= kernel_repeat:
                    self.SQ_attributes[SQ].append([])
                self.SQ_attributes[SQ][kernel_repeat].append(get_SQ_attributes(TRs_blocks, TRs_events, self.SQ_base[SQ_idx]))

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


    def get_SQ_rt_attributes(self):

        def get_blocks_rt_attributes(blocks_set, SQ_idx, SQ_repeat = False):
            # expect changes every TR, so can obtain step by subtracting the first block from the second block and considering the first block as strength and the number of blocks as factor
            strength = []
            step = []
            factor = []
            attribute = []
            event_name_store = []
            units = []
            gamma = 42.576e3 # to convert it into mT/m
            for block_num in blocks_set:
                # Access the base event attributes for the given block_num and event_type
                # event_type is an integer index (e.g., 1=RF, 2=GX, etc.)
                event_names = list(blocks_set[block_num].keys())
                for event_name in event_names:
                    factor.append(len(blocks_set[block_num][event_name]) + 1)  # Number of changes in this block_num + 1 for the base block
                    base_block = self.SQ_base[SQ_idx]['blocks'][block_num]
                    event_name_store.append(str(block_num)+ ': ' + event_name)
                    if event_name == 'RF':
                        base_attr = base_block.rf
                        attr_changes = list(blocks_set[block_num][event_name][0]['events_change_attributes'].keys())
                        for attr_change in attr_changes:
                            if attr_change == 'freq_offset':
                                attribute.append(attr_change)
                                strength.append(base_attr.freq_offset)
                                step.append(blocks_set[block_num][event_name][0]['events_change_attributes']['freq_offset'])
                                units.append('Hz')
                            elif attr_change == 'phase_offset':
                                attribute.append(attr_change)
                                strength.append(np.rad2deg(base_attr.phase_offset))
                                step.append(np.rad2deg(blocks_set[block_num][event_name][0]['events_change_attributes']['phase_offset']))
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
                                strength.append(base_attr.amplitude/gamma)
                                step.append(blocks_set[block_num][event_name][0]['events_change_attributes']['amplitude']/gamma)
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
                                strength.append(base_attr.amplitude/gamma)
                                step.append(blocks_set[block_num][event_name][0]['events_change_attributes']['amplitude']/gamma)
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
                                strength.append(base_attr.amplitude/gamma)
                                step.append(blocks_set[block_num][event_name][0]['events_change_attributes']['amplitude']/gamma)
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
                        attr_changes = list(blocks_set[block_num][event_name][0]['events_change_attributes'].keys())
                        for attr_change in attr_changes:
                            if attr_change == 'num_samples':
                                attribute.append(attr_change)
                                strength.append(base_attr.num_samples)
                                step.append(blocks_set[block_num][event_name][0]['events_change_attributes']['num_samples'])
                                units.append('samples')
                            elif attr_change == 'phase_offset':
                                attribute.append(attr_change)
                                strength.append(np.rad2deg(base_attr.phase_offset))
                                step.append(np.rad2deg(blocks_set[block_num][event_name][0]['events_change_attributes']['phase_offset']))
                                units.append('degrees')
                            else:
                                strength.append(0)
                                step.append(0)
                # Now you can use base_attr for further processing
            # if SQ_repeat:
            #     factor = int(self.SOR_kernel_repeats / (len(self.SOR_kernel) * len(self.SOR_kernel_SQ_repeats)))  # TODO: Think about this to be more general

            rt_attributes = {
                'block_event_name': event_name_store,
                'attribute': attribute,
                'units': units,
                'strength': strength,
                'step': step,
                'factor': factor
            }

            return rt_attributes
            


        self.SQ_rt_attributes = {}
        self.SQ_repeat_rt_attributes = {
            SQ_idx: {kernel_repeat: {} for kernel_repeat in range(self.SOR_kernel_repeats)}
            for SQ_idx in range(len(self.SOR_kernel))
        }
        for SQ_idx in range(len(self.SOR_kernel)):
            for kernel_repeat in range(self.SOR_kernel_repeats):
                if 'changes' not in self.SQ_attributes[SQ_idx][kernel_repeat][0]:
                    self.SQ_rt_attributes[SQ_idx] = False
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

                    self.SQ_rt_attributes[SQ_idx] = get_blocks_rt_attributes(blocks_set, SQ_idx)
                    # Only proceed if self.SQ_repeat_attributes has entries for this SQ_idx
                    if len(self.SQ_repeat_attributes) > SQ_idx and len(self.SQ_repeat_attributes[SQ_idx]) > kernel_repeat:
                        if len(self.SQ_repeat_attributes[SQ_idx][kernel_repeat]) > 0:
                            if 'changes' not in self.SQ_repeat_attributes[SQ_idx][kernel_repeat]:
                                self.SQ_repeat_rt_attributes[SQ_idx][kernel_repeat] = False
                            else:
                                blocks_changes = self.SQ_repeat_attributes[SQ_idx][kernel_repeat]['changes']
                                blocks_set = {}
                                for change in blocks_changes:
                                    block_num = change['block_num']
                                    event_name = change['event_name']
                                    if block_num not in blocks_set:
                                        blocks_set[block_num] = {}
                                    if event_name not in blocks_set[block_num]:
                                        blocks_set[block_num][event_name] = []
                                    blocks_set[block_num][event_name].append(change)
                                    self.SQ_repeat_rt_attributes[SQ_idx][kernel_repeat] = get_blocks_rt_attributes(blocks_set, SQ_idx, SQ_repeat=True)

        # Clean up the SQ_rt_attributes and SQ_repeat_rt_attributes to remove empty entries
        self.SQ_rt_attributes = {k: v for k, v in self.SQ_rt_attributes.items() if v is not False}
        self.SQ_repeat_rt_attributes = {
            sq_idx: {k: v for k, v in kernel_repeats.items() if v is not False and len(v) > 0}
            for sq_idx, kernel_repeats in self.SQ_repeat_rt_attributes.items()
            if any(v is not False and len(v) > 0 for v in kernel_repeats.values())
        }

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


          