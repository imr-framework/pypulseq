from types import SimpleNamespace

import numpy as np

from pypulseq.Sequence.sequence import Sequence


def main(seq: Sequence):
    prospa = str()
    dict_grad_codes = {'x': 'n1', 'y': 'n3', 'z': 'n5'}
    dict_grad_shims = {'x': 'n10', 'y': 'n8', 'z': 'n6'}

    prospa += "initpp(dir)\n"
    prospa += "gradon(n1, n10)\n"
    prospa += "gradon(n3, n8)\n"
    prospa += "gradon(n5, n6)\n"
    prospa += "delay(d11)\n"
    dict_grad_latest_amp = {'x': 'n10', 'y': 'n8', 'z': 'n6'}
    dict_grad_new_amp = {'x': 'n2', 'y': 'n4', 'z': 'n6'}

    rf_flips = []

    for block_counter in range(1, len(seq.dict_block_events) + 1):
        block = seq.get_block(block_counter)
        attributes = set(dir(block)) - set(dir(SimpleNamespace))
        events = np.array([getattr(block, attr) for attr in attributes])
        ordered_delays = np.array([e.delay for e in events])
        sort_idx = np.argsort(ordered_delays)
        ordered_events = events[sort_idx]

        arr_grad_duration = np.zeros(len(ordered_events))
        dict_grad_duration_remaining = dict()
        open_grads = []
        time_elapsed = 0

        for j in range(len(ordered_events)):
            event = ordered_events[j]

            delay = event.delay  # initial delay
            if delay > 0 and delay > time_elapsed:
                delay = delay - time_elapsed
                delay = delay * 1e6  # Prospa has delay in nanoseconds; minimum is 250 ns
                prospa += f'delay({delay})'
                prospa += '\n'

            if event.type == 'trap':
                latest_amp = dict_grad_latest_amp[event.channel]
                new_amp = dict_grad_new_amp[event.channel]

                # Cycle through remaining events in this block and see if there is an ADC
                # If yes, we want to gradramp to a special amplitude
                idx_to_check = set(range(len(ordered_events))) - {j}
                for new_j in idx_to_check:
                    new_event = ordered_events[new_j]
                    if new_event.type == 'adc':
                        new_amp = 'n9'

                prospa += f'gradramp({dict_grad_codes[event.channel]}, ' \
                          f'{latest_amp}, ' \
                          f'{new_amp}, ' \
                          f'n12, d12)'
                prospa += '\n'
                arr_grad_duration[j] = event.rise_time + event.flat_time + event.fall_time
                dict_grad_latest_amp[event.channel] = new_amp
                dict_grad_new_amp[event.channel] = latest_amp

                # If we are in the last event and not in the last block in this iteration, then we add delay statement
                if j == len(ordered_events) - 1 and block_counter - 1 != len(seq.dict_block_events) - 1:
                    grad_duration = arr_grad_duration.min()
                    grad_duration = grad_duration * 1e6  # See earlier comment about delays
                    prospa += f'delay({grad_duration})'
                    prospa += '\n'

                    dict_grad_duration_remaining[event.channel] = arr_grad_duration.max() - arr_grad_duration.min()

                open_grads.append(event.channel)
            elif event.type == 'rf':
                signal = event.signal
                if not any([np.all(s == signal) for s in rf_flips]):  # Have we encountered this RF before?
                    rf_flips.append(signal)
                # Hardcoded as d1 after meeting with Tom on 12/10/2020 because it is a GUI-driven param
                duration = event.t[-1]
                prospa += f'pulse(mode, a{len(rf_flips)}, p{len(rf_flips)}, d1)'
                prospa += '\n'
                time_elapsed += duration
            elif event.type == 'adc':
                num_samples = event.num_samples
                prospa += f'acquire("overwrite", {num_samples})'
                prospa += '\n'

        for g in open_grads:
            if g not in dict_grad_duration_remaining:
                latest_amp = dict_grad_latest_amp[g]
                new_amp = dict_grad_shims[g]
                prospa += f'gradramp({dict_grad_codes[g]}, ' \
                          f'{latest_amp}, ' \
                          f'{new_amp}, ' \
                          f'n12, d12)'
                dict_grad_latest_amp[g] = new_amp
                dict_grad_new_amp[g] = latest_amp
                prospa += '\n'

        for g in open_grads:
            if g in dict_grad_duration_remaining:
                grad_duration = dict_grad_duration_remaining[g]
                grad_duration = grad_duration * 1e6  # See earlier comment about delays
                prospa += f'delay({grad_duration})'
                prospa += '\n'

                dict_grad_duration_remaining = dict()  # clear dictionary

                latest_amp = dict_grad_latest_amp[g]
                new_amp = dict_grad_shims[g]
                prospa += f'gradramp({dict_grad_codes[g]}, ' \
                          f'{latest_amp}, ' \
                          f'{new_amp}, ' \
                          f'n12, d12)'
                dict_grad_latest_amp[g] = new_amp
                dict_grad_new_amp[g] = latest_amp
                prospa += '\n'

    return prospa
