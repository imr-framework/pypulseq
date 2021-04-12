from types import SimpleNamespace

import numpy as np

from pypulseq.block_to_events import block_to_events
from pypulseq.calc_duration import calc_duration
from pypulseq.compress_shape import compress_shape
from pypulseq.decompress_shape import decompress_shape
from pypulseq.supported_labels import get_supported_labels


def add_block(self, block_index: int, *args: SimpleNamespace) -> None:
    """
    Inserts PyPulseq block of sequence events into `self.dict_block_events` at position `block_index`. Also performs
    gradient checks.

    Parameters
    ----------
    block_index : int
        Index at which `SimpleNamespace` objects have to be inserted into `self.dict_block_events`.
    args : iterable[SimpleNamespace]
        Iterable of `SimpleNamespace` objects to be added to `self.dict_block_events`.

    Raises
    ------
    ValueError
        If trigger event that is passed is of unsupported control event type.
        If delay is set for a gradient even that starts with a non-zero amplitude.
    RuntimeError
        If two consecutive gradients to not have the same amplitude at the connection point.
        If the first gradient in the block does not start with 0.
        If a gradient that doesn't end at zero is not aligned to the block boundary.
    """
    events = block_to_events(args)
    block_duration = calc_duration(*events)
    self.dict_block_events[block_index] = np.zeros(7, dtype=np.int)
    duration = 0

    check_g = {}  # Key-value mapping of index and  pairs of gradients/times
    extensions = []

    for event in events:
        if event.type == 'rf':
            mag = np.abs(event.signal)
            amplitude = np.max(mag)
            mag = np.divide(mag, amplitude)
            # Following line of code is a workaround for numpy's divide functions returning NaN when mathematical
            # edge cases are encountered (eg. divide by 0)
            mag[np.isnan(mag)] = 0
            phase = np.angle(event.signal)
            phase[phase < 0] += 2 * np.pi
            phase /= 2 * np.pi

            mag_shape = compress_shape(mag)
            data = np.insert(mag_shape.data, 0, mag_shape.num_samples)
            mag_id, found = self.shape_library.find(data)
            if not found:
                self.shape_library.insert(mag_id, data)

            phase_shape = compress_shape(phase)
            data = np.insert(phase_shape.data, 0, phase_shape.num_samples)
            phase_id, found = self.shape_library.find(data)
            if not found:
                self.shape_library.insert(phase_id, data)

            use = 0
            use_cases = {'excitation': 1, 'refocusing': 2, 'inversion': 3}
            if hasattr(event, 'use'):
                use = use_cases[event.use]

            data = [amplitude, mag_id, phase_id, event.delay, event.freq_offset, event.phase_offset, event.dead_time,
                    event.ringdown_time, use]
            data_id, found = self.rf_library.find(data)
            if not found:
                self.rf_library.insert(data_id, data)

            self.dict_block_events[block_index][1] = data_id
            duration = max(duration, len(mag) * self.rf_raster_time + event.delay)
        elif event.type == 'grad':
            channel_num = ['x', 'y', 'z'].index(event.channel)
            idx = 2 + channel_num

            check_g[channel_num] = SimpleNamespace()
            check_g[channel_num].idx = idx
            check_g[channel_num].start = np.array((event.delay + min(event.t), event.first))
            check_g[channel_num].stop = np.array(
                (event.delay + max(event.t) + self.system.grad_raster_time, event.last))

            amplitude = max(abs(event.waveform))
            if amplitude > 0:
                g = event.waveform / amplitude
            else:
                g = event.waveform
            shape = compress_shape(g)
            data = np.insert(shape.data, 0, shape.num_samples)
            shape_id, found = self.shape_library.find(data)
            if not found:
                self.shape_library.insert(shape_id, data)
            data = [amplitude, shape_id, event.delay, event.first, event.last]
            grad_id, found = self.grad_library.find(data)
            if not found:
                self.grad_library.insert(grad_id, data, 'g')
            self.dict_block_events[block_index][idx] = grad_id
            duration = max(duration, event.delay + len(g) * self.grad_raster_time)
        elif event.type == 'trap':
            channel_num = ['x', 'y', 'z'].index(event.channel)
            idx = 2 + channel_num

            check_g[channel_num] = SimpleNamespace()
            check_g[channel_num].idx = idx
            check_g[channel_num].start = np.array((0, 0))
            check_g[channel_num].stop = np.array((event.delay + event.rise_time + event.fall_time + event.flat_time, 0))

            data = [event.amplitude, event.rise_time, event.flat_time, event.fall_time, event.delay]
            trap_id, found = self.grad_library.find(data)
            if not found:
                self.grad_library.insert(trap_id, data, 't')
            self.dict_block_events[block_index][idx] = trap_id
            duration = max(duration, event.delay + event.rise_time + event.flat_time + event.fall_time)
        elif event.type == 'adc':
            data = [event.num_samples, event.dwell, max(event.delay, event.dead_time), event.freq_offset,
                    event.phase_offset, event.dead_time]
            adc_id, found = self.adc_library.find(data)
            if not found:
                self.adc_library.insert(adc_id, data)
            self.dict_block_events[block_index][5] = adc_id
            duration = max(duration, event.delay + event.num_samples * event.dwell + event.dead_time)
        elif event.type == 'delay':
            data = [event.delay]
            delay_id, found = self.delay_library.find(data)
            if not found:
                self.delay_library.insert(delay_id, data)
            self.dict_block_events[block_index][0] = delay_id
            duration = max(duration, event.delay)
        elif event.type == 'output' or event.type == 'trigger':
            event_type = ['output', 'trigger'].index(event.type) + 1
            if event_type == 1:
                # Trigger codes supported by the Siemens interpreter as of May 2019
                event_channel = ['osc0', 'osc1', 'ext1'].index(event.channel) + 1
            elif event_type == 2:
                # Trigger codes supported by the Siemens interpreter as of June 2019
                event_channel = ['physio1', 'physio2'].index(event.channel) + 1
            else:
                raise ValueError('Unsupported control event type.')

            data = [event_type, event_channel, event.delay, event.duration]
            trigger_id, found = self.trigger_library.find(data)
            if not found:
                self.trigger_library.insert(trigger_id, data)

            # Now we collect the list of extension objects and we will add it to the event table later
            ext = {'type': self.get_extension_type_ID('TRIGGERS'), 'ref': trigger_id}
            extensions.append(ext)
            duration = max(duration, event.delay + event.duration)
        elif event.type == 'labelset':
            label_id = get_supported_labels().index(event.label) + 1
            data = [event.value, label_id]
            label_id2, found = self.label_set_library.find(data)
            if not found:
                self.label_set_library.insert(label_id2, data)

            ext = {'type': self.get_extension_type_ID('LABELSET'), 'ref': label_id2}
            extensions.append(ext)
        elif event.type == 'labelinc':
            label_id = get_supported_labels().index(event.label) + 1
            data = [event.value, label_id]
            label_id2, found = self.label_inc_library.find(data)
            if not found:
                self.label_inc_library.insert(label_id2, data)

            ext = {'type': self.get_extension_type_ID('LABELINC'), 'ref': label_id2}
            extensions.append(ext)

    # =========
    # ADD EXTENSIONS
    # =========
    if len(extensions) > 0:
        """
        Add extensions now... but it's tricky actually we need to check whether the exactly the same list of extensions 
        already exists, otherwise we have to create a new one... ooops, we have a potential problem with the key 
        mapping then... The trick is that we rely on the sorting of the extension IDs and then we can always find the 
        last one in the list by setting the reference to the next to 0 and then proceed with the other elements.
        """
        sort_idx = np.argsort([e['ref'] for e in extensions])
        extensions = np.take(extensions, sort_idx)
        all_found = True
        extension_id = 0
        for i in range(len(extensions)):
            data = [extensions[i]['type'], extensions[i]['ref'], extension_id]
            extension_id, found = self.extensions_library.find(data)
            all_found = all_found and found
            if not found:
                break

        if not all_found:
            # Add the list
            extension_id = 0
            for i in range(len(extensions)):
                data = [extensions[i]['type'], extensions[i]['ref'], extension_id]
                extension_id, found = self.extensions_library.find(data)
                if not found:
                    self.extensions_library.insert(extension_id, data)

        # Now we add the ID
        self.dict_block_events[block_index][6] = extension_id

    # =========
    # PERFORM GRADIENT CHECKS
    # =========
    for grad_to_check in check_g.values():

        if abs(grad_to_check.start[1]) > self.system.max_slew * self.system.grad_raster_time:
            if grad_to_check.start[0] != 0:
                raise ValueError('No delay allowed for gradients which start with a non-zero amplitude')

            if block_index > 1:
                prev_id = self.dict_block_events[block_index - 1][grad_to_check.idx]
                if prev_id != 0:
                    prev_lib = self.grad_library.get(prev_id)
                    prev_dat = prev_lib['data']
                    prev_type = prev_lib['type']
                    if prev_type == 't':
                        raise RuntimeError(
                            'Two consecutive gradients need to have the same amplitude at the connection point')
                    elif prev_type == 'g':
                        last = prev_dat[4]
                        if abs(last - grad_to_check.start[1]) > self.system.max_slew * self.system.grad_raster_time:
                            raise RuntimeError(
                                'Two consecutive gradients need to have the same amplitude at the connection point')
            else:
                raise RuntimeError('First gradient in the the first block has to start at 0.')

        if grad_to_check.stop[1] > self.system.max_slew * self.system.grad_raster_time and abs(
                grad_to_check.stop[0] - block_duration) > 1e-7:
            raise RuntimeError("A gradient that doesn't end at zero needs to be aligned to the block boundary.")

    eps = np.finfo(np.float).eps
    assert abs(duration - block_duration) < eps
    self.arr_block_durations.append(block_duration)


def get_block(self, block_index: int) -> SimpleNamespace:
    """
    Returns PyPulseq block at `block_index` position in `self.dict_block_events`.

    Parameters
    ----------
    block_index : int
        Index of PyPulseq block to be retrieved from `self.dict_block_events`.

    Returns
    -------
    block : SimpleNamespace
        PyPulseq block at 'block_index' position in `self.dict_block_events`.

    Raises
    ------
    ValueError
        If a trigger event of an unsupported control type is encountered.
        If a label object of an unknown extension ID is encountered.
    """

    block = SimpleNamespace()
    event_ind = self.dict_block_events[block_index]

    if event_ind[0] > 0:  # Delay
        delay = SimpleNamespace()
        delay.type = 'delay'
        delay.delay = self.delay_library.data[event_ind[0]][0]
        block.delay = delay

    if event_ind[1] > 0:  # RF
        block.rf = self.rf_from_lib_data(self.rf_library.data[event_ind[1]])

    # Gradients
    grad_channels = ['gx', 'gy', 'gz']
    for i in range(1, len(grad_channels) + 1):
        if event_ind[2 + (i - 1)] > 0:
            grad, compressed = SimpleNamespace(), SimpleNamespace()
            grad_type = self.grad_library.type[event_ind[2 + (i - 1)]]
            lib_data = self.grad_library.data[event_ind[2 + (i - 1)]]
            grad.type = 'trap' if grad_type == 't' else 'grad'
            grad.channel = grad_channels[i - 1][1]
            if grad.type == 'grad':
                amplitude = lib_data[0]
                shape_id = lib_data[1]
                delay = lib_data[2]
                shape_data = self.shape_library.data[shape_id]
                compressed.num_samples = shape_data[0]
                compressed.data = shape_data[1:]
                g = decompress_shape(compressed)
                grad.waveform = amplitude * g
                grad.t = np.arange(g.size) * self.grad_raster_time
                grad.delay = delay
                if len(lib_data) > 4:
                    grad.first = lib_data[3]
                    grad.last = lib_data[4]
                else:
                    grad.first = grad.waveform[0]
                    grad.last = grad.waveform[-1]
            else:
                if max(lib_data.shape) < 5:  # added by GT
                    grad.amplitude, grad.rise_time, grad.flat_time, grad.fall_time = [lib_data[x] for x in range(4)]
                    grad.delay = 0
                else:
                    grad.amplitude, grad.rise_time, grad.flat_time, grad.fall_time, grad.delay = [lib_data[x] for x in
                                                                                                  range(5)]
                grad.area = grad.amplitude * (grad.flat_time + grad.rise_time / 2 + grad.fall_time / 2)
                grad.flat_area = grad.amplitude * grad.flat_time
            setattr(block, grad_channels[i - 1], grad)
    # ADC
    if event_ind[5] > 0:
        lib_data = self.adc_library.data[event_ind[5]]
        if len(lib_data) < 6:
            lib_data = np.append(lib_data, 0)

        adc = SimpleNamespace()
        adc.num_samples, adc.dwell, adc.delay, adc.freq_offset, adc.phase_offset, adc.dead_time = [lib_data[x] for x in
                                                                                                   range(6)]
        adc.num_samples = int(adc.num_samples)
        adc.type = 'adc'
        block.adc = adc

    # Triggers
    if event_ind[6] > 0:
        # We have extensions - triggers, labels, etc.
        next_ext_id = event_ind[6]
        while next_ext_id != 0:
            ext_data = self.extensions_library.data[next_ext_id]
            # Format: ext_type, ext_id, next_ext_id
            ext_type = self.get_extension_type_string(ext_data[0])

            if ext_type == 'TRIGGERS':
                trigger_types = ['output', 'trigger']
                data = self.trigger_library.data[ext_data[1]]
                trigger = SimpleNamespace()
                trigger.type = trigger_types[int(data[0])]
                if data[0] == 0:
                    trigger_channels = ['osc0', 'osc1', 'ext1']
                    trigger.channel = trigger_channels[int(data[1])]
                elif data[0] == 1:
                    trigger_channels = ['physio1', 'physio2']
                    trigger.channel = trigger_channels[int(data[1])]
                else:
                    raise ValueError('Unsupported trigger event type')

                trigger.delay = data[2]
                trigger.duration = data[3]
                # Allow for multiple triggers per block
                if hasattr(block, 'trigger'):
                    block.trigger[len(block.trigger)] = trigger
                else:
                    block.trigger = {0: trigger}
            elif ext_type == 'LABELSET' or ext_type == 'LABELINC':
                label = SimpleNamespace()
                label.type = ext_type.lower()
                supported_labels = get_supported_labels()
                if ext_type == 'LABELSET':
                    data = self.label_set_library.data[ext_data[1]]
                else:
                    data = self.label_inc_library.data[ext_data[1]]

                label.label = supported_labels[data[1] - 1]
                label.value = data[0]
                # Allow for multiple labels per block
                if hasattr(block, 'label'):
                    block.label[len(block.label)] = label
                else:
                    block.label = {0: label}
            else:
                raise RuntimeError(f'Unknown extension ID {ext_data[0]}')

            next_ext_id = ext_data[2]

    return block
