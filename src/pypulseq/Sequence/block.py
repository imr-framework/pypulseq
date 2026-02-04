import math
import warnings
from types import SimpleNamespace
from typing import List, Tuple, Union

import numpy as np

from pypulseq.block_to_events import block_to_events
from pypulseq.compress_shape import compress_shape
from pypulseq.decompress_shape import decompress_shape
from pypulseq.event_lib import EventLibrary
from pypulseq.Sequence.ext_grad_check import ext_grad_check
from pypulseq.supported_labels_rf_use import get_supported_labels
from pypulseq.utils.tracing import trace_enabled


def set_block(self, block_index: int, *args: Union[SimpleNamespace, float]) -> None:
    """
    Replace block at index with new block provided as block structure, add sequence block, or create a new block
    from events and store at position specified by index. The block or events are provided in uncompressed form and
    will be stored in the compressed, non-redundant internal libraries.

    See Also
    --------
    - `pypulseq.Sequence.sequence.Sequence.get_block()`
    - `pypulseq.Sequence.sequence.Sequence.add_block()`

    Parameters
    ----------
    block_index : int
        Index at which block is replaced.
    args : SimpleNamespace
        Block or events to be replaced/added or created at `block_index`.
        If a floating point number is provided, it is interpreted as the duration of the block.

    Raises
    ------
    ValueError
        If trigger event that is passed is of unsupported control event type.
        If delay is set for a gradient even that starts with a non-zero amplitude.
    RuntimeError
        If two consecutive gradients to not have the same amplitude at the connection point.
        If the first gradient in the block does not start with 0.
        If a gradient that doesn't end at zero is not aligned to the block boundary.
        If multiple soft_delay extensions are used in a block.
        If a soft delay extension is used in a block of zero duration.
        If a soft delay extension is used in a block containing conventional events.
    """
    events = block_to_events(*args)
    new_block = np.zeros(7, dtype=np.int32)
    duration = 0

    check_g = {
        0: SimpleNamespace(start=(0, 0), stop=(0, 0)),
        1: SimpleNamespace(start=(0, 0), stop=(0, 0)),
        2: SimpleNamespace(start=(0, 0), stop=(0, 0)),
    }  # Key-value mapping of index and  pairs of gradients/times
    extensions = []

    for event in events:
        if not isinstance(event, float):  # If event is not a block duration
            if event.type == 'rf':
                if new_block[1] != 0:
                    raise ValueError('Multiple RF events were specified in set_block')

                if hasattr(event, 'id'):
                    rf_id = event.id
                else:
                    rf_id, _ = register_rf_event(self, event)

                new_block[1] = rf_id
                duration = max(duration, event.shape_dur + event.delay + event.ringdown_time)

                if trace_enabled() and hasattr(event, 'trace'):
                    self.block_trace[block_index].rf = event.trace
            elif event.type == 'grad':
                channel_num = ['x', 'y', 'z'].index(event.channel)
                idx = 2 + channel_num

                if new_block[idx] != 0:
                    raise ValueError(f'Multiple {event.channel.upper()} gradient events were specified in set_block')

                grad_start = (
                    event.delay + math.floor(event.tt[0] / self.grad_raster_time + 1e-10) * self.grad_raster_time
                )
                grad_duration = (
                    event.delay + math.ceil(event.tt[-1] / self.grad_raster_time - 1e-10) * self.grad_raster_time
                )

                check_g[channel_num] = SimpleNamespace()
                check_g[channel_num].idx = idx
                check_g[channel_num].start = (grad_start, event.first)
                check_g[channel_num].stop = (grad_duration, event.last)

                if hasattr(event, 'id'):
                    grad_id = event.id
                else:
                    grad_id, _ = register_grad_event(self, event)

                new_block[idx] = grad_id
                duration = max(duration, grad_duration)

                if trace_enabled() and hasattr(event, 'trace'):
                    setattr(self.block_trace[block_index], 'g' + event.channel, event.trace)
            elif event.type == 'trap':
                channel_num = ['x', 'y', 'z'].index(event.channel)
                idx = 2 + channel_num

                if new_block[idx] != 0:
                    raise ValueError(f'Multiple {event.channel.upper()} gradient events were specified in set_block')

                if hasattr(event, 'id'):
                    trap_id = event.id
                else:
                    trap_id = register_grad_event(self, event)

                new_block[idx] = trap_id
                duration = max(duration, event.delay + event.rise_time + event.flat_time + event.fall_time)

                if trace_enabled() and hasattr(event, 'trace'):
                    setattr(self.block_trace[block_index], 'g' + event.channel, event.trace)
            elif event.type == 'adc':
                if new_block[5] != 0:
                    raise ValueError('Multiple ADC events were specified in set_block')

                if hasattr(event, 'id'):
                    adc_id = event.id
                else:
                    adc_id, _ = register_adc_event(self, event)

                new_block[5] = adc_id
                duration = max(duration, event.delay + event.num_samples * event.dwell + event.dead_time)

                if trace_enabled() and hasattr(event, 'trace'):
                    self.block_trace[block_index].adc = event.trace
            elif event.type == 'delay':
                duration = max(duration, event.delay)
            elif event.type in ['output', 'trigger']:
                if hasattr(event, 'id'):
                    event_id = event.id
                else:
                    event_id = register_control_event(self, event)

                ext = {'type': self.get_extension_type_ID('TRIGGERS'), 'ref': event_id}
                extensions.append(ext)
                duration = max(duration, event.delay + event.duration)
            elif event.type in ['labelset', 'labelinc']:
                if hasattr(event, 'id'):
                    label_id = event.id
                else:
                    label_id = register_label_event(self, event)

                ext = {
                    'type': self.get_extension_type_ID(event.type.upper()),
                    'ref': label_id,
                }
                extensions.append(ext)
            elif event.type == 'soft_delay':
                if hasattr(event, 'id'):
                    event_id = event.id
                else:
                    event_id = register_soft_delay_event(self, event)

                duration = max(duration, event.default_duration)
                ext = {'type': self.get_extension_type_ID('DELAYS'), 'ref': event_id}
                extensions.append(ext)
            else:
                raise ValueError(f'Unknown event type {event.type} passed to set_block().')
        else:
            # Delay given as floating number (internal use only, e.g., from get_block())
            duration = max(duration, event)

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
            data = (extensions[i]['type'], extensions[i]['ref'], extension_id)
            extension_id, found = self.extensions_library.find(data)
            all_found = all_found and found
            if not found:
                break

        if not all_found:
            # Add the list
            extension_id = 0
            for i in range(len(extensions)):
                data = (extensions[i]['type'], extensions[i]['ref'], extension_id)
                extension_id, found = self.extensions_library.find(data)
                if not found:
                    self.extensions_library.insert(extension_id, data)

        # Sanity checks for the soft delays
        if 'DELAYS' in self.extension_string_idx:
            n_soft_delays = sum([1 for e in extensions if e['type'] == self.get_extension_type_ID('DELAYS')])
            if n_soft_delays:
                if n_soft_delays > 1:
                    raise RuntimeError('Only one soft delay extension is allowed per block.')
                if not duration:
                    raise RuntimeError(
                        'Soft delay extension can only be used in conjunction with blocks of non-zero duration.'
                    )  # otherwise the gradient checks get tedious
                if new_block[1:5].any():
                    raise RuntimeError(
                        'Soft delay extension can only be used in empty blocks (blocks containing no conventional events such as RF, adc or gradients).'
                    )
        # Now we add the ID
        new_block[6] = extension_id

    # =========
    # PERFORM GRADIENT CHECKS
    # =========
    ext_grad_check(self, block_index, check_g, duration)

    self.block_events[block_index] = new_block
    self.block_durations[block_index] = float(duration)


def get_raw_block_content_IDs(self, block_index: int) -> SimpleNamespace:
    """
    Returns PyPulseq block content IDs at `block_index` position in `self.block_events`.

    No block events are created, only the IDs of the objects are returned.

    Parameters
    ----------
    block_index : int
        Index of PyPulseq block to be retrieved from `self.block_events`.

    Returns
    -------
    block : SimpleNamespace
        PyPulseq block content IDs at 'block_index' position in `self.block_events`.
    """
    raw_block = SimpleNamespace(block_duration=0, rf=0, gx=0, gy=0, gz=0, adc=0, ext=[])
    event_ind = self.block_events[block_index]

    # Extensions
    if event_ind[6] > 0:
        next_ext_id = event_ind[6]
        while next_ext_id != 0:
            ext_data = self.extensions_library.data[next_ext_id]
            raw_block.ext.append(ext_data[:2])
            next_ext_id = ext_data[2]
        raw_block.ext = np.stack(raw_block.ext, axis=-1)

    # RF
    if event_ind[1] > 0:
        raw_block.rf = event_ind[1]

    # Gradients
    grad_channels = ['gx', 'gy', 'gz']
    for i in range(len(grad_channels)):
        if event_ind[2 + i] > 0:
            setattr(raw_block, grad_channels[i], event_ind[2 + i])

    # ADC
    if event_ind[5] > 0:
        raw_block.adc = event_ind[5]

    return raw_block


def get_block(self, block_index: int, add_IDs: bool = False) -> SimpleNamespace:
    """
    Returns PyPulseq block at `block_index` position in `self.block_events`.

    The block is created from the sequence data with all events and shapes decompressed.

    Parameters
    ----------
    block_index : int
        Index of PyPulseq block to be retrieved from `self.block_events`.
    add_IDs : bool, optional
        Add IDs to block structure. The default is `False`.

    Returns
    -------
    block : SimpleNamespace
        PyPulseq block at 'block_index' position in `self.block_events`.

    Raises
    ------
    ValueError
        If a trigger event of an unsupported control type is encountered.
        If a label object of an unknown extension ID is encountered.
    """
    # Check if block exists in the block cache. If so, return that
    if self.use_block_cache and block_index in self.block_cache:
        return self.block_cache[block_index]

    block = SimpleNamespace()
    attrs = ['block_duration', 'rf', 'gx', 'gy', 'gz', 'adc', 'label', 'soft_delay']
    values = [None] * len(attrs)
    for att, val in zip(attrs, values):
        setattr(block, att, val)
    raw_block = get_raw_block_content_IDs(self, block_index)

    # Extensions
    if len(raw_block.ext) > 0:
        # We have extensions - triggers, labels, etc.
        # Format: ext_type, ext_id, next_ext_id

        # Unpack trigger(s)
        trig_ext = raw_block.ext[1, raw_block.ext[0] == self.get_extension_type_ID('TRIGGERS', update=False)]
        if trig_ext.shape[-1] > 0:
            trigger_types = ['output', 'trigger']
            for i in range(trig_ext.shape[-1]):
                data = self.trigger_library.data[trig_ext[i]]
                trigger = SimpleNamespace()
                trigger.type = trigger_types[int(data[0]) - 1]
                if data[0] == 1:
                    trigger_channels = ['osc0', 'osc1', 'ext1']
                    trigger.channel = trigger_channels[int(data[1]) - 1]
                elif data[0] == 2:
                    trigger_channels = ['physio1', 'physio2']
                    trigger.channel = trigger_channels[int(data[1]) - 1]
                else:
                    raise ValueError('Unsupported trigger event type')
                trigger.delay = data[2]
                trigger.duration = data[3]
                if add_IDs:
                    trigger.id = trig_ext[i]
                # Allow for multiple triggers per block
                if hasattr(block, 'trigger'):
                    block.trigger[i] = trigger
                else:
                    block.trigger = {0: trigger}

        # Unpack labels
        lid_set = self.get_extension_type_ID('LABELSET', update=False)
        lid_inc = self.get_extension_type_ID('LABELINC', update=False)
        supported_labels = get_supported_labels()
        label_ext = raw_block.ext[:, np.logical_or(raw_block.ext[0] == lid_set, raw_block.ext[0] == lid_inc)]
        if label_ext.shape[-1] > 0:
            for i in range(label_ext.shape[-1]):
                label = SimpleNamespace()
                if label_ext[0, i] == lid_set:
                    label.type = 'labelset'
                    data = self.label_set_library.data[label_ext[1, i]]
                else:
                    label.type = 'labelinc'
                    data = self.label_inc_library.data[label_ext[1, i]]
                label.label = supported_labels[int(data[1] - 1)]
                label.value = data[0]
                if add_IDs:
                    label.id = label_ext[1, i]
                # Allow for multiple labels per block
                if block.label is not None:
                    block.label[i] = label
                else:
                    block.label = {0: label}

        # Reverse the order of labels, because extensions are saved as a reversed linked list
        if block.label is not None:
            block.label = dict(enumerate(reversed(block.label.values())))

        # Unpack Soft Delays
        delay_ext = raw_block.ext[:, raw_block.ext[0] == self.get_extension_type_ID('DELAYS', update=False)]
        if delay_ext.shape[-1] > 0:
            if delay_ext.shape[-1] > 1:
                raise ValueError('Only one soft delay extension object per block is allowed')
            data = self.soft_delay_library.data[delay_ext[1].item()]
            block.soft_delay = SimpleNamespace()
            block.soft_delay.type = 'soft_delay'
            block.soft_delay.numID = data[0]
            block.soft_delay.offset = data[1]
            block.soft_delay.factor = data[2]
            block.soft_delay.hint = data[3]
            block.soft_delay.default_duration = self.block_durations[block_index]
            if add_IDs:
                block.soft_delay.id = delay_ext[1].item()

        if trig_ext.shape[-1] + label_ext.shape[-1] != raw_block.ext.shape[-1]:
            for i in range(raw_block.ext.shape[1]):
                ext_id = raw_block.ext[0, i]
                if (
                    ext_id != self.get_extension_type_ID('TRIGGERS', update=False)
                    and ext_id != self.get_extension_type_ID('LABELSET', update=False)
                    and ext_id != self.get_extension_type_ID('LABELINC', update=False)
                    and ext_id != self.get_extension_type_ID('DELAYS', update=False)
                ):
                    warnings.warn(f'Unknown extension ID {ext_id}')

    # RF
    if raw_block.rf:  # RF
        if len(self.rf_library.type) >= raw_block.rf:
            block.rf = self.rf_from_lib_data(self.rf_library.data[raw_block.rf], self.rf_library.type[raw_block.rf])
        else:
            block.rf = self.rf_from_lib_data(self.rf_library.data[raw_block.rf], 'u')  # Undefined type/use
        if add_IDs:
            block.rf.id = raw_block.rf

    # Gradients
    grad_channels = ['gx', 'gy', 'gz']
    for i in range(len(grad_channels)):
        grad_id = getattr(raw_block, grad_channels[i])

        if grad_id:
            grad, compressed = SimpleNamespace(), SimpleNamespace()
            grad_type = self.grad_library.type[grad_id]
            lib_data = self.grad_library.data[grad_id]
            grad.type = 'trap' if grad_type == 't' else 'grad'
            grad.channel = grad_channels[i][1]
            if grad.type == 'grad':
                amplitude = lib_data[0]
                shape_id = lib_data[3]  # change in v150: changed from lib_data[1] to lib_data[3]
                time_id = lib_data[4]  # change in v150: changed from lib_data[2] to lib_data[4]
                delay = lib_data[5]  # change in v150: changed from lib_data[3] to lib_data[5]
                shape_data = self.shape_library.data[shape_id]
                compressed.num_samples = shape_data[0]
                compressed.data = shape_data[1:]
                g = decompress_shape(compressed)
                grad.waveform = amplitude * g

                if time_id == 0:
                    grad.tt = (np.arange(1, len(g) + 1) - 0.5) * self.grad_raster_time
                    t_end = len(g) * self.grad_raster_time
                    grad.area = sum(grad.waveform) * self.grad_raster_time
                elif time_id == -1:
                    # Gradient with oversampling by a factor of 2
                    grad.tt = 0.5 * (np.arange(1, len(g) + 1)) * self.grad_raster_time
                    if len(grad.tt) != len(grad.waveform):
                        raise ValueError(
                            f'Mismatch between time shape length ({len(grad.tt)}) and gradient shape length ({len(grad.waveform)}).'
                        )
                    if len(grad.waveform) % 2 != 1:
                        raise ValueError('Oversampled gradient waveforms must have odd number of samples')
                    t_end = (len(g) + 1) * self.grad_raster_time
                    grad.area = sum(grad.waveform[::2]) * self.grad_raster_time  # remove oversampling
                else:
                    t_shape_data = self.shape_library.data[time_id]
                    compressed.num_samples = t_shape_data[0]
                    compressed.data = t_shape_data[1:]
                    grad.tt = decompress_shape(compressed) * self.grad_raster_time
                    if len(grad.tt) != len(grad.waveform):
                        raise ValueError(
                            f'Mismatch between time shape length ({len(grad.tt)}) and gradient shape length ({len(grad.waveform)}).'
                        )
                    t_end = grad.tt[-1]
                    grad.area = 0.5 * sum((grad.tt[1:] - grad.tt[:-1]) * (grad.waveform[1:] + grad.waveform[:-1]))

                grad.shape_id = shape_id
                grad.time_id = time_id
                grad.delay = delay
                grad.shape_dur = t_end
                grad.first = lib_data[1]  # change in v150 - we always have first/last now
                grad.last = lib_data[2]  # change in v150 - we always have first/last now
                if add_IDs:
                    grad.shape_IDs = [shape_id, time_id]
            else:
                grad.amplitude = lib_data[0]
                grad.rise_time = lib_data[1]
                grad.flat_time = lib_data[2]
                grad.fall_time = lib_data[3]
                grad.delay = lib_data[4]
                grad.area = grad.amplitude * (grad.flat_time + grad.rise_time / 2 + grad.fall_time / 2)
                grad.flat_area = grad.amplitude * grad.flat_time

            if add_IDs:
                grad.id = grad_id

            setattr(block, grad_channels[i], grad)

    # ADC
    if raw_block.adc:
        lib_data = self.adc_library.data[raw_block.adc]
        shape_id_phase_modulation = lib_data[7]
        if shape_id_phase_modulation:
            shape_data = self.shape_library.data[shape_id_phase_modulation]
            compressed = SimpleNamespace()
            compressed.num_samples = shape_data[0]
            compressed.data = shape_data[1:]
            phase_shape = decompress_shape(compressed)
        else:
            phase_shape = np.array([], dtype=float)

        adc = SimpleNamespace()
        adc.num_samples = lib_data[0]
        adc.dwell = lib_data[1]
        adc.delay = lib_data[2]
        adc.freq_ppm = lib_data[3]
        adc.phase_ppm = lib_data[4]
        adc.freq_offset = lib_data[5]
        adc.phase_offset = lib_data[6]
        adc.phase_modulation = phase_shape
        adc.dead_time = self.system.adc_dead_time
        adc.num_samples = int(adc.num_samples)
        adc.type = 'adc'

        if add_IDs:
            adc.id = raw_block.adc

        block.adc = adc

    block.block_duration = self.block_durations[block_index]

    # Enter block into the block cache
    if self.use_block_cache:
        self.block_cache[block_index] = block

    return block


def register_adc_event(self, event: EventLibrary) -> Tuple[int, int]:
    """

    Parameters
    ----------
    event : SimpleNamespace
        ADC event to be registered.

    Returns
    -------
    int, int
        ID of registered ADC event, shape ID
    """
    surely_new = False

    # Handle phase modulation
    if not hasattr(event, 'phase_modulation') or event.phase_modulation is None or len(event.phase_modulation) == 0:
        shape_id = 0
    else:
        if hasattr(event, 'shape_id'):
            shape_id = event.shape_id
        else:
            phase_shape = compress_shape(np.asarray(event.phase_modulation).flatten())
            shape_data = np.concatenate(([phase_shape.num_samples], phase_shape.data))
            shape_id, shape_found = self.shape_library.find_or_insert(shape_data)
            if not shape_found:
                surely_new = True

    # Construct the ADC event data
    data = (
        event.num_samples,
        event.dwell,
        max(event.delay, event.dead_time),
        event.freq_ppm,
        event.phase_ppm,
        event.freq_offset,
        event.phase_offset,
        shape_id,
        event.dead_time,
    )

    # Insert or find/insert into libraryAdd commentMore actions
    if surely_new:
        adc_id = self.adc_library.insert(0, data)
    else:
        adc_id, found = self.adc_library.find_or_insert(data)

        # Clear block cache if overwritten
        if self.use_block_cache and found:
            self.block_cache.clear()

    # Optional mapping
    if hasattr(event, 'name'):
        self.adc_id_to_name_map[adc_id] = event.name

    return adc_id, shape_id


def register_control_event(self, event: SimpleNamespace) -> int:
    """

    Parameters
    ----------
    event : SimpleNamespace
        Control event to be registered.

    Returns
    -------
    int
        ID of registered control event.
    """
    event_type = ['output', 'trigger'].index(event.type)
    if event_type == 0:
        # Trigger codes supported by the Siemens interpreter as of May 2019
        event_channel = ['osc0', 'osc1', 'ext1'].index(event.channel)
    elif event_type == 1:
        # Trigger codes supported by the Siemens interpreter as of June 2019
        event_channel = ['physio1', 'physio2'].index(event.channel)
    else:
        raise ValueError('Unsupported control event type')

    data = (event_type + 1, event_channel + 1, event.delay, event.duration)
    control_id, found = self.trigger_library.find_or_insert(new_data=data)

    # Clear block cache because trigger was overwritten
    # TODO: Could find only the blocks that are affected by the changes
    if self.use_block_cache and found:
        self.block_cache.clear()

    return control_id


def register_grad_event(self, event: SimpleNamespace) -> Union[int, Tuple[int, List[int]]]:
    """
    Parameters
    ----------
    event : SimpleNamespace
        Gradient event to be registered.

    Returns
    -------
    int, [int, ...]
        For gradient events: ID of registered gradient event, list of shape IDs
    int
        For trapezoid gradient events: ID of registered gradient event
    """
    may_exist = True
    any_changed = False

    if event.type == 'grad':
        amplitude = np.max(np.abs(event.waveform))
        if amplitude > 0:
            fnz = event.waveform[np.nonzero(event.waveform)[0][0]]
            amplitude *= np.sign(fnz) if fnz != 0 else 1

        # Shape ID initialization
        if hasattr(event, 'shape_IDs'):
            shape_IDs = event.shape_IDs
        else:
            shape_IDs = [0, 0]

            # Shape for waveform
            g = event.waveform / amplitude if amplitude != 0 else event.waveform
            c_shape = compress_shape(g)
            s_data = np.concatenate(([c_shape.num_samples], c_shape.data))
            shape_IDs[0], found = self.shape_library.find_or_insert(s_data)
            may_exist = may_exist and found
            any_changed = any_changed or found

            # Shape for timing
            c_time = compress_shape(event.tt / self.grad_raster_time)
            t_data = np.concatenate(([c_time.num_samples], c_time.data))

            if len(c_time.data) == 4 and np.allclose(c_time.data, [0.5, 1, 1, c_time.num_samples - 3]):
                # Standard raster → leave shape_IDs[1] as 0
                pass
            elif len(c_time.data) == 3 and np.allclose(c_time.data, [0.5, 0.5, c_time.num_samples - 2]):
                # Half-raster → set to -1 as special flag
                shape_IDs[1] = -1
            else:
                shape_IDs[1], found = self.shape_library.find_or_insert(t_data)
                may_exist = may_exist and found
                any_changed = any_changed or found

        # Updated data layout to match MATLAB v1.5.0 ordering
        data = (amplitude, event.first, event.last, *shape_IDs, event.delay)

    elif event.type == 'trap':
        data = (
            event.amplitude,
            event.rise_time,
            event.flat_time,
            event.fall_time,
            event.delay,
        )
    else:
        raise ValueError('Unknown gradient type passed to register_grad_event()')

    if may_exist:
        grad_id, found = self.grad_library.find_or_insert(new_data=data, data_type=event.type[0])
        any_changed = any_changed or found
    else:
        grad_id = self.grad_library.insert(0, data, event.type[0])

    # Clear block cache because grad event or shapes were overwritten
    # TODO: Could find only the blocks that are affected by the changes
    if self.use_block_cache and any_changed:
        self.block_cache.clear()

    if hasattr(event, 'name'):
        self.grad_id_to_name_map[grad_id] = event.name

    if event.type == 'grad':
        return grad_id, shape_IDs
    elif event.type == 'trap':
        return grad_id


def register_label_event(self, event: SimpleNamespace) -> int:
    """
    Parameters
    ----------
    event : SimpleNamespace
        ID of label event to be registered.

    Returns
    -------
    int
        ID of registered label event.
    """
    label_id = get_supported_labels().index(event.label) + 1
    data = (event.value, label_id)
    if event.type == 'labelset':
        label_id, found = self.label_set_library.find_or_insert(new_data=data)
    elif event.type == 'labelinc':
        label_id, found = self.label_inc_library.find_or_insert(new_data=data)
    else:
        raise ValueError('Unsupported label type passed to register_label_event()')

    # Clear block cache because label event was overwritten
    # TODO: Could find only the blocks that are affected by the changes
    if self.use_block_cache and found:
        self.block_cache.clear()

    return label_id


def register_soft_delay_event(self, event: SimpleNamespace) -> int:
    """
    Parameters
    ----------
    event : SimpleNamespace
        ID of soft delay event to be registered.

    Returns
    -------
    int
        ID of registered soft delay event.
    """
    # Auto-assign numID based on hint - each unique hint gets a unique numID
    if event.hint in self.soft_delay_hints:
        # Reuse existing numID for this hint
        assigned_numID = self.soft_delay_hints[event.hint]
        if event.numID is not None and event.numID != assigned_numID:
            raise ValueError(
                f"Soft delay hint '{event.hint}' is already assigned to numID {assigned_numID}. "
                f'Cannot use numID {event.numID}. Consider using a different hint or omitting numID.'
            )
        event.numID = assigned_numID
    else:
        # Assign new numID for this hint
        if event.numID is None:
            # Auto-assign next available numID
            event.numID = max([-1, *self.soft_delay_hints.values()]) + 1
        else:
            # Check if user-provided numID is already taken
            if event.numID in self.soft_delay_hints.values():
                existing_hint = next(hint for hint, num_id in self.soft_delay_hints.items() if num_id == event.numID)
                raise ValueError(
                    f"numID {event.numID} is already used by soft delay '{existing_hint}'. "
                    f'Use a different numID or omit it for auto-assignment.'
                )

        self.soft_delay_hints[event.hint] = event.numID

    data = (event.numID, event.offset, event.factor, event.hint)
    soft_delay_id, found = self.soft_delay_library.find_or_insert(new_data=data)
    if self.use_block_cache and found:
        self.block_cache.clear()
    return soft_delay_id


def register_rf_event(self, event: SimpleNamespace) -> Tuple[int, List[int]]:
    """
    Parameters
    ----------
    event : SimpleNamespace
        RF event to be registered.

    Returns
    -------
    int, [int, ...]
        ID of registered RF event, list of shape IDs
    """
    mag = np.abs(event.signal)
    amplitude = np.max(mag)
    mag /= amplitude
    # Following line of code is a workaround for numpy's divide functions returning NaN when mathematical
    # edge cases are encountered (eg. divide by 0)
    mag[np.isnan(mag)] = 0
    phase = np.angle(event.signal)
    phase[phase < 0] += 2 * np.pi
    phase /= 2 * np.pi
    may_exist = True

    if hasattr(event, 'shape_IDs'):
        shape_IDs = event.shape_IDs
    else:
        shape_IDs = [0, 0, 0]

        mag_shape = compress_shape(mag)
        data = np.concatenate(([mag_shape.num_samples], mag_shape.data))
        shape_IDs[0], found = self.shape_library.find_or_insert(data)
        may_exist = may_exist & found

        phase_shape = compress_shape(phase)
        data = np.concatenate(([phase_shape.num_samples], phase_shape.data))
        shape_IDs[1], found = self.shape_library.find_or_insert(data)
        may_exist = may_exist & found

        t_regular = (np.floor(event.t / self.rf_raster_time) == np.arange(len(event.t))).all()

        if t_regular:
            shape_IDs[2] = 0
        else:
            time_shape = compress_shape(event.t / self.rf_raster_time)
            data = [time_shape.num_samples, *time_shape.data]
            shape_IDs[2], found = self.shape_library.find_or_insert(data)
            may_exist = may_exist & found

    use = 'u'  # Undefined
    if hasattr(event, 'use'):
        if event.use in [
            'excitation',
            'refocusing',
            'inversion',
            'saturation',
            'preparation',
        ]:
            use = event.use[0]
        else:
            use = 'u'
    else:
        raise ValueError('Parameter "use" is not optional since v1.5.0')

    data = (
        amplitude,
        *shape_IDs,
        event.center,
        event.delay,
        event.freq_ppm,
        event.phase_ppm,
        event.freq_offset,
        event.phase_offset,
    )

    if may_exist:
        rf_id, found = self.rf_library.find_or_insert(new_data=data, data_type=use)

        # Clear block cache because RF event was overwritten
        # TODO: Could find only the blocks that are affected by the changes
        if self.use_block_cache and found:
            self.block_cache.clear()
    else:
        rf_id = self.rf_library.insert(key_id=0, new_data=data, data_type=use)

    if hasattr(event, 'name'):
        self.rf_id_to_name_map[rf_id] = event.name

    return rf_id, shape_IDs
