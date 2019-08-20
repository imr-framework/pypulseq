from types import SimpleNamespace

import numpy as np

from pypulseq.calc_duration import calc_duration
from pypulseq.compress_shape import compress_shape
from pypulseq.decompress_shape import decompress_shape


def add_block(self, block_index: int, *args):
    """
    Inserts pulse sequence events into `self.block_events` at position `block_index`. Also performs gradient checks.

    Parameters
    ----------
    block_index : int
        Index at which `SimpleNamespace` events have to be inserted into `self.block_events`.
    args : list
        List of `SimpleNamespace` pulse sequence events to be added to `self.block_events`.
    """

    block_duration = calc_duration(*args)
    self.block_events[block_index] = np.zeros(6, dtype=np.int)
    duration = 0

    check_g = {}

    for event in args:
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

            data = np.array((amplitude, mag_id, phase_id, event.delay, event.freq_offset, event.phase_offset,
                             event.dead_time, event.ringdown_time, use))
            data_id, found = self.rf_library.find(data)
            if not found:
                self.rf_library.insert(data_id, data)

            self.block_events[block_index][1] = data_id
            duration = max(duration,
                           max(
                               mag.shape) * self.rf_raster_time + event.dead_time + event.ringdown_time + event.delay)
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
            data = np.array([amplitude, shape_id, event.delay, event.first, event.last])
            grad_id, found = self.grad_library.find(data)
            if not found:
                self.grad_library.insert(grad_id, data, 'g')
            idx = 2 + channel_num
            self.block_events[block_index][idx] = grad_id
            duration = max(duration, len(g) * self.grad_raster_time)
        elif event.type == 'trap':
            channel_num = ['x', 'y', 'z'].index(event.channel)
            idx = 2 + channel_num
            check_g[channel_num] = SimpleNamespace()
            check_g[channel_num].idx = idx
            check_g[channel_num].start = np.array((0, 0))
            check_g[channel_num].stop = np.array((event.delay + event.rise_time + event.fall_time + event.flat_time, 0))
            data = np.array([event.amplitude, event.rise_time, event.flat_time, event.fall_time, event.delay])
            trap_id, found = self.grad_library.find(data)
            if not found:
                self.grad_library.insert(trap_id, data, 't')
            self.block_events[block_index][idx] = trap_id
            duration = max(duration, event.delay + event.rise_time + event.flat_time + event.fall_time)
        elif event.type == 'adc':
            data = np.array(
                [event.num_samples, event.dwell, max(event.delay, event.dead_time), event.freq_offset,
                 event.phase_offset, event.dead_time])
            adc_id, found = self.adc_library.find(data)
            if not found:
                self.adc_library.insert(adc_id, data)
            self.block_events[block_index][5] = adc_id
            duration = max(duration, event.delay + event.num_samples * event.dwell + event.dead_time)
        elif event.type == 'delay':
            data = np.array([event.delay])
            delay_id, found = self.delay_library.find(data)
            if not found:
                self.delay_library.insert(delay_id, data)
            self.block_events[block_index][0] = delay_id
            duration = max(duration, event.delay)

    # =========
    # PERFORM GRADIENT CHECKS
    # =========
    for cg_temp in check_g.keys():
        cg = check_g[cg_temp]

        if abs(cg.start[1]) > self.system.max_slew * self.system.grad_raster_time:
            if cg.start[0] != 0:
                raise ValueError('No delay allowed for gradients which start with a non-zero amplitude')

            if block_index > 1:
                prev_id = self.block_events[block_index - 1][cg.idx]
                if prev_id != 0:
                    prev_lib = self.grad_library.get(prev_id)
                    prev_dat = prev_lib['data']
                    prev_type = prev_lib['type']
                    if prev_type == 't':
                        raise Exception(
                            'Two consecutive gradients need to have the same amplitude at the connection point')
                    elif prev_type == 'g':
                        last = prev_dat[4]
                        if abs(last - cg.start[1]) > self.system.max_slew * self.system.grad_raster_time:
                            raise Exception(
                                'Two consecutive gradients need to have the same amplitude at the connection point')
            else:
                raise Exception('First gradient in the the first block has to start at 0.')
        if cg.stop[1] > self.system.max_slew * self.system.grad_raster_time and abs(cg.stop[0] - block_duration) > 1e-7:
            raise Exception('A gradient that doesnt end at zero needs to be aligned to the block boundary.')


def get_block(self, block_index: int) -> SimpleNamespace:
    """
    Returns Pulseq block at `block_index` position in `self.block_events`.

    Parameters
    ----------
    block_index : int
        Index of Pulseq block to be retrieved from `self.block_events`.

    Returns
    -------
    block : SimpleNamespace
        Pulseq block at 'block_index' position in `self.block_events`.
    """

    block = SimpleNamespace()
    event_ind = self.block_events[block_index]
    if event_ind[0] > 0:
        delay = SimpleNamespace()
        delay.type = 'delay'
        delay.delay = self.delay_library.data[event_ind[0]][0]
        block.delay = delay
    elif event_ind[1] > 0:
        block.rf = self.rf_from_lib_data(self.rf_library.data[event_ind[1]])
    grad_channels = ['gx', 'gy', 'gz']
    for i in range(1, len(grad_channels) + 1):
        if event_ind[2 + (i - 1)] > 0:
            grad, compressed = SimpleNamespace(), SimpleNamespace()
            type = self.grad_library.type[event_ind[2 + (i - 1)]]
            lib_data = self.grad_library.data[event_ind[2 + (i - 1)]]
            grad.type = 'trap' if type == 't' else 'grad'
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
                grad.amplitude, grad.rise_time, grad.flat_time, grad.fall_time, grad.delay = [lib_data[x] for x in
                                                                                              range(5)]
                grad.area = grad.amplitude * (grad.flat_time + grad.rise_time / 2 + grad.fall_time / 2)
                grad.flat_area = grad.amplitude * grad.flat_time
            setattr(block, grad_channels[i - 1], grad)

    if event_ind[5] > 0:
        lib_data = self.adc_library.data[event_ind[5]]
        if max(lib_data.shape) < 6:
            lib_data = np.append(lib_data, 0)
        adc = SimpleNamespace()
        adc.num_samples, adc.dwell, adc.delay, adc.freq_offset, adc.phase_offset, adc.dead_time = [lib_data[x] for x in
                                                                                                   range(6)]
        adc.num_samples = int(adc.num_samples)
        adc.type = 'adc'
        block.adc = adc
    return block
