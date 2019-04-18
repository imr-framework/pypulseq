import numpy as np

from pypulseq.compress_shape import compress_shape
from pypulseq.decompress_shape import decompress_shape
from pypulseq.holder import Holder


def add_block(self, block_index, *args):
    """
    Adds supplied list of Holder objects at position specified by block_index.

    Parameters
    ----------
    block_index : int
        Index at which Block has to be inserted.
    args : list
        List of Holder objects to be added as a Block.
    """

    self.block_events[block_index] = np.zeros(6)
    duration = 0
    for event in args:
        if event.type == 'rf':
            mag = np.abs(event.signal)
            amplitude = np.max(mag)
            mag = np.divide(mag, amplitude)
            # Following two lines of code are workarounds for numpy's divide functions returning NaN when mathematical
            # edge cases are encountered (eg. divide by 0)
            mag[np.isnan(mag)] = 0
            phase = np.angle(event.signal)
            phase[np.where(phase < 0)] += 2 * np.pi
            phase /= 2 * np.pi

            mag_shape = compress_shape(mag)
            # data = np.array([[mag_shape.num_samples]])
            # data = np.append(data, mag_shape.data, axis=1)
            data = np.hstack((mag_shape.num_samples, np.ravel(mag_shape.data))).reshape((1, -1))
            mag_id, found = self.shape_library.find(data)
            if not found:
                self.shape_library.insert(mag_id, data, None)

            phase_shape = compress_shape(phase)
            data = np.hstack((phase_shape.num_samples, np.ravel(phase_shape.data))).reshape((1, -1))
            phase_id, found = self.shape_library.find(data)
            if not found:
                self.shape_library.insert(phase_id, data, None)

            data = np.array([amplitude, mag_id, phase_id, event.freq_offset, event.phase_offset, event.dead_time,
                             event.ring_down_time])
            data_id, found = self.rf_library.find(data)
            if not found:
                self.rf_library.insert(data_id, data, None)

            self.block_events[block_index][1] = data_id
            duration = max(duration, max(mag.shape) * self.rf_raster_time + event.dead_time + event.ring_down_time)
        elif event.type == 'grad':
            channel_num = ['x', 'y', 'z'].index(event.channel)
            amplitude = max(abs(event.waveform[0]))
            g = event.waveform / amplitude
            shape = compress_shape(g)
            data = np.array([[shape.num_samples]])
            data = np.append(data, shape.data, axis=1)
            shape_id, found = self.shape_library.find(data)
            if not found:
                self.shape_library.insert(shape_id, data, None)
            data = np.array([amplitude, shape_id])
            index, found = self.grad_library.find(data)
            if not found:
                self.grad_library.insert(index, data, 'g')
            idx = 2 + channel_num
            self.block_events[block_index][idx] = index
            duration = max(duration, len(g[0]) * self.grad_raster_time)
        elif event.type == 'trap':
            channel_num = ['x', 'y', 'z'].index(event.channel)
            data = np.array([event.amplitude, event.rise_time, event.flat_time, event.fall_time])
            index, found = self.grad_library.find(data)
            if not found:
                self.grad_library.insert(index, data, 't')
            idx = 2 + channel_num
            self.block_events[block_index][idx] = index
            duration = max(duration, event.rise_time + event.flat_time + event.fall_time)
        elif event.type == 'adc':
            data = np.array(
                [event.num_samples, event.dwell, event.delay, event.freq_offset, event.phase_offset, event.dead_time])
            index, found = self.adc_library.find(data)
            if not found:
                self.adc_library.insert(index, data, None)
            self.block_events[block_index][5] = index
            duration = max(duration, event.delay + event.num_samples * event.dwell + event.dead_time)
        elif event.type == 'delay':
            data = np.array([event.delay])
            index, found = self.delay_library.find(data)
            if not found:
                self.delay_library.insert(index, data, None)
            self.block_events[block_index][0] = index
            duration = max(duration, event.delay)


def get_block(self, block_index):
    """
    Returns Block at position specified by block_index.

    Parameters
    ----------
    block_index : int
        Index of Block to be retrieved.

    Returns
    -------
    block : dict
        Block at position specified by block_index.
    """

    block = {}
    event_ind = self.block_events[block_index]
    if event_ind[0] > 0:
        delay = Holder()
        delay.type = 'delay'
        delay.delay = self.delay_library.data[event_ind[0]]
        block['delay'] = delay
    elif event_ind[1] > 0:
        rf = Holder()
        rf.type = 'rf'
        lib_data = self.rf_library.data[event_ind[1]]

        amplitude, mag_shape, phase_shape = lib_data[0], lib_data[1], lib_data[2]
        shape_data = self.shape_library.data[mag_shape]
        compressed = Holder()
        compressed.num_samples = shape_data[0][0]
        compressed.data = shape_data[0][1:]
        compressed.data = compressed.data.reshape((1, compressed.data.shape[0]))
        mag = decompress_shape(compressed)
        shape_data = self.shape_library.data[phase_shape]
        compressed.num_samples = shape_data[0][0]
        compressed.data = shape_data[0][1:]
        compressed.data = compressed.data.reshape((1, compressed.data.shape[0]))
        phase = decompress_shape(compressed)
        rf.signal = 1j * 2 * np.pi * phase
        rf.signal = amplitude * mag * np.exp(rf.signal)
        rf.t = [(x * self.rf_raster_time) for x in range(1, max(mag.shape) + 1)]
        rf.t = np.reshape(rf.t, (1, len(rf.t)))
        rf.freq_offset = lib_data[3]
        rf.phase_offset = lib_data[4]
        if max(lib_data.shape) < 6:
            lib_data = np.append(lib_data, 0)
        rf.dead_time = lib_data[5]

        if max(lib_data.shape) < 7:
            lib_data = np.append(lib_data, 0)
        rf.ring_down_time = lib_data[6]

        block['rf'] = rf
    grad_channels = ['gx', 'gy', 'gz']
    for i in range(1, len(grad_channels) + 1):
        if event_ind[2 + (i - 1)] > 0:
            grad, compressed = Holder(), Holder()
            type = self.grad_library.type[event_ind[2 + (i - 1)]]
            lib_data = self.grad_library.data[event_ind[2 + (i - 1)]]
            grad.type = 'trap' if type == 't' else 'grad'
            grad.channel = grad_channels[i - 1][1]
            if grad.type == 'grad':
                amplitude = lib_data[0]
                shape_id = lib_data[1]
                shape_data = self.shape_library.data[shape_id]
                compressed.num_samples = shape_data[0][0]
                compressed.data = np.array([shape_data[0][1:]])
                g = decompress_shape(compressed)
                grad.waveform = amplitude * g
                grad.t = np.array([[x * self.grad_raster_time for x in range(1, g.size + 1)]])
            else:
                grad.amplitude, grad.rise_time, grad.flat_time, grad.fall_time = [lib_data[x] for x in range(4)]
                grad.area = grad.amplitude * (grad.flat_time + grad.rise_time / 2 + grad.fall_time / 2)
                grad.flat_area = grad.amplitude * grad.flat_time
            block[grad_channels[i - 1]] = grad

    if event_ind[5] > 0:
        lib_data = self.adc_library.data[event_ind[5]]
        if max(lib_data.shape) < 6:
            lib_data = np.append(lib_data, 0)
        adc = Holder()
        adc.num_samples, adc.dwell, adc.delay, adc.freq_offset, adc.phase_offset, adc.dead_time = [lib_data[x] for x in
                                                                                                   range(6)]
        adc.type = 'adc'
        block['adc'] = adc
    return block
