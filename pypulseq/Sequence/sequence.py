import math
from types import SimpleNamespace

import numpy as np
from matplotlib import pyplot as plt

import pypulseq.Sequence.test_report
import pypulseq.check_timing
from pypulseq.Sequence import block
from pypulseq.Sequence.read_seq import read
from pypulseq.Sequence.write_seq import write
from pypulseq.calc_duration import calc_duration
from pypulseq.calc_rf_center import calc_rf_center
from pypulseq.decompress_shape import decompress_shape
from pypulseq.event_lib import EventLibrary
from pypulseq.opts import Opts
from pypulseq.points_to_waveform import points_to_waveform


class Sequence:
    def __init__(self, system=Opts()):
        self.version_major = 1
        self.version_minor = 2
        self.version_revision = 1
        self.system = system
        self.definitions = dict()
        # EventLibrary.data is a dict
        self.grad_library = EventLibrary()
        self.shape_library = EventLibrary()
        self.rf_library = EventLibrary()
        self.adc_library = EventLibrary()
        self.delay_library = EventLibrary()
        self.block_events = {}
        self.rf_raster_time = self.system.rf_raster_time
        self.grad_raster_time = self.system.grad_raster_time

    def __str__(self):
        s = "Sequence:"
        s += "\nshape_library: " + str(self.shape_library)
        s += "\nrf_library: " + str(self.rf_library)
        s += "\ngrad_library: " + str(self.grad_library)
        s += "\nadc_library: " + str(self.adc_library)
        s += "\ndelay_library: " + str(self.delay_library)
        s += "\nrf_raster_time: " + str(self.rf_raster_time)
        s += "\ngrad_raster_time: " + str(self.grad_raster_time)
        s += "\nblock_events: " + str(len(self.block_events))
        return s

    def duration(self):
        num_blocks = len(self.block_events)
        event_count = np.zeros(len(self.block_events[1]))
        duration = 0
        for i in range(num_blocks):
            block = self.get_block(i + 1)
            event_count += self.block_events[i + 1] > 0
            duration += calc_duration(block)

        return duration, num_blocks, event_count

    def check_timing(self):
        num_blocks = len(self.block_events)
        is_ok = True
        error_report = []
        for i in range(num_blocks):
            block = self.get_block(i + 1)
            event_names = ['rf', 'gx', 'gy', 'gz', 'adc', 'delay']
            ind = [hasattr(block, 'rf'), hasattr(block, 'gx'), hasattr(block, 'gy'), hasattr(block, 'gz'),
                   hasattr(block, 'adc'), hasattr(block, 'delay')]
            ev = [getattr(block, event_names[i]) for i in range(len(event_names)) if ind[i] == 1]
            res, rep = pypulseq.check_timing.check_timing(self, *ev)
            is_ok = is_ok and res
            if len(rep) != 0:
                error_report.append(f'Block: {i} - {rep}\n')

        return is_ok, error_report

    def test_report(self):
        pypulseq.Sequence.test_report.test_report(self)

    def set_definition(self, key, val):
        self.definitions[key] = val

    def get_definition(self, key):
        if key in self.definitions:
            return self.definitions[key]
        else:
            return None

    def add_block(self, *args):
        """
        Adds supplied list of Holder objects as a Block.

        Parameters
        ----------
        args : list
            List of Holder objects to be added as a Block.
        """

        block.add_block(self, len(self.block_events) + 1, *args)

    def get_block(self, block_index):
        """
        Returns Block at block_index.

        Parameters
        ----------
        block_index : int
            Index of required block.
        """

        return block.get_block(self, block_index)

    def read(self, file_path):
        read(self, file_path)

    def write(self, name):
        write(self, name)

    def rf_from_lib_data(self, lib_data):
        rf = SimpleNamespace()
        rf.type = 'rf'

        amplitude, mag_shape, phase_shape = lib_data[0], lib_data[1], lib_data[2]
        shape_data = self.shape_library.data[mag_shape]
        compressed = SimpleNamespace()
        compressed.num_samples = shape_data[0]
        compressed.data = shape_data[1:]
        mag = decompress_shape(compressed)
        shape_data = self.shape_library.data[phase_shape]
        compressed.num_samples = shape_data[0]
        compressed.data = shape_data[1:]
        phase = decompress_shape(compressed)
        rf.signal = 1j * 2 * np.pi * phase
        rf.signal = amplitude * mag * np.exp(rf.signal)
        rf.t = np.arange(1, max(mag.shape) + 1) * self.rf_raster_time

        rf.delay = lib_data[3]
        rf.freq_offset = lib_data[4]
        rf.phase_offset = lib_data[5]

        if max(lib_data.shape) < 6:
            lib_data = np.append(lib_data, 0)
        rf.dead_time = lib_data[6]

        if max(lib_data.shape) < 7:
            lib_data = np.append(lib_data, 0)
        rf.ringdown_time = lib_data[7]

        if max(lib_data.shape) < 8:
            lib_data = np.append(lib_data, 0)

        use_cases = {1: 'excitation', 2: 'refocusing', 3: 'inversion'}
        if lib_data[8] in use_cases:
            rf.use = use_cases[lib_data[8]]

        return rf

    def calculate_kspace(self, trajectory_delay=0):
        c_excitation = 0
        c_refocusing = 0
        c_adc_samples = 0

        for i in range(len(self.block_events)):
            block = self.get_block(i + 1)
            if hasattr(block, 'rf'):
                if not hasattr(block.rf, 'use') or block.rf.use != 'refocusing':
                    c_excitation += 1
                else:
                    c_refocusing += 1

            if hasattr(block, 'adc'):
                c_adc_samples += int(block.adc.num_samples)

        t_excitation = np.zeros(c_excitation)
        t_refocusing = np.zeros(c_refocusing)
        k_time = np.zeros(c_adc_samples)
        current_dur = 0
        c_excitation = 0
        c_refocusing = 0
        k_counter = 0
        traj_recon_delay = trajectory_delay

        for i in range(len(self.block_events)):
            block = self.get_block(i + 1)
            if hasattr(block, 'rf'):
                rf = block.rf
                rf_center, _ = calc_rf_center(rf)
                t = rf.delay + rf_center
                if not hasattr(block.rf, 'use') or block.rf.use != 'refocusing':
                    t_excitation[c_excitation] = current_dur + t
                    c_excitation += 1
                else:
                    t_refocusing[c_refocusing] = current_dur + t
                    c_refocusing += 1
            if hasattr(block, 'adc'):
                k_time[k_counter:k_counter + block.adc.num_samples] = np.arange(
                    block.adc.num_samples) * block.adc.dwell + block.adc.delay + current_dur + traj_recon_delay
                k_counter += block.adc.num_samples
            current_dur += calc_duration(block)

        gw = self.gradient_waveforms()
        i_excitation = np.round(t_excitation / self.grad_raster_time)
        i_refocusing = np.round(t_refocusing / self.grad_raster_time)
        k_traj = np.zeros(gw.shape)
        k = [0, 0, 0]
        for i in range(gw.shape[1]):
            k += gw[:, i] * self.grad_raster_time
            k_traj[:, i] = k
            if len(np.where(i_excitation == i + 1)[0]) >= 1:
                k = 0
                k_traj[:, i] = np.nan
            if len(np.where(i_refocusing == i + 1)[0]) >= 1:
                k = -k

        k_traj_adc = []
        for i in range(k_traj.shape[0]):
            k_traj_adc.append(np.interp(k_time, np.arange(1, k_traj.shape[1] + 1) * self.grad_raster_time, k_traj[i]))
        k_traj_adc = np.asarray(k_traj_adc)
        t_adc = k_time

        return k_traj_adc, k_traj, t_excitation, t_refocusing, t_adc

    def gradient_waveforms(self):
        duration, num_blocks, _ = self.duration()

        wave_length = math.ceil(duration / self.grad_raster_time)
        grad_channels = 3
        grad_waveforms = np.zeros((grad_channels, wave_length))
        grad_channels = ['gx', 'gy', 'gz']

        t0 = 0
        t0_n = 0
        for i in range(num_blocks):
            block = self.get_block(i + 1)
            for j in range(len(grad_channels)):
                if hasattr(block, grad_channels[j]):
                    grad = getattr(block, grad_channels[j])
                    if grad.type == 'grad':
                        nt_start = round((grad.delay + grad.t[0]) / self.grad_raster_time)
                        waveform = grad.waveform
                    else:
                        nt_start = round(grad.delay / self.grad_raster_time)
                        if abs(grad.flat_time) > np.finfo(float).eps:
                            t = np.cumsum([0, grad.rise_time, grad.flat_time, grad.fall_time])
                            trap_form = np.multiply([0, 1, 1, 0], grad.amplitude)
                        else:
                            t = np.cumsum([0, grad.rise_time, grad.fall_time])
                            trap_form = np.multiply([0, 1, 0], grad.amplitude)

                        tn = math.floor(t[-1] / self.grad_raster_time)
                        t = np.append(t, t[-1] + self.grad_raster_time)
                        trap_form = np.append(trap_form, 0)

                        if abs(grad.amplitude) > np.finfo(float).eps:
                            waveform = points_to_waveform(t, trap_form, self.grad_raster_time)
                        else:
                            waveform = np.zeros(tn + 1)

                    if waveform.size != np.sum(np.isfinite(waveform)):
                        raise Warning('Not all elements of the generated waveform are finite')

                    grad_waveforms[j, int(t0_n + nt_start):int(t0_n + nt_start + max(waveform.shape))] = waveform

            t0 += calc_duration(block)
            t0_n = round(t0 / self.grad_raster_time)

        return grad_waveforms

    def plot(self, time_range=(0, np.inf)):
        """
        Show Matplotlib plot of all Events in the Sequence object.

        Parameters
        ----------
        time_range : List
            Time range (x-axis limits) for plot to be shown. Default is 0 to infinity (entire plot shown).
        """

        fig1, fig2 = plt.figure(1), plt.figure(2)
        f11, f12, f13 = fig1.add_subplot(311), fig1.add_subplot(312), fig1.add_subplot(313)
        f2 = [fig2.add_subplot(311), fig2.add_subplot(312), fig2.add_subplot(313)]
        t0 = 0
        for iB in range(1, len(self.block_events) + 1):
            block = self.get_block(iB)
            is_valid = time_range[0] <= t0 <= time_range[1]
            if is_valid:
                if block is not None:
                    if 'adc' in block:
                        adc = block['adc']
                        t = adc.delay + [(x * adc.dwell) for x in range(0, int(adc.num_samples))]
                        f11.plot((t0 + t), np.zeros(len(t)))
                    if 'rf' in block:
                        rf = block['rf']
                        t = rf.t
                        f12.plot(np.squeeze(t0 + t), abs(rf.signal))
                        f13.plot(np.squeeze(t0 + t), np.angle(rf.signal))
                    grad_channels = ['gx', 'gy', 'gz']
                    for x in range(0, len(grad_channels)):
                        if grad_channels[x] in block:
                            grad = block[grad_channels[x]]
                            if grad.type == 'grad':
                                t = grad.t
                                waveform = 1e-3 * grad.waveform
                            else:
                                t = np.cumsum([0, grad.rise_time, grad.flat_time, grad.fall_time])
                                waveform = [1e-3 * grad.amplitude * x for x in [0, 1, 1, 0]]
                            f2[x].plot(np.squeeze(t0 + t), waveform)
            t0 += calc_duration(block)

        f11.set_ylabel('adc')
        f12.set_ylabel('rf mag hz')
        f13.set_ylabel('rf phase rad')
        [f2[x].set_ylabel(grad_channels[x]) for x in range(3)]
        # Setting display limits
        disp_range = [time_range[0], min(t0, time_range[1])]
        f11.set_xlim(disp_range)
        f12.set_xlim(disp_range)
        f13.set_xlim(disp_range)
        [x.set_xlim(disp_range) for x in f2]

        plt.show()
