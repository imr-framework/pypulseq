import math
from types import SimpleNamespace

import matplotlib as mpl
import numpy as np

mpl.use('TkAgg')
from matplotlib import pyplot as plt

from pypulseq.Sequence.test_report import test_report as ext_test_report
from pypulseq.check_timing import check_timing as ext_check_timing
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
        self.version_revision = 0
        self.system = system
        self.definitions = dict()
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
        """
        Get duration of this sequence.

        Returns
        -------
        duration : float
            Duration of this sequence in millis.
        num_blocks : int
            Number of blocks in this sequence.
        event_count : int
            Number of events in this sequence.
        """

        num_blocks = len(self.block_events)
        event_count = np.zeros(len(self.block_events[1]))
        duration = 0
        for i in range(num_blocks):
            block = self.get_block(i + 1)
            event_count += self.block_events[i + 1] > 0
            duration += calc_duration(block)

        return duration, num_blocks, event_count

    def check_timing(self):
        """
        Check timing of the events in each block based on grad raster time system limit.

        Returns
        -------
        is_ok : bool
            Boolean flag indicating timing errors.
        error_report : str
            Error report in case of timing errors.
        """
        num_blocks = len(self.block_events)
        is_ok = True
        error_report = []
        for i in range(num_blocks):
            block = self.get_block(i + 1)
            event_names = ['rf', 'gx', 'gy', 'gz', 'adc', 'delay']
            ind = [hasattr(block, 'rf'), hasattr(block, 'gx'), hasattr(block, 'gy'), hasattr(block, 'gz'),
                   hasattr(block, 'adc'), hasattr(block, 'delay')]
            ev = [getattr(block, event_names[i]) for i in range(len(event_names)) if ind[i] == 1]
            res, rep = ext_check_timing(self, *ev)
            is_ok = is_ok and res
            if len(rep) != 0:
                error_report.append(f'Block: {i} - {rep}\n')

        return is_ok, error_report

    def test_report(self):
        """
        Analyze the sequence and return a text report.
        """
        return ext_test_report(self)

    def set_definition(self, key: str, val: str):
        """
        Sets custom definition to the `Sequence`.

        Parameters
        ----------
        key : str
            Definition key.
        val : str
            Definition value.
        """
        self.definitions[key] = val

    def get_definition(self, key: str) -> str:
        """
        Retrieves definition identified by `key` from `Sequence`. Returns `None` if no matching definition is found.

        Parameters
        ----------
        key : str
            Key of definition to retrieve.

        Returns
        -------
        str
            Definition identified by `key` if found, else returns `None`.
        """
        if key in self.definitions:
            return self.definitions[key]
        else:
            return None

    def add_block(self, *args):
        """
        Adds event(s) as a block to `Sequence`.

        Parameters
        ----------
        args
            Event or list of events to be added as a block to `Sequence`.
        """
        block.add_block(self, len(self.block_events) + 1, *args)

    def get_block(self, block_index: int) -> SimpleNamespace:
        """
        Retrieves block of events identified by `block_index` from `Sequence`.

        Parameters
        ----------
        block_index : int
            Index of block to be retrieved from `Sequence`.

        Returns
        -------
        SimpleNamespace
            Block identified by `block_index`.
        """
        return block.get_block(self, block_index)

    def read(self, file_path: str):
        """
        Read `.seq` file from `file_path`.

        Parameters
        ----------
        file_path : str
            Path to `.seq` file to be read.
        """
        read(self, file_path)

    def write(self, name: str):
        """
        Writes the calling `Sequence` object as a `.seq` file with filename `name`.

        Parameters
        ----------
        name :str
            Filename of `.seq` file to be written to disk.
        """
        write(self, name)

    def rf_from_lib_data(self, lib_data):
        """
        Construct RF object from `lib_data`.

        Parameters
        ----------
        lib_data : list
            RF envelope.

        Returns
        -------
        rf : SimpleNamespace
            RF object constructed from lib_data.
        """
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

        if max(lib_data.shape) < 6:
            rf.delay = 0
            rf.freq_offset = lib_data[3]
            rf.phase_offset = lib_data[4]
            lib_data = np.append(lib_data, 0)
        else:
            rf.delay = lib_data[3]
            rf.freq_offset = lib_data[4]
            rf.phase_offset = lib_data[5]

        if max(lib_data.shape) < 7:
            lib_data = np.append(lib_data, 0)
        rf.dead_time = lib_data[6]

        if max(lib_data.shape) < 8:
            lib_data = np.append(lib_data, 0)
        rf.ringdown_time = lib_data[7]

        if max(lib_data.shape) < 9:
            lib_data = np.append(lib_data, 0)

        use_cases = {1: 'excitation', 2: 'refocusing', 3: 'inversion'}
        if lib_data[8] in use_cases:
            rf.use = use_cases[lib_data[8]]

        return rf

    def calculate_kspace(self, trajectory_delay: int = 0):
        """
        Calculates the k-space trajectory of the entire pulse sequence.

        Parameters
        ----------
        trajectory_delay : int
            Compensation factor in millis to align ADC and gradients in the reconstruction.

        Returns
        -------
        k_traj_adc : numpy.ndarray
            K-space trajectory sampled at `t_adc` timepoints.
        k_traj : numpy.ndarray
            K-space trajectory of the entire pulse sequence.
        t_excitation : numpy.ndarray
            Excitation timepoints.
        t_refocusing : numpy.ndarray
            Refocusing timepoints.
        t_adc : numpy.ndarray
            Sampling timepoints.
        """
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

    def gradient_waveforms(self) -> np.ndarray:
        """
        Decompress the entire gradient waveform. Returns an array of shape `gradient_axesxtimepoints`.
        `gradient_axes` is typically 3.

        Returns
        -------
        grad_waveforms : numpy.ndarray
            Decompressed gradient waveform.
        """
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

                    """
                    Matlab dynamically resizes arrays during slice assignment operation if assignment is out of bounds
                    Numpy does not
                    Following lines are a workaround
                    """
                    l1, l2 = int(t0_n + nt_start), int(t0_n + nt_start + max(waveform.shape))
                    if l2 > grad_waveforms.shape[1]:
                        grad_waveforms.resize((len(grad_channels), l2))
                    grad_waveforms[j, l1:l2] = waveform

            t0 += calc_duration(block)
            t0_n = round(t0 / self.grad_raster_time)

        return grad_waveforms

    def plot(self, type: str = 'Gradient', time_range=(0, np.inf), time_disp: str = 's', save: bool=False):
        """
        Plot `Sequence`.

        Parameters
        ----------
        type : str
            Gradients display type, must be one of either 'Gradient' or 'Kspace'.
        time_range : List
            Time range (x-axis limits) for plotting the sequence. Default is 0 to infinity (entire sequence).
        time_disp : str
            Time display type, must be one of `s`, `ms` or `us`.
        save_as : bool
            Boolean flag indicating if plots should be saved. The two figures will be saved as JPG with numerical
            suffixes to the filename 'seq_plot'.
        """

        valid_plot_types = ['Gradient', 'Kspace']
        valid_time_units = ['s', 'ms', 'us']
        if type not in valid_plot_types:
            raise Exception()
        if time_disp not in valid_time_units:
            raise Exception()

        fig1, fig2 = plt.figure(1), plt.figure(2)
        sp11 = fig1.add_subplot(311)
        sp12, sp13 = fig1.add_subplot(312, sharex=sp11), fig1.add_subplot(313, sharex=sp11)
        fig2_sp_list = [fig2.add_subplot(311, sharex=sp11), fig2.add_subplot(312, sharex=sp11),
                        fig2.add_subplot(313, sharex=sp11)]

        t_factor_list = [1, 1e3, 1e6]
        t_factor = t_factor_list[valid_time_units.index(time_disp)]
        t0 = 0
        for iB in range(1, len(self.block_events) + 1):
            block = self.get_block(iB)
            is_valid = time_range[0] <= t0 <= time_range[1]
            if is_valid:
                if hasattr(block, 'adc'):
                    adc = block.adc
                    t = adc.delay + [(x * adc.dwell) for x in range(0, int(adc.num_samples))]
                    sp11.plot((t0 + t), np.zeros(len(t)), 'rx')
                if hasattr(block, 'rf'):
                    rf = block.rf
                    tc, ic = calc_rf_center(rf)
                    t = rf.t + rf.delay
                    tc = tc + rf.delay
                    sp12.plot(t_factor * (t0 + t), abs(rf.signal))
                    sp13.plot(t_factor * (t0 + t), np.angle(
                        rf.signal * np.exp(1j * rf.phase_offset) * np.exp(1j * 2 * math.pi * rf.t * rf.freq_offset)),
                              t_factor * (t0 + tc), np.angle(rf.signal[ic] * np.exp(1j * rf.phase_offset) * np.exp(
                            1j * 2 * math.pi * rf.t[ic] * rf.freq_offset)), 'xb')
                grad_channels = ['gx', 'gy', 'gz']
                for x in range(0, len(grad_channels)):
                    if hasattr(block, grad_channels[x]):
                        grad = getattr(block, grad_channels[x])
                        if grad.type == 'grad':
                            # In place unpacking of grad.t with the starred expression
                            t = grad.delay + [0, *(grad.t + (grad.t[1] - grad.t[0]) / 2),
                                              grad.t[-1] + grad.t[1] - grad.t[0]]
                            waveform = np.array([grad.first, grad.last])
                            waveform = 1e-3 * np.insert(waveform, 1, grad.waveform)
                        else:
                            t = np.cumsum([0, grad.delay, grad.rise_time, grad.flat_time, grad.fall_time])
                            waveform = 1e-3 * grad.amplitude * np.array([0, 0, 1, 1, 0])
                        fig2_sp_list[x].plot(t_factor * (t0 + t), waveform)
            t0 += calc_duration(block)

        grad_plot_labels = ['x', 'y', 'z']
        sp11.set_ylabel('ADC')
        sp12.set_ylabel('RF mag (Hz)')
        sp13.set_ylabel('RF phase (rad)')
        [fig2_sp_list[x].set_ylabel(f'G{grad_plot_labels[x]} (kHz/m)') for x in range(3)]
        # Setting display limits
        disp_range = t_factor * np.array([time_range[0], min(t0, time_range[1])])
        sp11.set_xlim(disp_range)
        sp12.set_xlim(disp_range)
        sp13.set_xlim(disp_range)
        [x.set_xlim(disp_range) for x in fig2_sp_list]

        fig1.tight_layout()
        fig2.tight_layout()
        if save:
            fig1.savefig('seq_plot1.jpg')
            fig2.savefig('seq_plot2.jpg')
        plt.show()
