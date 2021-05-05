import math
from collections import OrderedDict
from types import SimpleNamespace
from typing import Tuple
from typing import Union
from warnings import warn

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt

from pypulseq import major, minor, revision
from pypulseq.Sequence import block
from pypulseq.Sequence import parula
from pypulseq.Sequence.read_seq import read
from pypulseq.Sequence.test_report import test_report as ext_test_report
from pypulseq.Sequence.write_seq import write
from pypulseq.calc_rf_center import calc_rf_center
from pypulseq.check_timing import check_timing as ext_check_timing
from pypulseq.decompress_shape import decompress_shape
from pypulseq.event_lib import EventLibrary
from pypulseq.opts import Opts
from pypulseq.points_to_waveform import points_to_waveform
from pypulseq.supported_labels import get_supported_labels


class Sequence:
    version_major: int = major
    version_minor: int = minor
    version_revision: int = revision

    def __init__(self, system=Opts()):
        # =========
        # EVENT LIBRARIES
        # =========
        self.adc_library = EventLibrary()  # Library of ADC events
        self.delay_library = EventLibrary()  # Library of delay events
        # Library of extension events. Extension events form single-linked zero-terminated lists
        self.extensions_library = EventLibrary()
        self.grad_library = EventLibrary()  # Library of gradient events
        self.label_inc_library = EventLibrary()  # Library of Label(inc) events (reference from the extensions library)
        self.label_set_library = EventLibrary()  # Library of Label(set) events (reference from the extensions library)
        self.rf_library = EventLibrary()  # Library of RF events
        self.shape_library = EventLibrary()  # Library of compressed shapes
        self.trigger_library = EventLibrary()  # Library of trigger events

        # =========
        # OTHER
        # =========
        self.system = system

        self.rf_raster_time = self.system.rf_raster_time  # RF raster time (system dependent)
        self.grad_raster_time = self.system.grad_raster_time  # Gradient raster time (system dependent)

        self.dict_block_events = OrderedDict()  # Event table
        self.dict_definitions = OrderedDict()  # Optional sequence dict_definitions

        self.arr_block_durations = []  # Cache of block durations
        self.arr_extension_numeric_idx = []  # numeric IDs of the used extensions
        self.arr_extension_string_idx = []  # string IDs of the used extensions

    def __str__(self):
        s = "Sequence:"
        s += "\nshape_library: " + str(self.shape_library)
        s += "\nrf_library: " + str(self.rf_library)
        s += "\ngrad_library: " + str(self.grad_library)
        s += "\nadc_library: " + str(self.adc_library)
        s += "\ndelay_library: " + str(self.delay_library)
        s += "\nextensions_library: " + str(self.extensions_library)  # inserted for trigger support by mveldmann
        s += "\nrf_raster_time: " + str(self.rf_raster_time)
        s += "\ngrad_raster_time: " + str(self.grad_raster_time)
        s += "\ndict_block_events: " + str(len(self.dict_block_events))
        return s

    def add_block(self, *args: SimpleNamespace) -> None:
        """
        Adds event(s) as a block to `Sequence`.

        Parameters
        ----------
        args
            Event or list of events to be added as a block to `Sequence`.
        """
        block.add_block(self, len(self.dict_block_events) + 1, *args)

    def calculate_kspace(self, trajectory_delay: int = 0) -> Tuple[np.array, np.array, np.array, np.array, np.array]:
        """
        Calculates the k-space trajectory of the entire pulse sequence.

        Parameters
        ----------
        trajectory_delay : int, default=0
            Compensation factor in millis to align ADC and gradients in the reconstruction.

        Returns
        -------
        k_traj_adc : numpy.array
            K-space trajectory sampled at `t_adc` timepoints.
        k_traj : numpy.array
            K-space trajectory of the entire pulse sequence.
        t_excitation : numpy.array
            Excitation timepoints.
        t_refocusing : numpy.array
            Refocusing timepoints.
        t_adc : numpy.array
            Sampling timepoints.
        """
        # Initialise the counters and accumulator objects
        count_excitation = 0
        count_refocusing = 0
        count_adc_samples = 0

        # Loop through the blocks to prepare preallocations
        for block_counter in range(len(self.dict_block_events)):
            block = self.get_block(block_counter + 1)
            if hasattr(block, 'rf'):
                if not hasattr(block.rf, 'use') or block.rf.use != 'refocusing':
                    count_excitation += 1
                else:
                    count_refocusing += 1

            if hasattr(block, 'adc'):
                count_adc_samples += int(block.adc.num_samples)

        t_excitation = np.zeros(count_excitation)
        t_refocusing = np.zeros(count_refocusing)
        k_time = np.zeros(count_adc_samples)
        current_duration = 0
        count_excitation = 0
        count_refocusing = 0
        kc_outer = 0
        traj_recon_delay = trajectory_delay

        # Go through the blocks and collect RF and ADC timing data
        for block_counter in range(len(self.dict_block_events)):
            block = self.get_block(block_counter + 1)

            if hasattr(block, 'rf'):
                rf = block.rf
                rf_center, _ = calc_rf_center(rf)
                t = rf.delay + rf_center
                if not hasattr(block.rf, 'use') or block.rf.use != 'refocusing':
                    t_excitation[count_excitation] = current_duration + t
                    count_excitation += 1
                else:
                    t_refocusing[count_refocusing] = current_duration + t
                    count_refocusing += 1

            if hasattr(block, 'adc'):
                _k_time = np.arange(block.adc.num_samples) + 0.5
                _k_time = _k_time * block.adc.dwell + block.adc.delay + current_duration + traj_recon_delay
                k_time[kc_outer:kc_outer + block.adc.num_samples] = _k_time
                kc_outer += block.adc.num_samples
            current_duration += self.arr_block_durations[block_counter]

        # Now calculate the actual k-space trajectory based on the gradient waveforms
        gw = self.gradient_waveforms()
        i_excitation = np.round(t_excitation / self.grad_raster_time)
        i_refocusing = np.round(t_refocusing / self.grad_raster_time)
        i_periods = np.sort([1, *(i_excitation + 1), *(i_refocusing + 1), gw.shape[1] + 1]).astype(np.int)
        # i_periods -= 1  # Python is 0-indexed
        ii_next_excitation = min(len(i_excitation), 1)
        ii_next_refocusing = min(len(i_refocusing), 1)
        k_traj = np.zeros_like(gw)
        k = np.zeros((3, 1))

        for i in range(len(i_periods) - 1):
            i_period_end = i_periods[i + 1] - 1
            k_period = np.concatenate((k, gw[:, i_periods[i] - 1:i_period_end] * self.grad_raster_time), axis=1)
            k_period = np.cumsum(k_period, axis=1)
            k_traj[:, i_periods[i] - 1:i_period_end] = k_period[:, 1:]
            k = k_period[:, -1]

            if ii_next_excitation > 0 and i_excitation[ii_next_excitation - 1] == i_period_end:
                k[:] = 0
                k_traj[:, i_period_end - 1] = np.nan
                ii_next_excitation = min(len(i_excitation), ii_next_excitation + 1)

            if ii_next_refocusing > 0 and i_refocusing[ii_next_refocusing - 1] == i_period_end:
                k = -k
                ii_next_refocusing = min(len(i_refocusing), ii_next_refocusing + 1)

            k = k.reshape((-1, 1))  # To be compatible with np.concatenate

        k_traj_adc = []
        for _k_traj_row in k_traj:
            result = np.interp(xp=np.array(range(1, k_traj.shape[1] + 1)) * self.grad_raster_time,
                               fp=_k_traj_row,
                               x=k_time)
            k_traj_adc.append(result)
        k_traj_adc = np.stack(k_traj_adc)
        t_adc = k_time

        return k_traj_adc, k_traj, t_excitation, t_refocusing, t_adc

    def check_timing(self) -> Tuple[bool, list]:
        """
        Check timing of the events in each block based on grad raster time system limit.

        Returns
        -------
        is_ok : bool
            Boolean flag indicating timing errors.
        error_report : str
            Error report in case of timing errors.
        """
        error_report = []
        is_ok = True
        num_blocks = len(self.dict_block_events)
        total_duration = 0

        for block_counter in range(num_blocks):
            block = self.get_block(block_counter + 1)
            event_names = ['rf', 'gx', 'gy', 'gz', 'adc', 'delay']
            ind = [hasattr(block, attr) for attr in event_names]
            events = [getattr(block, event_names[i]) for i in range(len(event_names)) if ind[i] == 1]
            res, rep, duration = ext_check_timing(self.system, *events)
            is_ok = is_ok and res
            if len(rep) != 0:
                error_report.append(f'Event: {block_counter} - {rep}\n')
            total_duration += duration

        # Check if all the gradients in the last block are ramped down properly
        if len(events) != 0:
            for e in range(len(events)):
                if not isinstance(events[e], list) and events[e].type == 'grad':
                    if events[e].last != 0:
                        error_report.append(
                            f'Event {block_counter} gradients do not ramp to 0 at the end of the sequence')

        self.set_definition('Total duration', total_duration)

        return is_ok, error_report

    def duration(self) -> Tuple[int, int, np.ndarray]:
        """
        Returns the total duration of this sequence, and the total count of blocks and events.

        Returns
        -------
        duration : int
            Duration of this sequence in millis.
        num_blocks : int
            Number of blocks in this sequence.
        event_count : numpy.ndarray
            Number of events in this sequence.
        """

        num_blocks = len(self.dict_block_events)
        event_count = np.zeros(len(self.dict_block_events[1]))
        duration = 0
        for block_counter in range(num_blocks):
            event_count += self.dict_block_events[block_counter + 1] > 0
            duration += self.arr_block_durations[block_counter]

        return duration, num_blocks, event_count

    def flip_grad_axis(self, axis: str) -> None:
        """
        Convenience function to invert all gradients along specified axis.

        Parameters
        ----------
        axis : str
            Gradients to invert or scale. Must be one of 'x', 'y' or 'z'.
        """
        self.mod_grad_axis(axis, modifier=-1)

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
            Event identified by `block_index`.
        """
        return block.get_block(self, block_index)

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
            Definition identified by `key` if found, else returns ''.
        """
        if key in self.dict_definitions:
            return self.dict_definitions[key]
        else:
            return ''

    def get_extension_type_ID(self, extension_string: str) -> int:
        """
        Get numeric extension ID for `extension_string`. Will automatically create a new ID if unknown.

        Parameters
        ----------
        extension_string : str
            Given string extension ID.

        Returns
        -------
        extension_id : int
            Numeric ID for given string extension ID.

        """
        if extension_string not in self.arr_extension_string_idx:
            if len(self.arr_extension_numeric_idx) == 0:
                extension_id = 1
            else:
                extension_id = 1 + max(self.arr_extension_numeric_idx)

            self.arr_extension_numeric_idx.append(extension_id)
            self.arr_extension_string_idx.append(extension_string)
            assert len(self.arr_extension_numeric_idx) == len(self.arr_extension_string_idx)
        else:
            num = self.arr_extension_string_idx.index(extension_string)
            extension_id = self.arr_extension_numeric_idx[num]

        return extension_id

    def get_extension_type_string(self, extension_id: int) -> str:
        """
        Get string extension ID for `extension_id`.

        Parameters
        ----------
        extension_id : int
            Given numeric extension ID.

        Returns
        -------
        extension_str : str
            String ID for the given numeric extension ID.

        Raises
        ------
        ValueError
            If given numeric extension ID is unknown.
        """
        if extension_id in self.arr_extension_numeric_idx:
            num = self.arr_extension_numeric_idx.index(extension_id)
        else:
            raise ValueError(f'Extension for the given ID - {extension_id} - is unknown.')

        extension_str = self.arr_extension_string_idx[num]
        return extension_str

    def gradient_waveforms(self) -> np.ndarray:
        """
        Decompress the entire gradient waveform. Returns an array of shape `gradient_axes x timepoints`.
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
        for block_counter in range(num_blocks):
            block = self.get_block(block_counter + 1)
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
                            waveform = points_to_waveform(times=t, amplitudes=trap_form,
                                                          grad_raster_time=self.grad_raster_time)
                        else:
                            waveform = np.zeros(tn + 1)

                    if len(waveform) != np.sum(np.isfinite(waveform)):
                        warn('Not all elements of the generated waveform are finite')

                    """
                    Matlab dynamically resizes arrays during slice assignment operation if assignment is out of bounds
                    Numpy does not; following is a workaround
                    """
                    l1, l2 = int(t0_n + nt_start), int(t0_n + nt_start + len(waveform))
                    if l2 > grad_waveforms.shape[1]:
                        z = np.zeros((grad_waveforms.shape[0], l2 - grad_waveforms.shape[1]))
                        grad_waveforms = np.hstack((grad_waveforms, z))
                    grad_waveforms[j, l1:l2] = waveform

            t0 += self.arr_block_durations[block_counter]
            t0_n = round(t0 / self.grad_raster_time)

        return grad_waveforms

    def mod_grad_axis(self, axis: str, modifier: int) -> None:
        """
        Invert or scale all gradients along the corresponding axis/channel. The function acts on all gradient objects
        already added to the sequence object.

        Parameters
        ----------
        axis : str
            Gradients to invert or scale. Must be one of 'x', 'y' or 'z'.
        modifier : int
            Scaling value.

        Raises
        ------
        ValueError
            If invalid `axis` is passed. Must be one of 'x', 'y','z'.
        RuntimeError
            If same gradient event is used on multiple axes.
        """
        if axis not in ['x', 'y', 'z']:
            raise ValueError(f"Invalid axis. Must be one of 'x', 'y','z'. Passed: {axis}")

        channel_num = ['x', 'y', 'z'].index(axis)
        other_channels = [0, 1, 2]
        other_channels.remove(channel_num)

        # Go through all event table entries and list gradient objects in the library
        all_grad_events = np.array(list(self.dict_block_events.values()))
        all_grad_events = all_grad_events[:, 2:5]

        selected_events = np.unique(all_grad_events[:, channel_num])
        selected_events = selected_events[selected_events != 0]
        other_events = np.unique(all_grad_events[:, other_channels])
        if len(np.intersect1d(selected_events, other_events)) > 0:
            raise RuntimeError('mod_grad_axis does not yet support the same gradient event used on multiple axes.')

        for i in range(len(selected_events)):
            self.grad_library.data[selected_events[i]][0] *= modifier
            if self.grad_library.type[selected_events[i]] == 'g' and self.grad_library.lengths[selected_events[i]] == 5:
                # Need to update first and last fields
                self.grad_library.data[selected_events[i]][3] *= modifier
                self.grad_library.data[selected_events[i]][4] *= modifier

    def plot(self, label: str = str(), save: bool = False, time_range=(0, np.inf), time_disp: str = 's',
             plot_type: str = 'Gradient') -> None:
        """
        Plot `Sequence`.

        Parameters
        ----------
        label : str, defualt=str()

        save : bool, default=False
            Boolean flag indicating if plots should be saved. The two figures will be saved as JPG with numerical
            suffixes to the filename 'seq_plot'.
        time_range : iterable, default=(0, np.inf)
            Time range (x-axis limits) for plotting the sequence. Default is 0 to infinity (entire sequence).
        time_disp : str, default='s'
            Time display type, must be one of `s`, `ms` or `us`.
        plot_type : str, default='Gradient'
            Gradients display type, must be one of either 'Gradient' or 'Kspace'.
        """
        mpl.rcParams['lines.linewidth'] = 0.75  # Set default Matplotlib linewidth

        valid_plot_types = ['Gradient', 'Kspace']
        valid_time_units = ['s', 'ms', 'us']
        valid_labels = get_supported_labels()
        if plot_type not in valid_plot_types:
            raise ValueError('Unsupported plot type')
        if not all([isinstance(x, (int, float)) for x in time_range]) or len(time_range) != 2:
            raise ValueError('Invalid time range')
        if time_disp not in valid_time_units:
            raise ValueError('Unsupported time unit')

        fig1, fig2 = plt.figure(1), plt.figure(2)
        sp11 = fig1.add_subplot(311)
        sp12, sp13 = fig1.add_subplot(312, sharex=sp11), fig1.add_subplot(313, sharex=sp11)
        fig2_subplots = [fig2.add_subplot(311, sharex=sp11), fig2.add_subplot(312, sharex=sp11),
                         fig2.add_subplot(313, sharex=sp11)]

        t_factor_list = [1, 1e3, 1e6]
        t_factor = t_factor_list[valid_time_units.index(time_disp)]

        t0 = 0
        label_defined = False
        label_idx_to_plot = []
        label_legend_to_plot = []
        label_store = dict()
        for i in range(len(valid_labels)):
            label_store[valid_labels[i]] = 0
            if label.upper() == valid_labels[i]:
                label_idx_to_plot.append(i)
                label_legend_to_plot.append(valid_labels[i])

        if len(label_idx_to_plot) != 0:
            p = parula.main(len(label_idx_to_plot) + 1)
            label_colors_to_plot = p(np.arange(len(label_idx_to_plot)))

        for block_counter in range(len(self.dict_block_events)):
            block = self.get_block(block_counter + 1)
            is_valid = time_range[0] <= t0 <= time_range[1]
            if is_valid:
                if hasattr(block, 'label'):
                    for i in range(len(block.label)):
                        if block.label[i].type == 'labelinc':
                            label_store[block.label[i].label] += block.label[i].value
                        else:
                            label_store[block.label[i].label] = block.label[i].value
                    label_defined = True

                if hasattr(block, 'adc'):
                    adc = block.adc
                    # From Pulseq: According to the information from Klaus Scheffler and indirectly from Siemens this
                    # is the present convention - the samples are shifted by 0.5 dwell
                    t = adc.delay + (np.arange(int(adc.num_samples)) + 0.5) * adc.dwell
                    sp11.plot(t_factor * (t0 + t), np.zeros(len(t)), 'rx')
                    sp13.plot(t_factor * (t0 + t),
                              np.angle(np.exp(1j * adc.phase_offset) * np.exp(1j * 2 * np.pi * t * adc.freq_offset)),
                              'b.')

                    if label_defined and len(label_idx_to_plot) != 0:
                        cycler = mpl.cycler(color=label_colors_to_plot)
                        sp11.set_prop_cycle(cycler)
                        label_store_arr = list(label_store.values())
                        lbl_vals = np.take(label_store_arr, label_idx_to_plot)
                        t = t0 + adc.delay + (adc.num_samples - 1) / 2 * adc.dwell
                        p = sp11.plot(t_factor * t, lbl_vals, '.')
                        if len(label_legend_to_plot) != 0:
                            sp11.legend(p, label_legend_to_plot, loc='upper left')
                            label_legend_to_plot = []

                if hasattr(block, 'rf'):
                    rf = block.rf
                    tc, ic = calc_rf_center(rf)
                    t = rf.t + rf.delay
                    tc = tc + rf.delay
                    sp12.plot(t_factor * (t0 + t), np.abs(rf.signal))
                    sp13.plot(t_factor * (t0 + t), np.angle(rf.signal * np.exp(1j * rf.phase_offset)
                                                            * np.exp(1j * 2 * math.pi * rf.t * rf.freq_offset)),
                              t_factor * (t0 + tc), np.angle(rf.signal[ic] * np.exp(1j * rf.phase_offset)
                                                             * np.exp(1j * 2 * math.pi * rf.t[ic] * rf.freq_offset)),
                              'xb')

                grad_channels = ['gx', 'gy', 'gz']
                for x in range(len(grad_channels)):
                    if hasattr(block, grad_channels[x]):
                        grad = getattr(block, grad_channels[x])
                        if grad.type == 'grad':
                            # In place unpacking of grad.t with the starred expression
                            t = grad.delay + [0, *(grad.t + (grad.t[1] - grad.t[0]) / 2),
                                              grad.t[-1] + grad.t[1] - grad.t[0]]
                            waveform = 1e-3 * np.array((grad.first, *grad.waveform, grad.last))
                        else:
                            t = np.cumsum([0, grad.delay, grad.rise_time, grad.flat_time, grad.fall_time])
                            waveform = 1e-3 * grad.amplitude * np.array([0, 0, 1, 1, 0])
                        fig2_subplots[x].plot(t_factor * (t0 + t), waveform)
            t0 += self.arr_block_durations[block_counter]

        grad_plot_labels = ['x', 'y', 'z']
        sp11.set_ylabel('ADC')
        sp12.set_ylabel('RF mag (Hz)')
        sp13.set_ylabel('RF/ADC phase (rad)')
        sp13.set_xlabel('t(s)')
        for x in range(3):
            _label = grad_plot_labels[x]
            fig2_subplots[x].set_ylabel(f'G{_label} (kHz/m)')
        fig2_subplots[-1].set_xlabel('t(s)')

        # Setting display limits
        disp_range = t_factor * np.array([time_range[0], min(t0, time_range[1])])
        [x.set_xlim(disp_range) for x in [sp11, sp12, sp13, *fig2_subplots]]

        fig1.tight_layout()
        fig2.tight_layout()
        if save:
            fig1.savefig('seq_plot1.jpg')
            fig2.savefig('seq_plot2.jpg')
        plt.show()

    def read(self, file_path: str) -> None:
        """
        Read `.seq` file from `file_path`.

        Parameters
        ----------
        file_path : str
            Path to `.seq` file to be read.
        """
        read(self, file_path)

    def rf_from_lib_data(self, lib_data: list) -> SimpleNamespace:
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
        rf.signal = amplitude * mag * np.exp(1j * 2 * np.pi * phase)
        rf.t = np.arange(1, len(mag) + 1) * self.rf_raster_time

        rf.delay = lib_data[3]
        rf.freq_offset = lib_data[4]
        rf.phase_offset = lib_data[5]

        if len(lib_data) < 7:
            lib_data = np.append(lib_data, 0)
        rf.dead_time = lib_data[6]

        if len(lib_data.shape) < 8:
            lib_data = np.append(lib_data, 0)
        rf.ringdown_time = lib_data[7]

        if len(lib_data.shape) < 9:
            lib_data = np.append(lib_data, 0)

        use_cases = {1: 'excitation', 2: 'refocusing', 3: 'inversion'}
        if lib_data[8] in use_cases:
            rf.use = use_cases[lib_data[8]]

        return rf

    def set_definition(self, key: str, val: Union[int, list, np.ndarray, str, tuple]) -> None:
        """
        Sets custom definition to the `Sequence`.

        Parameters
        ----------
        key : str
            Definition key.
        val : int, list, numpy.ndarray, str or tuple
            Definition value.
        """
        if key == 'FOV':
            if max(val) > 1:
                text = 'Definition FOV uses values exceeding 1 m.'
                text += 'New Pulseq interpreters expect values in units of meters.'
                warn(text)

        self.dict_definitions[key] = val

    def set_extension_string_ID(self, extension_str: str, extension_id: int) -> None:
        """
        Set numeric ID for the given string extension ID.

        Parameters
        ----------
        extension_str : str
            Given string extension ID.
        extension_id : int
            Given numeric extension ID.

        Raises
        ------
        ValueError
            If given numeric or string extension ID is not unique.
        """
        if extension_str in self.arr_extension_string_idx or extension_id in self.arr_extension_numeric_idx:
            raise ValueError('Numeric or string ID is not unique')

        self.arr_extension_numeric_idx.append(extension_id)
        self.arr_extension_string_idx.append(extension_str)
        assert len(self.arr_extension_numeric_idx) == len(self.arr_extension_string_idx)

    def test_report(self) -> str:
        """
        Analyze the sequence and return a text report.
        """
        return ext_test_report(self)

    def write(self, name: str) -> None:
        """
        Writes the calling `Sequence` object as a `.seq` file with filename `name`.

        Parameters
        ----------
        name :str
            Filename of `.seq` file to be written to disk.
        """
        write(self, name)
