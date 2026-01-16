import math
from collections import OrderedDict
from copy import deepcopy
from types import SimpleNamespace
from typing import Any, List, Tuple, Union
from warnings import warn

try:
    from typing import Self
except ImportError:
    from typing import TypeVar

    Self = TypeVar('Self', bound='Sequence')

import numpy as np
from scipy.interpolate import PPoly

from pypulseq import __version__, eps
from pypulseq.calc_rf_center import calc_rf_center
from pypulseq.check_timing import check_timing as ext_check_timing
from pypulseq.check_timing import print_error_report
from pypulseq.decompress_shape import decompress_shape
from pypulseq.event_lib import EventLibrary
from pypulseq.opts import Opts
from pypulseq.Sequence import block
from pypulseq.Sequence.calc_grad_spectrum import calculate_gradient_spectrum
from pypulseq.Sequence.calc_pns import calc_pns
from pypulseq.Sequence.ext_test_report import ext_test_report
from pypulseq.Sequence.install import detect_scanner
from pypulseq.Sequence.read_seq import read
from pypulseq.Sequence.write_seq import write as write_seq
from pypulseq.Sequence.write_seq import write_v141 as write_seq_v141
from pypulseq.utils.cumsum import cumsum
from pypulseq.utils.paper_plot import paper_plot as ext_paper_plot
from pypulseq.utils.seq_plot import SeqPlot
from pypulseq.utils.tracing import format_trace, trace, trace_enabled

major, minor, revision = __version__.split('.')[:3]


class Sequence:
    """
    Generate sequences and read/write sequence files. This class defines properties and methods to define a complete MR
    sequence including RF pulses, gradients, ADC events, etc. The class provides an implementation of the open MR
    sequence format defined by the Pulseq project. See http://pulseq.github.io/.

    See also `demo_read.py`, `demo_write.py`.
    """

    version_major = int(major)
    version_minor = int(minor)
    version_revision = revision

    def __init__(self, system: Union[Opts, None] = None, use_block_cache: bool = True):
        if system is None:
            system = Opts()

        # =========
        # EVENT LIBRARIES
        # =========
        self.adc_library = EventLibrary()
        self.delay_library = EventLibrary()
        self.extensions_library = EventLibrary()
        self.grad_library = EventLibrary()
        self.label_inc_library = EventLibrary()
        self.label_set_library = EventLibrary()
        self.rf_library = EventLibrary()
        self.shape_library = EventLibrary(numpy_data=True)
        self.trigger_library = EventLibrary()
        self.soft_delay_library = EventLibrary()

        # =========
        # OTHER
        # =========
        self.system = system

        self.block_events = OrderedDict()
        self.block_trace = OrderedDict()
        self.use_block_cache = use_block_cache
        self.block_cache = {}
        self.next_free_block_ID = 1
        self.definitions = {}

        self.rf_raster_time = self.system.rf_raster_time
        self.grad_raster_time = self.system.grad_raster_time
        self.adc_raster_time = self.system.adc_raster_time
        self.block_duration_raster = self.system.block_duration_raster
        self.set_definition('AdcRasterTime', self.adc_raster_time)
        self.set_definition('BlockDurationRaster', self.block_duration_raster)
        self.set_definition('GradientRasterTime', self.grad_raster_time)
        self.set_definition('RadiofrequencyRasterTime', self.rf_raster_time)
        self.signature_type = ''
        self.signature_file = ''
        self.signature_value = ''
        self.rf_id_to_name_map = {}
        self.adc_id_to_name_map = {}
        self.grad_id_to_name_map = {}

        self.block_durations = {}
        self.extension_numeric_idx = []
        self.extension_string_idx = []
        self.soft_delay_hints = {}

    def __str__(self) -> str:
        s = 'Sequence:'
        s += '\nshape_library: ' + str(self.shape_library)
        s += '\nrf_library: ' + str(self.rf_library)
        s += '\ngrad_library: ' + str(self.grad_library)
        s += '\nadc_library: ' + str(self.adc_library)
        s += '\ndelay_library: ' + str(self.delay_library)
        s += '\nextensions_library: ' + str(self.extensions_library)
        s += '\nrf_raster_time: ' + str(self.rf_raster_time)
        s += '\ngrad_raster_time: ' + str(self.grad_raster_time)
        s += '\nblock_events: ' + str(len(self.block_events))
        return s

    def adc_times(self, time_range: Union[List[float], None] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return time points of ADC sampling points.

        Returns
        -------
        t_adc: np.ndarray
            Contains times of all ADC sample points.
        fp_adc : np.ndarray
            Contains frequency and phase offsets of each ADC object (not samples).
        """
        # Collect ADC timing data
        t_adc = []
        fp_adc = []

        curr_dur = 0
        if time_range is None:
            blocks = self.block_events
        else:
            if len(time_range) != 2:
                raise ValueError('Time range must be list of two elements')
            if time_range[0] > time_range[1]:
                raise ValueError('End time of time_range must be after begin time')

            # Calculate end times of each block
            bd = np.array(list(self.block_durations.values()))
            t = np.cumsum(bd)
            # Search block end times for start of time range
            begin_block = np.searchsorted(t, time_range[0])
            # Search block begin times for end of time range
            end_block = np.searchsorted(t - bd, time_range[1], side='right')
            blocks = list(self.block_durations.keys())[begin_block:end_block]
            curr_dur = t[begin_block] - bd[begin_block]

        for block_counter in blocks:
            block = self.get_block(block_counter)

            if block.adc is not None:
                t_adc.append((np.arange(block.adc.num_samples) + 0.5) * block.adc.dwell + block.adc.delay + curr_dur)
                fp_adc.append([block.adc.freq_offset, block.adc.phase_offset])

            curr_dur += self.block_durations[block_counter]

        if t_adc == []:
            # If there are no ADCs, make sure the output is the right shape
            t_adc = np.zeros(0)
            fp_adc = np.zeros((0, 2))
        else:
            t_adc = np.concatenate(t_adc)
            fp_adc = np.array(fp_adc)

        return t_adc, fp_adc

    def add_block(self, *args: SimpleNamespace) -> None:
        """
        Add a new block/multiple events to the sequence.

        Parameters
        ----------
        *args : SimpleNamespace
            Event objects to be added as a block to the sequence. For delays,
            use `pypulseq.make_delay()` instead of raw float values.

        See Also
        --------
        pypulseq.make_delay : Create delay events
        pypulseq.make_adc : Create ADC events
        pypulseq.make_trapezoid : Create trapezoid gradient events
        pypulseq.make_sinc_pulse : Create sinc RF pulse events
        pypulseq.make_soft_delay : Create soft delay events
        """
        # Validate that no raw floats are passed directly by users
        for arg in args:
            if isinstance(arg, float):
                raise ValueError(
                    f'Raw float values are not allowed in add_block(). Use pp.make_delay({arg}) for delays.'
                )

        if trace_enabled():
            self.block_trace[self.next_free_block_ID] = SimpleNamespace(block=trace())

        block.set_block(self, self.next_free_block_ID, *args)
        self.next_free_block_ID += 1

    def calculate_gradient_spectrum(
        self,
        max_frequency: float = 2000.0,
        window_width: float = 0.05,
        frequency_oversampling: float = 3.0,
        time_range: Union[List[float], None] = None,
        plot: bool = True,
        combine_mode: str = 'max',
        use_derivative: bool = False,
        acoustic_resonances: Union[List[dict], None] = None,
    ) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculates the gradient spectrum of the sequence. Returns a spectrogram
        for each gradient channel, as well as a root-sum-squares combined
        spectrogram.

        Works by splitting the sequence into windows that are 'window_width'
        long and calculating the fourier transform of each window. Windows
        overlap 50% with the previous and next window. When 'combine_mode' is
        not 'none', all windows are combined into one spectrogram.

        Parameters
        ----------
        max_frequency : float, optional
            Maximum frequency to include in spectrograms. The default is 2000.
        window_width : float, optional
            Window width (in seconds). The default is 0.05.
        frequency_oversampling : float, optional
            Oversampling in the frequency dimension, higher values make
            smoother spectrograms. The default is 3.
        time_range : List[float], optional
            Time range over which to calculate the spectrograms as a list of
            two timepoints (in seconds) (e.g. [1, 1.5])
            The default is None.
        plot : bool, optional
            Whether to plot the spectograms. The default is True.
        combine_mode : str, optional
            How to combine all windows into one spectrogram, options:
                'max', 'mean', 'sos' (root-sum-of-squares), 'none' (no combination)
            The default is 'max'.
        use_derivative : bool, optional
            Whether the use the derivative of the gradient waveforms instead of the
            gradient waveforms for the gradient spectrum calculations. The default
            is False
        acoustic_resonances : List[dict], optional
            Acoustic resonances as a list of dictionaries with 'frequency' and
            'bandwidth' elements. Only used when plot==True. The default is [].

        Returns
        -------
        spectrograms : List[np.ndarray]
            List of spectrograms per gradient channel.
        spectrogram_sos : np.ndarray
            Root-sum-of-squares combined spectrogram over all gradient channels.
        frequencies : np.ndarray
            Frequency axis of the spectrograms.
        times : np.ndarray
            Time axis of the spectrograms (only relevant when combine_mode == 'none').

        """
        if acoustic_resonances is None:
            acoustic_resonances = []

        return calculate_gradient_spectrum(
            self,
            max_frequency=max_frequency,
            window_width=window_width,
            frequency_oversampling=frequency_oversampling,
            time_range=time_range,
            plot=plot,
            combine_mode=combine_mode,
            use_derivative=use_derivative,
            acoustic_resonances=acoustic_resonances,
        )

    def calculate_kspace(
        self,
        trajectory_delay: Union[float, List[float], np.ndarray] = 0.0,
        gradient_offset: Union[float, List[float], np.ndarray] = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray, List[float], List[float], np.ndarray]:
        """
        Calculates the k-space trajectory of the entire pulse sequence.

        Parameters
        ----------
        trajectory_delay : float or list or numpy.ndarray, default=0
            Compensation factor in seconds (s) to align ADC and gradients in the reconstruction.
            If trajectory_delay is a single value, this value will be used for all gradient channels.
            If trajectory_delay is a list or array, it is expected to have the same length as the number of gradient
            channels and the first element is applied to the first gradient channel, the second to the second, and so on.
        gradient_offset : float or list or numpy.ndarray, default=0
            Simulates background gradients (specified in Hz/m)
            If gradient_offset is a single value, this value will be used for all gradient channels.
            If gradient_offset is a list or array, it is expected to have the same length as the number of gradient
            channels and the first element is applied to the first gradient channel, the second to the second, and so on.

        Returns
        -------
        k_traj_adc : numpy.array
            K-space trajectory sampled at `t_adc` timepoints.
        k_traj : numpy.array
            K-space trajectory of the entire pulse sequence.
        t_excitation : List[float]
            Excitation timepoints.
        t_refocusing : List[float]
            Refocusing timepoints.
        t_adc : numpy.array
            Sampling timepoints.
        """
        if np.any(np.abs(trajectory_delay) > 100e-6):
            raise Warning(f'Trajectory delay of {trajectory_delay * 1e6} us is suspiciously high')

        total_duration = sum(self.block_durations.values())

        t_excitation, fp_excitation, t_refocusing, _ = self.rf_times()
        t_adc, _ = self.adc_times()

        # Convert data to piecewise polynomials
        gw_pp = self.get_gradients(trajectory_delay, gradient_offset)
        ng = len(gw_pp)

        # Calculate slice positions.
        # For now we entirely rely on the excitation -- ignoring complicated interleaved refocused sequences
        if len(t_excitation) > 0:
            # Position in x, y, z
            slice_pos = np.zeros((ng, len(t_excitation)))
            for j in range(ng):
                if gw_pp[j] is None:
                    slice_pos[j] = np.nan
                else:
                    # Check for divisions by zero to avoid numpy warning
                    divisor = np.array(gw_pp[j](t_excitation))
                    slice_pos[j, divisor != 0.0] = fp_excitation[0, divisor != 0.0] / divisor[divisor != 0.0]
                    slice_pos[j, divisor == 0.0] = np.nan

            slice_pos[~np.isfinite(slice_pos)] = 0  # Reset undefined to 0
        else:
            slice_pos = []

        # Integrate waveforms as PPs to produce gradient moments
        gm_pp = []
        tc = []
        for i in range(ng):
            if gw_pp[i] is None:
                gm_pp.append(None)
                continue

            gm_pp.append(gw_pp[i].antiderivative())
            tc.append(gm_pp[i].x)
            # "Sample" ramps for display purposes.  Otherwise piecewise-linear display (plot) fails
            ii = np.flatnonzero(np.abs(gm_pp[i].c[0, :]) > 1e-7 * self.system.max_slew)

            # Do nothing if there are no ramps
            if ii.shape[0] == 0:
                continue

            starts = np.int64(np.floor((gm_pp[i].x[ii] + eps) / self.grad_raster_time))
            ends = np.int64(np.ceil((gm_pp[i].x[ii + 1] - eps) / self.grad_raster_time))

            # Create all ranges starts[0]:ends[0], starts[1]:ends[1], etc.
            lengths = ends - starts + 1
            inds = np.ones((lengths).sum())
            # Calculate output index where each range will start
            start_inds = np.cumsum(np.concatenate(([0], lengths[:-1])))
            # Create element-wise differences that will cumsum into
            # the final indices: [starts[0], 1, 1, starts[1]-starts[0]-lengths[0]+1, 1, etc.]
            inds[start_inds] = np.concatenate(([starts[0]], np.diff(starts) - lengths[:-1] + 1))

            tc.append(np.cumsum(inds) * self.grad_raster_time)
        if tc != []:
            tc = np.concatenate(tc)

        t_acc = 1e-10  # Temporal accuracy
        t_acc_inv = 1 / t_acc
        # tc = self.__flatten_jagged_arr(tc)
        t_ktraj = t_acc * np.unique(
            np.round(
                t_acc_inv
                * np.array(
                    [
                        *tc,
                        0,
                        *np.asarray(t_excitation) - 2 * self.rf_raster_time,
                        *np.asarray(t_excitation) - self.rf_raster_time,
                        *t_excitation,
                        *np.asarray(t_refocusing) - self.rf_raster_time,
                        *t_refocusing,
                        *t_adc,
                        total_duration,
                    ]
                )
            )
        )

        i_excitation = np.searchsorted(t_ktraj, t_acc * np.round(t_acc_inv * np.asarray(t_excitation)))
        i_refocusing = np.searchsorted(t_ktraj, t_acc * np.round(t_acc_inv * np.asarray(t_refocusing)))
        i_adc = np.searchsorted(t_ktraj, t_acc * np.round(t_acc_inv * np.asarray(t_adc)))

        i_periods = np.unique([0, *i_excitation, *i_refocusing, len(t_ktraj) - 1])
        if len(i_excitation) > 0:
            ii_next_excitation = 0
        else:
            ii_next_excitation = -1
        if len(i_refocusing) > 0:
            ii_next_refocusing = 0
        else:
            ii_next_refocusing = -1

        k_traj = np.zeros((ng, len(t_ktraj)))
        for i in range(ng):
            if gw_pp[i] is None:
                continue

            it = np.where(
                np.logical_and(
                    t_ktraj >= t_acc * round(t_acc_inv * gm_pp[i].x[0]),
                    t_ktraj <= t_acc * round(t_acc_inv * gm_pp[i].x[-1]),
                )
            )[0]
            k_traj[i, it] = gm_pp[i](t_ktraj[it])
            if t_ktraj[it[-1]] < t_ktraj[-1]:
                k_traj[i, it[-1] + 1 :] = k_traj[i, it[-1]]

        # Convert gradient moments to k-space positions
        dk = -k_traj[:, 0]
        for i in range(len(i_periods) - 1):
            i_period = i_periods[i]
            i_period_end = i_periods[i + 1]
            if ii_next_excitation >= 0 and i_excitation[ii_next_excitation] == i_period:
                if abs(t_ktraj[i_period] - t_excitation[ii_next_excitation]) > t_acc:
                    raise Warning(
                        f'abs(t_ktraj[i_period]-t_excitation[ii_next_excitation]) < {t_acc} failed for ii_next_excitation={ii_next_excitation} error={t_ktraj(i_period) - t_excitation(ii_next_excitation)}'
                    )
                dk = -k_traj[:, i_period]
                if i_period > 0:
                    # Use nans to mark the excitation points since they interrupt the plots
                    k_traj[:, i_period - 1] = np.nan
                # -1 on len(i_excitation) for 0-based indexing
                ii_next_excitation = min(len(i_excitation) - 1, ii_next_excitation + 1)
            elif ii_next_refocusing >= 0 and i_refocusing[ii_next_refocusing] == i_period:
                # dk = -k_traj[:, i_period]
                dk = -2 * k_traj[:, i_period] - dk
                # -1 on len(i_excitation) for 0-based indexing
                ii_next_refocusing = min(len(i_refocusing) - 1, ii_next_refocusing + 1)

            k_traj[:, i_period:i_period_end] = k_traj[:, i_period:i_period_end] + dk[:, None]

        k_traj[:, i_period_end] = k_traj[:, i_period_end] + dk
        k_traj_adc = k_traj[:, i_adc]

        return k_traj_adc, k_traj, t_excitation, t_refocusing, t_adc

    def calculate_kspacePP(
        self,
        trajectory_delay: Union[float, List[float], np.ndarray] = 0,
        gradient_offset: Union[float, List[float], np.ndarray] = 0,
    ) -> Tuple[np.array, np.array, np.array, np.array, np.array]:
        warn(
            'Sequence.calculate_kspacePP has been deprecated, use calculate_kspace instead',
            DeprecationWarning,
            stacklevel=2,
        )
        return self.calculate_kspace(trajectory_delay, gradient_offset)

    def calculate_pns(
        self,
        hardware: SimpleNamespace,
        time_range: Union[List[float], None] = None,
        do_plots: bool = True,
    ) -> Tuple[bool, np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate PNS using safe model implementation by Szczepankiewicz and Witzel
        See http://github.com/filip-szczepankiewicz/safe_pns_prediction

        Returns pns levels due to respective axes (normalized to 1 and not to 100#)

        Parameters
        ----------
        hardware : SimpleNamespace
            Hardware specifications. See safe_example_hw() from
            the safe_pns_prediction package. Alternatively a text file
            in the .asc format (Siemens) can be passed, e.g. for Prisma
            it is MP_GPA_K2309_2250V_951A_AS82.asc (we leave it as an
            exercise to the interested user to find were these files
            can be acquired from)
        do_plots : bool, optional
            Plot the results from the PNS calculations. The default is True.

        Returns
        -------
        ok : bool
            Boolean flag indicating whether peak PNS is within acceptable limits
        pns_norm : numpy.array [N]
            PNS norm over all gradient channels, normalized to 1
        pns_components : numpy.array [Nx3]
            PNS levels per gradient channel
        t_pns : np.array [N]
            Time axis for the pns_norm and pns_components arrays
        """
        return calc_pns(self, hardware, time_range=time_range, do_plots=do_plots)

    def check_timing(self, print_errors=False) -> Tuple[bool, List[SimpleNamespace]]:
        """
        Checks timing of all blocks and objects in the sequence optionally returns the detailed error log.

        Returns
        -------
        is_ok : bool
            Boolean flag indicating timing errors.
        error_report : List[SimpleNamespace]
            Error report in case of timing errors.
        """
        is_ok, error_report = ext_check_timing(self)

        if not is_ok and print_errors:
            print_error_report(self, error_report)

        return is_ok, error_report

    def duration(self) -> Tuple[int, int, np.ndarray]:
        """
        Returns the total duration of this sequence, and the total count of blocks and events.

        Returns
        -------
        duration : int
            Duration of this sequence in seconds (s).
        num_blocks : int
            Number of blocks in this sequence.
        event_count : np.ndarray
            Number of events in this sequence.
        """
        num_blocks = len(self.block_events)
        event_count = np.zeros(len(next(iter(self.block_events.values()))))
        duration = 0
        for block_counter in self.block_events:
            event_count += self.block_events[block_counter] > 0
            duration += self.block_durations[block_counter]

        return duration, num_blocks, event_count

    def evaluate_labels(self, init: Union[dict, None] = None, evolution: str = 'none') -> dict:
        """
        Evaluate label values of the entire sequence.

        When no evolution is given, returns the label values at the end of the
        sequence. Returns a dictionary with keys named after the labels used
        in the sequence. Only the keys corresponding to the labels actually
        used are created.
        E.g. labels['LIN'] == 4

        When evolution is given, labels are tracked through the sequence. See
        below for options for different types of evolutions. The resulting
        dictionary will contain arrays of the label values.
        E.g. labels['LIN'] == np.array([0,1,2,3,4])

        Initial values for the labels can be given with the 'init' parameter.
        Useful if evaluating labels block-by-block.

        Parameters
        ----------
        init : dict, optional
            Dictionary containing initial label values. The default is None.
        evolution : str, optional
            Flag to specify tracking of label evolutions.
            Must be one of: 'none', 'adc', 'label', 'blocks' (default = 'none')
            'blocks': Return label values for all blocks.
            'adc':    Return label values only for blocks containing ADC events.
            'label':  Return label values only for blocks where labels are
                      manipulated.

        Returns
        -------
        labels : dict
            Dictionary containing label values.
            If evolution == 'none', the dictionary values only contains the
            final label value.
            Otherwise, the dictionary values are arrays of label evolutions.
            Only the labels that are used in the sequence are created in the
            dictionary.

        """
        labels = init or {}
        label_evolution = []

        # TODO: MATLAB implementation includes block_range parameter. But in
        #       general we cannot assume linear block ordering. Could include
        #       time_range like in other sequence functions. Or a blocks
        #       parameter to specify which blocks to loop over?
        for block_counter in self.block_events:
            block = self.get_block(block_counter)

            if block.label is not None:
                # Current block has labels
                for lab in block.label.values():
                    if lab.type == 'labelinc':
                        # Increment label
                        if lab.label not in labels:
                            labels[lab.label] = 0

                        labels[lab.label] += lab.value
                    else:
                        # Set label
                        labels[lab.label] = lab.value

                if evolution == 'label':
                    label_evolution.append(dict(labels))

            if evolution == 'blocks' or (evolution == 'adc' and block.adc is not None):
                label_evolution.append(dict(labels))

        # Convert evolutions into label dictionary
        if len(label_evolution) > 0:
            for lab in labels:
                labels[lab] = np.array([e.get(lab, 0) for e in label_evolution])

        return labels

    import numpy as np

    def find_block_by_time(self, t: float) -> int:
        """
        Find the index of the block containing time `t`.

        Parameters
        ----------
        t : float
            Time (in seconds) to locate within the sequence.

        Returns
        -------
        int or None
            Index of the block that contains the given time, or None if out of range.
        """
        cumsum_durations = np.cumsum(list(self.block_durations.values()))
        block_index = np.searchsorted(cumsum_durations, t, side='right').item()

        if block_index >= len(self.block_durations):
            return None

        if self.block_durations[block_index] <= 0:
            raise ValueError('Block duration cannot be negative')

        return block_index

    def flip_grad_axis(self, axis: str) -> None:
        """
        Invert all gradients along the corresponding axis/channel. The function acts on all gradient objects already
        added to the sequence object.

        Parameters
        ----------
        axis : str
            Gradients to invert or scale. Must be one of 'x', 'y' or 'z'.
        """
        self.mod_grad_axis(axis, modifier=-1)

    def get_raw_block_content_IDs(self, block_index: int) -> SimpleNamespace:
        """
        Returns PyPulseq block content IDs at `block_index` position in `self.block_events`.

        No block events are created, only the IDs of the objects are returned.

        See Also
        --------
        - `pypulseq.Sequence.sequence.Sequence.get_block()`.

        Parameters
        ----------
        block_index : int
            Index of block to be retrieved from `Sequence`.

        Returns
        -------
        SimpleNamespace
            PyPulseq block content IDs at 'block_index' position in `self.block_events`.
        """
        return block.get_raw_block_content_IDs(self, block_index)

    def get_block(self, block_index: int) -> SimpleNamespace:
        """
        Return a block of the sequence  specified by the index. The block is created from the sequence data with all
        events and shapes decompressed.

        See Also
        --------
        - `pypulseq.Sequence.sequence.Sequence.set_block()`.
        - `pypulseq.Sequence.sequence.Sequence.add_block()`.

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
        Return value of the definition specified by the key. These definitions can be added manually or read from the
        header of a sequence file defined in the sequence header. An empty array is returned if the key is not defined.

        See also `pypulseq.Sequence.sequence.Sequence.set_definition()`.

        Parameters
        ----------
        key : str
            Key of definition to retrieve.

        Returns
        -------
        str
            Definition identified by `key` if found, else returns ''.
        """
        if key in self.definitions:
            return self.definitions[key]
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
        if extension_string not in self.extension_string_idx:
            if len(self.extension_numeric_idx) == 0:
                extension_id = 1
            else:
                extension_id = 1 + max(self.extension_numeric_idx)

            self.extension_numeric_idx.append(extension_id)
            self.extension_string_idx.append(extension_string)
            assert len(self.extension_numeric_idx) == len(self.extension_string_idx)
        else:
            num = self.extension_string_idx.index(extension_string)
            extension_id = self.extension_numeric_idx[num]

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
        if extension_id in self.extension_numeric_idx:
            num = self.extension_numeric_idx.index(extension_id)
        else:
            raise ValueError(f'Extension for the given ID - {extension_id} - is unknown.')

        extension_str = self.extension_string_idx[num]
        return extension_str

    def get_gradients(
        self,
        trajectory_delay: Union[float, List[float], np.ndarray] = 0,
        gradient_offset: Union[float, List[float], np.ndarray] = 0,
        time_range: Union[List[float], None] = None,
    ) -> List[PPoly]:
        """
        Get all gradient waveforms of the sequence in a piecewise-polynomial
        format (scipy PPoly). Gradient values can be accessed easily at one or
        more timepoints using `gw_pp[channel](t)` (where t is a float, list of
        floats, or numpy array). Note that PPoly objects return nan for
        timepoints outside the waveform.

        Parameters
        ----------
        trajectory_delay : float or list or numpy.ndarray, default=0
            Compensation factor in seconds (s) to align ADC and gradients in the reconstruction.
            If trajectory_delay is a single value, this value will be used for all gradient channels.
            If trajectory_delay is a list or array, it is expected to have the same length as the number of gradient
            channels and the first element is applied to the first gradient channel, the second to the second, and so on.
        gradient_offset : float or list or numpy.ndarray, default=0
            Simulates background gradients (specified in Hz/m)
            If gradient_offset is a single value, this value will be used for all gradient channels.
            If gradient_offset is a list or array, it is expected to have the same length as the number of gradient
            channels and the first element is applied to the first gradient channel, the second to the second, and so on.

        Returns
        -------
        gw_pp : List[PPoly]
            List of gradient waveforms for each of the gradient channels,
            expressed as scipy PPoly objects.
        """
        if np.any(np.abs(trajectory_delay) > 100e-6):
            raise Warning(f'Trajectory delay of {trajectory_delay * 1e6} us is suspiciously high')

        total_duration = sum(self.block_durations.values())

        gw_data = self.waveforms(time_range=time_range)
        ng = len(gw_data)

        # Gradient delay handling
        if isinstance(trajectory_delay, (int, float)):
            gradient_delays = [trajectory_delay] * ng
        else:
            assert len(trajectory_delay) == ng  # Need to have same number of gradient channels
            gradient_delays = trajectory_delay

        # Gradient offset handling
        if isinstance(gradient_offset, (int, float)):
            gradient_offset = [gradient_offset] * ng
        else:
            assert len(gradient_offset) == ng  # Need to have same number of gradient channels

        # Convert data to piecewise polynomials
        gw_pp = []
        for j in range(ng):
            wave_cnt = gw_data[j].shape[1]
            if wave_cnt == 0:
                if np.abs(gradient_offset[j]) <= eps:
                    gw_pp.append(None)
                    continue
                else:
                    gw = np.array(([0, total_duration], [0, 0]))
            else:
                gw = gw_data[j]

            # Now gw contains the waveform from the current axis
            if np.abs(gradient_delays[j]) > eps:
                gw[0] = gw[0] - gradient_delays[j]  # Anisotropic gradient delay support
            if not np.all(np.isfinite(gw)):
                raise Warning('Not all elements of the generated waveform are finite.')

            teps = 1e-12
            _temp1 = np.array(([gw[0, 0] - 2 * teps, gw[0, 0] - teps], [0, 0]))
            _temp2 = np.array(([gw[0, -1] + teps, gw[0, -1] + 2 * teps], [0, 0]))
            gw = np.hstack((_temp1, gw, _temp2))

            if np.abs(gradient_offset[j]) > eps:
                gw[1, :] += gradient_offset[j]

            gw[1][gw[1] == -0.0] = 0.0

            gw_pp.append(PPoly(np.stack((np.diff(gw[1]) / np.diff(gw[0]), gw[1][:-1])), gw[0], extrapolate=True))
        return gw_pp

    def install(self, target: Union[str, None] = None, clear_cache: bool = False, **kwargs: Any) -> None:
        """Install a sequence to a target scanner.

        The sequence will be installed to a scanner specified by `target`. If `target` is not specified,
        all known scanners will be attempted to be detected. Targets supported by PyPulseq:
            siemens: All siemens targets below
            siemens_nx: Siemens Numaris X
            siemens_n4: Siemens Numaris 4
            siemens_n4_2: Siemens Numeris 4 on IP 192.168.2.2
            siemens_n4_3: Siemens Numeris 4 on IP 192.168.2.3
        Once a scanner is successfully detected, this result will be cached so
        `install` will operate more quickly on successive calls. The cache can
        be cleared by specifying `clear_cache=True`.
        Parameters
        ----------
        target : str, optional
            Target scanner. The default is None.
        clear_cache : bool, optional
            Clear the scanner detection cache. The default is False.
        **kwargs : Any
            Keyword arguments to be passed to the target's `install` function.
        Raises
        ------
        RuntimeError
            If the scanner could not be detected, or if the installation failed.
        """
        name, definition = detect_scanner(target, clear_cache=clear_cache)

        if definition is None:
            raise RuntimeError('Scanner could not be detected')

        if not definition.install(self, **kwargs):
            raise RuntimeError('Sequence install failed')

        print(f'Sequence installed correctly on target `{name}`')

    def mod_grad_axis(self, axis: str, modifier: float) -> None:
        """
        Invert or scale all gradients along the corresponding axis/channel.

        The function acts on all gradient objects already added to the sequence object.
        This modifies the amplitude of gradients while preserving timing parameters
        (rise_time, flat_time, fall_time, delay). The gradient area scales proportionally
        with the amplitude.

        Parameters
        ----------
        axis : str
            Gradients to invert or scale. Must be one of 'x', 'y' or 'z'.
        modifier : float
            Scaling value.

        Raises
        ------
        ValueError
            If invalid `axis` is passed. Must be one of 'x', 'y','z'.
        RuntimeError
            If same gradient event is used on multiple axes.

        Examples
        --------
        Disable phase encoding gradients (y-axis):

        >>> seq.mod_grad_axis('y', 0.0)

        Invert readout gradients (x-axis):

        >>> seq.mod_grad_axis('x', -1.0)

        Reduce slice selection gradients by half (z-axis):

        >>> seq.mod_grad_axis('z', 0.5)

        Double all gradients on x-axis:

        >>> seq.mod_grad_axis('x', 2.0)

        Notes
        -----
        - Only amplitude-related parameters are modified (amplitude, area, first, last)
        - Timing parameters remain unchanged (rise_time, flat_time, fall_time, delay)
        - For arbitrary gradients, the entire waveform is scaled
        - For trapezoid gradients, only the amplitude is scaled
        - Setting modifier to 0.0 effectively disables gradients on that axis
        """
        if axis not in ['x', 'y', 'z']:
            raise ValueError(f"Invalid axis. Must be one of 'x', 'y','z'. Passed: {axis}")

        channel_num = ['x', 'y', 'z'].index(axis)
        other_channels = [0, 1, 2]
        other_channels.remove(channel_num)

        # Go through all event table entries and list gradient objects in the library
        if len(self.block_events) == 0:
            # Empty sequence - nothing to modify
            return

        all_grad_events = np.array(list(self.block_events.values()))
        all_grad_events = all_grad_events[:, 2:5]

        selected_events = np.unique(all_grad_events[:, channel_num])
        selected_events = selected_events[selected_events != 0]
        other_events = np.unique(all_grad_events[:, other_channels])
        if len(np.intersect1d(selected_events, other_events)) > 0:
            raise RuntimeError('mod_grad_axis does not yet support the same gradient event used on multiple axes.')

        for i in range(len(selected_events)):
            event_id = selected_events[i]
            old_data = self.grad_library.data[event_id]
            grad_type = self.grad_library.type[event_id]

            # Convert tuple to list for modification
            grad_data = list(old_data)
            grad_data[0] *= modifier

            if grad_type == 'g' and len(grad_data) == 6:
                # Need to update first and last fields for arbitrary gradients
                # Data structure: (amplitude, first, last, shape_ID1, shape_ID2, delay) # changed in v1.5.x
                grad_data[1] *= modifier  # first # changed in v1.5.x
                grad_data[2] *= modifier  # last # changed in v1.5.x

            # Use EventLibrary.update() to properly maintain keymap integrity
            new_data = tuple(grad_data)
            self.grad_library.update(event_id, old_data, new_data, grad_type)

        # Clear block cache to ensure get_block() uses the modified gradient data
        if self.use_block_cache:
            self.block_cache.clear()

    def paper_plot(
        self,
        time_range: Tuple[float] = (0, np.inf),
        line_width: float = 1.2,
        axes_color: Tuple[float] = (0.5, 0.5, 0.5),
        rf_color: str = 'black',
        gx_color: str = 'blue',
        gy_color: str = 'red',
        gz_color: Tuple[float] = (0, 0.5, 0.3),
        rf_plot: str = 'abs',
    ):
        """
        Plot sequence using paper-style formatting (minimalist, high-contrast layout).

        Parameters
        ----------
        time_range : iterable, default=(0, np.inf)
            Time range (x-axis limits) for plotting the sequence.
            Default is 0 to infinity (entire sequence).
        line_width : float, default=1.2
            Line width used in plots.
        axes_color : color, default=(0.5, 0.5, 0.5)
            Color of horizontal zero axes (e.g., gray).
        rf_color : color, default='black'
            Color for RF and ADC events.
        gx_color : color, default='blue'
            Color for gradient X waveform.
        gy_color : color, default='red'
            Color for gradient Y waveform.
        gz_color : color, default=(0, 0.5, 0.3)
            Color for gradient Z waveform.
        rf_plot : {'abs', 'real', 'imag'}, default='abs'
            Determines how to plot RF waveforms (magnitude, real or imaginary part).

        """
        ext_paper_plot(self, time_range, line_width, axes_color, rf_color, gx_color, gy_color, gz_color, rf_plot)

    def plot(
        self,
        label: str = str(),
        show_blocks: bool = False,
        save: bool = False,
        time_range=(0, np.inf),
        time_disp: str = 's',
        grad_disp: str = 'kHz/m',
        plot_now: bool = True,
        clear: bool = True,
        overlay: SeqPlot = None,
        stacked: bool = False,
        show_guides: bool = False,
    ) -> SeqPlot:
        """
        Plot `Sequence`.

        Parameters
        ----------
        label : str, default=str()
            Plot label values for ADC events: in this example for LIN and REP labels; other valid labels are accepted as
            a comma-separated list.
        show_blocks : bool, default=False
            Boolean flag to indicate if grid and tick labels at the block boundaries are to be plotted.
        save : bool, default=False
            Boolean flag indicating if plots should be saved. The two figures will be saved as JPG with numerical
            suffixes to the filename 'seq_plot'.
        time_range : iterable, default=(0, np.inf)
            Time range (x-axis limits) for plotting the sequence. Default is 0 to infinity (entire sequence).
        time_disp : str, default='s'
            Time display type, must be one of 's', 'ms' or 'us'.
        grad_disp : str, default='kHz/m'
            Gradient display unit, must be one of 'kHz/m' or 'mT/m'.
        plot_now : bool, default=True
            If True, function immediately shows the plots, blocking the rest of the code until plots are exited.
            If False, plots are shown when plt.show() is called. Useful if plots are to be modified.
        clear : bool, default=True
            If True, clear existing figures before plotting (default behavior).
            If False, overlay on existing figures 1 and 2 for sequence comparison.
        overlay : SeqPlot or None, default=None
            If provided, overlay this plot on the figures from the given SeqPlot object. Overrides fig1, fig2, and sets clear=False.
        stacked : bool, default=False
            If True, plot all channels (ADC, RF mag, RF phase, Gx, Gy, Gz) in a single stacked figure (MATLAB Pulseq style).
            If False, use separate figures for RF/ADC and gradients.
        show_guides : bool, default=False
            If True, enable dynamic vertical hairline guides that follow the cursor. Requires `mplcursors`.

        Returns
        -------
        SeqPlot
            SeqPlot handle.
        """
        return SeqPlot(
            self,
            label,
            show_blocks,
            save,
            time_range,
            time_disp,
            grad_disp,
            plot_now,
            clear,
            overlay,
            stacked,
            show_guides,
        )

    def read(self, file_path: str, detect_rf_use: bool = False, remove_duplicates: bool = True) -> None:
        """
        Read `.seq` file from `file_path`.

        Parameters
        ----------
        detect_rf_use
        file_path : str
            Path to `.seq` file to be read.
        remove_duplicates : bool, default=True
            Remove duplicate events from the sequence after reading.
        """
        if self.use_block_cache:
            self.block_cache.clear()

        read(self, path=file_path, detect_rf_use=detect_rf_use, remove_duplicates=remove_duplicates)

        # Initialize next free block ID
        self.next_free_block_ID = (max(self.block_events) + 1) if self.block_events else 1

    def register_adc_event(self, event: EventLibrary) -> int:
        return block.register_adc_event(self, event)

    def register_grad_event(self, event: SimpleNamespace) -> Union[int, Tuple[int, int]]:
        return block.register_grad_event(self, event)

    def register_label_event(self, event: SimpleNamespace) -> int:
        return block.register_label_event(self, event)

    def register_rf_event(self, event: SimpleNamespace) -> Tuple[int, List[int]]:
        return block.register_rf_event(self, event)

    def register_soft_delay_event(self, event: SimpleNamespace) -> int:
        return block.register_soft_delay_event(self, event)

    def remove_duplicates(self, in_place: bool = False) -> 'Sequence':
        """
        Removes duplicate events from the shape and event libraries contained
        in this sequence.

        Parameters
        ----------
        in_place : bool, optional
            If true, removes the duplicates from the current sequence.
            Otherwise, a copy is created. The default is False.

        Returns
        -------
        seq_copy : Sequence
            If `in_place`, returns self. Otherwise returns a copy of the
            sequence.
        """
        if in_place:
            seq_copy = self
        else:
            # Avoid copying block_cache for performance
            tmp = self.block_cache
            self.block_cache = {}
            seq_copy = deepcopy(self)
            self.block_cache = tmp

        # Find duplicate in shape library
        seq_copy.shape_library, mapping = seq_copy.shape_library.remove_duplicates(9)

        # Remap shape IDs of arbitrary gradient events
        for grad_id in seq_copy.grad_library.data:
            if seq_copy.grad_library.type[grad_id] == 'g':
                data = seq_copy.grad_library.data[grad_id]
                new_data = (*data[0:3], mapping[data[3]], mapping[data[4]], data[5])
                if data != new_data:
                    seq_copy.grad_library.update(grad_id, None, new_data)

        # Remap shape IDs of RF events
        for rf_id in seq_copy.rf_library.data:
            data = seq_copy.rf_library.data[rf_id]
            new_data = (data[0], mapping[data[1]], mapping[data[2]], mapping[data[3]], *data[4:])
            if data != new_data:
                seq_copy.rf_library.update(rf_id, None, new_data)

        # Filter duplicates in gradient library
        seq_copy.grad_library, mapping = seq_copy.grad_library.remove_duplicates((6, -6, -6, -6, -6, -6))

        # Remap gradient event IDs
        for block_id in seq_copy.block_events:
            seq_copy.block_events[block_id][2] = mapping[seq_copy.block_events[block_id][2]]
            seq_copy.block_events[block_id][3] = mapping[seq_copy.block_events[block_id][3]]
            seq_copy.block_events[block_id][4] = mapping[seq_copy.block_events[block_id][4]]

        # Filter duplicates in RF library
        seq_copy.rf_library, mapping = seq_copy.rf_library.remove_duplicates((6, 0, 0, 0, 6, 6, 6, 6, 6, 6))

        # Remap RF event IDs
        for block_id in seq_copy.block_events:
            seq_copy.block_events[block_id][1] = mapping[seq_copy.block_events[block_id][1]]

        # Filter duplicates in ADC library
        seq_copy.adc_library, mapping = seq_copy.adc_library.remove_duplicates((0, -9, -6, 6, 6, 6, 6, 6, 6))

        # Remap ADC event IDs
        for block_id in seq_copy.block_events:
            seq_copy.block_events[block_id][5] = mapping[seq_copy.block_events[block_id][5]]

        return seq_copy

    def rf_from_lib_data(self, lib_data: list, use: str = '') -> SimpleNamespace:
        """
        Construct RF object from `lib_data`.

        Parameters
        ----------
        lib_data : list
            RF envelope.
        use : str, default=''
            RF event use.

        Returns
        -------
        rf : SimpleNamespace
            RF object constructed from `lib_data`.
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
        rf.signal = amplitude * mag * np.exp(1j * 2 * math.pi * phase)
        time_shape = lib_data[3]
        if time_shape > 0:
            shape_data = self.shape_library.data[time_shape]
            compressed.num_samples = shape_data[0]
            compressed.data = shape_data[1:]
            rf.t = decompress_shape(compressed) * self.rf_raster_time
            rf.shape_dur = math.ceil((rf.t[-1] - eps) / self.rf_raster_time) * self.rf_raster_time
        else:  # Generate default time raster on the fly
            rf.t = (np.arange(1, len(rf.signal) + 1) - 0.5) * self.rf_raster_time
            rf.shape_dur = len(rf.signal) * self.rf_raster_time

        rf.center = lib_data[4]  # v150: new field
        rf.delay = lib_data[5]  # v150: changed from lib_data[4] to lib_data[5]
        rf.freq_ppm = lib_data[6]  # v150: new field
        rf.phase_ppm = lib_data[7]  # v150: new field
        rf.freq_offset = lib_data[8]  # v150: changed from lib_data[5] to lib_data[8]
        rf.phase_offset = lib_data[9]  # v150: changed from lib_data[6] to lib_data[9]

        rf.dead_time = self.system.rf_dead_time
        rf.ringdown_time = self.system.rf_ringdown_time

        # TODO: fixme : use map built from pp.get_supported_rf_uses()
        use_cases = {
            'e': 'excitation',
            'r': 'refocusing',
            'i': 'inversion',
            's': 'saturation',
            'p': 'preparation',
            'o': 'other',
        }
        rf.use = use_cases.get(use, 'undefined')

        return rf

    def rf_times(
        self, time_range: Union[List[float], None] = None
    ) -> Tuple[List[float], np.ndarray, List[float], np.ndarray, np.ndarray]:
        """
        Return time points of excitations and refocusings.

        Returns
        -------
        t_excitation : List[float]
            Contains time moments of the excitation RF pulses
        fp_excitation : np.ndarray
            Contains frequency and phase offsets of the excitation RF pulses
        t_refocusing : List[float]
            Contains time moments of the refocusing RF pulses
        fp_refocusing : np.ndarray
            Contains frequency and phase offsets of the excitation RF pulses
        """
        # Collect RF timing data
        t_excitation = []
        fp_excitation = []
        t_refocusing = []
        fp_refocusing = []

        curr_dur = 0
        if time_range is None:
            blocks = self.block_events
        else:
            if len(time_range) != 2:
                raise ValueError('Time range must be list of two elements')
            if time_range[0] > time_range[1]:
                raise ValueError('End time of time_range must be after begin time')

            # Calculate end times of each block
            bd = np.array(list(self.block_durations.values()))
            t = np.cumsum(bd)
            # Search block end times for start of time range
            begin_block = np.searchsorted(t, time_range[0])
            # Search block begin times for end of time range
            end_block = np.searchsorted(t - bd, time_range[1], side='right')
            blocks = list(self.block_durations.keys())[begin_block:end_block]
            curr_dur = t[begin_block] - bd[begin_block]

        for block_counter in blocks:
            block = self.get_block(block_counter)

            if block.rf is not None:
                rf = block.rf

                tc = calc_rf_center(rf)[0]
                t = rf.delay + tc

                full_freq_offset = rf.freq_offset + rf.freq_ppm * 1e-6 * self.system.gamma * self.system.B0
                full_phase_offset = rf.phase_offset + rf.phase_ppm * 1e-6 * self.system.gamma * self.system.B0
                full_phase_offset = full_phase_offset + 2 * math.pi * full_freq_offset * tc

                if not hasattr(rf, 'use') or block.rf.use in [
                    'excitation',
                    'undefined',
                ]:
                    t_excitation.append(curr_dur + t)
                    fp_excitation.append([full_freq_offset, full_phase_offset])
                elif block.rf.use == 'refocusing':
                    t_refocusing.append(curr_dur + t)
                    fp_refocusing.append([full_freq_offset, full_phase_offset])

            curr_dur += self.block_durations[block_counter]

        if len(t_excitation) != 0:
            fp_excitation = np.array(fp_excitation).T
        else:
            fp_excitation = np.empty((2, 0))

        if len(t_refocusing) != 0:
            fp_refocusing = np.array(fp_refocusing).T
        else:
            fp_refocusing = np.empty((2, 0))

        return t_excitation, fp_excitation, t_refocusing, fp_refocusing

    def set_block(self, block_index: int, *args: SimpleNamespace) -> None:
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
        """
        if trace_enabled():
            self.block_trace[block_index] = SimpleNamespace(block=trace())

        block.set_block(self, block_index, *args)

        if block_index >= self.next_free_block_ID:
            self.next_free_block_ID = block_index + 1

    def set_definition(self, key: str, value: Union[float, int, list, np.ndarray, str, tuple]) -> None:
        """
        Modify a custom definition of the sequence. Set the user definition 'key' to value 'value'. If the definition
        does not exist it will be created.

        See also `pypulseq.Sequence.sequence.Sequence.get_definition()`.

        Parameters
        ----------
        key : str
            Definition key.
        value : int, list, np.ndarray, str or tuple
            Definition value.
        """
        if key == 'FOV' and np.max(value) > 1:
            text = 'Definition FOV uses values exceeding 1 m. '
            text += 'New Pulseq interpreters expect values in units of meters.'
            warn(text)

        self.definitions[key] = value

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
        if extension_str in self.extension_string_idx or extension_id in self.extension_numeric_idx:
            raise ValueError('Numeric or string ID is not unique')

        self.extension_numeric_idx.append(extension_id)
        self.extension_string_idx.append(extension_str)
        assert len(self.extension_numeric_idx) == len(self.extension_string_idx)

    def apply_soft_delay(self, **kwargs):
        """
        Apply soft delay values to modify block durations in the sequence.

        This method updates the durations of blocks containing soft delay events
        based on the provided values. The new block duration is calculated using
        the formula: duration = (input_value / factor) + offset.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments where keys are soft delay hint strings and values
            are the desired delay values in seconds. Not all soft delays in the
            sequence need to be specified.

        Raises
        ------
        ValueError
            If a specified soft delay hint does not exist in the sequence.
        ValueError
            If the calculated block duration would be negative.
        ValueError
            If soft delay hint and numeric ID mapping is inconsistent.

        Examples
        --------
        Apply single soft delay:

        >>> seq.apply_soft_delay(TE=40e-3)  # Set TE to 40ms

        Apply multiple soft delays:

        >>> seq.apply_soft_delay(TE=50e-3, TR=2.0)  # Set TE to 50ms, TR to 2s

        See Also
        --------
        pypulseq.make_soft_delay : Create soft delay events

        Notes
        -----
        - Only soft delays present in the sequence can be modified
        - Block durations are automatically rounded to the block duration raster
        - A warning is issued if substantial rounding occurs
        """

        # Go through all the blocks and update durations, at the same time checking the consistency of the soft delays
        sd_str2numID = {}
        sd_numID2hint = {}
        sd_warns = {}
        for block_counters in self.block_events:
            block = self.get_block(block_counters)
            if block.soft_delay is not None:
                # Check the numeric ID consistency
                if block.soft_delay.hint not in sd_str2numID:
                    sd_str2numID[block.soft_delay.hint] = block.soft_delay.numID
                else:
                    if sd_str2numID[block.soft_delay.hint] != block.soft_delay.numID:
                        raise ValueError(
                            f"Soft delay in block {block_counters} with numeric ID {block.soft_delay.numID} and string hint '{block.soft_delay.hint}' is inconsistent with the previous occurrences of the same string hint"
                        )

                if block.soft_delay.numID not in sd_numID2hint:
                    sd_numID2hint[block.soft_delay.numID] = block.soft_delay.hint
                else:
                    if sd_numID2hint[block.soft_delay.numID] != block.soft_delay.hint:
                        raise ValueError(
                            f"Soft delay in block {block_counters} with numeric ID {block.soft_delay.numID} and string hint '{block.soft_delay.hint}' is inconsistent with the previous occurrences of the same numeric ID"
                        )

                if block.soft_delay.hint in kwargs:
                    # Calculate the new block duration
                    new_dur_ru = (
                        kwargs[block.soft_delay.hint] / block.soft_delay.factor + block.soft_delay.offset
                    ) / self.system.block_duration_raster
                    new_dur = round(new_dur_ru) * self.system.block_duration_raster
                    # Check if rounding error is significant (threshold: 0.5 microseconds)
                    rounding_threshold = 0.5e-6
                    rounding_error = abs(new_dur - new_dur_ru * self.system.block_duration_raster)
                    if rounding_error > rounding_threshold and block.soft_delay.numID not in sd_warns:
                        warn(
                            f"Soft delay '{block.soft_delay.hint}' in block {block_counters}: "
                            f'Duration rounded by {rounding_error * 1e6:.1f} μs to align with raster time '
                            f'({self.system.block_duration_raster * 1e6:.1f} μs). '
                            f'This warning is shown only once per soft delay ID.'
                        )
                        sd_warns[block.soft_delay.numID] = True

                    if new_dur < 0:
                        raise ValueError(
                            f"Soft delay '{block.soft_delay.hint}' in block {block_counters}: "
                            f'Calculated duration is negative ({new_dur * 1e6:.1f} μs). '
                            f'Check the offset ({block.soft_delay.offset * 1e6:.1f} μs) and factor '
                            f'({block.soft_delay.factor}) parameters.'
                        )

                    self.block_durations[block_counters] = new_dur

        # Now check if there are some input soft delays which haven't been found in the sequence
        all_input_hints = kwargs.keys()
        for hint in all_input_hints:
            if hint not in sd_str2numID:
                available_hints = list(sd_str2numID.keys())
                raise ValueError(
                    f"Soft delay '{hint}' not found in sequence. "
                    f'Available soft delays: {available_hints if available_hints else "none"}'
                )

    def test_report(self) -> str:
        """
        Analyze the sequence and return a text report.
        """
        return ext_test_report(self)

    def waveforms(self, append_RF: bool = False, time_range: Union[List[float], None] = None) -> Tuple[np.ndarray]:
        """
        Decompress the entire gradient waveform. Returns gradient waveforms as a tuple of `np.ndarray` of
        `gradient_axes` (typically 3) dimensions. Each `np.ndarray` contains timepoints and the corresponding
        gradient amplitude values.

        Parameters
        ----------
        append_RF : bool, default=False
            Boolean flag to indicate if RF wave shapes are to be appended after the gradients.

        Returns
        -------
        wave_data : np.ndarray
        """
        grad_channels = ['gx', 'gy', 'gz']

        # Collect shape pieces
        if append_RF:
            shape_channels = len(grad_channels) + 1  # Last 'channel' is RF
        else:
            shape_channels = len(grad_channels)

        shape_pieces = [[] for _ in range(shape_channels)]
        out_len = np.zeros(shape_channels)  # Last 'channel' is RF

        curr_dur = 0
        if time_range is None:
            blocks = self.block_events
        else:
            if len(time_range) != 2:
                raise ValueError('Time range must be list of two elements')
            if time_range[0] > time_range[1]:
                raise ValueError('End time of time_range must be after begin time')

            # Calculate end times of each block
            bd = np.array(list(self.block_durations.values()))
            t = np.cumsum(bd)
            # Search block end times for start of time range
            begin_block = np.searchsorted(t, time_range[0])
            # Search block begin times for end of time range
            end_block = np.searchsorted(t - bd, time_range[1], side='right')
            blocks = list(self.block_durations.keys())[begin_block:end_block]
            curr_dur = t[begin_block] - bd[begin_block]

        for block_counter in blocks:
            block = self.get_block(block_counter)

            for j in range(len(grad_channels)):
                grad = getattr(block, grad_channels[j])
                if grad is not None:  # Gradients
                    if grad.type == 'grad':
                        # Check if we have an extended trapezoid or an arbitrary gradient on a regular raster
                        tt_rast = grad.tt / self.grad_raster_time + 0.5
                        if np.all(np.abs(tt_rast - np.arange(1, len(tt_rast) + 1)) < eps):  # Arbitrary gradient
                            """
                            Arbitrary gradient: restore & recompress shape - if we had a trapezoid converted to shape we
                            have to find the "corners" and we can eliminate internal samples on the straight segments
                            but first we have to restore samples on the edges of the gradient raster intervals for that
                            we need the first sample.
                            """

                            # TODO: Implement restoreAdditionalShapeSamples
                            #       https://github.com/pulseq/pulseq/blob/master/matlab/%2Bmr/restoreAdditionalShapeSamples.m

                            out_len[j] += len(grad.tt) + 2
                            shape_pieces[j].append(
                                np.array(
                                    [
                                        curr_dur
                                        + grad.delay
                                        + np.concatenate(([0], grad.tt, [grad.tt[-1] + self.grad_raster_time / 2])),
                                        np.concatenate(([grad.first], grad.waveform, [grad.last])),
                                    ]
                                )
                            )
                        else:  # Extended trapezoid
                            out_len[j] += len(grad.tt)
                            shape_pieces[j].append(
                                np.array(
                                    [
                                        curr_dur + grad.delay + grad.tt,
                                        grad.waveform,
                                    ]
                                )
                            )
                    else:
                        if abs(grad.flat_time) > eps:
                            out_len[j] += 4
                            _temp = np.vstack(
                                (
                                    cumsum(
                                        curr_dur + grad.delay,
                                        grad.rise_time,
                                        grad.flat_time,
                                        grad.fall_time,
                                    ),
                                    grad.amplitude * np.array([0, 1, 1, 0]),
                                )
                            )
                            shape_pieces[j].append(_temp)
                        else:
                            if abs(grad.rise_time) > eps and abs(grad.fall_time) > eps:
                                out_len[j] += 3
                                _temp = np.vstack(
                                    (
                                        cumsum(curr_dur + grad.delay, grad.rise_time, grad.fall_time),
                                        grad.amplitude * np.array([0, 1, 0]),
                                    )
                                )
                                shape_pieces[j].append(_temp)
                            else:
                                if abs(grad.amplitude) > eps:
                                    print(
                                        'Warning: "empty" gradient with non-zero magnitude detected in block {}'.format(
                                            block_counter
                                        )
                                    )

            if block.rf is not None:  # RF
                rf = block.rf
                full_freq_offset = rf.freq_offset + rf.freq_ppm * 1e-6 * self.system.gamma * self.system.B0
                full_phase_offset = rf.phase_offset + rf.phase_ppm * 1e-6 * self.system.gamma * self.system.B0
                if append_RF:
                    rf_piece = np.array(
                        [
                            curr_dur + rf.delay + rf.t,
                            rf.signal * np.exp(1j * (full_phase_offset + 2 * math.pi * full_freq_offset * rf.t)),
                        ]
                    )
                    out_len[-1] += len(rf.t)

                    if abs(rf.signal[0]) > 0:
                        pre = np.array([[rf_piece[0, 0] - 0.1 * self.system.rf_raster_time], [0]])
                        rf_piece = np.hstack((pre, rf_piece))
                        out_len[-1] += pre.shape[1]

                    if abs(rf.signal[-1]) > 0:
                        post = np.array([[rf_piece[0, -1] + 0.1 * self.system.rf_raster_time], [0]])
                        rf_piece = np.hstack((rf_piece, post))
                        out_len[-1] += post.shape[1]

                    shape_pieces[-1].append(rf_piece)

            curr_dur += self.block_durations[block_counter]

        # Collect wave data
        wave_data = []

        for j in range(shape_channels):
            if shape_pieces[j] == []:
                wave_data.append(np.zeros((2, 0)))
                continue

            # If the first element of the next shape has the same time as
            # the last element of the previous shape, drop the first
            # element of the next shape.
            shape_pieces[j] = [shape_pieces[j][0]] + [
                cur if prev[0, -1] + eps < cur[0, 0] else cur[:, 1:]
                for prev, cur in zip(shape_pieces[j][:-1], shape_pieces[j][1:], strict=False)
            ]

            wave_data.append(np.concatenate(shape_pieces[j], axis=1))

            rftdiff = np.diff(wave_data[j][0])
            if np.any(rftdiff < eps):
                raise Warning('Time vector elements are not monotonically increasing.')

        return wave_data

    def waveforms_and_times(
        self, append_RF: bool = False, time_range: Union[List[float], None] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Decompress the entire gradient waveform. Returns gradient waveforms as a tuple of `np.ndarray` of
        `gradient_axes` (typically 3) dimensions. Each `np.ndarray` contains timepoints and the corresponding
        gradient amplitude values. Additional return values are time points of excitations, refocusings and ADC
        sampling points.

        Parameters
        ----------
        append_RF : bool, default=False
            Boolean flag to indicate if RF wave shapes are to be appended after the gradients.

        Returns
        -------
        wave_data : np.ndarray
        tfp_excitation : np.ndarray
            Contains time moments, frequency and phase offsets of the excitation RF pulses (similar for `
            tfp_refocusing`).
        tfp_refocusing : np.ndarray
        t_adc: np.ndarray
            Contains times of all ADC sample points.
        fp_adc : np.ndarray
            Contains frequency and phase offsets of each ADC object (not samples).
        """
        wave_data = self.waveforms(append_RF=append_RF, time_range=time_range)
        t_excitation, fp_excitation, t_refocusing, fp_refocusing = self.rf_times(time_range=time_range)
        t_adc, fp_adc = self.adc_times(time_range=time_range)

        # Join times, frequency and phases of RF pulses for compatibility with previous implementation
        tfp_excitation = np.concatenate((np.array(t_excitation)[None], fp_excitation), axis=0)
        tfp_refocusing = np.concatenate((np.array(t_refocusing)[None], fp_refocusing), axis=0)

        return wave_data, tfp_excitation, tfp_refocusing, t_adc, fp_adc

    def write(
        self,
        name: str,
        create_signature: bool = True,
        remove_duplicates: bool = True,
        check_timing: bool = True,
        v141_compat: bool = False,
    ) -> Union[str, None]:
        """
        Write the sequence data to the given filename using the open file format for MR sequences.

        See also `pypulseq.Sequence.read_seq.read()`.

        Parameters
        ----------
        name : str
            Filename of `.seq` file to be written to disk.
        create_signature : bool, default=True
            Boolean flag to indicate if the file has to be signed.
        remove_duplicates : bool, default=True
            Remove duplicate events from the sequence before writing
        v141_compat: bool, default=False
            Write the sequence in v1.4.1 compatible file format.

        Returns
        -------
        signature or None : If create_signature is True, it returns the written .seq file's signature as a string,
        otherwise it returns None. Note that, if remove_duplicates is True, signature belongs to the
        deduplicated sequences signature, and not the Sequence that is stored in the Sequence object.
        """
        # Check if there are any timing errors in the sequence
        if check_timing:
            is_ok, error_report = self.check_timing()
            if not is_ok:
                warn(f'write(): {len(error_report)} timing errors found in the sequence', stacklevel=2)

        # Calculate sequence duration and stored it in the TotalDuration definition
        self.set_definition('TotalDuration', sum(self.block_durations.values()))

        # Check whether all gradients in the last block are ramped down properly
        last_block_id = next(reversed(self.block_events))
        last_block = self.get_block(last_block_id)
        for channel, event in zip(('x', 'y', 'z'), (last_block.gx, last_block.gy, last_block.gz), strict=False):
            if (
                event is not None
                and event.type == 'grad'
                and abs(event.last) > self.system.max_slew * self.system.grad_raster_time
            ):
                warn_msg = f'write(): Gradient on channel {channel} in last sequence block does not ramp down to 0'

                if trace_enabled():
                    trace = self.block_trace.get(last_block_id, None)

                    if hasattr(trace, 'block'):
                        warn_msg += '\nLast block defined here:\n' + format_trace(trace.block)
                    if hasattr(trace, 'g' + channel):
                        warn_msg += f'\n`g{channel}` defined here:\n' + format_trace(getattr(trace, 'g' + channel))

                warn(warn_msg, stacklevel=2)

        # Write the sequence
        if v141_compat:
            signature = write_seq_v141(self, name, create_signature, remove_duplicates)
        else:
            signature = write_seq(self, name, create_signature, remove_duplicates)

        # Return the sequence md5 signature if requested
        if signature is not None:
            self.signature_type = 'md5'
            self.signature_file = 'text'
            self.signature_value = signature
            return signature
        else:
            return None
