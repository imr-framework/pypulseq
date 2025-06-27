import itertools
import math

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt

from pypulseq.calc_rf_center import calc_rf_center
from pypulseq.Sequence import parula
from pypulseq.supported_labels_rf_use import get_supported_labels
from pypulseq.utils.cumsum import cumsum

try:
    import mplcursors

    __MPLCURSORS_AVAILABLE__ = True
except ImportError:
    __MPLCURSORS_AVAILABLE__ = False


class SeqPlot:
    """
    Interactive plotter for a Pulseq `Sequence` object.

    Parameters
    ----------
    seq : Sequence
        The Pulseq sequence object to plot.
    label : str, default=str()
        Plot label values for ADC events: in this example for LIN and REP labels; other valid labes are accepted as
        a comma-separated list.
    save : bool, default=False
        Boolean flag indicating if plots should be saved. The two figures will be saved as JPG with numerical
        suffixes to the filename 'seq_plot'.
    show_blocks : bool, default=False
        Boolean flag to indicate if grid and tick labels at the block boundaries are to be plotted.
    time_range : iterable, default=(0, np.inf)
        Time range (x-axis limits) for plotting the sequence. Default is 0 to infinity (entire sequence).
    time_disp : str, default='s'
        Time display type, must be one of `s`, `ms` or `us`.
    grad_disp : str, default='s'
        Gradient display unit, must be one of `kHz/m` or `mT/m`.
    plot_now : bool, default=True
        If true, function immediately shows the plots, blocking the rest of the code until plots are exited.
        If false, plots are shown when plt.show() is called. Useful if plots are to be modified.
    plot_type : str, default='Gradient'
        Gradients display type, must be one of either 'Gradient' or 'Kspace'.

    Attributes
    ----------
    fig1 : matplotlib.figure.Figure
        Figure containing RF and ADC channels.
    fig2 : matplotlib.figure.Figure
        Figure containing Gradient or K-space channels.
    ax1 : matplotlib.axes.Axes
        Axes for fig1.
    ax2 : matplotlib.axes.Axes
        Axes for fig2.
    """

    MARGIN = 6
    MY1 = 45
    MX1 = 70
    MX2 = 5

    def __init__(
        self,
        seq,
        label: str = str(),
        show_blocks: bool = False,
        save: bool = False,
        time_range=(0, np.inf),
        time_disp: str = 's',
        grad_disp: str = 'kHz/m',
        plot_now: bool = True,
    ):
        self.seq = seq
        self.fig1, self.fig2 = _seq_plot(
            seq,
            label=label,
            save=save,
            show_blocks=show_blocks,
            time_range=time_range,
            time_disp=time_disp,
            grad_disp=grad_disp,
        )
        self.ax1 = self.fig1.axes[0]
        self.ax2 = self.fig2.axes[0]

        self.fig1.canvas.mpl_connect('resize_event', self._on_resize)
        self.fig2.canvas.mpl_connect('resize_event', self._on_resize)

        if plot_now:
            self.show()

    def show(self):
        """Show the figures and enable interactive data tips (if mplcursors is available)."""
        plt.show()
        if __MPLCURSORS_AVAILABLE__:
            self._setup_cursor(self.fig1)
            self._setup_cursor(self.fig2)

    def _setup_cursor(self, fig):
        for ax in fig.axes:
            lines = ax.get_lines()
            cursor = mplcursors.cursor(lines, hover=False)
            cursor.connect('add', lambda sel, ax=ax: self._on_datatip(sel, ax))

    def _on_datatip(self, sel, ax):
        x, y = sel.target
        artist = sel.artist
        ylabel = ax.get_ylabel().lower()

        if ylabel.startswith('adc') or (
            ylabel.startswith('rf/adc') and artist.get_linestyle() == 'none' and artist.get_marker() == '.'
        ):
            field = 'adc'
        else:
            field = ylabel[:2]

        t0 = artist.get_xdata()[0] if hasattr(artist, 'get_xdata') else x
        block_index = self.seq.find_block_by_time(t0)
        rb = self.seq.get_raw_block_content_IDs(block_index)

        lines_txt = [f't: {x:.3f}', f'Y: {y:.3f}']

        val = getattr(rb, field, None)
        if val is not None:
            try:
                if field[0] == 'a':
                    name = self.seq.adc_id2name_map[val]
                elif field[0] == 'r':
                    name = self.seq.rf_id2name_map[val]
                else:
                    name = self.seq.grad_id2name_map[val]

                lines_txt.append(f"blk: {block_index} {field}_id: {val} '{name}'")
            except Exception:
                lines_txt.append(f'blk: {block_index} {field}_id: {val}')
        else:
            lines_txt.append(f'blk: {block_index}')

        sel.annotation.set_text('\n'.join(lines_txt))
        self._update_guides()

    def _update_guides(self):
        """Refresh the figure display after interaction or data tip selection."""
        for ax in (self.ax1, self.ax2):
            ax.relim()
            ax.autoscale_view()

        for fig in (self.fig1, self.fig2):
            fig.canvas.draw_idle()

    def _on_resize(self, event):
        fig = event.canvas.figure
        w_px, h_px = fig.get_size_inches() * fig.dpi
        left = self.MX1
        bottom = self.MY1
        right = w_px - self.MX2
        top = h_px - self.MARGIN

        for ax in fig.axes:
            ax.set_position([left / w_px, bottom / h_px, (right - left) / w_px, (top - bottom) / h_px])
        fig.canvas.draw_idle()


def _seq_plot(
    seq,
    label,
    show_blocks,
    save,
    time_range,
    time_disp,
    grad_disp,
):
    mpl.rcParams['lines.linewidth'] = 0.75  # Set default Matplotlib linewidth

    valid_time_units = ['s', 'ms', 'us']
    valid_grad_units = ['kHz/m', 'mT/m']
    valid_labels = get_supported_labels()
    if not all(isinstance(x, (int, float)) for x in time_range) or len(time_range) != 2:
        raise ValueError('Invalid time range')
    if time_disp not in valid_time_units:
        raise ValueError('Unsupported time unit')

    if grad_disp not in valid_grad_units:
        raise ValueError('Unsupported gradient unit. Supported gradient units are: ' + str(valid_grad_units))

    fig1, fig2 = plt.figure(), plt.figure()
    sp11 = fig1.add_subplot(311)
    sp12 = fig1.add_subplot(312, sharex=sp11)
    sp13 = fig1.add_subplot(313, sharex=sp11)
    fig2_subplots = [
        fig2.add_subplot(311, sharex=sp11),
        fig2.add_subplot(312, sharex=sp11),
        fig2.add_subplot(313, sharex=sp11),
    ]

    t_factor_list = [1, 1e3, 1e6]
    t_factor = t_factor_list[valid_time_units.index(time_disp)]

    g_factor_list = [1e-3, 1e3 / seq.system.gamma]
    g_factor = g_factor_list[valid_grad_units.index(grad_disp)]

    t0 = 0
    label_defined = False
    label_idx_to_plot = []
    label_legend_to_plot = []
    label_store = {}
    for i in range(len(valid_labels)):
        label_store[valid_labels[i]] = 0
        if valid_labels[i] in label.upper():
            label_idx_to_plot.append(i)
            label_legend_to_plot.append(valid_labels[i])

    if len(label_idx_to_plot) != 0:
        p = parula.main(len(label_idx_to_plot) + 1)
        label_colors_to_plot = p(np.arange(len(label_idx_to_plot)))
        cycler = mpl.cycler(color=label_colors_to_plot)
        sp11.set_prop_cycle(cycler)

    # Block timings
    block_edges = np.cumsum([0] + [x[1] for x in sorted(seq.block_durations.items())])
    block_edges_in_range = block_edges[(block_edges >= time_range[0]) * (block_edges <= time_range[1])]
    if show_blocks:
        for sp in [sp11, sp12, sp13, *fig2_subplots]:
            sp.set_xticks(t_factor * block_edges_in_range)
            sp.set_xticklabels(sp.get_xticklabels(), rotation=90)

    for block_counter in seq.block_events:
        block = seq.get_block(block_counter)
        is_valid = time_range[0] <= t0 + seq.block_durations[block_counter] and t0 <= time_range[1]
        if is_valid:
            if getattr(block, 'label', None) is not None:
                for i in range(len(block.label)):
                    if block.label[i].type == 'labelinc':
                        label_store[block.label[i].label] += block.label[i].value
                    else:
                        label_store[block.label[i].label] = block.label[i].value
                label_defined = True

            if getattr(block, 'adc', None) is not None:  # ADC
                adc = block.adc
                # From Pulseq: According to the information from Klaus Scheffler and indirectly from Siemens this
                # is the present convention - the samples are shifted by 0.5 dwell
                t = adc.delay + (np.arange(int(adc.num_samples)) + 0.5) * adc.dwell
                sp11.plot(t_factor * (t0 + t), np.zeros(len(t)), 'rx')

                if adc.phase_modulation is None or len(adc.phase_modulation) == 0:
                    phase_modulation = 0
                else:
                    phase_modulation = adc.phase_modulation

                full_freq_offset = np.atleast_1d(adc.freq_offset + adc.freq_ppm * 1e-6 * seq.system.B0)
                full_phase_offset = np.atleast_1d(
                    adc.phase_offset + adc.phase_offset * 1e-6 * seq.system.B0 + phase_modulation
                )

                sp13.plot(
                    t_factor * (t0 + t),
                    np.angle(np.exp(1j * full_phase_offset) * np.exp(1j * 2 * np.pi * t * full_freq_offset)),
                    'b.',
                    markersize=0.25,
                )

                if label_defined and len(label_idx_to_plot) != 0:
                    arr_label_store = list(label_store.values())
                    lbl_vals = np.take(arr_label_store, label_idx_to_plot)
                    t = t0 + adc.delay + (adc.num_samples - 1) / 2 * adc.dwell
                    _t = [t_factor * t] * len(lbl_vals)
                    # Plot each label individually to retrieve each corresponding Line2D object
                    p = itertools.chain.from_iterable(
                        [sp11.plot(__t, _lbl_vals, '.') for __t, _lbl_vals in zip(_t, lbl_vals)]
                    )
                    if len(label_legend_to_plot) != 0:
                        sp11.legend(list(p), label_legend_to_plot, loc='upper left')
                        label_legend_to_plot = []

            if getattr(block, 'rf', None) is not None:  # RF
                rf = block.rf
                time_center, index_center = calc_rf_center(rf)
                time = rf.t
                signal = rf.signal

                if signal.shape[0] == 2 and rf.freq_offset != 0:
                    num_samples = min(int(abs(rf.freq_offset)), 256)
                    time = np.linspace(time[0], time[-1], num_samples)
                    signal = np.linspace(signal[0], signal[-1], num_samples)

                if abs(signal[0]) != 0:
                    signal = np.concatenate(([0], signal))
                    time = np.concatenate(([time[0]], time))
                    index_center += 1

                if abs(signal[-1]) != 0:
                    signal = np.concatenate((signal, [0]))
                    time = np.concatenate((time, [time[-1]]))

                signal_is_real = max(np.abs(np.imag(signal))) / max(np.abs(np.real(signal))) < 1e-6

                full_freq_offset = rf.freq_offset + rf.freq_ppm * 1e-6 * seq.system.B0
                full_phase_offset = rf.phase_offset + rf.phase_ppm * 1e-6 * seq.system.B0

                # If off-resonant and rectangular (2 samples), interpolate the pulse
                if len(signal) == 2 and full_freq_offset != 0:
                    num_interp = min(int(abs(full_freq_offset)), 256)
                    time = np.linspace(time[0], time[-1], num_interp)
                    signal = np.linspace(signal[0], signal[-1], num_interp)
                if abs(signal[0]) != 0:  # fix strangely looking phase / amplitude in the beginning
                    signal = np.concatenate([[0], signal])
                    time = np.concatenate([[time[0]], time])
                if abs(signal[-1]) != 0:  # fix strangely looking phase / amplitude at the end
                    signal = np.concatenate([signal, [0]])
                    time = np.concatenate([time, [time[-1]]])

                # Compute time vector with delay applied
                time_with_delay = t_factor * (t0 + time + rf.delay)
                time_center_with_delay = t_factor * (t0 + time_center + rf.delay)

                # Choose plot behavior based on realness of signal
                if signal_is_real:
                    # Plot real part of signal
                    sp12.plot(time_with_delay, np.real(signal))

                    # Include sign(real(signal)) factor like MATLAB
                    phase_corrected = (
                        signal
                        * np.sign(np.real(signal))
                        * np.exp(1j * full_phase_offset)
                        * np.exp(1j * 2 * math.pi * time * full_freq_offset)
                    )
                    sc_corrected = (
                        signal[index_center]
                        * np.exp(1j * full_phase_offset)
                        * np.exp(1j * 2 * math.pi * time[index_center] * full_freq_offset)
                    )

                    sp13.plot(
                        time_with_delay,
                        np.angle(phase_corrected),
                        time_center_with_delay,
                        np.angle(sc_corrected),
                        'xb',
                    )
                else:
                    # Plot magnitude of complex signal
                    sp12.plot(time_with_delay, np.abs(signal))

                    # Plot angle of complex signal
                    phase_corrected = (
                        signal * np.exp(1j * full_phase_offset) * np.exp(1j * 2 * math.pi * time * full_freq_offset)
                    )
                    sc_corrected = (
                        signal[index_center]
                        * np.exp(1j * full_phase_offset)
                        * np.exp(1j * 2 * math.pi * time[index_center] * full_freq_offset)
                    )

                    sp13.plot(
                        time_with_delay,
                        np.angle(phase_corrected),
                        time_center_with_delay,
                        np.angle(sc_corrected),
                        'xb',
                    )

            grad_channels = ['gx', 'gy', 'gz']
            for x in range(len(grad_channels)):  # Gradients
                if getattr(block, grad_channels[x], None) is not None:
                    grad = getattr(block, grad_channels[x])
                    if grad.type == 'grad':
                        # We extend the shape by adding the first and the last points in an effort of making the
                        # display a bit less confusing...
                        time = grad.delay + np.array([0, *grad.tt, grad.shape_dur])
                        waveform = g_factor * np.array((grad.first, *grad.waveform, grad.last))
                    else:
                        time = np.array(
                            cumsum(
                                0,
                                grad.delay,
                                grad.rise_time,
                                grad.flat_time,
                                grad.fall_time,
                            )
                        )
                        waveform = g_factor * grad.amplitude * np.array([0, 0, 1, 1, 0])
                    fig2_subplots[x].plot(t_factor * (t0 + time), waveform)
        t0 += seq.block_durations[block_counter]

    grad_plot_labels = ['x', 'y', 'z']
    sp11.set_ylabel('ADC')
    sp12.set_ylabel('RF mag (Hz)')
    sp13.set_ylabel('RF/ADC phase (rad)')
    sp13.set_xlabel(f't ({time_disp})')
    for x in range(3):
        _label = grad_plot_labels[x]
        fig2_subplots[x].set_ylabel(f'G{_label} ({grad_disp})')
    fig2_subplots[-1].set_xlabel(f't ({time_disp})')

    # Setting display limits
    disp_range = t_factor * np.array([time_range[0], min(t0, time_range[1])])
    [x.set_xlim(disp_range) for x in [sp11, sp12, sp13, *fig2_subplots]]

    # Grid on
    for sp in [sp11, sp12, sp13, *fig2_subplots]:
        sp.grid()

    fig1.tight_layout()
    fig2.tight_layout()
    if save:
        fig1.savefig('seq_plot1.jpg')
        fig2.savefig('seq_plot2.jpg')

    return fig1, fig2
