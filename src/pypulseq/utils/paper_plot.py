from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

from pypulseq import eps


def paper_plot(
    seq,
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
    seq : Sequence
        The Pulseq sequence object to plot.
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
    # Get waveform data
    wave_data, _, _, t_adc, _ = seq.waveforms_and_times(append_RF=True, time_range=time_range)

    # Max amplitudes for scaling
    if wave_data[0].size + wave_data[1].size + wave_data[2].size:
        gwm = np.max(np.abs(np.concatenate(wave_data[:3], axis=1)), axis=1)
    else:
        gwm = (-eps, eps)
    if t_adc.size:
        gwm[0] = max(gwm[0], t_adc[-1])
    if wave_data[3].size:
        rfm = np.max(np.abs(wave_data[3]), axis=1)
    else:
        rfm = (-eps, eps)

    # Handle complex RF
    if rf_plot == 'real':
        rf_waveform = np.real(wave_data[3][1])
    elif rf_plot == 'imag':
        rf_waveform = np.imag(wave_data[3][1])
    else:
        rf_waveform = np.abs(wave_data[3][1])
    wave_data[3] = np.stack((wave_data[3][0].real, rf_waveform), axis=0)

    # Clean waveforms by inserting NaNs between zero plateaus
    for i in range(4):
        data = wave_data[i]
        j = data.shape[1] - 1
        while j > 0:
            if data[1, j] == 0 and data[1, j - 1] == 0:
                midpoint = 0.5 * (data[0, j] + data[0, j - 1])
                data = np.hstack([data[:, :j], np.array([[midpoint], [np.nan]]), data[:, j:]])
                wave_data[i] = data
            j -= 1

    # Create figure
    fig = plt.figure(figsize=(12, 10), constrained_layout=True)
    fig.patch.set_facecolor('white')
    spec = fig.add_gridspec(nrows=4, ncols=1, hspace=0.0)
    axes = []

    def format_axis(ax, xlim, ylim):
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor('white')
        ax.spines[:].set_visible(False)

    # ADC
    ax = fig.add_subplot(spec[0])
    ax.vlines(t_adc, ymin=0, ymax=rfm[1] / 5, color=rf_color, lw=line_width / 4, zorder=2.5)

    # RF
    ax.plot([-0.01 * gwm[0], 1.01 * gwm[0]], [0, 0], color=axes_color, lw=line_width / 5)
    ax.plot(wave_data[3][0], wave_data[3][1], color=rf_color, lw=line_width)

    # Format RF + ADC
    format_axis(ax, [-0.03 * gwm[0], 1.03 * gwm[0]], [-1.03 * rfm[1], 1.03 * rfm[1]])
    axes.append(ax)

    # Gradient Z
    ax = fig.add_subplot(spec[1])
    ax.plot([-0.01 * gwm[0], 1.01 * gwm[0]], [0, 0], color=axes_color, lw=line_width / 5)
    ax.plot(wave_data[2][0], wave_data[2][1], color=gz_color, lw=line_width)
    format_axis(ax, [-0.03 * gwm[0], 1.03 * gwm[0]], [-1.03 * gwm[1], 1.03 * gwm[1]])
    axes.append(ax)

    # Gradient Y
    ax = fig.add_subplot(spec[2])
    ax.plot([-0.01 * gwm[0], 1.01 * gwm[0]], [0, 0], color=axes_color, lw=line_width / 5)
    ax.plot(wave_data[1][0], wave_data[1][1], color=gy_color, lw=line_width)
    format_axis(ax, [-0.03 * gwm[0], 1.03 * gwm[0]], [-1.03 * gwm[1], 1.03 * gwm[1]])
    axes.append(ax)

    # Gradient X
    ax = fig.add_subplot(spec[3])
    ax.plot([-0.01 * gwm[0], 1.01 * gwm[0]], [0, 0], color=axes_color, lw=line_width / 5)
    ax.plot(wave_data[0][0], wave_data[0][1], color=gx_color, lw=line_width)
    format_axis(ax, [-0.03 * gwm[0], 1.03 * gwm[0]], [-1.03 * gwm[1], 1.03 * gwm[1]])
    axes.append(ax)

    # Link X-axes (time axis)
    for ax in axes[1:]:
        ax.sharex(axes[0])
