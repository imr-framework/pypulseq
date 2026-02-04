from __future__ import annotations

import typing

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import convolve, windows

try:
    import sounddevice as sd

    _HAS_SOUNDDEVICE = True
except ImportError:
    _HAS_SOUNDDEVICE = False

if typing.TYPE_CHECKING:
    from pypulseq.Sequence.sequence import Sequence


def seq_sound(
    seq: Sequence,
    time_range=(0.0, np.inf),
    channel_weights: tuple = (1.0, 1.0, 1.0),
    sample_rate: int = 44100,
    only_produce_sound_data: bool = False,
) -> np.ndarray:
    """
    Play out the sequence through the system speaker and return the sound data.

    Parameters
    ----------
    seq : Sequence
        The Pulseq sequence object to played.
    time_range : iterable, default=(0, np.inf)
        Time range for playing the sequence.
        Default is 0 to infinity (entire sequence).
    channel_weights : tuple, default=(1.0, 1.0, 1.0)
        Weight for each gradient channel.
    sample_rate : int, default=44100
        Sampling rate for audio waveform
    only_produce_sound_data : bool, default=False
        If True, skip the playing part and only produce
        the sound data

    Returns
    -------
    sound_data : np.ndarray
        Sound data of shape (2, N)
    """
    # --- get waveform data ---
    gw_data = seq.waveforms_and_times(False, time_range)
    total_duration = np.sum(seq.block_durations)

    dwell_time = 1.0 / sample_rate
    sound_length = int(np.floor(total_duration / dwell_time)) + 1

    sound_data = np.zeros((2, sound_length), dtype=float)

    t_out = np.arange(sound_length) * dwell_time

    # --- X channel → channel 1 ---
    if gw_data[0] is not None and gw_data[0].size > 0:
        f = interp1d(
            gw_data[0][0],
            gw_data[0][1] * channel_weights[0],
            kind='linear',
            bounds_error=False,
            fill_value=0.0,
        )
        sound_data[0, :] = f(t_out)

    # --- Y channel → channel 2 ---
    if gw_data[1] is not None and gw_data[1].size > 0:
        f = interp1d(
            gw_data[1][0],
            gw_data[1][1] * channel_weights[1],
            kind='linear',
            bounds_error=False,
            fill_value=0.0,
        )
        sound_data[1, :] = f(t_out)

    # --- Z channel → split equally ---
    if gw_data[2] is not None and gw_data[2].size > 0:
        f = interp1d(
            gw_data[2][0],
            0.5 * gw_data[2][1] * channel_weights[2],
            kind='linear',
            bounds_error=False,
            fill_value=0.0,
        )
        tmp = f(t_out)
        sound_data[0, :] += tmp
        sound_data[1, :] += tmp

    # --- Gaussian smoothing (same as MATLAB gausswin + conv) ---
    gw_len = int(round(sample_rate / 6000) * 2 + 1)
    gw = windows.gaussian(gw_len, std=gw_len / 6)
    gw /= np.sum(gw)

    sound_data[0, :] = convolve(sound_data[0, :], gw, mode='same')
    sound_data[1, :] = convolve(sound_data[1, :], gw, mode='same')

    # --- normalize ---
    max_val = np.max(np.abs(sound_data))
    if max_val > 0:
        sound_data = 0.95 * sound_data / max_val

    # --- play sound ---
    if not only_produce_sound_data:
        duration = sound_length * dwell_time
        print(f'playing out the sequence waveform, duration {duration:.1f}s')

        if not _HAS_SOUNDDEVICE:
            raise RuntimeError('sounddevice not installed')

        pad = np.zeros((2, sample_rate // 2))
        play_data = np.hstack([pad, sound_data, pad]).T  # (N, 2)
        sd.play(play_data, samplerate=sample_rate)
        sd.wait()

    return sound_data
