from typing import Tuple, List

import numpy as np
from scipy.signal import spectrogram
from matplotlib import pyplot as plt


def calculate_gradient_spectrum(
        obj, max_frequency: float = 2000,
        window_width: float = 0.05,
        frequency_oversampling: float = 3,
        time_range: List[float] = None,
        plot: bool = True,
        combine_mode: str = 'max',
        acoustic_resonances: List[dict] = []
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
    dt = obj.system.grad_raster_time # time raster
    nwin = round(window_width / dt)
    nfft = round(frequency_oversampling*nwin)

    # Get gradients as piecewise-polynomials
    gw_pp = obj.get_gradients(time_range=time_range)
    ng = len(gw_pp)
    max_t = max(g.x[-1] for g in gw_pp if g != None)
    
    # Determine sampling points
    if time_range == None:
        nt = int(np.ceil(max_t/dt))
        t = (np.arange(nt) + 0.5)*dt
    else:
        tmax = min(time_range[1], max_t) - max(time_range[0], 0)
        nt = int(np.ceil(tmax/dt))
        t = max(time_range[0], 0) + (np.arange(nt) + 0.5)*dt
    
    # Sample gradients
    gw = np.zeros((ng,t.shape[0]))
    for i in range(ng):
        if gw_pp[i] != None:
            gw[i] = gw_pp[i](t)
    
    # Calculate spectrogram for each gradient channel
    spectrograms = []
    frequencies = None
    spectrogram_sos = 0
    
    for i in range(ng):
        # Use scipy to calculate the spectrograms
        freq, times, Sxx = spectrogram(gw[i],
                                       fs=1/dt,
                                       mode='magnitude',
                                       nperseg=nwin,
                                       noverlap=nwin//2,
                                       nfft=nfft,
                                       detrend='constant',
                                       window=('tukey', 1))
        mask = freq<max_frequency
        
        # Accumulate spectrum for all gradient channels
        spectrogram_sos += Sxx[mask]**2
        
        # Combine spectrogram over time axis
        if combine_mode == 'max':
            s = Sxx[mask].max(axis=1)
        elif combine_mode == 'mean':
            s = Sxx[mask].mean(axis=1)
        elif combine_mode == 'sos':
            s = np.sqrt((Sxx[mask]**2).sum(axis=1))
        elif combine_mode == 'none':
            s = Sxx[mask]
        else:
            raise ValueError(f'Unknown value for combine_mode: {combine_mode}, must be one of [max, mean, sos, none]')
        
        frequencies = freq[mask]
        spectrograms.append(s)
    
    # Root-sum-of-squares combined spectrogram for all gradient channels
    spectrogram_sos = np.sqrt(spectrogram_sos)
    if combine_mode == 'max':
        spectrogram_sos = spectrogram_sos.max(axis=1)
    elif combine_mode == 'mean':
        spectrogram_sos = spectrogram_sos.mean(axis=1)
    elif combine_mode == 'sos':
        spectrogram_sos = np.sqrt((spectrogram_sos**2).sum(axis=1))
    
    # Plot spectrograms and acoustic resonances if specified
    if plot:
        if combine_mode != 'none':
            plt.figure()
            plt.xlabel('Frequency (Hz)')
            # According to spectrogram documentation y unit is (Hz/m)^2 / Hz = Hz/m^2, is this meaningful?
            for s in spectrograms:
                plt.plot(frequencies, s)
            plt.plot(frequencies, spectrogram_sos)
            plt.legend(['x', 'y', 'z', 'sos'])
    
            for res in acoustic_resonances:
                plt.axvline(res['frequency'], color='k', linestyle='-')
                plt.axvline(res['frequency'] - res['bandwidth']/2, color='k', linestyle='--')
                plt.axvline(res['frequency'] + res['bandwidth']/2, color='k', linestyle='--')
        else:
            for s, c in zip(spectrograms, ['X', 'Y', 'Z']):
                plt.figure()
                plt.title(f'Spectrum {c}')
                plt.xlabel('Time (s)')
                plt.ylabel('Frequency (Hz)')
                plt.imshow(abs(s[::-1]), extent=(times[0], times[-1], frequencies[0], frequencies[-1]),
                           aspect=(times[-1]-times[0])/(frequencies[-1]-frequencies[0]))
                
                for res in acoustic_resonances:
                    plt.axhline(res['frequency'], color='r', linestyle='-')
                    plt.axhline(res['frequency'] - res['bandwidth']/2, color='r', linestyle='--')
                    plt.axhline(res['frequency'] + res['bandwidth']/2, color='r', linestyle='--')
            
            plt.figure()
            plt.title('Total spectrum')
            plt.xlabel('Time (s)')
            plt.ylabel('Frequency (Hz)')
            plt.imshow(abs(spectrogram_sos[::-1]), extent=(times[0], times[-1], frequencies[0], frequencies[-1]),
                       aspect=(times[-1]-times[0])/(frequencies[-1]-frequencies[0]))
            
            for res in acoustic_resonances:
                plt.axhline(res['frequency'], color='r', linestyle='-')
                plt.axhline(res['frequency'] - res['bandwidth']/2, color='r', linestyle='--')
                plt.axhline(res['frequency'] + res['bandwidth']/2, color='r', linestyle='--')
                
    return spectrograms, spectrogram_sos, frequencies, times
