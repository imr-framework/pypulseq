import numpy as np


def points_to_waveform(times: np.ndarray, amplitudes: np.ndarray, grad_raster_time: float):
    """
    1D interpolate amplitude values `amplitudes` at time indices `times` as per the gradient raster time
    `grad_raster_time` to generate a gradient waveform.

    Parameters
    ----------
    times : numpy.ndarray
        Time indices.
    amplitudes : numpy.ndarray
        Amplitude values at time indices `times`.
    grad_raster_time : float
        Gradient raster time.

    Returns
    -------
    waveform : numpy.ndarray
        Gradient waveform.
    """
    grd = np.arange(start=round(min(times) / grad_raster_time),
                    stop=round(max(times) / grad_raster_time)) * grad_raster_time
    waveform = np.interp(x=grd + grad_raster_time / 2, xp=times, fp=amplitudes)

    return waveform
