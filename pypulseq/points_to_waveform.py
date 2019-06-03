import numpy as np


def points_to_waveform(times, amplitudes, grad_raster_time):
    grd = np.arange(start=round(min(times) / grad_raster_time),
                    stop=round(max(times) / grad_raster_time)) * grad_raster_time
    waveform = np.interp(x=grd + grad_raster_time / 2, xp=times, fp=amplitudes)

    return waveform
