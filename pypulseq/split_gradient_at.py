import numpy as np

from pypulseq.make_extended_trapezoid import make_extended_trapezoid
from pypulseq.opts import Opts


def split_gradient_at(grad: np.ndarray, time_point: float, system: Opts = Opts()):
    """
    Split gradient waveform `grad` into two at time point `time_point`.

    Parameters
    ----------
    grad : numpy.ndarray
        Gradient waveform to be split into two gradient waveforms.
    time_point : float, optional
        Time point at which `grad` will be split into two gradient waveforms.
    system : Opts, optional
        System limits. Default is a system limits object initialised to default values.
    Returns
    -------
    grad1, grad2 : numpy.ndarray
        Gradient waveforms after splitting.
    """
    grad_raster_time = system.grad_raster_time

    time_index = round(time_point / grad_raster_time)
    time_point = time_index * grad_raster_time
    time_index += 1

    if grad.type == 'trap':
        ch = grad.channel
        grad.delay = round(grad.delay / grad_raster_time) * grad_raster_time
        grad.rise_time = round(grad.rise_time / grad_raster_time) * grad_raster_time
        grad.flat_time = round(grad.flat_time / grad_raster_time) * grad_raster_time
        grad.fall_time = round(grad.fall_time / grad_raster_time) * grad_raster_time

        if grad.flat_time == 0:
            times = [0, grad.rise_time, grad.rise_time + grad.fall_time]
            amplitudes = [0, grad.amplitude, 0]
        else:
            times = [0, grad.rise_time, grad.rise_time + grad.flat_time,
                     grad.rise_time + grad.flat_time + grad.fall_time]
            amplitudes = [0, grad.amplitude, grad.amplitude, 0]

        if time_point < grad.delay:
            times = np.insert(grad.delay + times, 0, 0)
            amplitudes = [0, amplitudes]
            grad.delay = 0

        amplitudes = np.array(amplitudes)
        times = np.array(times)

        amp_tp = np.interp(x=time_point, xp=times, fp=amplitudes)
        times1 = np.append(times[np.where(times < time_point)], time_point)
        amplitudes1 = np.append(amplitudes[np.where(times < time_point)], amp_tp)
        times2 = np.insert(times[times > time_point], 0, time_point) - time_point
        amplitudes2 = np.insert(amplitudes[times > time_point], 0, amp_tp)

        grad1 = make_extended_trapezoid(channel=ch, system=system, times=times1, amplitudes=amplitudes1,
                                        skip_check=True)
        grad1.delay = grad.delay
        grad2 = make_extended_trapezoid(channel=ch, system=system, times=times2, amplitudes=amplitudes2,
                                        skip_check=True)
        grad2.delay = time_point
        return grad1, grad2
    elif grad.type == 'grad':
        if time_index == 1 or time_index >= len(grad.t):
            return grad
        else:
            grad1 = grad
            grad2 = grad
            grad1.last = grad.waveform[time_index]
            grad2.first = grad.waveform[time_index]
            grad2.delay = grad.delay + grad.t[time_index]
            grad1.t = grad.t[:time_index]
            grad1.waveform = grad.waveform[:time_index]
            grad2.t = grad.t[time_index:]
            grad2.waveform = grad.waveform[time_index:]
            return grad1, grad2
    else:
        raise ValueError('Splitting of unsupported event.')
