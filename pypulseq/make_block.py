import numpy as np

from pypulseq.holder import Holder
from pypulseq.make_trap import make_trapezoid
from pypulseq.opts import Opts


def make_block_pulse(kwargs, nargout=1):
    """
    Makes a Holder object for an RF pulse Event.

    Parameters
    ----------
    kwargs : dict
        Key value mappings of RF Event parameters_params and values.
    nargout: int
        Number of output arguments to be returned. Default is 1, only RF Event is returned. Passing any number greater
        than 1 will return the Gz Event along with the RF Event.

    Returns
    -------
    Tuple consisting of:
    rf : Holder
        RF Event configured based on supplied kwargs.
    gz : Holder
        Slice select trapezoidal gradient Event.
    """

    flip_angle = kwargs.get("flip_angle")
    system = kwargs.get("system", Opts())
    duration = kwargs.get("duration", 0)
    freq_offset = kwargs.get("freq_offset", 0)
    phase_offset = kwargs.get("phase_offset", 0)
    time_bw_product = kwargs.get("time_bw_product", 4)
    bandwidth = kwargs.get("bandwidth", 0)
    max_grad = kwargs.get("max_grad", 0)
    max_slew = kwargs.get("max_slew", 0)
    slice_thickness = kwargs.get("slice_thickness", 0)

    if duration == 0:
        if time_bw_product > 0:
            duration = time_bw_product / bandwidth
        elif bandwidth > 0:
            duration = 1 / (4 * bandwidth)
        else:
            raise ValueError('Either bandwidth or duration must be defined')

    BW = 1 / (4 * duration)
    N = round(duration / 1e-6)
    t = [x * system.rf_raster_time for x in range(N)]
    signal = flip_angle / (2 * np.pi) / duration * np.ones(len(t))

    rf = Holder()
    rf.type = 'rf'
    rf.signal = signal
    rf.t = t
    rf.freq_offset = freq_offset
    rf.phase_offset = phase_offset
    rf.dead_time = system.rf_dead_time
    rf.ring_down_time = system.rf_ring_down_time

    fill_time = 0
    if nargout > 1:
        if slice_thickness < 0:
            raise ValueError('Slice thickness must be provided')

        if max_grad > 0:
            system.max_grad = max_grad
        if max_slew > 0:
            system.max_slew = max_slew

        amplitude = BW / slice_thickness
        area = amplitude * duration
        kwargs_for_trap = {'channel': 'z', 'system': system, 'flat_time': duration, 'flat_area': area}
        gz = make_trapezoid(kwargs_for_trap)

        fill_time = gz.rise_time
        t_fill = np.array([x * 1e-6 for x in range(int(round(fill_time / 1e-6)))])
        rf.t = np.array([t_fill, rf.t + t_fill[-1], t_fill + rf.t[-1] + t_fill[-1]])
        rf.signal = np.array([np.zeros(t_fill.size), rf.signal, np.zeros(t_fill.size)])

    if fill_time < rf.dead_time:
        fill_time = rf.dead_time - fill_time
        t_fill = np.array([x * 1e-6 for x in range(int(round(fill_time / 1e-6)))])
        rf.t = np.insert(rf.t, 0, t_fill) + t_fill[-1]
        rf.t = np.reshape(rf.t, (1, len(rf.t)))
        rf.signal = np.insert(rf.signal, 0, np.zeros(t_fill.size))
        rf.signal = np.reshape(rf.signal, (1, len(rf.signal)))

    if rf.ring_down_time > 0:
        t_fill = np.arange(1, round(rf.ring_down_time / 1e-6) + 1) * 1e-6
        rf.t = [rf.t, rf.t[-1] + t_fill]
        rf.signal = [rf.signal, np.zeros(len(t_fill))]

    if nargout > 1:
        return rf, gz
    else:
        return rf
