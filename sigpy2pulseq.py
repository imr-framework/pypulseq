import math
from types import SimpleNamespace
from typing import Tuple, Union

import numpy as np
import pypulseq as pp
import sigpy.mri.rf as rf_ext
import sigpy.plot as pl
from matplotlib import pyplot as plt
from pypulseq.make_trap_pulse import make_trapezoid
from pypulseq.opts import Opts
from scipy import integrate


def sig_2_seq(pulse: float, flip_angle: float, delay: float = 0, duration: float = 0,
              freq_offset: float = 0, center_pos: float = 0.5, max_grad: float = 0, max_slew: float = 0,
              phase_offset: float = 0, return_gz: bool = True, slice_thickness: float = 0, system: Opts = Opts(),
              time_bw_product: float = 4, rf_freq: float = 0,
              use: str = str()) -> Union[SimpleNamespace,
                                         Tuple[SimpleNamespace, SimpleNamespace,
                                               SimpleNamespace]]:
    """
    Creates a radio-frequency sinc pulse event using the sigpy rf pulse library and optionally accompanying slice select, slice select rephasing
    trapezoidal gradient events.

    Parameters
    ----------
    pulse : float
        Sigpy designed RF pulse
    flip_angle : float
        Flip angle in radians.
    apodization : float, optional, default=0
        Apodization.
    center_pos : float, optional, default=0.5
        Position of peak.5 (midway).
    delay : float, optional, default=0
        Delay in milliseconds (ms).
    duration : float, optional, default=0
        Duration in milliseconds (ms).
    freq_offset : float, optional, default=0
        Frequency offset in Hertz (Hz).
    max_grad : float, optional, default=0
        Maximum gradient strength of accompanying slice select trapezoidal event.
    max_slew : float, optional, default=0
        Maximum slew rate of accompanying slice select trapezoidal event.
    phase_offset : float, optional, default=0
        Phase offset in Hertz (Hz).
    return_gz:bool, default=False
        Boolean flag to indicate if slice-selective gradient has to be returned.
    slice_thickness : float, optional, default=0
        Slice thickness of accompanying slice select trapezoidal event. The slice thickness determines the area of the
        slice select event.
    system : Opts, optional
        System limits. Default is a system limits object initialised to default values.
    time_bw_product : float, optional, default=4
        Time-bandwidth product.
    rf_freq : float, optional, default = 0
        frequency offset
    use : str, optional, default=str()
        Use of radio-frequency sinc pulse. Must be one of 'excitation', 'refocusing' or 'inversion'.


    Returns
    -------
    rf : SimpleNamespace
        Radio-frequency sinc pulse event.
    gz : SimpleNamespace, optional
        Accompanying slice select trapezoidal gradient event. Returned only if `slice_thickness` is provided.
    gzr : SimpleNamespace, optional
        Accompanying slice select rephasing trapezoidal gradient event. Returned only if `slice_thickness` is provided.

    Raises
    ------
    ValueError
        If invalid `use` parameter was passed. Must be one of 'excitation', 'refocusing' or 'inversion'.
        If `return_gz=True` and `slice_thickness` was not provided.
    """

    valid_use_pulses = ['excitation', 'refocusing', 'inversion']
    if use != '' and use not in valid_use_pulses:
        raise ValueError(
            f"Invalid use parameter. Must be one of 'excitation', 'refocusing' or 'inversion'. Passed: {use}")
    if np.sum(rf_freq) == 0:
        # Step 1: Get signal, t
        [signal, t] = get_signal(pulse=pulse, duration=duration, flip_angle=flip_angle, system=system)

        rfp = SimpleNamespace()
        rfp.type = 'rf'
        rfp.signal = signal
        rfp.t = t
        rfp.freq_offset = rf_freq
        rfp.phase_offset = phase_offset
        rfp.dead_time = system.rf_dead_time
        rfp.ringdown_time = system.rf_ringdown_time
        rfp.delay = delay

        if use != '':
            rfp.use = use

        if rfp.dead_time > rfp.delay:
            rfp.delay = rfp.dead_time

        if return_gz:
            if slice_thickness == 0:
                raise ValueError('Slice thickness must be provided')

            if max_grad > 0:
                system.max_grad = max_grad

            if max_slew > 0:
                system.max_slew = max_slew
            BW = time_bw_product / duration
            amplitude = BW / slice_thickness
            area = amplitude * duration
            gz = make_trapezoid(channel='z', system=system, flat_time=duration, flat_area=area)
            gzr = make_trapezoid(channel='z', system=system, area=-area * (1 - center_pos) - 0.5 * (gz.area - area))

            if rfp.delay > gz.rise_time:
                gz.delay = math.ceil((rfp.delay - gz.rise_time) / system.grad_raster_time) * system.grad_raster_time

            if rfp.delay < (gz.rise_time + gz.delay):
                rfp.delay = gz.rise_time + gz.delay

        if rfp.ringdown_time > 0:
            t_fill = np.arange(1, round(rfp.ringdown_time / 1e-6) + 1) * 1e-6
            rfp.t = np.concatenate((rfp.t, rfp.t[-1] + t_fill))
            rfp.signal = np.concatenate((rfp.signal, np.zeros(len(t_fill))))

        # Following 2 lines of code are workarounds for numpy returning 3.14... for np.angle(-0.00...)
        negative_zero_indices = np.where(rfp.signal == -0.0)
        rfp.signal[negative_zero_indices] = 0

        if return_gz:
            return rfp, gz, gzr, pulse
        else:
            return rfp
    else:
        print('Working on a FM pulse')
        rf_train = get_RF_train(am_pulse=pulse, fm_pulse=rf_freq, duration=duration, system=system,
                                flip_angle=flip_angle)

        return rf_train


def get_signal(pulse: float, duration: float, flip_angle: float, system: Opts = Opts()):
    # N = int(round(duration / 1e-6))
    N = int(round(duration / system.rf_raster_time))
    t = np.arange(1, N + 1) * system.rf_raster_time
    flip = np.sum(pulse) * system.rf_raster_time * 2 * np.pi
    signal = pulse * flip_angle / flip
    return signal, t


def disp_pulse(am_pulse: float, fm_pulse: float = 0, tb=4, duration=3e-3, system: Opts = Opts()):
    if np.sum(fm_pulse) == 0:
        pl.LinePlot(am_pulse)

        # Simulate it
        # x =np.arange(-20 * tb, 20 * tb, 40 * tb / 2000)
        a = 20
        x = np.arange(-a * tb, a * tb, 40 * tb / 2000)
        [a, b] = rf_ext.sim.abrm(am_pulse,
                                 x,
                                 True)
        Mxy = 2 * np.multiply(np.conj(a), b)
        pl.LinePlot(Mxy)
    else:
        T = duration
        Ns = am_pulse.shape[0]
        t = np.linspace(-T / 2, T / 2, Ns, endpoint=True) * 1000
        plt.figure()
        plt.plot(t, np.abs(am_pulse))
        plt.xlabel('ms')
        plt.ylabel('a.u.')
        plt.title('|AM|')
        plt.figure()
        plt.plot(t, fm_pulse / 1000)
        plt.xlabel('ms')
        plt.ylabel('kHz')
        plt.title('FM')

        # Simulate it
        b1 = np.arange(0, 1, 0.01)  # b1 grid we simulate the pulse over, Gauss
        b1 = np.reshape(b1, (np.size(b1), 1))
        a = np.zeros(np.shape(b1), dtype='complex')
        b = np.zeros(np.shape(b1), dtype='complex')
        for ii in range(0, np.size(b1)):
            [a[ii], b[ii]] = rf_ext.sim.abrm_nd(2 * np.pi * (T / t.shape[0]) * 4258 * b1[ii] * am_pulse, np.ones(1),
                                                T / t.shape[0] * np.reshape(fm_pulse, (np.size(fm_pulse), 1)))
        Mz = 1 - 2 * np.abs(b) ** 2
        plt.figure()
        plt.plot(b1, Mz)
        plt.xlabel('Gauss')
        plt.ylabel('Mz')


def get_RF_train(am_pulse, fm_pulse, duration, system, flip_angle):
    #scale the amplitude according to pulseq requirements


    RF_train = []
    Ns = am_pulse.shape[0]
    Tseg = np.round(duration / Ns, 6)
    t_rf_raster = system.rf_raster_time
    system.rf_raster_time = Tseg
    [signal, _] = get_signal(pulse=am_pulse, duration=duration, flip_angle=flip_angle, system=system)


    t_block = np.round(0.5 * Tseg, 6)  # Fixed at 50% duty cycle
    system.rf_raster_time = t_rf_raster
    system.rf_ringdown_time = np.round(0.5 * Tseg, 6)  # To ensure 50% duty cycle

    ph = 0
    alpha = 0

    for n in range(1, Ns):
        flip_angle_block = 2 * math.pi * signal[n - 1] * t_block
        pulse = pp.make_block_pulse(flip_angle=flip_angle_block, duration=t_block,
                                    phase_offset=ph, system=system)


        rfp = SimpleNamespace()
        rfp.type = 'rf'
        rfp.signal = pulse.signal
        rfp.t = pulse.t
        rfp.freq_offset = pulse.freq_offset
        rfp.phase_offset = pulse.phase_offset
        rfp.dead_time = 0
        rfp.ringdown_time = system.rf_ringdown_time
        rfp.delay = 0

        RF_train.append(rfp)

        alpha = alpha + 2 * math.pi * signal[n] * (Tseg)
        del_ph = integrate.trapz([fm_pulse[n], fm_pulse[n - 1]], x=None, dx=t_block)

        ph = ph + del_ph * 2 * math.pi

    print(flip_angle)
    print(alpha)
    return RF_train
