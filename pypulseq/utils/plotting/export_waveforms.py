import math
import numpy as np
from matplotlib import pyplot as plt
from pypulseq.calc_rf_center import calc_rf_center
from pypulseq.Sequence.sequence import Sequence
from scipy.io import savemat

def export_waveforms(seq, time_range=(0, np.inf)):
    """
    Plot `Sequence`.

    Parameters
    ----------
    time_range : iterable, default=(0, np.inf)
        Time range (x-axis limits) for all waveforms. Default is 0 to infinity (entire sequence).

    Returns
    -------
    all_waveforms: dict
        Dictionary containing all sequence waveforms and time array(s)
        The keys are listed here:
        't_adc' - ADC timing array [seconds]
        't_rf' - RF timing array [seconds]
        ''
        'adc' - ADC complex signal (amplitude=1, phase=adc phase) [a.u.]
        'rf' - RF complex signal []
        'gx' - x gradient []
        'gy' - y gradient []
        'gz' - z gradient []


    """
    # Check time range validity
    if not all([isinstance(x, (int, float)) for x in time_range]) or len(time_range) != 2:
        raise ValueError('Invalid time range')

    t0 = 0
    adc_t_all = np.array([])
    adc_signal_all = np.array([],dtype=complex)
    rf_t_all =np.array([])
    rf_signal_all = np.array([],dtype=complex)
    rf_t_centers = np.array([])
    rf_signal_centers = np.array([],dtype=complex)
    gx_t_all = np.array([])
    gy_t_all = np.array([])
    gz_t_all = np.array([])
    gx_all = np.array([])
    gy_all = np.array([])
    gz_all = np.array([])


    for block_counter in range(len(seq.dict_block_events)): # For each block
        block = seq.get_block(block_counter + 1)  # Retrieve it
        is_valid = time_range[0] <= t0 <= time_range[1] # Check if "current time" is within requested range.
        if is_valid:
            # Case 1: ADC
            if hasattr(block, 'adc'):
                adc = block.adc # Get adc info
                # From Pulseq: According to the information from Klaus Scheffler and indirectly from Siemens this
                # is the present convention - the samples are shifted by 0.5 dwell # OK

                t = adc.delay + (np.arange(int(adc.num_samples)) + 0.5) * adc.dwell
                adc_t = t0 + t
                adc_signal = np.exp(1j * adc.phase_offset) * np.exp(1j * 2 * np.pi * t * adc.freq_offset)
                adc_t_all = np.append(adc_t_all, adc_t)
                adc_signal_all = np.append(adc_signal_all, adc_signal)

            if hasattr(block, 'rf'):
                rf = block.rf
                tc, ic = calc_rf_center(rf)
                t = rf.t + rf.delay
                tc = tc + rf.delay
                #
                # sp12.plot(t_factor * (t0 + t), np.abs(rf.signal))
                # sp13.plot(t_factor * (t0 + t), np.angle(rf.signal * np.exp(1j * rf.phase_offset)
                #                                         * np.exp(1j * 2 * math.pi * rf.t * rf.freq_offset)),
                #           t_factor * (t0 + tc), np.angle(rf.signal[ic] * np.exp(1j * rf.phase_offset)
                #                                          * np.exp(1j * 2 * math.pi * rf.t[ic] * rf.freq_offset)),
                #           'xb')

                rf_t = t0 + t
                rf = rf.signal * np.exp(1j * rf.phase_offset) \
                                                        * np.exp(1j * 2 * math.pi * rf.t * rf.freq_offset)
                rf_t_all = np.append(rf_t_all, rf_t)
                rf_signal_all = np.append(rf_signal_all, rf)
                rf_t_centers = np.append(rf_t_centers, rf_t[ic])
                rf_signal_centers = np.append(rf_signal_centers, rf[ic])

            grad_channels = ['gx', 'gy', 'gz']
            for x in range(len(grad_channels)): # Check each gradient channel: x, y, and z
                if hasattr(block, grad_channels[x]): # If this channel is on in current block
                    grad = getattr(block, grad_channels[x])
                    if grad.type == 'grad':# Arbitrary gradient option
                        # In place unpacking of grad.t with the starred expression
                        g_t = t0 +  grad.delay + [0, *(grad.t + (grad.t[1] - grad.t[0]) / 2),
                                              grad.t[-1] + grad.t[1] - grad.t[0]]
                        g = 1e-3 * np.array((grad.first, *grad.waveform, grad.last))
                    else: # Trapezoid gradient option
                        g_t = t0 + np.cumsum([0, grad.delay, grad.rise_time, grad.flat_time, grad.fall_time])
                        g = 1e-3 * grad.amplitude * np.array([0, 0, 1, 1, 0])

                    if grad.channel == 'x':
                        gx_t_all = np.append(gx_t_all, g_t)
                        gx_all = np.append(gx_all,g)
                    elif grad.channel == 'y':
                        gy_t_all = np.append(gy_t_all, g_t)
                        gy_all = np.append(gy_all,g)
                    elif grad.channel == 'z':
                        gz_t_all = np.append(gz_t_all, g_t)
                        gz_all = np.append(gz_all,g)


        t0 += seq.arr_block_durations[block_counter] # "Current time" gets updated to end of block just examined

    all_waveforms = {'t_adc': adc_t_all, 't_rf': rf_t_all, 't_rf_centers': rf_t_centers,
                     't_gx': gx_t_all, 't_gy':gy_t_all, 't_gz':gz_t_all,
                     'adc': adc_signal_all, 'rf': rf_signal_all, 'rf_centers': rf_signal_centers,'gx':gx_all, 'gy':gy_all, 'gz':gz_all,
                     'grad_unit': '[kHz/m]', 'rf_unit': '[Hz]', 'time_unit':'[seconds]'}

    return all_waveforms


if __name__ == '__main__':
    # Test with TSE sequence
    seq = Sequence()
    seq.read('tse_pypulseq.seq')

    # Plot with original function
    seq.plot(time_range=[2,2.25])

    # Save exported waveforms
    all_waveforms = export_waveforms(seq, time_range=[2,2.25])
    savemat('tse_waveforms.mat',all_waveforms)
