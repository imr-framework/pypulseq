import numpy as np
from pypulseq.convert import convert


def test_report(self):
    """
    Analyze the sequence and return a text report.
    """
    # Find RF pulses and list flip angles
    flip_angles_deg = []
    for k in self.rf_library.keys:
        lib_data = self.rf_library.data[k]
        rf = self.rf_from_lib_data(lib_data)
        flip_angles_deg.append(np.abs(np.sum(rf.signal) * rf.t[0] * 360))

    flip_angles_deg = np.unique(flip_angles_deg)

    # Calculate TE, TR
    duration, num_blocks, event_count = self.duration()

    k_traj_adc, k_traj, t_excitation, t_refocusing, t_adc = self.calculate_kspace()

    k_abs_adc = np.sqrt(np.sum(np.square(k_traj_adc), axis=0))
    k_abs_echo, index_echo = np.min(k_abs_adc), np.argmin(k_abs_adc)
    t_echo = t_adc[index_echo]
    t_ex_tmp = t_excitation[t_excitation < t_echo]
    TE = t_echo - t_ex_tmp[-1]

    if len(t_excitation) < 2:
        TR = duration
    else:
        t_ex_tmp1 = t_excitation[t_excitation > t_echo]
        if len(t_ex_tmp1) == 0:
            TR = t_ex_tmp[-1] - t_ex_tmp[-2]
        else:
            TR = t_ex_tmp1[0] - t_ex_tmp[-1]

    # Check sequence dimensionality and spatial resolution
    k_extent = np.max(np.abs(k_traj_adc), axis=1)
    k_scale = np.max(k_extent)
    if k_scale != 0:
        k_bins = 4e6
        k_threshold = k_scale / k_bins

        # Detect unused dimensions and delete them
        if np.any(k_extent < k_threshold):
            k_traj_adc = np.delete(k_traj_adc, np.where(k_extent < k_threshold), axis=0)
            k_extent = np.delete(k_extent, np.where(k_extent < k_threshold), axis=0)

        # Bin the k-space trajectory to detect repetitions / slices
        k_len = k_traj_adc.shape[1]
        k_repeat = np.zeros(k_len)

        k_map = dict()
        for i in range(k_len):
            l = k_bins + np.round(k_traj_adc[:, i] / k_threshold)
            key_string = ('{:.0f} ' * len(l)).format(*l)
            if key_string in k_map:
                k_repeat[i] = k_map[key_string] + 1
            else:
                k_repeat[i] = 1
            k_map[key_string] = k_repeat[i]

        repeats = np.max(k_repeat)

        k_traj_rep1 = k_traj_adc[:, k_repeat == 1]

        k_counters = np.zeros(k_traj_rep1.shape)
        dims = k_traj_rep1.shape[0]
        ordering = dict()
        for j in range(dims):
            c = 1
            k_map = dict()
            for i in range(k_traj_rep1.shape[1]):
                key = round(k_traj_rep1[j, i] / k_threshold)
                if key in k_map:
                    k_counters[j, i] = k_map[key]
                else:
                    k_counters[j, i] = c
                    k_map[key] = c
                    c += 1
            ordering[j] = k_map.values()

        unique_k_positions = np.max(k_counters, axis=1)
        is_cartesian = np.prod(unique_k_positions) == k_traj_rep1.shape[1]
    else:
        unique_k_positions = 1

    gw = self.gradient_waveforms()
    gws = (gw[:, 1:] - gw[:, :-1]) / self.system.grad_raster_time
    ga = np.max(np.abs(gw), axis=1)
    gs = np.max(np.abs(gws), axis=1)

    ga_abs = np.max(np.sqrt(np.sum(np.square(gw), axis=0)))
    gs_abs = np.max(np.sqrt(np.sum(np.square(gws), axis=0)))

    timing_ok, timing_error_report = self.check_timing()

    report = f'Number of blocks: {num_blocks}\n' \
        f'Number of events:\n' \
        f'RF: {event_count[1]:6.0f}\n' \
        f'Gx: {event_count[2]:6.0f}\n' \
        f'Gy: {event_count[3]:6.0f}\n' \
        f'Gz: {event_count[4]:6.0f}\n' \
        f'ADC: {event_count[5]:6.0f}\n' \
        f'Delay: {event_count[0]:6.0f}\n' \
        f'Sequence duration: {duration:.6f} s\n' \
        f'TE: {TE:.6f} s\n' \
        f'TR: {TR:.6f} s\n'
    report += 'Flip angle: ' + ('{:.02f} ' * len(flip_angles_deg)).format(*flip_angles_deg) + 'deg\n'
    report += 'Unique k-space positions (aka cols, rows, etc.): ' + ('{:.0f} ' * len(unique_k_positions)).format(
        *unique_k_positions) + '\n'

    if len(np.where(unique_k_positions > 1)):
        report += f'Dimensions: {len(k_extent)}\n'
        report += ('Spatial resolution: {:.02f} mm\n' * len(k_extent)).format(*(0.5 / k_extent * 1e3))
        report += f'Repetitions/slices/contrasts: {repeats}\n'

        if is_cartesian:
            report += 'Cartesian encoding trajectory detected\n'
        else:
            report += 'Non-cartesian/irregular encoding trajectory detected (eg: EPI, spiral, radial, etc.)\n'

    if timing_ok:
        report += 'Block timing check passed successfully\n'
    else:
        report += f'Block timing check failed. Error listing follows:\n {timing_error_report}'

    ga_converted = convert(from_value=ga, from_unit='Hz/m', to_unit='mT/m')
    gs_converted = convert(from_value=gs, from_unit='Hz/m/s', to_unit='T/m/s')
    report += 'Max gradient: ' + ('{:.0f} ' * len(ga)).format(*ga) + 'Hz/m == ' + (
            '{:.02f} ' * len(ga_converted)).format(*ga_converted) + 'mT/m\n'
    report += 'Max slew rate: ' + ('{:.0f} ' * len(gs)).format(*gs) + 'Hz/m/s == ' + (
            '{:.02f} ' * len(gs_converted)).format(*gs_converted) + 'mT/m/s\n'

    ga_abs_converted = convert(from_value=ga_abs, from_unit='Hz/m', to_unit='mT/m')
    gs_abs_converted = convert(from_value=gs_abs, from_unit='Hz/m/s', to_unit='T/m/s')
    report += f'Max absolute gradient: {ga_abs:.0f} Hz/m == {ga_abs_converted:.2f} mT/m\n'
    report += f'Max absolute slew rate: {gs_abs:g} Hz/m/s == {gs_abs_converted:.2f} T/m/s'

    return report
