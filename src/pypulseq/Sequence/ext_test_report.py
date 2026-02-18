from typing import Any, Dict, List, Union

import numpy as np

from pypulseq import eps
from pypulseq.convert import convert


def ext_test_report_data(self) -> Dict[str, Any]:
    """
    Analyze the sequence and return statistics as a dictionary.

    Returns
    -------
    data : dict
        Dictionary containing sequence statistics with the following keys:
        - num_blocks: int
        - event_count: dict with keys 'rf', 'gx', 'gy', 'gz', 'adc', 'delay'
        - duration: float (seconds)
        - TE: float (seconds)
        - TR: float (seconds)
        - flip_angles_deg: list of float
        - unique_k_positions: np.ndarray
        - dimensions: int (if applicable)
        - spatial_resolution_mm: list of float (if applicable)
        - repetitions: dict with 'median', 'min', 'max' (if applicable)
        - is_cartesian: bool (if applicable)
        - max_gradient: dict with 'per_channel_Hz_m', 'per_channel_mT_m', 'absolute_Hz_m', 'absolute_mT_m'
        - max_slew_rate: dict with 'per_channel_Hz_m_s', 'per_channel_T_m_s', 'absolute_Hz_m_s', 'absolute_T_m_s'
        - timing_ok: bool
        - timing_error_report: list
    """
    # Find RF pulses and list flip angles
    flip_angles_deg = []
    for k in self.rf_library.data:
        lib_data = self.rf_library.data[k]
        if len(self.rf_library.type) >= k:
            rf = self.rf_from_lib_data(lib_data, self.rf_library.type[k])
        else:
            rf = self.rf_from_lib_data(lib_data)
        flip_angles_deg.append(np.abs(np.sum(rf.signal[:-1] * (rf.t[1:] - rf.t[:-1]))) * 360)

    flip_angles_deg = np.unique(flip_angles_deg)

    # Calculate TE, TR
    duration, num_blocks, event_count = self.duration()

    k_traj_adc, _, t_excitation, _, t_adc = self.calculate_kspace()
    t_excitation = np.asarray(t_excitation)

    # remove all ADC events that come before the first RF event (noise scans or alike)
    t_adc = t_adc[t_adc > t_excitation[0]]

    k_abs_adc = np.sqrt(np.sum(np.square(k_traj_adc), axis=0))
    k_abs_echo, index_echo = np.min(k_abs_adc), np.argmin(k_abs_adc)
    t_echo = t_adc[index_echo]
    if k_abs_echo > eps:
        i2check = []
        # Check if ADC k-space trajectory has elements left and right to index_echo
        if index_echo > 1:
            i2check.append(index_echo - 1)
        if index_echo < len(k_abs_adc):
            i2check.append(index_echo + 1)

        for a in range(len(i2check)):
            v_i_to_0 = -k_traj_adc[:, index_echo]
            v_i_to_t = k_traj_adc[:, i2check[a]] - k_traj_adc[:, index_echo]
            # Project v_i_to_0 to v_i_to_t
            p_vit = np.matmul(v_i_to_0, v_i_to_t) / np.square(np.linalg.norm(v_i_to_t))
            if p_vit > 0:
                # We have found a bracket for the echo and the proportionality coefficient is p_vit
                t_echo = t_adc[index_echo] * (1 - p_vit) + t_adc[i2check[a]] * p_vit

    if len(t_excitation) != 0:
        t_ex_tmp = t_excitation[t_excitation < t_echo]
        TE = t_echo - t_ex_tmp[-1]
    else:
        TE = np.nan

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
    is_cartesian = False
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
        k_storage = np.zeros(k_len)
        k_storage_next = 0
        k_map = {}
        keys = np.round(k_traj_adc / k_threshold).astype(np.int32)
        for i in range(k_len):
            key_string = tuple(keys[:, i])
            k_storage_ind = k_map.get(key_string)
            if k_storage_ind is None:
                k_storage_ind = k_storage_next
                k_map[key_string] = k_storage_ind
                k_storage_next += 1
            k_storage[k_storage_ind] = k_storage[k_storage_ind] + 1
            k_repeat[i] = k_storage[k_storage_ind]

        repeats_max = np.max(k_storage[:k_storage_next])
        repeats_min = np.min(k_storage[:k_storage_next])
        repeats_median = np.median(k_storage[:k_storage_next])
        repeats_unique = np.unique(k_storage[:k_storage_next])
        counts_unique = np.zeros_like(repeats_unique)
        for i in range(len(repeats_unique)):
            counts_unique[i] = np.sum(repeats_unique[i] == k_storage[: k_storage_next - 1])

        k_traj_rep1 = k_traj_adc[:, k_repeat == 1]

        k_counters = np.zeros_like(k_traj_rep1)
        dims = k_traj_rep1.shape[0]

        keys = keys[:, k_repeat == 1]
        for j in range(dims):
            k_map = {}
            k_storage = np.zeros(k_len)
            k_storage_next = 0

            for i in range(k_traj_rep1.shape[1]):
                key = keys[j, i]
                k_storage_ind = k_map.get(key)
                if k_storage_ind is None:
                    k_storage_ind = k_map.get(key + 1)
                if k_storage_ind is None:
                    k_storage_ind = k_map.get(key - 1)
                if k_storage_ind is None:
                    k_storage_ind = k_storage_next
                    k_map[key] = k_storage_ind
                    k_storage_next += 1
                    k_storage[k_storage_ind] = k_traj_rep1[j, i]
                k_counters[j, i] = k_storage_ind

        unique_k_positions = np.max(k_counters, axis=1) + 1
        is_cartesian = np.prod(unique_k_positions) == k_traj_rep1.shape[1]
    else:
        unique_k_positions = np.ones(1)

    gw_data = self.waveforms()
    gws = [np.zeros_like(x) for x in gw_data]
    ga = np.zeros(len(gw_data))
    gs = np.zeros(len(gw_data))

    common_time = np.unique(np.concatenate(gw_data, axis=1)[0])

    # catch case where no gradients are present (e.g. FID)
    if all(x.size == 0 for x in gw_data):
        gw_ct = np.zeros(0)
        gs_ct = np.zeros(0)
    else:
        gw_ct = np.zeros((len(gw_data), len(common_time)))
        gs_ct = np.zeros((len(gw_data), len(common_time) - 1))

    for gc in range(len(gw_data)):
        if gw_data[gc].shape[1] > 0:
            # Slew
            gws[gc] = (gw_data[gc][1, 1:] - gw_data[gc][1, :-1]) / (gw_data[gc][0, 1:] - gw_data[gc][0, :-1])

            # Interpolate to common time
            gw_ct[gc] = np.interp(
                x=common_time,
                xp=gw_data[gc][0, :],
                fp=gw_data[gc][1, :],
                left=0,
                right=0,
            )

            # Sometimes there are very small steps in common_time:
            #   add 1e-10 to resolve instability (adding eps is too small)
            gs_ct[gc] = (gw_ct[gc][1:] - gw_ct[gc][:-1]) / (common_time[1:] - common_time[:-1] + 1e-10)

            # Max grad/slew per channel
            ga[gc] = np.max(np.abs(gw_data[gc][1:]))
            gs[gc] = np.max(np.abs(gws[gc]))

    ga_abs = np.max(np.sqrt(np.sum(np.square(gw_ct), axis=0)))
    gs_abs = np.max(np.sqrt(np.sum(np.square(gs_ct), axis=0)))

    timing_ok, timing_error_report = self.check_timing()

    # Convert gradient/slew values
    ga_converted = convert(from_value=ga, from_unit='Hz/m', to_unit='mT/m', gamma=self.system.gamma)
    gs_converted = convert(from_value=gs, from_unit='Hz/m/s', to_unit='T/m/s', gamma=self.system.gamma)
    ga_abs_converted = convert(from_value=ga_abs, from_unit='Hz/m', to_unit='mT/m', gamma=self.system.gamma)
    gs_abs_converted = convert(from_value=gs_abs, from_unit='Hz/m/s', to_unit='T/m/s', gamma=self.system.gamma)

    # Build the result dictionary
    data: Dict[str, Any] = {
        'num_blocks': num_blocks,
        'event_count': {
            'rf': int(event_count[1]),
            'gx': int(event_count[2]),
            'gy': int(event_count[3]),
            'gz': int(event_count[4]),
            'adc': int(event_count[5]),
            'delay': int(event_count[0]),
        },
        'duration': duration,
        'TE': TE,
        'TR': TR,
        'flip_angles_deg': list(flip_angles_deg),
        'unique_k_positions': unique_k_positions,
        'max_gradient': {
            'per_channel_Hz_m': list(ga),
            'per_channel_mT_m': list(ga_converted),
            'absolute_Hz_m': ga_abs,
            'absolute_mT_m': ga_abs_converted,
        },
        'max_slew_rate': {
            'per_channel_Hz_m_s': list(gs),
            'per_channel_T_m_s': list(gs_converted),
            'absolute_Hz_m_s': gs_abs,
            'absolute_T_m_s': gs_abs_converted,
        },
        'timing_ok': timing_ok,
        'timing_error_report': timing_error_report,
    }

    # Add optional fields if there are multiple k-space positions
    if np.any(unique_k_positions > 1):
        data['dimensions'] = len(k_extent)
        data['spatial_resolution_mm'] = list(0.5 / k_extent * 1e3)
        data['repetitions'] = {
            'median': repeats_median,
            'min': repeats_min,
            'max': repeats_max,
        }
        data['is_cartesian'] = is_cartesian

    return data


def ext_test_report_str(data: Dict[str, Any]) -> str:
    """
    Format test report data dictionary into a human-readable string.

    Parameters
    ----------
    data : dict
        Dictionary returned by ext_test_report_data().

    Returns
    -------
    report : str
        Formatted text report.
    """
    event_count = data['event_count']
    flip_angles_deg = data['flip_angles_deg']
    unique_k_positions = data['unique_k_positions']
    ga = data['max_gradient']['per_channel_Hz_m']
    ga_converted = data['max_gradient']['per_channel_mT_m']
    gs = data['max_slew_rate']['per_channel_Hz_m_s']
    gs_converted = data['max_slew_rate']['per_channel_T_m_s']

    report = (
        f"Number of blocks: {data['num_blocks']}\n"
        f'Number of events:\n'
        f"RF: {event_count['rf']:6.0f}\n"
        f"Gx: {event_count['gx']:6.0f}\n"
        f"Gy: {event_count['gy']:6.0f}\n"
        f"Gz: {event_count['gz']:6.0f}\n"
        f"ADC: {event_count['adc']:6.0f}\n"
        f"Delay: {event_count['delay']:6.0f}\n"
        f"Sequence duration: {data['duration']:.6f} s\n"
        f"TE: {data['TE']:.6f} s\n"
        f"TR: {data['TR']:.6f} s\n"
    )
    report += 'Flip angle: ' + ('{:.02f} ' * len(flip_angles_deg)).format(*flip_angles_deg) + 'deg\n'
    report += (
        'Unique k-space positions (aka cols, rows, etc.): '
        + ('{:.0f} ' * len(unique_k_positions)).format(*unique_k_positions)
        + '\n'
    )

    if 'dimensions' in data:
        k_extent = data['spatial_resolution_mm']
        repetitions = data['repetitions']
        report += f"Dimensions: {data['dimensions']}\n"
        report += ('Spatial resolution: {:.02f} mm\n' * len(k_extent)).format(*k_extent)
        report += (
            f"Repetitions/slices/contrasts: {repetitions['median']}; "
            f"range: [{repetitions['min']}, {repetitions['max']}]\n"
        )

        if data['is_cartesian']:
            report += 'Cartesian encoding trajectory detected\n'
        else:
            report += 'Non-cartesian/irregular encoding trajectory detected (eg: EPI, spiral, radial, etc.)\n'

    report += (
        'Max gradient: '
        + ('{:.0f} ' * len(ga)).format(*ga)
        + 'Hz/m == '
        + ('{:.02f} ' * len(ga_converted)).format(*ga_converted)
        + 'mT/m\n'
    )
    report += (
        'Max slew rate: '
        + ('{:.0f} ' * len(gs)).format(*gs)
        + 'Hz/m/s == '
        + ('{:.02f} ' * len(gs_converted)).format(*gs_converted)
        + 'T/m/s\n'
    )

    report += (
        f"Max absolute gradient: {data['max_gradient']['absolute_Hz_m']:.0f} Hz/m == "
        f"{data['max_gradient']['absolute_mT_m']:.2f} mT/m\n"
    )
    report += (
        f"Max absolute slew rate: {data['max_slew_rate']['absolute_Hz_m_s']:g} Hz/m/s == "
        f"{data['max_slew_rate']['absolute_T_m_s']:.2f} T/m/s"
    )

    timing_error_report = data['timing_error_report']
    if data['timing_ok']:
        report += '\nEvent timing check passed successfully\n'
    else:
        report += f'\nEvent timing check failed with {len(timing_error_report)} errors in total. \n'
        report += 'Details of the first up to 20 timing errors:'
        max_errors = min(20, len(timing_error_report))
        for line in timing_error_report[:max_errors]:
            report += f'\n{line}'
        if len(timing_error_report) > max_errors:
            report += '\n...'

    return report


def ext_test_report(self) -> str:
    """
    Analyze the sequence and return a text report.

    Returns
    -------
    report : str
        Formatted text report.
    """
    data = ext_test_report_data(self)
    return ext_test_report_str(data)

