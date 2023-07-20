import numpy as np

from pypulseq import eps
from pypulseq.convert import convert


def ext_test_report(self) -> str:
    """
    Analyze the sequence and return a text report.

    Returns
    -------
    report : str

    """
    # Find RF pulses and list flip angles
    flip_angles_deg = []
    for k in self.rf_library.keys:
        lib_data = self.rf_library.data[k]
        if len(self.rf_library.type) >= k:
            rf = self.rf_from_lib_data(lib_data, self.rf_library.type[k])
        else:
            rf = self.rf_from_lib_data(lib_data)
        flip_angles_deg.append(
            np.abs(np.sum(rf.signal[:-1] * (rf.t[1:] - rf.t[:-1]))) * 360
        )

    flip_angles_deg = np.unique(flip_angles_deg)

    # Calculate TE, TR
    duration, num_blocks, event_count = self.duration()

    k_traj_adc, k_traj, t_excitation, t_refocusing, t_adc = self.calculate_kspacePP()

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
        k_map = dict()
        for i in range(k_len):
            key_string = str(
                (k_bins + np.round(k_traj_adc[:, i] / k_threshold)).astype(np.int32)
            )
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
            counts_unique[i] = np.sum(
                repeats_unique[i] == k_storage[: k_storage_next - 1]
            )

        k_traj_rep1 = k_traj_adc[:, k_repeat == 1]

        k_counters = np.zeros_like(k_traj_rep1)
        dims = k_traj_rep1.shape[0]
        
        for j in range(dims):
            k_map = dict()
            k_storage = np.zeros(k_len)
            k_storage_next = 0

            for i in range(k_traj_rep1.shape[1]):
                key = np.round(k_traj_rep1[j, i] / k_threshold).astype(np.int32)
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
        unique_k_positions = 1

    # gw_data = self.gradient_waveforms()
    waveforms_and_times = self.waveforms_and_times()
    gw_data = waveforms_and_times[0]
    gws = np.zeros_like(gw_data)
    ga = np.zeros(len(gw_data))
    gs = np.zeros(len(gw_data))

    common_time = np.unique(np.concatenate(gw_data, axis=1)[0])
    gw_ct = np.zeros((len(gw_data), len(common_time)))
    gs_ct = np.zeros((len(gw_data), len(common_time) - 1))
    for gc in range(len(gw_data)):
        if gw_data[gc].shape[1] > 0:
            # Slew
            gws[gc] = (gw_data[gc][1, 1:] - gw_data[gc][1, :-1]) / (
                gw_data[gc][0, 1:] - gw_data[gc][0, :-1]
            )

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
            gs_ct[gc] = (gw_ct[gc][1:] - gw_ct[gc][:-1]) / (
                common_time[1:] - common_time[:-1] + 1e-10
            )

            # Max grad/slew per channel
            ga[gc] = np.max(np.abs(gw_data[gc][1:]))
            gs[gc] = np.max(np.abs(gws[gc]))

    ga_abs = np.max(np.sqrt(np.sum(np.square(gw_ct), axis=0)))
    gs_abs = np.max(np.sqrt(np.sum(np.square(gs_ct), axis=0)))

    timing_ok, timing_error_report = self.check_timing()

    report = (
        f"Number of blocks: {num_blocks}\n"
        f"Number of events:\n"
        f"RF: {event_count[1]:6.0f}\n"
        f"Gx: {event_count[2]:6.0f}\n"
        f"Gy: {event_count[3]:6.0f}\n"
        f"Gz: {event_count[4]:6.0f}\n"
        f"ADC: {event_count[5]:6.0f}\n"
        f"Delay: {event_count[0]:6.0f}\n"
        f"Sequence duration: {duration:.6f} s\n"
        f"TE: {TE:.6f} s\n"
        f"TR: {TR:.6f} s\n"
    )
    report += (
        "Flip angle: "
        + ("{:.02f} " * len(flip_angles_deg)).format(*flip_angles_deg)
        + "deg\n"
    )
    report += (
        "Unique k-space positions (aka cols, rows, etc.): "
        + ("{:.0f} " * len(unique_k_positions)).format(*unique_k_positions)
        + "\n"
    )

    if np.any(unique_k_positions > 1):
        report += f"Dimensions: {len(k_extent)}\n"
        report += ("Spatial resolution: {:.02f} mm\n" * len(k_extent)).format(
            *(0.5 / k_extent * 1e3)
        )
        report += f"Repetitions/slices/contrasts: {repeats_median}; range: [{repeats_min, repeats_max}]\n"

        if is_cartesian:
            report += "Cartesian encoding trajectory detected\n"
        else:
            report += "Non-cartesian/irregular encoding trajectory detected (eg: EPI, spiral, radial, etc.)\n"

    if timing_ok:
        report += "Event timing check passed successfully\n"
    else:
        report += (
            f"Event timing check failed. Error listing follows:\n {timing_error_report}"
        )

    ga_converted = convert(from_value=ga, from_unit="Hz/m", to_unit="mT/m")
    gs_converted = convert(from_value=gs, from_unit="Hz/m/s", to_unit="T/m/s")
    report += (
        "Max gradient: "
        + ("{:.0f} " * len(ga)).format(*ga)
        + "Hz/m == "
        + ("{:.02f} " * len(ga_converted)).format(*ga_converted)
        + "mT/m\n"
    )
    report += (
        "Max slew rate: "
        + ("{:.0f} " * len(gs)).format(*gs)
        + "Hz/m/s == "
        + ("{:.02f} " * len(ga_converted)).format(*gs_converted)
        + "T/m/s\n"
    )

    ga_abs_converted = convert(from_value=ga_abs, from_unit="Hz/m", to_unit="mT/m")
    gs_abs_converted = convert(from_value=gs_abs, from_unit="Hz/m/s", to_unit="T/m/s")
    report += (
        f"Max absolute gradient: {ga_abs:.0f} Hz/m == {ga_abs_converted:.2f} mT/m\n"
    )
    report += (
        f"Max absolute slew rate: {gs_abs:g} Hz/m/s == {gs_abs_converted:.2f} T/m/s"
    )

    return report
