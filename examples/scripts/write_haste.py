import math
import warnings

import numpy as np

import pypulseq as pp


def main(
    plot: bool = False,
    test_report: bool = False,
    write_seq: bool = False,
    seq_filename: str = 'haste_pypulseq.seq',
    *,
    fov: float = 256e-3,
    n_x: int = 64,
    n_y: int = 64,
    n_y_pre: int = 8,
    n_echo: int | None = None,
    n_slices: int = 1,
    rf_flip_deg: int = 180,
    slice_thickness: float = 5e-3,
    te: float = 12e-3,
    tr: float = 2000e-3,
):
    """Create a HASTE (Half-Fourier Acquisition Single-shot TSE) sequence.

    Parameters
    ----------
    plot : bool, optional
        Plot the sequence diagram. Default is False.
    test_report : bool, optional
        Print a test report. Default is False.
    write_seq : bool, optional
        Write the sequence to a .seq file. Default is False.
    seq_filename : str, optional
        Output filename for the .seq file. Default is 'haste_pypulseq.seq'.
    fov : float, optional
        Field of view in meters. Default is 256e-3.
    n_x : int, optional
        Number of readout samples. Default is 64.
    n_y : int, optional
        Number of phase encoding steps. Default is 64.
    n_y_pre : int, optional
        Number of pre-encoding lines. Default is 8.
    n_echo : int or None, optional
        Number of echoes. Default is None (n_y / 2 + n_y_pre).
    n_slices : int, optional
        Number of slices. Default is 1.
    rf_flip_deg : int, optional
        Refocusing flip angle in degrees. Default is 180.
    slice_thickness : float, optional
        Slice thickness in meters. Default is 5e-3.
    te : float, optional
        Echo time in seconds. Default is 12e-3.
    tr : float, optional
        Repetition time in seconds. Default is 2000e-3.

    Returns
    -------
    seq : pypulseq.Sequence
        The HASTE sequence object.
    """
    dG = 250e-6

    # Set system limits
    system = pp.Opts(
        max_grad=30,
        grad_unit='mT/m',
        max_slew=170,
        slew_unit='T/m/s',
        rf_ringdown_time=100e-6,
        rf_dead_time=100e-6,
        adc_dead_time=10e-6,
    )

    seq = pp.Sequence(system)

    if n_echo is None:
        n_echo = int(n_y / 2 + n_y_pre)
    if isinstance(rf_flip_deg, int):
        rf_flip_deg = np.zeros(n_echo) + rf_flip_deg

    sampling_time = 6.4e-3
    readout_time = sampling_time + 2 * system.adc_dead_time
    t_ex = 2.5e-3
    t_ex_wd = t_ex + system.rf_ringdown_time + system.rf_dead_time
    t_ref = 2e-3
    t_ref_wd = t_ref + system.rf_ringdown_time + system.rf_dead_time
    t_sp = 0.5 * (te - readout_time - t_ref_wd)
    t_sp_ex = 0.5 * (te - t_ex_wd - t_ref_wd)
    fsp_r = 1.0
    fsp_s = 0.5

    rf_ex_phase = math.pi / 2
    rf_ref_phase = 0

    # Create excitation pulse and gradient
    flip_ex = np.deg2rad(90)
    rf_ex, gz, _ = pp.make_sinc_pulse(
        flip_angle=flip_ex,
        system=system,
        duration=t_ex,
        slice_thickness=slice_thickness,
        apodization=0.5,
        time_bw_product=4,
        phase_offset=rf_ex_phase,
        return_gz=True,
        delay=system.rf_dead_time,
        use='excitation',
    )
    gs_ex = pp.make_trapezoid(
        channel='z',
        system=system,
        amplitude=gz.amplitude,
        flat_time=t_ex_wd,
        rise_time=dG,
    )

    flip_ref = np.deg2rad(rf_flip_deg[0])
    rf_ref, gz, _ = pp.make_sinc_pulse(
        flip_angle=flip_ref,
        system=system,
        duration=t_ref,
        slice_thickness=slice_thickness,
        apodization=0.5,
        time_bw_product=4,
        phase_offset=rf_ref_phase,
        use='refocusing',
        return_gz=True,
        delay=system.rf_dead_time,
    )
    gs_ref = pp.make_trapezoid(
        channel='z',
        system=system,
        amplitude=gs_ex.amplitude,
        flat_time=t_ref_wd,
        rise_time=dG,
    )

    ags_ex = gs_ex.area / 2
    gs_spr = pp.make_trapezoid(
        channel='z',
        system=system,
        area=ags_ex * (1 + fsp_s),
        duration=t_sp,
        rise_time=dG,
    )
    gs_spex = pp.make_trapezoid(channel='z', system=system, area=ags_ex * fsp_s, duration=t_sp_ex, rise_time=dG)

    delta_k = 1 / fov
    k_width = n_x * delta_k

    gr_acq = pp.make_trapezoid(
        channel='x',
        system=system,
        flat_area=k_width,
        flat_time=readout_time,
        rise_time=dG,
    )
    adc = pp.make_adc(num_samples=n_x, duration=sampling_time, delay=system.adc_dead_time, system=system)
    gr_spr = pp.make_trapezoid(channel='x', system=system, area=gr_acq.area * fsp_r, duration=t_sp, rise_time=dG)

    agr_spr = gr_spr.area
    agr_preph = gr_acq.area / 2 + agr_spr
    gr_preph = pp.make_trapezoid(channel='x', system=system, area=agr_preph, duration=t_sp_ex, rise_time=dG)

    n_ex = 1
    pe_order = np.arange(-n_y_pre, n_y + 1).T
    phase_areas = pe_order * delta_k

    # Split gradients and recombine into blocks
    gs1_times = np.array([0, gs_ex.rise_time])
    gs1_amp = np.array([0, gs_ex.amplitude])
    gs1 = pp.make_extended_trapezoid(channel='z', times=gs1_times, amplitudes=gs1_amp)

    gs2_times = np.array([0, gs_ex.flat_time])
    gs2_amp = np.array([gs_ex.amplitude, gs_ex.amplitude])
    gs2 = pp.make_extended_trapezoid(channel='z', times=gs2_times, amplitudes=gs2_amp)

    gs3_times = np.array(
        [
            0,
            gs_spex.rise_time,
            gs_spex.rise_time + gs_spex.flat_time,
            gs_spex.rise_time + gs_spex.flat_time + gs_spex.fall_time,
        ]
    )
    gs3_amp = np.array([gs_ex.amplitude, gs_spex.amplitude, gs_spex.amplitude, gs_ref.amplitude])
    gs3 = pp.make_extended_trapezoid(channel='z', times=gs3_times, amplitudes=gs3_amp)

    gs4_times = np.array([0, gs_ref.flat_time])
    gs4_amp = np.array([gs_ref.amplitude, gs_ref.amplitude])
    gs4 = pp.make_extended_trapezoid(channel='z', times=gs4_times, amplitudes=gs4_amp)

    gs5_times = np.array(
        [
            0,
            gs_spr.rise_time,
            gs_spr.rise_time + gs_spr.flat_time,
            gs_spr.rise_time + gs_spr.flat_time + gs_spr.fall_time,
        ]
    )
    gs5_amp = np.array([gs_ref.amplitude, gs_spr.amplitude, gs_spr.amplitude, 0])
    gs5 = pp.make_extended_trapezoid(channel='z', times=gs5_times, amplitudes=gs5_amp)

    gs7_times = np.array(
        [
            0,
            gs_spr.rise_time,
            gs_spr.rise_time + gs_spr.flat_time,
            gs_spr.rise_time + gs_spr.flat_time + gs_spr.fall_time,
        ]
    )
    gs7_amp = np.array([0, gs_spr.amplitude, gs_spr.amplitude, gs_ref.amplitude])
    gs7 = pp.make_extended_trapezoid(channel='z', times=gs7_times, amplitudes=gs7_amp)

    # Readout gradient
    gr3 = gr_preph

    gr5_times = np.array(
        [
            0,
            gr_spr.rise_time,
            gr_spr.rise_time + gr_spr.flat_time,
            gr_spr.rise_time + gr_spr.flat_time + gr_spr.fall_time,
        ]
    )
    gr5_amp = np.array([0, gr_spr.amplitude, gr_spr.amplitude, gr_acq.amplitude])
    gr5 = pp.make_extended_trapezoid(channel='x', times=gr5_times, amplitudes=gr5_amp)

    gr6_times = np.array([0, readout_time])
    gr6_amp = np.array([gr_acq.amplitude, gr_acq.amplitude])
    gr6 = pp.make_extended_trapezoid(channel='x', times=gr6_times, amplitudes=gr6_amp)

    gr7_times = np.array(
        [
            0,
            gr_spr.rise_time,
            gr_spr.rise_time + gr_spr.flat_time,
            gr_spr.rise_time + gr_spr.flat_time + gr_spr.fall_time,
        ]
    )
    gr7_amp = np.array([gr_acq.amplitude, gr_spr.amplitude, gr_spr.amplitude, 0])
    gr7 = pp.make_extended_trapezoid(channel='x', times=gr7_times, amplitudes=gr7_amp)

    # Fill-times
    tex = gs1.shape_dur + gs2.shape_dur + gs3.shape_dur
    tref = gs4.shape_dur + gs5.shape_dur + gs7.shape_dur + readout_time
    tend = gs4.shape_dur + gs5.shape_dur
    te_train = tex + n_echo * tref + tend
    tr_fill = (tr - n_slices * te_train) / n_slices

    tr_fill = system.grad_raster_time * round(tr_fill / system.grad_raster_time)
    if tr_fill < 0:
        tr_fill = 1e-3
        warnings.warn(f'TR too short, adapted to include all slices to: {1000 * n_slices * (te_train + tr_fill)} ms')
    else:
        print(f'TR fill: {1000 * tr_fill} ms')
    tr_delay = pp.make_delay(tr_fill)
    delay_end = pp.make_delay(5)

    for i_excitation in range(n_ex):
        for i_slice in range(n_slices):
            rf_ex.freq_offset = gs_ex.amplitude * slice_thickness * (i_slice - (n_slices - 1) / 2)
            rf_ref.freq_offset = gs_ref.amplitude * slice_thickness * (i_slice - (n_slices - 1) / 2)
            # Align the phase for off-center slices
            rf_ex.phase_offset = rf_ex_phase - 2 * math.pi * rf_ex.freq_offset * pp.calc_rf_center(rf_ex)[0]
            rf_ref.phase_offset = rf_ref_phase - 2 * math.pi * rf_ref.freq_offset * pp.calc_rf_center(rf_ref)[0]

            seq.add_block(gs1)
            seq.add_block(rf_ex, gs2)
            seq.add_block(gs3, gr3)

            for i_echo in range(n_echo):
                if i_excitation >= 0:
                    phase_area = phase_areas[i_echo]
                else:
                    phase_area = 0

                gp_pre = pp.make_trapezoid(
                    channel='y',
                    system=system,
                    area=phase_area,
                    duration=t_sp,
                    rise_time=dG,
                )
                gp_rew = pp.make_trapezoid(
                    channel='y',
                    system=system,
                    area=-phase_area,
                    duration=t_sp,
                    rise_time=dG,
                )

                seq.add_block(rf_ref, gs4)
                seq.add_block(gr5, gp_pre, gs5)

                if i_excitation >= 0:
                    seq.add_block(gr6, adc)
                else:
                    seq.add_block(gr6)

                seq.add_block(gr7, gp_rew, gs7)

            seq.add_block(gs4)
            seq.add_block(gs5)
            seq.add_block(tr_delay)

    seq.add_block(delay_end)

    ok, error_report = seq.check_timing()
    if ok:
        print('Timing check passed successfully')
    else:
        print('Timing check failed. Error listing follows:')
        [print(e) for e in error_report]

    if test_report:
        print(seq.test_report())

    if plot:
        seq.plot()

    seq.set_definition(key='FOV', value=[fov, fov, slice_thickness * n_slices])
    seq.set_definition(key='Name', value='haste')

    if write_seq:
        seq.write(seq_filename)

    return seq


if __name__ == '__main__':
    main(plot=True, write_seq=True)
