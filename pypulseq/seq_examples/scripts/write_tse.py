import math
import warnings

import numpy as np

import pypulseq as pp


def main(plot: bool, write_seq: bool, seq_filename: str = "tse_pypulseq.seq"):
    # ======
    # SETUP
    # ======
    dG = 250e-6

    # Set system limits
    system = pp.Opts(
        max_grad=32,
        grad_unit="mT/m",
        max_slew=130,
        slew_unit="T/m/s",
        rf_ringdown_time=100e-6,
        rf_dead_time=100e-6,
        adc_dead_time=10e-6,
    )

    seq = pp.Sequence(system)  # Create a new sequence object
    fov = 256e-3  # Define FOV and resolution
    Nx, Ny = 128, 128
    n_echo = 16  # Number of echoes
    n_slices = 1
    rf_flip = 180  # Flip angle
    if isinstance(rf_flip, int):
        rf_flip = np.zeros(n_echo) + rf_flip
    slice_thickness = 5e-3
    TE = 12e-3  # Echo time
    TR = 2000e-3  # Repetition time

    sampling_time = 6.4e-3
    readout_time = sampling_time + 2 * system.adc_dead_time
    t_ex = 2.5e-3
    t_exwd = t_ex + system.rf_ringdown_time + system.rf_dead_time
    t_ref = 2e-3
    t_refwd = t_ref + system.rf_ringdown_time + system.rf_dead_time
    t_sp = 0.5 * (TE - readout_time - t_refwd)
    t_spex = 0.5 * (TE - t_exwd - t_refwd)
    fsp_r = 1
    fsp_s = 0.5

    rf_ex_phase = np.pi / 2
    rf_ref_phase = 0

    # ======
    # CREATE EVENTS
    # ======
    flip_ex = 90 * np.pi / 180
    rf_ex, gz, _ = pp.make_sinc_pulse(
        flip_angle=flip_ex,
        system=system,
        duration=t_ex,
        slice_thickness=slice_thickness,
        apodization=0.5,
        time_bw_product=4,
        phase_offset=rf_ex_phase,
        return_gz=True,
    )
    gs_ex = pp.make_trapezoid(
        channel="z",
        system=system,
        amplitude=gz.amplitude,
        flat_time=t_exwd,
        rise_time=dG,
    )

    flip_ref = rf_flip[0] * np.pi / 180
    rf_ref, gz, _ = pp.make_sinc_pulse(
        flip_angle=flip_ref,
        system=system,
        duration=t_ref,
        slice_thickness=slice_thickness,
        apodization=0.5,
        time_bw_product=4,
        phase_offset=rf_ref_phase,
        use="refocusing",
        return_gz=True,
    )
    gs_ref = pp.make_trapezoid(
        channel="z",
        system=system,
        amplitude=gs_ex.amplitude,
        flat_time=t_refwd,
        rise_time=dG,
    )

    ags_ex = gs_ex.area / 2
    gs_spr = pp.make_trapezoid(
        channel="z",
        system=system,
        area=ags_ex * (1 + fsp_s),
        duration=t_sp,
        rise_time=dG,
    )
    gs_spex = pp.make_trapezoid(
        channel="z", system=system, area=ags_ex * fsp_s, duration=t_spex, rise_time=dG
    )

    delta_k = 1 / fov
    k_width = Nx * delta_k

    gr_acq = pp.make_trapezoid(
        channel="x",
        system=system,
        flat_area=k_width,
        flat_time=readout_time,
        rise_time=dG,
    )
    adc = pp.make_adc(
        num_samples=Nx, duration=sampling_time, delay=system.adc_dead_time
    )
    gr_spr = pp.make_trapezoid(
        channel="x",
        system=system,
        area=gr_acq.area * fsp_r,
        duration=t_sp,
        rise_time=dG,
    )

    agr_spr = gr_spr.area
    agr_preph = gr_acq.area / 2 + agr_spr
    gr_preph = pp.make_trapezoid(
        channel="x", system=system, area=agr_preph, duration=t_spex, rise_time=dG
    )

    # Phase-encoding
    n_ex = math.floor(Ny / n_echo)
    pe_steps = np.arange(1, n_echo * n_ex + 1) - 0.5 * n_echo * n_ex - 1
    if divmod(n_echo, 2)[1] == 0:
        pe_steps = np.roll(pe_steps, [0, int(-np.round(n_ex / 2))])
    pe_order = pe_steps.reshape((n_ex, n_echo), order="F").T
    phase_areas = pe_order * delta_k

    # Split gradients and recombine into blocks
    gs1_times = np.array([0, gs_ex.rise_time])
    gs1_amp = np.array([0, gs_ex.amplitude])
    gs1 = pp.make_extended_trapezoid(channel="z", times=gs1_times, amplitudes=gs1_amp)

    gs2_times = np.array([0, gs_ex.flat_time])
    gs2_amp = np.array([gs_ex.amplitude, gs_ex.amplitude])
    gs2 = pp.make_extended_trapezoid(channel="z", times=gs2_times, amplitudes=gs2_amp)

    gs3_times = np.array(
        [
            0,
            gs_spex.rise_time,
            gs_spex.rise_time + gs_spex.flat_time,
            gs_spex.rise_time + gs_spex.flat_time + gs_spex.fall_time,
        ]
    )
    gs3_amp = np.array(
        [gs_ex.amplitude, gs_spex.amplitude, gs_spex.amplitude, gs_ref.amplitude]
    )
    gs3 = pp.make_extended_trapezoid(channel="z", times=gs3_times, amplitudes=gs3_amp)

    gs4_times = np.array([0, gs_ref.flat_time])
    gs4_amp = np.array([gs_ref.amplitude, gs_ref.amplitude])
    gs4 = pp.make_extended_trapezoid(channel="z", times=gs4_times, amplitudes=gs4_amp)

    gs5_times = np.array(
        [
            0,
            gs_spr.rise_time,
            gs_spr.rise_time + gs_spr.flat_time,
            gs_spr.rise_time + gs_spr.flat_time + gs_spr.fall_time,
        ]
    )
    gs5_amp = np.array([gs_ref.amplitude, gs_spr.amplitude, gs_spr.amplitude, 0])
    gs5 = pp.make_extended_trapezoid(channel="z", times=gs5_times, amplitudes=gs5_amp)

    gs7_times = np.array(
        [
            0,
            gs_spr.rise_time,
            gs_spr.rise_time + gs_spr.flat_time,
            gs_spr.rise_time + gs_spr.flat_time + gs_spr.fall_time,
        ]
    )
    gs7_amp = np.array([0, gs_spr.amplitude, gs_spr.amplitude, gs_ref.amplitude])
    gs7 = pp.make_extended_trapezoid(channel="z", times=gs7_times, amplitudes=gs7_amp)

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
    gr5 = pp.make_extended_trapezoid(channel="x", times=gr5_times, amplitudes=gr5_amp)

    gr6_times = np.array([0, readout_time])
    gr6_amp = np.array([gr_acq.amplitude, gr_acq.amplitude])
    gr6 = pp.make_extended_trapezoid(channel="x", times=gr6_times, amplitudes=gr6_amp)

    gr7_times = np.array(
        [
            0,
            gr_spr.rise_time,
            gr_spr.rise_time + gr_spr.flat_time,
            gr_spr.rise_time + gr_spr.flat_time + gr_spr.fall_time,
        ]
    )
    gr7_amp = np.array([gr_acq.amplitude, gr_spr.amplitude, gr_spr.amplitude, 0])
    gr7 = pp.make_extended_trapezoid(channel="x", times=gr7_times, amplitudes=gr7_amp)

    # Fill-times
    t_ex = pp.calc_duration(gs1) + pp.calc_duration(gs2) + pp.calc_duration(gs3)
    t_ref = (
        pp.calc_duration(gs4)
        + pp.calc_duration(gs5)
        + pp.calc_duration(gs7)
        + readout_time
    )
    t_end = pp.calc_duration(gs4) + pp.calc_duration(gs5)

    TE_train = t_ex + n_echo * t_ref + t_end
    TR_fill = (TR - n_slices * TE_train) / n_slices
    # Round to gradient raster
    TR_fill = system.grad_raster_time * np.round(TR_fill / system.grad_raster_time)
    if TR_fill < 0:
        TR_fill = 1e-3
        warnings.warn(
            f"TR too short, adapted to include all slices to: {1000 * n_slices * (TE_train + TR_fill)} ms"
        )
    else:
        print(f"TR fill: {1000 * TR_fill} ms")
    delay_TR = pp.make_delay(TR_fill)

    # ======
    # CONSTRUCT SEQUENCE
    # ======
    for k_ex in range(n_ex + 1):
        for s in range(n_slices):
            rf_ex.freq_offset = (
                gs_ex.amplitude * slice_thickness * (s - (n_slices - 1) / 2)
            )
            rf_ref.freq_offset = (
                gs_ref.amplitude * slice_thickness * (s - (n_slices - 1) / 2)
            )
            rf_ex.phase_offset = (
                rf_ex_phase
                - 2 * np.pi * rf_ex.freq_offset * pp.calc_rf_center(rf_ex)[0]
            )
            rf_ref.phase_offset = (
                rf_ref_phase
                - 2 * np.pi * rf_ref.freq_offset * pp.calc_rf_center(rf_ref)[0]
            )

            seq.add_block(gs1)
            seq.add_block(gs2, rf_ex)
            seq.add_block(gs3, gr3)

            for k_echo in range(n_echo):
                if k_ex > 0:
                    phase_area = phase_areas[k_echo, k_ex - 1]
                else:
                    phase_area = 0.0  # 0.0 and not 0 because -phase_area should successfully result in negative zero

                gp_pre = pp.make_trapezoid(
                    channel="y",
                    system=system,
                    area=phase_area,
                    duration=t_sp,
                    rise_time=dG,
                )
                gp_rew = pp.make_trapezoid(
                    channel="y",
                    system=system,
                    area=-phase_area,
                    duration=t_sp,
                    rise_time=dG,
                )
                seq.add_block(gs4, rf_ref)
                seq.add_block(gs5, gr5, gp_pre)
                if k_ex > 0:
                    seq.add_block(gr6, adc)
                else:
                    seq.add_block(gr6)

                seq.add_block(gs7, gr7, gp_rew)

            seq.add_block(gs4)
            seq.add_block(gs5)
            seq.add_block(delay_TR)

    (
        ok,
        error_report,
    ) = seq.check_timing()  # Check whether the timing of the sequence is correct
    if ok:
        print("Timing check passed successfully")
    else:
        print("Timing check failed. Error listing follows:")
        [print(e) for e in error_report]

    # ======
    # VISUALIZATION
    # ======
    if plot:
        seq.plot()

    # =========
    # WRITE .SEQ
    # =========
    if write_seq:
        seq.write(seq_filename)


if __name__ == "__main__":
    main(plot=True, write_seq=True)
