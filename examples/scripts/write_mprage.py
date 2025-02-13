from types import SimpleNamespace

import numpy as np

import pypulseq as pp


def main(plot: bool = False, write_seq: bool = False, seq_filename: str = 'mprage_pypulseq.seq'):
    # ======
    # SETUP
    # ======

    # Set system limits
    system = pp.Opts(
        max_grad=24,
        grad_unit='mT/m',
        max_slew=100,
        slew_unit='T/m/s',
        rf_ringdown_time=20e-6,
        rf_dead_time=100e-6,
        adc_dead_time=10e-6,
    )

    seq = pp.Sequence(system)  # Create a new sequence object

    alpha = 7  # Flip angle
    ro_dur = 5017.6e-6
    ro_os = 1  # Readout oversampling
    ro_spoil = 3  # Additional k-max excursion for RO spoiling
    TI = 1.1
    TR_out = 2.5

    rf_spoiling_inc = 117
    rf_len = 100e-6
    ax = SimpleNamespace()  # Encoding axes

    fov = np.array([192, 240, 256]) * 1e-3  # Define FOV and resolution
    N = [48, 60, 64]
    ax.d1 = 'z'  # Fastest dimension (readout)
    ax.d2 = 'x'  # Second-fastest dimension (inner phase-encoding loop)
    xyz = ['x', 'y', 'z']
    ax.d3 = np.setdiff1d(xyz, [ax.d1, ax.d2])[0]
    ax.n1 = xyz.index(ax.d1)
    ax.n2 = xyz.index(ax.d2)
    ax.n3 = xyz.index(ax.d3)

    # Create alpha-degree hard pulse and gradient
    rf = pp.make_block_pulse(flip_angle=alpha * np.pi / 180, system=system, duration=rf_len, delay=system.rf_dead_time)
    rf180 = pp.make_adiabatic_pulse(
        pulse_type='hypsec', system=system, duration=10.24e-3, dwell=1e-5, delay=system.rf_dead_time
    )

    # Define other gradients and ADC events
    deltak = 1 / fov
    gro = pp.make_trapezoid(
        channel=ax.d1,
        amplitude=N[ax.n1] * deltak[ax.n1] / ro_dur,
        flat_time=np.ceil((ro_dur + system.adc_dead_time) / system.grad_raster_time) * system.grad_raster_time,
        system=system,
    )
    adc = pp.make_adc(
        num_samples=N[ax.n1] * ro_os,
        duration=ro_dur,
        delay=gro.rise_time,
        system=system,
    )
    #  First 0.5 is necessary to account for the Siemens sampling in the center of the dwell periods
    gro_pre = pp.make_trapezoid(
        channel=ax.d1,
        area=-gro.amplitude * (adc.dwell * (adc.num_samples / 2 + 0.5) + 0.5 * gro.rise_time),
        system=system,
    )
    gpe1 = pp.make_trapezoid(channel=ax.d2, area=-deltak[ax.n2] * (N[ax.n2] / 2), system=system)  # Maximum PE1 gradient
    gpe2 = pp.make_trapezoid(channel=ax.d3, area=-deltak[ax.n3] * (N[ax.n3] / 2), system=system)  # Maximum PE2 gradient
    # Spoil with 4x cycles per voxel
    gsl_sp = pp.make_trapezoid(channel=ax.d3, area=np.max(deltak * N) * 4, duration=10e-3, system=system)

    # We cut the RO gradient into two parts for the optimal spoiler timing
    gro1, gro_Sp = pp.split_gradient_at(grad=gro, time_point=gro.rise_time + gro.flat_time)
    # Gradient spoiling
    if ro_spoil > 0:
        gro_Sp = pp.make_extended_trapezoid_area(
            channel=gro.channel,
            grad_start=gro.amplitude,
            grad_end=0,
            area=deltak[ax.n1] / 2 * N[ax.n1] * ro_spoil,
            system=system,
        )[0]

    # Calculate timing of the fast loop. We will have two blocks in the inner loop:
    # 1: spoilers/rewinders + RF
    # 2: prewinder, phase enconding + readout
    rf.delay = pp.calc_duration(gro_Sp, gpe1, gpe2)
    gro_pre, _, _ = pp.align(right=[gro_pre, gpe1, gpe2])
    gro1.delay = pp.calc_duration(gro_pre)
    adc.delay = gro1.delay + gro.rise_time
    gro1 = pp.add_gradients(grads=[gro1, gro_pre], system=system)
    TR_inner = pp.calc_duration(rf) + pp.calc_duration(gro1)  # For TI delay
    # pe_steps -- control reordering
    pe1_steps = ((np.arange(N[ax.n2])) - N[ax.n2] / 2) / N[ax.n2] * 2
    pe2_steps = ((np.arange(N[ax.n3])) - N[ax.n3] / 2) / N[ax.n3] * 2
    # TI calc
    TI_delay = (
        np.round(
            (
                TI
                - (np.where(pe1_steps == 0)[0][0]) * TR_inner
                - (pp.calc_duration(rf180) - pp.calc_rf_center(rf180)[0] - rf180.delay)
                - rf.delay
                - pp.calc_rf_center(rf)[0]
            )
            / system.block_duration_raster
        )
        * system.block_duration_raster
    )
    TR_out_delay = TR_out - TR_inner * N[ax.n2] - TI_delay - pp.calc_duration(rf180)

    # All LABELS / counters an flags are automatically initialized to 0 in the beginning, no need to define initial 0's
    # so we will just increment LIN after the ADC event (e.g. during the spoiler)
    label_inc_lin = pp.make_label(type='INC', label='LIN', value=1)
    label_inc_par = pp.make_label(type='INC', label='PAR', value=1)
    label_reset_par = pp.make_label(type='SET', label='PAR', value=0)

    # NOTE: The follow registration calls are commented out because they make
    #       one of the sequence unit tests fail.

    # # Pre-register objects that do not change while looping
    # result = seq.register_grad_event(gsl_sp)
    # gsl_sp.id = result if isinstance(result, int) else result[0]
    # result = seq.register_grad_event(gro_Sp)
    # gro_Sp.id = result if isinstance(result, int) else result[0]
    # result = seq.register_grad_event(gro1)
    # gro1.id = result if isinstance(result, int) else result[0]
    # # Phase of the RF object will change, therefore we only pre-register the shapes
    # _, rf.shape_IDs = seq.register_rf_event(rf)
    # rf180.id, rf180.shape_IDs = seq.register_rf_event(rf180)
    # label_inc_par.id = seq.register_label_event(label_inc_par)

    # Sequence
    for j in range(N[ax.n3]):
        seq.add_block(rf180)
        seq.add_block(pp.make_delay(TI_delay), gsl_sp)
        rf_phase = 0
        rf_inc = 0
        # Pre-register PE events that repeat in the inner loop
        gpe2je = pp.scale_grad(grad=gpe2, scale=pe2_steps[j])
        gpe2je.id = seq.register_grad_event(gpe2je)
        gpe2jr = pp.scale_grad(grad=gpe2, scale=-pe2_steps[j])
        gpe2jr.id = seq.register_grad_event(gpe2jr)

        for i in range(N[ax.n2]):
            rf.phase_offset = rf_phase / 180 * np.pi
            adc.phase_offset = rf_phase / 180 * np.pi
            rf_inc = np.mod(rf_inc + rf_spoiling_inc, 360.0)
            rf_phase = np.mod(rf_phase + rf_inc, 360.0)

            if i == 0:
                seq.add_block(rf)
            else:
                seq.add_block(
                    rf,
                    gro_Sp,
                    pp.scale_grad(grad=gpe1, scale=-pe1_steps[i - 1]),
                    gpe2jr,
                    label_inc_par,
                )
            seq.add_block(adc, gro1, pp.scale_grad(grad=gpe1, scale=pe1_steps[i]), gpe2je)
        seq.add_block(gro_Sp, pp.make_delay(TR_out_delay), label_reset_par, label_inc_lin)

    # ======
    # VISUALIZATION
    # ======
    if plot:
        seq.plot(time_range=[0, TR_out * 2], label='PAR')

    # =========
    # WRITE .SEQ
    # =========
    if write_seq:
        seq.write(seq_filename)

    return seq


if __name__ == '__main__':
    main(plot=True, write_seq=True)
