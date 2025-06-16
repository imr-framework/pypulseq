import math

import numpy as np

import pypulseq as pp


def main(plot: bool = False, write_seq: bool = False, seq_filename: str = 'gre_label_softdelay.seq'):
    # ======
    # SETUP
    # ======
    fov = 224e-3  # Define FOV and resolution
    Nx = 64
    Ny = Nx
    alpha = 7  # Flip angle
    slice_thickness = 3e-3  # Slice thickness
    n_slices = 1
    TR = 20e-3  # Repetition time
    max_TE = 8e-3
    rf_spoiling_inc = 117  # RF spoiling increment
    ro_duration = 3.2e-3  # ADC duration

    # Set system limits
    system = pp.Opts(
        max_grad=28,
        grad_unit='mT/m',
        max_slew=150,
        slew_unit='T/m/s',
        rf_ringdown_time=20e-6,
        rf_dead_time=100e-6,
        adc_dead_time=10e-6,
    )

    seq = pp.Sequence(system)  # Create a new sequence object

    # ======
    # CREATE EVENTS
    # ======
    # Create alpha-degree slice selection pulse and gradient
    rf, gz, _ = pp.make_sinc_pulse(
        flip_angle=alpha * np.pi / 180,
        duration=3e-3,
        slice_thickness=slice_thickness,
        apodization=0.5,
        time_bw_product=4,
        system=system,
        return_gz=True,
        delay=system.rf_dead_time,
    )

    # Define other gradients and ADC events
    delta_k = 1 / fov
    gx = pp.make_trapezoid(channel='x', flat_area=Nx * delta_k, flat_time=ro_duration, system=system)
    adc = pp.make_adc(num_samples=Nx, duration=gx.flat_time, delay=gx.rise_time, system=system)
    gx_pre = pp.make_trapezoid(channel='x', area=-gx.area / 2, duration=1e-3, system=system)
    gz_reph = pp.make_trapezoid(channel='z', area=-gz.area / 2, duration=1e-3, system=system)
    phase_areas = -(np.arange(Ny) - Ny / 2) * delta_k

    # Gradient spoiling
    gx_spoil = pp.make_trapezoid(channel='x', area=2 * Nx * delta_k, system=system)
    gz_spoil = pp.make_trapezoid(channel='z', area=4 / slice_thickness, system=system)

    # Calculate timing
    min_TE = (
        math.ceil(
            (pp.calc_duration(gx_pre) + gz.fall_time + gz.flat_time / 2 + pp.calc_duration(gx) / 2)
            / seq.grad_raster_time
        )
        * seq.grad_raster_time
    )

    min_TR = (
        math.ceil(
            (gz.fall_time + gz.flat_time / 2 + max_TE + pp.calc_duration(gx) / 2 + pp.calc_duration(gx_spoil, gz_spoil))
            / seq.grad_raster_time
        )
        * seq.grad_raster_time
    )  # + whatever the soft TE delay is.

    rf_phase = 0
    rf_inc = 0

    seq.add_block(pp.make_label(label='REV', type='SET', value=1))

    # ======
    # CONSTRUCT SEQUENCE
    # ======
    # Loop over slices
    for s in range(n_slices):
        rf.freq_offset = gz.amplitude * slice_thickness * (s - (n_slices - 1) / 2)
        # Loop over phase encodes and define sequence blocks
        for i in range(Ny):
            rf.phase_offset = rf_phase / 180 * np.pi
            adc.phase_offset = rf_phase / 180 * np.pi
            rf_inc = divmod(rf_inc + rf_spoiling_inc, 360.0)[1]
            rf_phase = divmod(rf_phase + rf_inc, 360.0)[1]

            seq.add_block(rf, gz)
            gy_pre = pp.make_trapezoid(
                channel='y',
                area=phase_areas[i],
                duration=pp.calc_duration(gx_pre),
                system=system,
            )
            seq.add_block(gx_pre, gy_pre, gz_reph)
            # Formula is duration = input / factor + offset, to know the default TE
            # we do the inverse: input = (duration - offset) * factor
            # so default TE is (10e-6 - (-min_TE)) * 1.0 = 10e-6 + min_TE
            seq.add_block(pp.make_soft_delay(hint='TE', offset=-min_TE, factor=1.0))
            seq.add_block(gx, adc)
            gy_pre.amplitude = -gy_pre.amplitude
            spoil_block_contents = [gx_spoil, gy_pre, gz_spoil]
            if i != Ny - 1:
                spoil_block_contents.append(pp.make_label(type='INC', label='LIN', value=1))
            else:
                spoil_block_contents.extend(
                    [
                        pp.make_label(type='SET', label='LIN', value=0),
                        pp.make_label(type='INC', label='SLC', value=1),
                    ]
                )
            seq.add_block(*spoil_block_contents)

            # Now, whatever we add to TE, we need to subtract from TR delay to keep the TR constant
            # Also, our min TR can be max TE + rest of the sequence.
            # Let's see if default duration is consistent with the input:
            # (max_TE - min_TE - 10e-6 - max_TE) * -1.0 = min_TE + 10e-6, checks out.
            seq.add_block(
                pp.make_soft_delay(hint='TE', offset=max_TE, factor=-1.0, default_duration=max_TE - min_TE - 10e-6)
            )
            # Finally the TR
            # (TR - min_TR - max_TE + min_TE - (-min_TR)) * 1.0 = TR + min_TE - max_TE
            # From previous line, we have max_TE - min_TE - 10e-6 duration, sum them up:
            # TR + min_TE - max_TE + max_TE - min_TE -10e-6 = TR - 10e-6, we ended up with default TR.
            seq.add_block(
                pp.make_soft_delay(
                    hint='TR', offset=-min_TR, factor=1.0, default_duration=TR - min_TR - max_TE + min_TE
                )
            )

    ok, error_report = seq.check_timing()

    if ok:
        print('Timing check passed successfully')
    else:
        print('Timing check failed. Error listing follows:')
        [print(e) for e in error_report]

    # ======
    # VISUALIZATION
    # ======
    if plot:
        seq.plot(label='lin', time_range=np.array([0, 32]) * TR, time_disp='ms')

    # =========
    # WRITE .SEQ
    # =========
    if write_seq:
        # Prepare the sequence output for the scanner
        seq.set_definition(key='FOV', value=[fov, fov, slice_thickness * n_slices])
        seq.set_definition(key='Name', value='gre_label')

        seq.write(seq_filename)

    return seq


if __name__ == '__main__':
    main(plot=True, write_seq=True)
