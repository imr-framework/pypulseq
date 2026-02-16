"""
Demo low-performance EPI sequence without ramp-sampling.
In addition, it demonstrates how the LABEL extension can be used to set data header values, which can be used either in
combination with integrated image reconstruction or to guide the off-line reconstruction tools.
"""

import numpy as np

import pypulseq as pp


def main(
    plot: bool = False,
    test_report: bool = False,
    write_seq: bool = False,
    seq_filename: str = 'epi_label_pypulseq.seq',
    *,
    fov: float | tuple[float, float] = 220e-3,
    n_x: int = 64,
    n_y: int = 64,
    slice_thickness: float = 3e-3,
    n_slices: int = 7,
    n_reps: int = 4,
    n_navigator: int = 3,
):
    """Create an EPI sequence with labels for data header control.

    Parameters
    ----------
    plot : bool, optional
        Plot the sequence diagram. Default is False.
    test_report : bool, optional
        Print a test report. Default is False.
    write_seq : bool, optional
        Write the sequence to a .seq file. Default is False.
    seq_filename : str, optional
        Output filename for the .seq file. Default is 'epi_label_pypulseq.seq'.
    fov : float or tuple of float, optional
        Field of view in meters. If a single value, it is used for both x and y.
        If a tuple, it is (fov_x, fov_y). Default is 220e-3.
    n_x : int, optional
        Number of readout samples. Default is 64.
    n_y : int, optional
        Number of phase encoding steps. Default is 64.
    slice_thickness : float, optional
        Slice thickness in meters. Default is 3e-3.
    n_slices : int, optional
        Number of slices. Default is 7.
    n_reps : int, optional
        Number of repetitions. Default is 4.
    n_navigator : int, optional
        Number of navigator lines. Default is 3.

    Returns
    -------
    seq : pypulseq.Sequence
        The EPI sequence object.
    """
    fov_x, fov_y = (fov, fov) if isinstance(fov, (int, float)) else fov

    # Set system limits
    system = pp.Opts(
        max_grad=32,
        grad_unit='mT/m',
        max_slew=130,
        slew_unit='T/m/s',
        rf_ringdown_time=30e-6,
        rf_dead_time=100e-6,
    )

    seq = pp.Sequence(system)

    # Create 90 degree slice selection pulse and gradient
    rf, gz, _ = pp.make_sinc_pulse(
        flip_angle=np.pi / 2,
        system=system,
        duration=3e-3,
        slice_thickness=slice_thickness,
        apodization=0.5,
        time_bw_product=4,
        return_gz=True,
        delay=system.rf_dead_time,
        use='excitation',
    )

    # Define trigger
    trig = pp.make_trigger(channel='physio1', duration=2000e-6)

    # Define other gradients and ADC events
    delta_kx = 1 / fov_x
    delta_ky = 1 / fov_y
    k_width = n_x * delta_kx
    adc_dwell = 4e-6
    adc_duration = n_x * adc_dwell
    gx_flat_time = adc_duration
    gx_flat_time = np.ceil(gx_flat_time * 1e5) * 1e-5  # Round-up to the gradient raster
    gx = pp.make_trapezoid(
        channel='x',
        system=system,
        amplitude=k_width / adc_duration,
        flat_time=gx_flat_time,
    )
    adc = pp.make_adc(
        num_samples=n_x,
        duration=adc_duration,
        delay=gx.rise_time + gx_flat_time / 2 - (adc_duration - adc_dwell) / 2,
    )

    # Pre-phasing gradients
    pre_time = 8e-4
    gx_pre = pp.make_trapezoid(channel='x', system=system, area=-gx.area / 2, duration=pre_time)
    gz_reph = pp.make_trapezoid(channel='z', system=system, area=-gz.area / 2, duration=pre_time)
    gy_pre = pp.make_trapezoid(channel='y', system=system, area=n_y / 2 * delta_ky, duration=pre_time)

    # Phase blip in the shortest possible time
    gy_blip_duration = 2 * np.sqrt(delta_ky / system.max_slew)
    gy_blip_duration = np.ceil(gy_blip_duration / 10e-6) * 10e-6
    gy = pp.make_trapezoid(channel='y', system=system, area=-delta_ky, duration=gy_blip_duration)

    gz_spoil = pp.make_trapezoid(channel='z', system=system, area=delta_kx * n_x * 4)

    # Loop over repetitions and slices
    for _i_rep in range(n_reps):
        seq.add_block(trig, pp.make_label(type='SET', label='SLC', value=0))
        for i_slice in range(n_slices):
            rf.freq_offset = gz.amplitude * slice_thickness * (i_slice - (n_slices - 1) / 2)
            # Compensate for the slice-offset induced phase
            rf.phase_offset = -rf.freq_offset * pp.calc_rf_center(rf)[0]
            seq.add_block(rf, gz)
            seq.add_block(
                gx_pre,
                gz_reph,
                pp.make_label(type='SET', label='NAV', value=1),
                pp.make_label(type='SET', label='LIN', value=np.round(n_y / 2)),
            )
            for i_nav in range(n_navigator):
                seq.add_block(
                    gx,
                    adc,
                    pp.make_label(type='SET', label='REV', value=gx.amplitude < 0),
                    pp.make_label(type='SET', label='SEG', value=gx.amplitude < 0),
                    pp.make_label(type='SET', label='AVG', value=i_nav + 1 == 3),
                )
                if i_nav + 1 != n_navigator:
                    # Dummy blip pulse to maintain identical RO gradient timing and the corresponding eddy currents
                    seq.add_block(pp.make_delay(pp.calc_duration(gy)))

                gx.amplitude = -gx.amplitude  # Reverse polarity of read gradient

            # Reset lin/nav/avg
            seq.add_block(
                gy_pre,
                pp.make_label(type='SET', label='LIN', value=0),
                pp.make_label(type='SET', label='NAV', value=0),
                pp.make_label(type='SET', label='AVG', value=0),
            )

            for _ in range(n_y):
                seq.add_block(
                    pp.make_label(type='SET', label='REV', value=gx.amplitude < 0),
                    pp.make_label(type='SET', label='SEG', value=gx.amplitude < 0),
                )
                seq.add_block(gx, adc)  # Read one line of k-space
                # Phase blip
                seq.add_block(gy, pp.make_label(type='INC', label='LIN', value=1))
                gx.amplitude = -gx.amplitude  # Reverse polarity of read gradient

            seq.add_block(
                gz_spoil,
                pp.make_delay(0.1),
                pp.make_label(type='INC', label='SLC', value=1),
            )
            if np.remainder(n_navigator + n_y, 2) != 0:
                gx.amplitude = -gx.amplitude

        seq.add_block(pp.make_label(type='INC', label='REP', value=1))

    ok, error_report = seq.check_timing()
    if ok:
        print('Timing check passed successfully')
    else:
        print('Timing check failed. Error listing follows:')
        [print(e) for e in error_report]

    if test_report:
        print(seq.test_report())

    if plot:
        seq.plot(time_range=(0, 0.1), time_disp='ms', label='SEG, LIN, SLC')

    seq.set_definition(key='FOV', value=[fov_x, fov_y, slice_thickness * n_slices])
    seq.set_definition(key='Name', value='epi_lbl')

    if write_seq:
        seq.write(seq_filename)

    return seq


if __name__ == '__main__':
    main(plot=True, write_seq=True)
