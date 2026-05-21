import numpy as np

import pypulseq as pp


def main(
    plot: bool = False,
    test_report: bool = False,
    write_seq: bool = False,
    seq_filename: str = 'gre_label_pypulseq.seq',
    *,
    fov: float | tuple[float, float] = 224e-3,
    n_x: int = 64,
    n_y: int | None = None,
    flip_angle_deg: float = 7.0,
    slice_thickness: float = 3e-3,
    n_slices: int = 1,
    te: float = 4.3e-3,
    tr: float = 10e-3,
    readout_duration: float = 3.2e-3,
):
    """Create a GRE sequence with labels for data header control.

    Parameters
    ----------
    plot : bool, optional
        Plot the sequence diagram. Default is False.
    test_report : bool, optional
        Print a test report. Default is False.
    write_seq : bool, optional
        Write the sequence to a .seq file. Default is False.
    seq_filename : str, optional
        Output filename for the .seq file. Default is 'gre_label_pypulseq.seq'.
    fov : float or tuple of float, optional
        Field of view in meters. If a single value, it is used for both x and y.
        If a tuple, it is (fov_x, fov_y). Default is 224e-3.
    n_x : int, optional
        Number of readout samples. Default is 64.
    n_y : int or None, optional
        Number of phase encoding steps. Default is None (same as n_x).
    flip_angle_deg : float, optional
        Flip angle in degrees. Default is 7.0.
    slice_thickness : float, optional
        Slice thickness in meters. Default is 3e-3.
    n_slices : int, optional
        Number of slices. Default is 1.
    te : float, optional
        Echo time in seconds. Default is 4.3e-3.
    tr : float, optional
        Repetition time in seconds. Default is 10e-3.
    readout_duration : float, optional
        ADC readout duration in seconds. Default is 3.2e-3.

    Returns
    -------
    seq : pypulseq.Sequence
        The GRE sequence object.
    """
    fov_x, fov_y = (fov, fov) if isinstance(fov, (int, float)) else fov
    if n_y is None:
        n_y = n_x
    rf_spoiling_inc = 117

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

    seq = pp.Sequence(system)

    # Create slice selection pulse and gradient
    rf, gz, _ = pp.make_sinc_pulse(
        flip_angle=np.deg2rad(flip_angle_deg),
        duration=3e-3,
        slice_thickness=slice_thickness,
        apodization=0.5,
        time_bw_product=4,
        system=system,
        return_gz=True,
        delay=system.rf_dead_time,
        use='excitation',
    )

    # Define other gradients and ADC events
    delta_kx = 1 / fov_x
    delta_ky = 1 / fov_y
    gx = pp.make_trapezoid(channel='x', flat_area=n_x * delta_kx, flat_time=readout_duration, system=system)
    adc = pp.make_adc(num_samples=n_x, duration=gx.flat_time, delay=gx.rise_time, system=system)
    gx_pre = pp.make_trapezoid(channel='x', area=-gx.area / 2, duration=1e-3, system=system)
    gz_reph = pp.make_trapezoid(channel='z', area=-gz.area / 2, duration=1e-3, system=system)
    phase_areas = -(np.arange(n_y) - n_y / 2) * delta_ky

    # Gradient spoiling
    gx_spoil = pp.make_trapezoid(channel='x', area=2 * n_x * delta_kx, system=system)
    gz_spoil = pp.make_trapezoid(channel='z', area=4 / slice_thickness, system=system)

    # Calculate timing
    te_delay = te - pp.calc_duration(gx_pre) - gz.fall_time - gz.flat_time / 2 - pp.calc_duration(gx) / 2
    te_delay = np.ceil(te_delay / seq.grad_raster_time) * seq.grad_raster_time

    tr_delay = tr - pp.calc_duration(gz) - pp.calc_duration(gx_pre) - pp.calc_duration(gx) - te_delay
    tr_delay = np.ceil(tr_delay / seq.grad_raster_time) * seq.grad_raster_time
    assert np.all(te_delay >= 0)
    assert np.all(tr_delay >= pp.calc_duration(gx_spoil, gz_spoil))

    rf_phase = 0
    rf_inc = 0

    seq.add_block(pp.make_label(label='REV', type='SET', value=1))

    # Loop over slices
    for i_slice in range(n_slices):
        rf.freq_offset = gz.amplitude * slice_thickness * (i_slice - (n_slices - 1) / 2)
        # Loop over phase encodes and define sequence blocks
        for i_phase in range(n_y):
            rf.phase_offset = rf_phase / 180 * np.pi
            adc.phase_offset = rf_phase / 180 * np.pi
            rf_inc = divmod(rf_inc + rf_spoiling_inc, 360.0)[1]
            rf_phase = divmod(rf_phase + rf_inc, 360.0)[1]

            seq.add_block(rf, gz)
            gy_pre = pp.make_trapezoid(
                channel='y',
                area=phase_areas[i_phase],
                duration=pp.calc_duration(gx_pre),
                system=system,
            )
            seq.add_block(gx_pre, gy_pre, gz_reph)
            seq.add_block(pp.make_delay(te_delay))
            seq.add_block(gx, adc)
            gy_pre.amplitude = -gy_pre.amplitude

            # Create labels
            if i_phase != n_y - 1:
                labels = [pp.make_label(type='INC', label='LIN', value=1)]
            else:
                labels = [
                    pp.make_label(type='SET', label='LIN', value=0),
                    pp.make_label(type='INC', label='SLC', value=1),
                ]

            seq.add_block(pp.make_delay(tr_delay), gx_spoil, gy_pre, gz_spoil, *labels)

    ok, error_report = seq.check_timing()

    if ok:
        print('Timing check passed successfully')
    else:
        print('Timing check failed. Error listing follows:')
        [print(e) for e in error_report]

    if test_report:
        print(seq.test_report())

    if plot:
        seq.plot(label='lin', time_range=np.array([0, 32]) * tr, time_disp='ms')

    seq.set_definition(key='FOV', value=[fov_x, fov_y, slice_thickness * n_slices])
    seq.set_definition(key='Name', value='gre_label')

    if write_seq:
        seq.write(seq_filename)

    return seq


if __name__ == '__main__':
    main(plot=True, write_seq=True)
