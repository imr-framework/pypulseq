import numpy as np

import pypulseq as pp


def main(
    plot: bool = False,
    test_report: bool = False,
    write_seq: bool = False,
    seq_filename: str = 'gre_radial_pypulseq.seq',
    *,
    fov: float = 260e-3,
    n_x: int = 64,
    flip_angle_deg: float = 10,
    slice_thickness: float = 3e-3,
    te: float = 8e-3,
    tr: float = 20e-3,
    n_spokes: int = 60,
    n_dummy: int = 20,
):
    """Create a radial gradient echo (GRE) sequence.

    Parameters
    ----------
    plot : bool, optional
        Plot the sequence diagram. Default is False.
    test_report : bool, optional
        Print a test report. Default is False.
    write_seq : bool, optional
        Write the sequence to a .seq file. Default is False.
    seq_filename : str, optional
        Output filename for the .seq file. Default is 'gre_radial_pypulseq.seq'.
    fov : float, optional
        Field of view in meters. Default is 260e-3.
    n_x : int, optional
        Number of readout samples. Default is 64.
    flip_angle_deg : float, optional
        Flip angle in degrees. Default is 10.
    slice_thickness : float, optional
        Slice thickness in meters. Default is 3e-3.
    te : float, optional
        Echo time in seconds. Default is 8e-3.
    tr : float, optional
        Repetition time in seconds. Default is 20e-3.
    n_spokes : int, optional
        Number of radial spokes. Default is 60.
    n_dummy : int, optional
        Number of dummy scans. Default is 20.

    Returns
    -------
    seq : pypulseq.Sequence
        The radial GRE sequence object.
    """
    spoke_angle_increment = np.pi / n_spokes
    rf_spoiling_inc = 117

    # Set system limits
    system = pp.Opts(
        max_grad=28,
        grad_unit='mT/m',
        max_slew=120,
        slew_unit='T/m/s',
        rf_ringdown_time=20e-6,
        rf_dead_time=100e-6,
        adc_dead_time=10e-6,
    )

    seq = pp.Sequence(system)

    # Create slice selection pulse and gradient
    rf, gz, _ = pp.make_sinc_pulse(
        apodization=0.5,
        duration=4e-3,
        flip_angle=np.deg2rad(flip_angle_deg),
        slice_thickness=slice_thickness,
        system=system,
        time_bw_product=4,
        return_gz=True,
        delay=system.rf_dead_time,
        use='excitation',
    )

    # Define other gradients and ADC events
    delta_k = 1 / fov
    gx = pp.make_trapezoid(channel='x', flat_area=n_x * delta_k, flat_time=6.4e-3 / 5, system=system)
    adc = pp.make_adc(num_samples=n_x, duration=gx.flat_time, delay=gx.rise_time, system=system)
    gx_pre = pp.make_trapezoid(channel='x', area=-gx.area / 2 - delta_k / 2, duration=2e-3, system=system)
    gz_reph = pp.make_trapezoid(channel='z', area=-gz.area / 2, duration=2e-3, system=system)

    # Gradient spoiling
    gx_spoil = pp.make_trapezoid(channel='x', area=0.5 * n_x * delta_k, system=system)
    gz_spoil = pp.make_trapezoid(channel='z', area=4 / slice_thickness, system=system)

    # Calculate timing
    te_delay = te - pp.calc_duration(gx_pre) - gz.fall_time - gz.flat_time / 2 - pp.calc_duration(gx) / 2
    te_delay = np.ceil(te_delay / seq.grad_raster_time) * seq.grad_raster_time

    tr_delay = tr - pp.calc_duration(gx_pre) - pp.calc_duration(gz) - pp.calc_duration(gx) - te_delay
    tr_delay = np.ceil(tr_delay / seq.grad_raster_time) * seq.grad_raster_time
    assert np.all(tr_delay) > pp.calc_duration(gx_spoil, gz_spoil)

    rf_phase = 0
    rf_inc = 0

    for i_spoke in range(-n_dummy, n_spokes + 1):
        rf.phase_offset = rf_phase / 180 * np.pi
        adc.phase_offset = rf_phase / 180 * np.pi

        rf_inc = divmod(rf_inc + rf_spoiling_inc, 360.0)[1]
        rf_phase = divmod(rf_inc + rf_phase, 360.0)[1]

        seq.add_block(rf, gz)
        phi = spoke_angle_increment * (i_spoke - 1)
        seq.add_block(*pp.rotate(gx_pre, gz_reph, angle=phi, axis='z'))
        seq.add_block(pp.make_delay(te_delay))
        if i_spoke > 0:
            seq.add_block(*pp.rotate(gx, adc, angle=phi, axis='z'))
        else:
            seq.add_block(*pp.rotate(gx, angle=phi, axis='z'))
        seq.add_block(*pp.rotate(gx_spoil, gz_spoil, pp.make_delay(tr_delay), angle=phi, axis='z'))

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

    seq.set_definition(key='FOV', value=[fov, fov, slice_thickness])
    seq.set_definition(key='Name', value='gre_rad')

    if write_seq:
        seq.write(seq_filename)

    return seq


if __name__ == '__main__':
    main(plot=True, write_seq=True)
