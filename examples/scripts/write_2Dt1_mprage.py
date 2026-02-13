import numpy as np

import pypulseq as pp


def main(
    plot: bool = False,
    test_report: bool = False,
    write_seq: bool = False,
    seq_filename: str = '2d_mprage_pypulseq.seq',
    *,
    n_x: int = 128,
    n_y: int = 128,
    n_slices: int = 3,
    fov: float = 220e-3,
    slice_thickness: float = 5e-3,
    slice_gap: float = 15e-3,
    te: float = 13e-3,
    ti: float = 140e-3,
    tr: float = 65e-3,
):
    """Create a 2D T1-weighted MPRAGE sequence.

    Parameters
    ----------
    plot : bool, optional
        Plot the sequence diagram. Default is False.
    test_report : bool, optional
        Print a test report. Default is False.
    write_seq : bool, optional
        Write the sequence to a .seq file. Default is False.
    seq_filename : str, optional
        Output filename for the .seq file. Default is '2d_mprage_pypulseq.seq'.
    n_x : int, optional
        Number of readout samples. Default is 128.
    n_y : int, optional
        Number of phase encoding steps. Default is 128.
    n_slices : int, optional
        Number of slices. Default is 3.
    fov : float, optional
        Field of view in meters. Default is 220e-3.
    slice_thickness : float, optional
        Slice thickness in meters. Default is 5e-3.
    slice_gap : float, optional
        Slice gap in meters. Default is 15e-3.
    te : float, optional
        Echo time in seconds. Default is 13e-3.
    ti : float, optional
        Inversion time in seconds. Default is 140e-3.
    tr : float, optional
        Repetition time in seconds. Default is 65e-3.

    Returns
    -------
    seq : pypulseq.Sequence
        The 2D MPRAGE sequence object.
    """
    # Set system limits
    system = pp.Opts(
        max_grad=32,
        grad_unit='mT/m',
        max_slew=130,
        slew_unit='T/m/s',
        grad_raster_time=10e-6,
        rf_ringdown_time=10e-6,
        rf_dead_time=100e-6,
    )
    seq = pp.Sequence(system)

    delta_z = n_slices * slice_gap
    rf_offset = 0
    z = np.linspace((-delta_z / 2), (delta_z / 2), n_slices) + rf_offset

    # Create excitation and preparation pulses
    rf, gz, _ = pp.make_sinc_pulse(
        flip_angle=np.deg2rad(12),
        system=system,
        duration=2e-3,
        slice_thickness=slice_thickness,
        apodization=0.5,
        time_bw_product=4,
        return_gz=True,
        use='excitation',
    )

    rf_prep = pp.make_block_pulse(
        flip_angle=np.deg2rad(90),
        system=system,
        duration=500e-6,
        time_bw_product=4,
        use='preparation',
    )

    # Readout
    delta_k = 1 / fov
    k_width = n_x * delta_k
    readout_time = 6.4e-3
    gx = pp.make_trapezoid(channel='x', system=system, flat_area=k_width, flat_time=readout_time)
    adc = pp.make_adc(num_samples=n_x, duration=gx.flat_time, delay=gx.rise_time)

    # Prephase and rephase
    phase_areas = (np.arange(n_y) - (n_y / 2)) * delta_k
    gy_pre = pp.make_trapezoid(channel='y', system=system, area=phase_areas[-1], duration=2e-3)
    gx_pre = pp.make_trapezoid(channel='x', system=system, area=-gx.area / 2, duration=2e-3)
    gz_reph = pp.make_trapezoid(channel='z', system=system, area=-gz.area / 2, duration=2e-3)

    # Spoilers
    pre_time = 8e-4
    gx_spoil = pp.make_trapezoid(channel='x', system=system, area=gz.area * 4, duration=pre_time * 4)
    gy_spoil = pp.make_trapezoid(channel='y', system=system, area=gz.area * 4, duration=pre_time * 4)
    gz_spoil = pp.make_trapezoid(channel='z', system=system, area=gz.area * 4, duration=pre_time * 4)

    # Calculate delays
    # Align delays to the block-duration raster to avoid timing assertion errors
    bd_raster = system.block_duration_raster

    te_delay_val = te - pp.calc_duration(rf) / 2 - pp.calc_duration(gy_pre) - pp.calc_duration(gx) / 2
    te_delay_val = np.round(te_delay_val / bd_raster) * bd_raster
    te_delay = pp.make_delay(te_delay_val)

    ti_delay_val = ti - pp.calc_duration(rf_prep) / 2 - pp.calc_duration(gx_spoil)
    ti_delay_val = np.round(ti_delay_val / bd_raster) * bd_raster
    ti_delay = pp.make_delay(ti_delay_val)

    tr_delay_val = tr - pp.calc_duration(rf) / 2 - pp.calc_duration(gx) / 2 - pp.calc_duration(gy_pre) - te
    tr_delay_val = np.round(tr_delay_val / bd_raster) * bd_raster
    tr_delay = pp.make_delay(tr_delay_val)

    for i_slice in range(n_slices):
        freq_offset = gz.amplitude * z[i_slice]
        rf.freq_offset = freq_offset

        for i_phase in range(n_y):
            seq.add_block(rf_prep)
            seq.add_block(gx_spoil, gy_spoil, gz_spoil)
            seq.add_block(ti_delay)
            seq.add_block(rf, gz)
            gy_pre = pp.make_trapezoid(channel='y', system=system, area=phase_areas[i_phase], duration=2e-3)
            seq.add_block(gx_pre, gy_pre, gz_reph)
            seq.add_block(te_delay)
            seq.add_block(gx, adc)
            gy_pre = pp.make_trapezoid(channel='y', system=system, area=-phase_areas[i_phase], duration=2e-3)
            seq.add_block(gx_spoil, gy_pre)
            seq.add_block(tr_delay)

    if test_report:
        print(seq.test_report())

    if plot:
        seq.plot()

    seq.set_definition(key='FOV', value=[fov, fov, slice_thickness * n_slices])
    seq.set_definition(key='Name', value='2D T1 MPRAGE')

    if write_seq:
        seq.write(seq_filename)

    return seq


if __name__ == '__main__':
    main(plot=True, write_seq=True)
