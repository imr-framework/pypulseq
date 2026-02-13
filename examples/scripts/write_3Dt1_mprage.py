import numpy as np

import pypulseq as pp


def main(
    plot: bool = False,
    test_report: bool = False,
    write_seq: bool = False,
    seq_filename: str = '256_3d_t1_mprage_pypulseq.seq',
    *,
    n_x: int = 256,
    n_y: int = 256,
    n_z: int = 32,
    fov: float = 256e-3,
    fov_z: float = 256e-3,
    te: float = 4e-3,
    ti: float = 140e-3,
    tr: float = 10e-3,
    t_recovery: float = 1e-3,
):
    """Create a 3D T1-weighted MPRAGE sequence.

    Parameters
    ----------
    plot : bool, optional
        Plot the sequence diagram. Default is False.
    test_report : bool, optional
        Print a test report. Default is False.
    write_seq : bool, optional
        Write the sequence to a .seq file. Default is False.
    seq_filename : str, optional
        Output filename for the .seq file. Default is '256_3d_t1_mprage_pypulseq.seq'.
    n_x : int, optional
        Number of readout samples. Default is 256.
    n_y : int, optional
        Number of phase encoding steps. Default is 256.
    n_z : int, optional
        Number of partition encoding steps. Default is 32.
    fov : float, optional
        Field of view in meters. Default is 256e-3.
    fov_z : float, optional
        Field of view in the partition direction in meters. Default is 256e-3.
    te : float, optional
        Echo time in seconds. Default is 4e-3.
    ti : float, optional
        Inversion time in seconds. Default is 140e-3.
    tr : float, optional
        Repetition time in seconds. Default is 10e-3.
    t_recovery : float, optional
        Recovery time in seconds. Default is 1e-3.

    Returns
    -------
    seq : pypulseq.Sequence
        The 3D MPRAGE sequence object.
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

    # Create excitation and preparation pulses
    rf = pp.make_block_pulse(
        flip_angle=np.deg2rad(12),
        system=system,
        duration=250e-6,
        time_bw_product=4,
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
    readout_time = 3.5e-3
    gx = pp.make_trapezoid(channel='x', system=system, flat_area=k_width, flat_time=readout_time)
    adc = pp.make_adc(num_samples=n_x, duration=gx.flat_time, delay=gx.rise_time)

    # Prephase and rephase
    delta_kz = 1 / fov_z
    phase_areas = (np.arange(n_y) - (n_y / 2)) * delta_k
    slice_areas = (np.arange(n_z) - (n_z / 2)) * delta_kz

    gx_pre = pp.make_trapezoid(channel='x', system=system, area=-gx.area / 2, duration=2e-3)
    gy_pre = pp.make_trapezoid(channel='y', system=system, area=phase_areas[-1], duration=2e-3)

    # Spoilers
    pre_time = 6.4e-4
    gx_spoil = pp.make_trapezoid(
        channel='x',
        system=system,
        area=(4 * np.pi) / (42.576e6 * delta_k * 1e-3) * 42.576e6,
        duration=pre_time * 6,
    )
    gy_spoil = pp.make_trapezoid(
        channel='y',
        system=system,
        area=(4 * np.pi) / (42.576e6 * delta_k * 1e-3) * 42.576e6,
        duration=pre_time * 6,
    )
    gz_spoil = pp.make_trapezoid(
        channel='z',
        system=system,
        area=(4 * np.pi) / (42.576e6 * delta_kz * 1e-3) * 42.576e6,
        duration=pre_time * 6,
    )

    # Extended trapezoids: gx, gx_spoil
    t_gx_extended = np.array([0, gx.rise_time, gx.flat_time, (gx.rise_time * 2) + gx.flat_time + gx.fall_time])
    amp_gx_extended = np.array([0, gx.amplitude, gx.amplitude, gx_spoil.amplitude])
    t_gx_spoil_extended = np.array(
        [
            0,
            gx_spoil.rise_time + gx_spoil.flat_time,
            gx_spoil.rise_time + gx_spoil.flat_time + gx_spoil.fall_time,
        ]
    )
    amp_gx_spoil_extended = np.array([gx_spoil.amplitude, gx_spoil.amplitude, 0])

    gx_extended = pp.make_extended_trapezoid(channel='x', times=t_gx_extended, amplitudes=amp_gx_extended)
    gx_spoil_extended = pp.make_extended_trapezoid(
        channel='x', times=t_gx_spoil_extended, amplitudes=amp_gx_spoil_extended
    )

    # Calculate delays
    # Align delays to the block-duration raster to avoid timing assertion errors
    bd_raster = system.block_duration_raster

    te_delay = te - pp.calc_duration(rf) / 2 - pp.calc_duration(gx_pre) - pp.calc_duration(gx) / 2
    te_delay = np.round(te_delay / bd_raster) * bd_raster

    ti_delay = ti - pp.calc_duration(rf_prep) / 2 - pp.calc_duration(gx_spoil)
    ti_delay = np.round(ti_delay / bd_raster) * bd_raster

    tr_delay = tr - pp.calc_duration(rf) - pp.calc_duration(gx_pre) - pp.calc_duration(gx) - pp.calc_duration(gx_spoil)
    tr_delay = np.round(tr_delay / bd_raster) * bd_raster

    for i_phase in range(n_y):
        gy_pre = pp.make_trapezoid(channel='y', system=system, area=phase_areas[i_phase], duration=2e-3)

        seq.add_block(rf_prep)
        seq.add_block(gx_spoil, gy_spoil, gz_spoil)
        seq.add_block(pp.make_delay(ti_delay))

        for i_partition in range(n_z):
            gz_pre = pp.make_trapezoid(channel='z', system=system, area=slice_areas[i_partition], duration=2e-3)
            gz_reph = pp.make_trapezoid(channel='z', system=system, area=-slice_areas[i_partition], duration=2e-3)

            seq.add_block(rf)
            seq.add_block(gx_pre, gy_pre, gz_pre)
            # Skip TE: readout_time = 3.5e3 --> TE = -2.168404344971009e-19
            # seq.add_block(pp.make_delay(te_delay))
            seq.add_block(gx_extended, adc)
            seq.add_block(gx_spoil_extended, gz_reph)
            seq.add_block(pp.make_delay(tr_delay))

        seq.add_block(pp.make_delay(t_recovery))

    if test_report:
        print(seq.test_report())

    if plot:
        seq.plot(time_range=(0, ti + tr + 2e-3))

    seq.set_definition(key='FOV', value=[fov, fov, fov_z])
    seq.set_definition(key='Name', value='3D T1 MPRAGE')

    if write_seq:
        seq.write(seq_filename)

    return seq


if __name__ == '__main__':
    main(plot=True, write_seq=True)
