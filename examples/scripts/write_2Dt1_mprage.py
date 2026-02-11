from math import pi

import numpy as np

import pypulseq as pp


def main(
    plot: bool = False,
    write_seq: bool = False,
    seq_filename: str = '2d_mprage_pypulseq.seq',
    *,
    Nx: int = 128,
    Ny: int = 128,
    n_slices: int = 3,
    fov: float = 220e-3,
    slice_thickness: float = 5e-3,
    slice_gap: float = 15e-3,
    TE: float = 13e-3,
    TI: float = 140e-3,
    TR: float = 65e-3,
):
    # ======
    # SETUP
    # ======
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

    # =========
    # RF90, RF180
    # =========
    flip = 12 * pi / 180
    rf, gz, _ = pp.make_sinc_pulse(
        flip_angle=flip,
        system=system,
        duration=2e-3,
        slice_thickness=slice_thickness,
        apodization=0.5,
        time_bw_product=4,
        return_gz=True,
        use='excitation',
    )

    flip90 = 90 * pi / 180
    rf90 = pp.make_block_pulse(
        flip_angle=flip90,
        system=system,
        duration=500e-6,
        time_bw_product=4,
        use='preparation',
    )

    # =========
    # Readout
    # =========
    delta_k = 1 / fov
    k_width = Nx * delta_k
    readout_time = 6.4e-3
    gx = pp.make_trapezoid(channel='x', system=system, flat_area=k_width, flat_time=readout_time)
    adc = pp.make_adc(num_samples=Nx, duration=gx.flat_time, delay=gx.rise_time)

    # =========
    # Prephase and Rephase
    # =========
    phase_areas = (np.arange(Ny) - (Ny / 2)) * delta_k
    gy_pre = pp.make_trapezoid(channel='y', system=system, area=phase_areas[-1], duration=2e-3)

    gx_pre = pp.make_trapezoid(channel='x', system=system, area=-gx.area / 2, duration=2e-3)

    gz_reph = pp.make_trapezoid(channel='z', system=system, area=-gz.area / 2, duration=2e-3)

    # =========
    # Spoilers
    # =========
    pre_time = 8e-4
    gx_spoil = pp.make_trapezoid(channel='x', system=system, area=gz.area * 4, duration=pre_time * 4)
    gy_spoil = pp.make_trapezoid(channel='y', system=system, area=gz.area * 4, duration=pre_time * 4)
    gz_spoil = pp.make_trapezoid(channel='z', system=system, area=gz.area * 4, duration=pre_time * 4)

    # =========
    # Delays
    # =========
    # Ensure that block delays are aligned to the block-duration raster to avoid
    # assertion errors when writing the sequence (`write_seq` expects each block
    # duration to be an integer multiple of `system.block_duration_raster`).
    bd_raster = system.block_duration_raster

    delay_TE_val = TE - pp.calc_duration(rf) / 2 - pp.calc_duration(gy_pre) - pp.calc_duration(gx) / 2
    delay_TE_val = np.round(delay_TE_val / bd_raster) * bd_raster
    delay_TE = pp.make_delay(delay_TE_val)

    delay_TI_val = TI - pp.calc_duration(rf90) / 2 - pp.calc_duration(gx_spoil)
    delay_TI_val = np.round(delay_TI_val / bd_raster) * bd_raster
    delay_TI = pp.make_delay(delay_TI_val)

    delay_TR_val = TR - pp.calc_duration(rf) / 2 - pp.calc_duration(gx) / 2 - pp.calc_duration(gy_pre) - TE
    delay_TR_val = np.round(delay_TR_val / bd_raster) * bd_raster
    delay_TR = pp.make_delay(delay_TR_val)

    for j in range(n_slices):
        freq_offset = gz.amplitude * z[j]
        rf.freq_offset = freq_offset

        for i in range(Ny):
            seq.add_block(rf90)
            seq.add_block(gx_spoil, gy_spoil, gz_spoil)
            seq.add_block(delay_TI)
            seq.add_block(rf, gz)
            gy_pre = pp.make_trapezoid(channel='y', system=system, area=phase_areas[i], duration=2e-3)
            seq.add_block(gx_pre, gy_pre, gz_reph)
            seq.add_block(delay_TE)
            seq.add_block(gx, adc)
            gy_pre = pp.make_trapezoid(channel='y', system=system, area=-phase_areas[i], duration=2e-3)
            seq.add_block(gx_spoil, gy_pre)
            seq.add_block(delay_TR)

    seq.set_definition(key='FOV', value=[fov, fov, slice_thickness * n_slices])
    seq.set_definition(key='Name', value='2D T1 MPRAGE')

    if plot:
        seq.plot()

    if write_seq:
        seq.write(seq_filename)

    return seq


if __name__ == '__main__':
    main(plot=True, write_seq=True)
