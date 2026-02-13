import math

import numpy as np

import pypulseq as pp


def main(
    plot: bool = False,
    test_report: bool = False,
    write_seq: bool = False,
    seq_filename: str = 'epi_se_pypulseq.seq',
    *,
    fov: float = 256e-3,
    n_x: int = 64,
    n_y: int = 64,
    slice_thickness: float = 3e-3,
    te: float = 60e-3,
):
    """Create a spin-echo EPI sequence.

    Parameters
    ----------
    plot : bool, optional
        Plot the sequence diagram. Default is False.
    test_report : bool, optional
        Print a test report. Default is False.
    write_seq : bool, optional
        Write the sequence to a .seq file. Default is False.
    seq_filename : str, optional
        Output filename for the .seq file. Default is 'epi_se_pypulseq.seq'.
    fov : float, optional
        Field of view in meters. Default is 256e-3.
    n_x : int, optional
        Number of readout samples. Default is 64.
    n_y : int, optional
        Number of phase encoding steps. Default is 64.
    slice_thickness : float, optional
        Slice thickness in meters. Default is 3e-3.
    te : float, optional
        Echo time in seconds. Default is 60e-3.

    Returns
    -------
    seq : pypulseq.Sequence
        The EPI sequence object.
    """
    # Set system limits
    system = pp.Opts(
        max_grad=32,
        grad_unit='mT/m',
        max_slew=130,
        slew_unit='T/m/s',
        rf_ringdown_time=20e-6,
        rf_dead_time=100e-6,
        adc_dead_time=20e-6,
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

    # Define other gradients and ADC events
    delta_k = 1 / fov
    k_width = n_x * delta_k
    readout_time = 3.2e-4
    gx = pp.make_trapezoid(channel='x', system=system, flat_area=k_width, flat_time=readout_time)
    adc = pp.make_adc(num_samples=n_x, system=system, duration=gx.flat_time, delay=gx.rise_time)

    # Pre-phasing gradients
    pre_time = 8e-4
    gz_reph = pp.make_trapezoid(channel='z', system=system, area=-gz.area / 2, duration=pre_time)
    # Do not need minus for in-plane prephasers because of the spin-echo (position reflection in k-space)
    gx_pre = pp.make_trapezoid(channel='x', system=system, area=gx.area / 2 - delta_k / 2, duration=pre_time)
    gy_pre = pp.make_trapezoid(channel='y', system=system, area=n_y / 2 * delta_k, duration=pre_time)

    # Phase blip in shortest possible time
    gy_blip_duration = 2 * math.sqrt(delta_k / system.max_slew)
    gy_blip_duration = math.ceil(gy_blip_duration / 10e-6) * 10e-6
    gy = pp.make_trapezoid(channel='y', system=system, area=delta_k, duration=gy_blip_duration)

    # Refocusing pulse with spoiling gradients
    rf180 = pp.make_block_pulse(
        flip_angle=np.pi,
        delay=system.rf_dead_time,
        system=system,
        duration=500e-6,
        use='refocusing',
    )
    gz_spoil = pp.make_trapezoid(channel='z', system=system, area=gz.area * 2, duration=3 * pre_time)

    # Calculate delay time
    duration_to_center = (n_x / 2 + 0.5) * pp.calc_duration(gx) + n_y / 2 * pp.calc_duration(gy)
    rf_center_incl_delay = rf.delay + pp.calc_rf_center(rf)[0]
    rf180_center_incl_delay = rf180.delay + pp.calc_rf_center(rf180)[0]
    te_delay_1 = (
        te / 2
        - pp.calc_duration(gz)
        + rf_center_incl_delay
        - pre_time
        - pp.calc_duration(gz_spoil)
        - rf180_center_incl_delay
    )
    te_delay_2 = (
        te / 2 - pp.calc_duration(rf180) + rf180_center_incl_delay - pp.calc_duration(gz_spoil) - duration_to_center
    )

    # Construct sequence
    seq.add_block(rf, gz)
    seq.add_block(gx_pre, gy_pre, gz_reph)
    seq.add_block(pp.make_delay(te_delay_1))
    seq.add_block(gz_spoil)
    seq.add_block(rf180)
    seq.add_block(gz_spoil)
    seq.add_block(pp.make_delay(te_delay_2))
    for _ in range(n_y):
        seq.add_block(gx, adc)  # Read one line of k-space
        seq.add_block(gy)  # Phase blip
        gx.amplitude = -gx.amplitude  # Reverse polarity of read gradient
    seq.add_block(pp.make_delay(1e-4))

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
    seq.set_definition(key='Name', value='epi_se')

    if write_seq:
        seq.write(seq_filename)

    return seq


if __name__ == '__main__':
    main(plot=True, write_seq=True)
