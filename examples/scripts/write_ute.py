"""
A very basic UTE-like sequence, without ramp-sampling, ramp-RF. Achieves TE in the range of 300-400 us
"""

from copy import copy

import numpy as np
from matplotlib import pyplot as plt

import pypulseq as pp


def main(
    plot: bool = False,
    test_report: bool = False,
    write_seq: bool = False,
    seq_filename: str = 'ute_pypulseq.seq',
    *,
    fov: float = 250e-3,
    n_x: int = 64,
    flip_angle_deg: float = 10,
    slice_thickness: float = 3e-3,
    tr: float = 10e-3,
    n_spokes: int = 32,
    readout_duration: float = 2.56e-3,
    readout_oversampling: int = 2,
    readout_asymmetry: float = 1.0,
):
    """Create a basic UTE-like sequence.

    A very basic UTE-like sequence, without ramp-sampling or ramp-RF.
    Achieves TE in the range of 300-400 us.

    Parameters
    ----------
    plot : bool, optional
        Plot the sequence diagram. Default is False.
    test_report : bool, optional
        Print a test report. Default is False.
    write_seq : bool, optional
        Write the sequence to a .seq file. Default is False.
    seq_filename : str, optional
        Output filename for the .seq file. Default is 'ute_pypulseq.seq'.
    fov : float, optional
        Field of view in meters. Default is 250e-3.
    n_x : int, optional
        Number of readout samples. Default is 64.
    flip_angle_deg : float, optional
        Flip angle in degrees. Default is 10.
    slice_thickness : float, optional
        Slice thickness in meters. Default is 3e-3.
    tr : float, optional
        Repetition time in seconds. Default is 10e-3.
    n_spokes : int, optional
        Number of radial spokes. Default is 32.
    readout_duration : float, optional
        ADC readout duration in seconds. Default is 2.56e-3.
    readout_oversampling : int, optional
        Readout oversampling factor. Default is 2.
    readout_asymmetry : float, optional
        Readout asymmetry factor. Default is 1.0.

    Returns
    -------
    seq : pypulseq.Sequence
        The UTE sequence object.
    """
    spoke_angle_increment = 2 * np.pi / n_spokes
    rf_spoiling_inc = 117

    # Set system limits
    system = pp.Opts(
        max_grad=28,
        grad_unit='mT/m',
        max_slew=100,
        slew_unit='T/m/s',
        rf_ringdown_time=20e-6,
        rf_dead_time=100e-6,
        adc_dead_time=10e-6,
    )

    seq = pp.Sequence(system)

    # Create slice selection pulse and gradient
    rf, gz, gz_reph = pp.make_sinc_pulse(
        flip_angle=np.deg2rad(flip_angle_deg),
        duration=1e-3,
        slice_thickness=slice_thickness,
        apodization=0.5,
        time_bw_product=2,
        center_pos=1,
        system=system,
        return_gz=True,
        delay=system.rf_dead_time,
        use='excitation',
    )

    # Align RO asymmetry to ADC samples
    n_x_oversampled = np.round(readout_oversampling * n_x)
    readout_asymmetry = pp.round_half_up(readout_asymmetry * n_x_oversampled / 2) / n_x_oversampled * 2

    # Define other gradients and ADC events
    delta_k = 1 / fov / (1 + readout_asymmetry)
    ro_area = n_x * delta_k
    gx = pp.make_trapezoid(channel='x', flat_area=ro_area, flat_time=readout_duration, system=system)
    adc = pp.make_adc(num_samples=n_x_oversampled, duration=gx.flat_time, delay=gx.rise_time, system=system)
    gx_pre = pp.make_trapezoid(
        channel='x',
        area=-(gx.area - ro_area) / 2 - gx.amplitude * adc.dwell / 2 - ro_area / 2 * (1 - readout_asymmetry),
        system=system,
    )

    # Gradient spoiling
    gx_spoil = pp.make_trapezoid(channel='x', area=0.2 * n_x * delta_k, system=system)

    # Calculate timing
    te = (
        gz.fall_time
        + pp.calc_duration(gx_pre, gz_reph)
        + gx.rise_time
        + adc.dwell * n_x_oversampled / 2 * (1 - readout_asymmetry)
    )
    tr_delay = tr - pp.calc_duration(gx_pre, gz_reph) - pp.calc_duration(gz) - pp.calc_duration(gx)
    tr_delay = np.ceil(tr_delay / seq.grad_raster_time) * seq.grad_raster_time
    assert np.all(tr_delay >= pp.calc_duration(gx_spoil))

    print(f'TE = {te * 1e6:.0f} us')

    if pp.calc_duration(gz_reph) > pp.calc_duration(gx_pre):
        gx_pre.delay = pp.calc_duration(gz_reph) - pp.calc_duration(gx_pre)

    rf_phase = 0
    rf_inc = 0

    for i_spoke in range(n_spokes):
        for _c in range(2):
            rf.phase_offset = rf_phase / 180 * np.pi
            adc.phase_offset = rf_phase / 180 * np.pi
            rf_inc = np.mod(rf_inc + rf_spoiling_inc, 360.0)
            rf_phase = np.mod(rf_phase + rf_inc, 360.0)

            gz.amplitude = -gz.amplitude  # Alternate GZ amplitude
            gz_reph.amplitude = -gz_reph.amplitude

            seq.add_block(rf, gz)
            phi = spoke_angle_increment * i_spoke

            gpc = copy(gx_pre)
            gps = copy(gx_pre)
            gpc.amplitude = gx_pre.amplitude * np.cos(phi)
            gps.amplitude = gx_pre.amplitude * np.sin(phi)
            gps.channel = 'y'

            grc = copy(gx)
            grs = copy(gx)
            grc.amplitude = gx.amplitude * np.cos(phi)
            grs.amplitude = gx.amplitude * np.sin(phi)
            grs.channel = 'y'

            gsc = copy(gx_spoil)
            gss = copy(gx_spoil)
            gsc.amplitude = gx_spoil.amplitude * np.cos(phi)
            gss.amplitude = gx_spoil.amplitude * np.sin(phi)
            gss.channel = 'y'

            seq.add_block(gpc, gps, gz_reph)
            seq.add_block(grc, grs, adc)
            seq.add_block(gsc, gss, pp.make_delay(tr_delay))

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

        # Plot gradients to check for gaps and optimality of the timing
        gw = seq.waveforms_and_times()[0]
        plt.figure()
        plt.plot(gw[0][0], gw[0][1], gw[1][0], gw[1][1], gw[2][0], gw[2][1])
        plt.show()

    seq.set_definition(key='FOV', value=[fov, fov, slice_thickness])
    seq.set_definition(key='Name', value='ute')

    if write_seq:
        seq.write(seq_filename)

    return seq


if __name__ == '__main__':
    main(plot=True, write_seq=True)
