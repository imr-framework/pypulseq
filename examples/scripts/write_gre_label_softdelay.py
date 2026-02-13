"""
Advanced GRE sequence with soft delays for TE/TR optimization.

This example demonstrates a complete gradient echo (GRE) imaging sequence with
soft delays for dynamic TE and TR adjustment. This is a complex example showing
advanced soft delay usage with mathematical relationships between delays.

For a simpler introduction to soft delays, see: soft_delay_simple_example.py

Soft Delay Strategy in this example:
-----------------------------------
1. TE Delays: Two soft delays with the same 'TE' hint but different factors:
   - First TE delay: Simple positive relationship (factor=1.0, offset=-te_min)
   - Second TE delay: Inverse relationship (factor=-1.0, offset=te_max)
   - This allows TE adjustment while maintaining constant TR

2. TR Delay: Compensates for TE changes to maintain overall TR
   - Uses offset=-tr_min to ensure minimum TR constraints

Mathematical Relationships:
--------------------------
The delays are designed so that:
- Increasing TE extends the first delay and shortens the second delay
- TR delay compensates to maintain constant total TR
- All timing constraints (te_min, te_max, tr_min) are respected

This demonstrates advanced soft delay usage for complex timing optimization
where multiple delays interact to maintain sequence constraints.
"""

import numpy as np

import pypulseq as pp


def main(
    plot: bool = False,
    test_report: bool = False,
    write_seq: bool = False,
    seq_filename: str = 'gre_label_softdelay.seq',
    *,
    fov: float = 224e-3,
    n_x: int = 64,
    n_y: int | None = None,
    flip_angle_deg: float = 7.0,
    slice_thickness: float = 3e-3,
    n_slices: int = 1,
    tr: float = 20e-3,
    te_max: float = 8e-3,
    readout_duration: float = 3.2e-3,
):
    """Create a GRE sequence with labels and soft delays for TE/TR optimization.

    This example demonstrates a gradient echo (GRE) imaging sequence with soft
    delays for dynamic TE and TR adjustment. Two TE soft delays with opposite
    factors allow shifting the echo time while maintaining constant TR.

    Parameters
    ----------
    plot : bool, optional
        Plot the sequence diagram. Default is False.
    test_report : bool, optional
        Print a test report. Default is False.
    write_seq : bool, optional
        Write the sequence to a .seq file. Default is False.
    seq_filename : str, optional
        Output filename for the .seq file. Default is 'gre_label_softdelay.seq'.
    fov : float, optional
        Field of view in meters. Default is 224e-3.
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
    tr : float, optional
        Repetition time in seconds. Default is 20e-3.
    te_max : float, optional
        Maximum echo time in seconds. Default is 8e-3.
    readout_duration : float, optional
        ADC readout duration in seconds. Default is 3.2e-3.

    Returns
    -------
    seq : pypulseq.Sequence
        The GRE sequence object.
    """
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

    # Create a new sequence object
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
    delta_k = 1 / fov
    gx = pp.make_trapezoid(channel='x', flat_area=n_x * delta_k, flat_time=readout_duration, system=system)
    adc = pp.make_adc(num_samples=n_x, duration=gx.flat_time, delay=gx.rise_time, system=system)
    gx_pre = pp.make_trapezoid(channel='x', area=-gx.area / 2, duration=1e-3, system=system)
    gz_reph = pp.make_trapezoid(channel='z', area=-gz.area / 2, duration=1e-3, system=system)
    phase_areas = -(np.arange(n_y) - n_y / 2) * delta_k

    # Gradient spoiling
    gx_spoil = pp.make_trapezoid(channel='x', area=2 * n_x * delta_k, system=system)
    gz_spoil = pp.make_trapezoid(channel='z', area=4 / slice_thickness, system=system)

    # Calculate timing
    te_min = pp.calc_duration(gx_pre) + gz.fall_time + gz.flat_time / 2 + pp.calc_duration(gx) / 2
    te_min = np.ceil(te_min / seq.grad_raster_time) * seq.grad_raster_time

    tr_min = gz.fall_time + gz.flat_time / 2 + te_max + pp.calc_duration(gx) / 2 + pp.calc_duration(gx_spoil, gz_spoil)
    tr_min = np.ceil(tr_min / seq.grad_raster_time) * seq.grad_raster_time

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

            # FIRST TE SOFT DELAY: Extends with increasing TE
            # Formula: duration = (TE_input / factor) + offset = (TE_input / 1.0) + (-te_min)
            # This creates a delay that increases linearly with TE input
            # Default duration: 10 μs (minimum block duration)
            # When TE=te_min: duration = te_min + (-te_min) = 0 (but clamped to 10 μs minimum)
            # When TE>te_min: duration increases proportionally
            seq.add_block(pp.make_soft_delay(hint='TE', offset=-te_min, factor=1.0))
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

            seq.add_block(gx_spoil, gy_pre, gz_spoil, *labels)

            # SECOND TE SOFT DELAY: Shrinks with increasing TE (maintains constant total TE)
            # Formula: duration = (TE_input / factor) + offset = (TE_input / -1.0) + te_max
            # This creates an inverse relationship: as TE increases, this delay decreases
            # Default duration: te_max - te_min - 10μs
            # When TE=te_min: duration = (-te_min) + te_max = te_max - te_min
            # When TE=te_max: duration = (-te_max) + te_max = 0 (clamped to 10μs minimum)
            # Combined with first TE delay: total TE delay remains constant, only distribution changes
            seq.add_block(
                pp.make_soft_delay(hint='TE', offset=te_max, factor=-1.0, default_duration=te_max - te_min - 10e-6)
            )

            # TR SOFT DELAY: Maintains constant TR regardless of TE changes
            # Formula: duration = (TR_input / factor) + offset = (TR_input / 1.0) + (-tr_min)
            # Since TE delays have constant total duration, TR delay only needs to handle TR changes
            # Default duration: tr - tr_min - (te_max - te_min)
            # This ensures total sequence duration = tr regardless of TE/TR settings
            seq.add_block(
                pp.make_soft_delay(
                    hint='TR', offset=-tr_min, factor=1.0, default_duration=tr - tr_min - te_max + te_min
                )
            )

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

    seq.set_definition(key='FOV', value=[fov, fov, slice_thickness * n_slices])
    seq.set_definition(key='Name', value='gre_label_softdelay')

    if write_seq:
        seq.write(seq_filename)

    return seq


if __name__ == '__main__':
    main(plot=True, write_seq=True)
