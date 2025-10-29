"""
Advanced GRE sequence with soft delays for TE/TR optimization.

This example demonstrates a complete gradient echo (GRE) imaging sequence with
soft delays for dynamic TE and TR adjustment. This is a complex example showing
advanced soft delay usage with mathematical relationships between delays.

For a simpler introduction to soft delays, see: soft_delay_simple_example.py

Soft Delay Strategy in this example:
-----------------------------------
1. TE Delays: Two soft delays with the same 'TE' hint but different factors:
   - First TE delay: Simple positive relationship (factor=1.0, offset=-min_TE)
   - Second TE delay: Inverse relationship (factor=-1.0, offset=max_TE)
   - This allows TE adjustment while maintaining constant TR

2. TR Delay: Compensates for TE changes to maintain overall TR
   - Uses offset=-min_TR to ensure minimum TR constraints

Mathematical Relationships:
--------------------------
The delays are designed so that:
- Increasing TE extends the first delay and shortens the second delay
- TR delay compensates to maintain constant total TR
- All timing constraints (min_TE, max_TE, min_TR) are respected

This demonstrates advanced soft delay usage for complex timing optimization
where multiple delays interact to maintain sequence constraints.
"""

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

            # FIRST TE SOFT DELAY: Extends with increasing TE
            # Formula: duration = (TE_input / factor) + offset = (TE_input / 1.0) + (-min_TE)
            # This creates a delay that increases linearly with TE input
            # Default duration: 10μs (minimum block duration)
            # When TE=min_TE: duration = min_TE + (-min_TE) = 0 (but clamped to 10μs minimum)
            # When TE>min_TE: duration increases proportionally
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

            # SECOND TE SOFT DELAY: Shrinks with increasing TE (maintains constant total TE)
            # Formula: duration = (TE_input / factor) + offset = (TE_input / -1.0) + max_TE
            # This creates an inverse relationship: as TE increases, this delay decreases
            # Default duration: max_TE - min_TE - 10μs
            # When TE=min_TE: duration = (-min_TE) + max_TE = max_TE - min_TE
            # When TE=max_TE: duration = (-max_TE) + max_TE = 0 (clamped to 10μs minimum)
            # Combined with first TE delay: total TE delay remains constant, only distribution changes
            seq.add_block(
                pp.make_soft_delay(hint='TE', offset=max_TE, factor=-1.0, default_duration=max_TE - min_TE - 10e-6)
            )

            # TR SOFT DELAY: Maintains constant TR regardless of TE changes
            # Formula: duration = (TR_input / factor) + offset = (TR_input / 1.0) + (-min_TR)
            # Since TE delays have constant total duration, TR delay only needs to handle TR changes
            # Default duration: TR - min_TR - (max_TE - min_TE)
            # This ensures total sequence duration = TR regardless of TE/TR settings
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
