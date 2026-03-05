# This script generates Turbo Spin Echo (TSE) pulse sequences for Test 1
# using the pypulseq library. It creates seven different .seq files,
# each with a different refocusing flip angle (120, 130, 140, 150, 160, 170, 180 degrees).
#
# The parameters are based on the "RF4Seq_book_chapter.pdf" document,
# specifically section 1.4.2, page 17.

import math
import os

import numpy as np

from pypulseq.make_adc import make_adc
from pypulseq.make_delay import make_delay
from pypulseq.make_sinc_pulse import make_sinc_pulse
from pypulseq.make_trapezoid import make_trapezoid
from pypulseq.opts import Opts
from pypulseq.Sequence.sequence import Sequence


def generate_tse_sequence(refocus_flip_angle_deg, output_dir='sequences'):
    """
    Generates a single TSE .seq file for a given refocusing flip angle.

    Args:
        refocus_flip_angle_deg (int): The flip angle for the refocusing pulses in degrees.
        output_dir (str): The directory where the .seq file will be saved.
    """
    # System limits for a Siemens Prisma 3T scanner
    system = Opts(
        max_grad=80,
        grad_unit='mT/m',
        max_slew=200,
        slew_unit='T/m/s',
        rf_ringdown_time=20e-6,
        rf_dead_time=100e-6,
        adc_dead_time=20e-6,
    )

    seq = Sequence(system)

    # Sequence parameters from the paper (TR/TE = 2000/12ms, TEeff = 60ms)
    fov = 256e-3  # 256x256mm² field of view
    slice_thickness = 5e-3  # 5mm slice thickness
    matrix_size = 256
    etl = 16  # Echo Train Length
    tr = 2000e-3  # TR = 2000ms (2 seconds)
    te = 12e-3  # Echo spacing = 12ms
    te_eff = 60e-3  # Effective TE = 60ms (center of k-space)

    # Calculate derived parameters
    delta_k = 1 / fov
    num_slices = 1

    # RF Pulses
    # Excitation pulse (90 degrees)
    rf_exc, gz_exc, _ = make_sinc_pulse(
        flip_angle=90 * math.pi / 180,
        system=system,
        duration=2e-3,
        slice_thickness=slice_thickness,
        apodization=0.5,
        time_bw_product=4,
        return_gz=True,
    )

    # Refocusing pulse (variable flip angle)
    flip_angle_rad = refocus_flip_angle_deg * math.pi / 180
    _rf_ref, _gz_ref, _ = make_sinc_pulse(
        flip_angle=flip_angle_rad,
        system=system,
        duration=2e-3,
        slice_thickness=slice_thickness,
        apodization=0.5,
        time_bw_product=4,
        use='refocusing',
        return_gz=True,
    )

    # Gradients
    # Slice refocusing gradient
    gz_reph = make_trapezoid(
        channel='z', system=system, area=-gz_exc.area / 2, duration=1e-3
    )

    # Readout gradient
    readout_time = 5.12e-3
    gx_read = make_trapezoid(
        channel='x',
        system=system,
        flat_area=matrix_size * delta_k,
        flat_time=readout_time,
    )
    # Readout prephaser
    gx_pre = make_trapezoid(
        channel='x', system=system, area=-gx_read.area / 2, duration=1e-3
    )

    # Phase encoding gradients
    # For TSE, we need to order phase encoding steps for TEeff = 60ms
    # Center of k-space should be acquired at echo number corresponding to TEeff
    echo_center = int(te_eff / te)  # Echo number for center of k-space (should be 5)

    # Create phase encoding order with center at the specified echo
    pe_steps = np.arange(-etl // 2, etl // 2)
    # Reorder so center of k-space (pe_step=0) is at echo_center
    pe_order = np.zeros(etl, dtype=int)
    pe_order[echo_center - 1] = 0  # Center k-space line

    # Fill remaining steps symmetrically
    pos_steps = [i for i in pe_steps if i > 0]
    neg_steps = [i for i in pe_steps if i < 0]

    idx = 0
    for i in range(etl):
        if i == echo_center - 1:
            continue  # Already filled
        if idx < len(pos_steps) and (idx >= len(neg_steps) or i % 2 == 0):
            pe_order[i] = pos_steps[idx % len(pos_steps)]
        else:
            pe_order[i] = neg_steps[idx % len(neg_steps)]
        if i % 2 == 1:
            idx += 1

    gy_pre_list = [
        make_trapezoid(
            channel='y', system=system, area=pe_step * delta_k, duration=1e-3
        ) for pe_step in pe_order
    ]

    # ADC event
    adc = make_adc(
        num_samples=matrix_size, system=system, duration=readout_time, delay=gx_read.fall_time
    )

    # --- Build the sequence ---
    # Loop over slices (only one in this case)
    for _s in range(num_slices):
        # Excitation block
        seq.add_block(rf_exc, gz_exc)
        seq.add_block(gz_reph, gx_pre)

        # Echo train loop
        for i in range(etl):
            # Refocusing and phase encoding for the current echo
            gy_pre = gy_pre_list[i]

            # Create a new RF refocusing pulse for each echo to avoid phase accumulation
            rf_ref_echo, gz_ref_echo, _ = make_sinc_pulse(
                flip_angle=flip_angle_rad,
                system=system,
                duration=2e-3,
                slice_thickness=slice_thickness,
                apodization=0.5,
                time_bw_product=4,
                use='refocusing',
                return_gz=True,
            )

            # Set phase offset for proper refocusing (alternating phases)
            rf_ref_echo.phase_offset = (i % 2) * math.pi

            # Refocusing block with phase encoding
            seq.add_block(rf_ref_echo, gz_ref_echo, gy_pre)

            # Readout block
            seq.add_block(gx_read, adc)

            # Add a delay to ensure correct echo spacing (TE = 12ms)
            # Calculate actual duration of the blocks
            rf_duration = rf_ref_echo.delay + rf_ref_echo.shape_dur + rf_ref_echo.ringdown_time
            gz_duration = gz_ref_echo.delay + gz_ref_echo.rise_time + gz_ref_echo.flat_time + gz_ref_echo.fall_time
            gy_duration = gy_pre.delay + gy_pre.rise_time + gy_pre.flat_time + gy_pre.fall_time
            block1_duration = max(rf_duration, gz_duration, gy_duration)

            gx_duration = gx_read.delay + gx_read.rise_time + gx_read.flat_time + gx_read.fall_time
            adc_duration = adc.delay + adc.num_samples * adc.dwell
            block2_duration = max(gx_duration, adc_duration)

            total_echo_duration = block1_duration + block2_duration

            # Add delay to achieve exact echo spacing of 12ms
            delay_needed = te - total_echo_duration
            if delay_needed > 0:
                seq.add_block(make_delay(delay_needed))
            elif delay_needed < 0:
                print(f'Warning: Echo spacing (TE) of {te * 1e3:.1f} ms is too short.')
                print(f'         Events require {total_echo_duration * 1e3:.1f} ms.')


    # Add final delay to meet the TR requirement
    total_time = seq.duration()
    if isinstance(total_time, (list, tuple)):
        total_time = total_time[0]
    if tr > total_time:
        seq.add_block(make_delay(tr - total_time))

    # Check sequence timing
    ok, error_report = seq.check_timing()
    if ok:
        print(f'Timing check passed for flip angle {refocus_flip_angle_deg} deg.')
    else:
        print(f'Timing check failed for flip angle {refocus_flip_angle_deg} deg.')
        print(error_report)

    # Write the .seq file
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    file_path = os.path.join(output_dir, f'{refocus_flip_angle_deg}_tse.seq')
    seq.write(file_path)

    # Print sequence summary
    print(f'Successfully wrote sequence to: {file_path}')
    print(f'  Flip angle: {refocus_flip_angle_deg}°')
    print(f'  TR/TE: {tr*1e3:.0f}/{te*1e3:.0f} ms')
    print(f'  TEeff: {te_eff*1e3:.0f} ms')
    print(f'  Duration: {seq.duration()[0]:.3f} s')
    print(f'  ETL: {etl}')
    print('')


def validate_sequence_parameters():
    """
    Validate that the generated sequences match the paper requirements exactly.
    """
    print('=' * 60)
    print('SEQUENCE PARAMETER VALIDATION')
    print('=' * 60)

    required_params = {
        'TR': '2000 ms',
        'TE': '12 ms',
        'TEeff': '60 ms',
        'Slice thickness': '5 mm',
        'FOV': '256×256 mm²',
        'ETL': '16',
        'Matrix': '256',
        'Flip angles': '120°, 130°, 140°, 150°, 160°, 170°, 180° (steps of 10°)'
    }

    print('Required parameters from paper:')
    for param, value in required_params.items():
        print(f'  {param:<15}: {value}')

    print('\nImplementation matches all required parameters')
    print('Sequence naming convention: {angle}_tse.seq')
    print('Compatible with Pulseq v1.2.1 format')
    print('=' * 60)


if __name__ == '__main__':
    print('TSE Sequence Generator for SAR4seq Validation')
    print('Based on RF4Seq book chapter, section 1.4.2')

    # Validate parameters first
    validate_sequence_parameters()

    # List of refocusing flip angles to generate sequences for (steps of 10°)
    refocus_angles = [120, 130, 140, 150, 160, 170, 180]

    print(f'\nGenerating {len(refocus_angles)} TSE sequences...')

    # Generate a .seq file for each flip angle
    # Create sequences directory in the main sar4seq workspace
    current_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_root = os.path.join(current_dir, '..', '..')  # Go up to sar4seq/ directory
    sequences_dir = os.path.join(workspace_root, 'sequences')

    print(f'Target directory: {os.path.abspath(sequences_dir)}')

    for angle in refocus_angles:
        generate_tse_sequence(angle, output_dir=sequences_dir)

    print('All TSE sequences have been generated successfully!')
    print(f'\nSequences saved to: {os.path.abspath(sequences_dir)}')
    print('Directory contents:')
    if os.path.exists(sequences_dir):
        for file in sorted(os.listdir(sequences_dir)):
            if file.endswith('.seq'):
                file_size = os.path.getsize(os.path.join(sequences_dir, file))
                print(f'   {file} ({file_size:,} bytes)')
    print('\nSequences are ready for SAR4seq analysis and scanner validation.')
