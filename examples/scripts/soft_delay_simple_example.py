"""
Simple example demonstrating soft delay functionality in pypulseq.

This example shows how to create and use soft delays for dynamic timing adjustment
without recompiling the sequence. Soft delays are useful for optimizing timing
parameters like TE, TR, or other delays at the scanner.
"""

import math

import pypulseq as pp


def main():
    """Create a simple sequence with soft delays to demonstrate the functionality."""

    # Create sequence with default system limits
    seq = pp.Sequence()

    print('=== Simple Soft Delay Example ===\n')

    # ======================================
    # Example 1: Basic TE soft delay
    # ======================================
    print('1. Creating a basic TE soft delay...')

    # Create a soft delay for echo time (TE) adjustment
    # The scanner interface will show this as "TE" with default value 5ms
    te_delay = pp.make_soft_delay('TE', default_duration=5e-3)

    print(f"   - Created TE delay with hint: '{te_delay.hint}'")
    print(f'   - Default duration: {te_delay.default_duration * 1000:.1f} ms')

    # Add to sequence - block duration automatically becomes the default_duration
    # numID is assigned when added to sequence
    seq.add_block(te_delay)
    print(f'   - Auto-assigned numID: {te_delay.numID}')
    print(f'   - Block duration: {seq.block_durations[1] * 1000:.1f} ms\n')

    # ======================================
    # Example 2: TR soft delay with scaling
    # ======================================
    print('2. Creating a TR soft delay with scaling...')

    # Create a TR delay with offset and scaling
    # Formula: final_duration = (user_input / factor) + offset
    # Here: TR input in ms, but we want seconds, so factor=1000
    tr_delay = pp.make_soft_delay(
        'TR',
        factor=1000.0,  # Convert ms input to seconds
        offset=-10e-3,  # Subtract 10ms overhead
        default_duration=100e-3,
    )  # Default 100ms

    print(f'   - Created TR delay with factor: {tr_delay.factor}')
    print(f'   - Offset: {tr_delay.offset * 1000:.1f} ms')

    seq.add_block(tr_delay)
    print(f'   - Auto-assigned numID: {tr_delay.numID}')
    print(f'   - Block duration: {seq.block_durations[2] * 1000:.1f} ms\n')

    # ======================================
    # Example 3: Multiple delays with same hint
    # ======================================
    print('3. Creating multiple delays with same hint...')

    # Multiple TE delays automatically share the same numID
    te_delay2 = pp.make_soft_delay('TE', default_duration=8e-3)  # Different default
    te_delay3 = pp.make_soft_delay('TE', default_duration=12e-3)  # Different default

    seq.add_block(te_delay2)
    seq.add_block(te_delay3)

    print(f'   - TE delay #2 numID: {te_delay2.numID} (reuses same ID)')
    print(f'   - TE delay #3 numID: {te_delay3.numID} (reuses same ID)')
    print(f'   - Block durations: {seq.block_durations[3] * 1000:.1f} ms, {seq.block_durations[4] * 1000:.1f} ms\n')

    # ======================================
    # Example 4: Applying soft delays
    # ======================================
    print('4. Applying soft delay values...')

    print('   Before applying:')
    for i, duration in seq.block_durations.items():
        print(f'     Block {i}: {duration * 1000:.1f} ms')

    # Apply new values - this updates all blocks with matching hints
    seq.apply_soft_delay(
        TE=15e-3,  # Set all TE delays to 15ms
        TR=250,  # Set TR to 250ms (will be processed as: (250/1000) - 0.01 = 0.24s)
    )

    print('\n   After applying TE=15ms, TR=250ms:')
    for i, duration in seq.block_durations.items():
        print(f'     Block {i}: {duration * 1000:.1f} ms')

    # ======================================
    # Example 5: Realistic sequence context
    # ======================================
    print('\n5. Soft delays in a realistic sequence context...')

    # Create a simple imaging sequence with soft delays
    seq_realistic = pp.Sequence()

    # RF pulse
    rf_pulse = pp.make_block_pulse(flip_angle=30 * math.pi / 180, duration=1e-3)

    # Gradients
    gx_readout = pp.make_trapezoid('x', area=1000, duration=5e-3)
    gy_phase = pp.make_trapezoid('y', area=500, duration=2e-3)

    # ADC
    adc = pp.make_adc(num_samples=128, duration=4e-3)

    # Soft delays for timing optimization
    te_delay_real = pp.make_soft_delay('TE', default_duration=10e-3)
    tr_delay_real = pp.make_soft_delay('TR', default_duration=50e-3)

    # Build sequence: RF -> TE delay -> Phase encoding -> Readout+ADC -> TR delay
    seq_realistic.add_block(rf_pulse)
    seq_realistic.add_block(te_delay_real)
    seq_realistic.add_block(gy_phase)
    seq_realistic.add_block(gx_readout, adc)
    seq_realistic.add_block(tr_delay_real)

    print(f'   - Created realistic sequence with {len(seq_realistic.block_durations)} blocks')
    print(f'   - Total sequence duration: {sum(seq_realistic.block_durations.values()) * 1000:.1f} ms')

    # Optimize timing
    seq_realistic.apply_soft_delay(TE=8e-3, TR=40e-3)
    print(f'   - After optimization: {sum(seq_realistic.block_durations.values()) * 1000:.1f} ms')

    print('\n=== Example Complete ===')
    print('\nKey takeaways:')
    print('• Soft delays enable runtime timing adjustment without recompiling')
    print("• Use descriptive hints like 'TE', 'TR', 'TI' for scanner interface")
    print('• Multiple delays with same hint automatically share numID')
    print('• Default duration becomes the initial block duration')
    print('• Apply delays with seq.apply_soft_delay(HINT=value)')
    print('• Use factor/offset for unit conversion and timing adjustments')


if __name__ == '__main__':
    main()
