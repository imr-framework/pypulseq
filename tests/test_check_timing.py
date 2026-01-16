import pypulseq as pp

# System settings
system = pp.Opts(
    max_grad=28,
    grad_unit='mT/m',
    max_slew=200,
    slew_unit='T/m/s',
    rf_ringdown_time=20e-6,
    rf_dead_time=100e-6,
    adc_dead_time=10e-6,
)

# System with ringdown and dead times set to 0 to introduce timing errors
system_broken = pp.Opts(
    max_grad=28,
    grad_unit='mT/m',
    max_slew=200,
    slew_unit='T/m/s',
    rf_ringdown_time=0e-6,
    rf_dead_time=0e-6,
    adc_dead_time=0e-6,
)


# Check whether there are no errors in the timing error report for the given blocks
def blocks_not_in_error_report(error_report, blocks):
    return all(error.block not in blocks for error in error_report)


# Check whether a given timing error exists in the report
def exists_in_error_report(error_report, block, event, field, error_type):
    for error in error_report:
        if error.block == block and error.event == event and error.field == field and error.error_type == error_type:
            return True
    return False


# Test whether check_timing catches all different timing errors that can occur
def test_check_timing():
    seq = pp.Sequence(system=system)

    # Add events with possible timing errors
    rf = pp.make_sinc_pulse(flip_angle=1, duration=1e-3, delay=system.rf_dead_time, system=system)
    seq.add_block(rf)  # Block 1: No error

    rf = pp.make_sinc_pulse(flip_angle=1, duration=1e-3, system=system_broken)
    seq.add_block(rf)  # Block 2: RF_DEAD_TIME, RF_RINGDOWN_TIME, BLOCK_DURATION_MISMATCH

    adc = pp.make_adc(num_samples=100, duration=1e-3, delay=system.adc_dead_time, system=system)
    seq.add_block(adc)  # Block 3: No error

    adc = pp.make_adc(num_samples=123, duration=1e-3, delay=system.adc_dead_time, system=system)
    seq.add_block(adc)  # Block 4: RASTER

    adc = pp.make_adc(num_samples=100, duration=1e-3, system=system_broken)
    seq.add_block(adc)  # Block 5: ADC_DEAD_TIME, POST_ADC_DEAD_TIME, BLOCK_DURATION_MISMATCH

    gx = pp.make_trapezoid(channel='x', area=1, duration=1, system=system)
    seq.add_block(gx)  # Block 6: No error

    gx = pp.make_trapezoid(channel='x', area=1, duration=1.00001e-3, system=system)
    seq.add_block(gx)  # Block 7: RASTER

    gx = pp.make_trapezoid(channel='x', area=1, rise_time=1e-6, flat_time=1e-3, fall_time=3e-6, system=system)
    seq.add_block(gx)  # Block 8: RASTER

    gx = pp.make_trapezoid(channel='x', area=1, duration=1e-3, delay=-1e-5, system=system)
    seq.add_block(gx)  # Block 9: NEGATIVE_DELAY

    # Check timing errors
    _, error_report = seq.check_timing()

    # Check whether the timing error report is as expected
    assert blocks_not_in_error_report(error_report, [1, 3, 6]), 'No timing errors expected on blocks 1, 3, and 6'

    assert exists_in_error_report(error_report, 2, event='rf', field='delay', error_type='RF_DEAD_TIME')
    assert exists_in_error_report(error_report, 2, event='rf', field='duration', error_type='RF_RINGDOWN_TIME')
    assert exists_in_error_report(
        error_report, 2, event='block', field='duration', error_type='BLOCK_DURATION_MISMATCH'
    )

    assert exists_in_error_report(error_report, 4, event='adc', field='dwell', error_type='RASTER')

    assert exists_in_error_report(error_report, 5, event='adc', field='delay', error_type='ADC_DEAD_TIME')
    assert exists_in_error_report(error_report, 5, event='adc', field='duration', error_type='POST_ADC_DEAD_TIME')
    assert exists_in_error_report(
        error_report, 5, event='block', field='duration', error_type='BLOCK_DURATION_MISMATCH'
    )

    assert exists_in_error_report(error_report, 7, event='block', field='duration', error_type='RASTER')
    assert exists_in_error_report(error_report, 7, event='gx', field='flat_time', error_type='RASTER')

    assert exists_in_error_report(error_report, 8, event='block', field='duration', error_type='RASTER')
    assert exists_in_error_report(error_report, 8, event='gx', field='rise_time', error_type='RASTER')
    assert exists_in_error_report(error_report, 8, event='gx', field='fall_time', error_type='RASTER')

    assert exists_in_error_report(error_report, 9, event='gx', field='delay', error_type='NEGATIVE_DELAY')

    assert len(error_report) == 13, 'Total number of timing errors was expected to be 12'
