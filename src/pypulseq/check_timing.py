from types import SimpleNamespace
from typing import Any, List, Tuple

from pypulseq import Sequence, eps
from pypulseq.calc_duration import calc_duration
from pypulseq.utils.tracing import format_trace

error_messages = {
    'RASTER': '{value*multiplier:.2f} {unit} does not align to {raster} (Nearest valid value: {value_rounded*multiplier:.0f} {unit}, error: {error*multiplier:.2f} {unit})',
    'ADC_DEAD_TIME': 'ADC delay is smaller than ADC dead time ({value*multiplier:.2f} {unit} < {dead_time*multiplier:.0f} {unit})',
    'POST_ADC_DEAD_TIME': 'Post-ADC dead time exceeds block duration ({value*multiplier:.2f} {unit} + {dead_time*multiplier:.0f} {unit} > {duration*multiplier} {unit})',
    'BLOCK_DURATION_MISMATCH': 'Inconsistency between the stored block duration ({duration*multiplier:.2f} {unit}) and the content of the block ({value*multiplier:.2f} {unit})',
    'RF_DEAD_TIME': 'Delay of {value*multiplier:.2f} {unit} is smaller than the RF dead time {dead_time*multiplier:.0f} {unit}',
    'RF_RINGDOWN_TIME': 'Time between the end of the RF pulse at {value*multiplier:.2f} {unit} and the end of the block at {duration * multiplier:.2f} {unit} is shorter than rf_ringdown_time ({ringdown_time*multiplier:.0f} {unit})',
    'NEGATIVE_DELAY': 'Delay is negative {value*multiplier:.2f} {unit}',
    'SOFT_DELAY_FACTOR': 'Soft delay {hint}/{numID} has the factor parameter as zero, which is invalid.',
    'SOFT_DELAY_DUR_INCONSISTENCY': 'Soft delay {hint}/{numID} default duration derived from this block ({value*1e6} us) is inconsistent with the previous default.',
    'SOFT_DELAY_HINT_INCONSISTENCY': 'Soft delay {hint}/{numID}: Soft delays with the same numeric ID are expected to share the same text hint but previous hint recorded is {prev_hint}.',
    'SOFT_DELAY_INVALID_NUMID': 'Soft delay {hint}/{numID} has an invalid numeric ID {numID}. Numeric IDs must be positive integers.',
    'ADC_DWELL_RASTER': 'ADC: dwell time ({dwell:.2f} {unit}) is not a multiple of sys.adc_raster_time ({value*multiplier:.2f} {unit})',
    'ADC_NUM_SAMPLES_DIV': 'ADC: num_samples ({num_samples} {unit}) is not a multiple of sys.adc_samples_divisor ({value})',
    'GRAD_START_NONZERO': '{channel} gradient starts at non-zero {first:.3f} but no continuation',
    'GRAD_END_NONZERO_DURATION': '{channel} gradient ends at non-zero {last:.3f}us but does not reach block end',
    'GRAD_PREV_BLOCK_NOT_CONSUMED': 'Previous block gradient in {channel} not consumed before this block',
}


def check_timing(seq: Sequence) -> Tuple[bool, List[SimpleNamespace]]:
    error_report: List[SimpleNamespace] = []
    grad_book = {}
    soft_delay_state = {}

    def div_check(a: float, b: float, event: str, field: str, raster: str):
        """
        Checks whether `a` can be divided by `b` to an accuracy of 1e-9.
        """
        c = a / b
        c_rounded = round(c)
        is_ok = abs(c - c_rounded) < 1e-6

        if not is_ok:
            error_report.append(
                SimpleNamespace(
                    block=block_counter,
                    event=event,
                    field=field,
                    value=a,
                    value_rounded=c_rounded * b,
                    error=(a - c_rounded * b),
                    raster=raster,
                    error_type='RASTER',
                )
            )

    # Loop over all blocks
    for block_counter in seq.block_events:
        block = seq.get_block(block_counter)

        # Check block duration
        duration = calc_duration(block)
        div_check(
            duration, seq.system.block_duration_raster, event='block', field='duration', raster='block_duration_raster'
        )

        if abs(duration - seq.block_durations[block_counter]) > eps:
            error_report.append(
                SimpleNamespace(
                    block=block_counter,
                    event='block',
                    field='duration',
                    error_type='BLOCK_DURATION_MISMATCH',
                    value=duration,
                    duration=seq.block_durations[block_counter],
                )
            )
            duration = seq.block_durations[block_counter]

        # Check block events
        for event, e in block.__dict__.items():
            if e is None or isinstance(e, (float, int)):  # Special handling for block_duration
                continue
            elif not isinstance(e, (dict, SimpleNamespace)):
                raise ValueError('Wrong data type of variable arguments, list[SimpleNamespace] expected.')

            if isinstance(e, list) and len(e) > 1:
                # For now this is only the case for arrays of extensions, but we cannot actually check extensions anyway...
                continue

            if hasattr(e, 'type') and e.type == 'rf':
                raster = seq.system.rf_raster_time
                raster_str = 'rf_raster_time'
            elif hasattr(e, 'type') and e.type == 'adc':
                # note that ADC samples must be on ADC raster time, but the ADC start time must be on RF raster time!
                # see https://github.com/pulseq/pulseq/blob/master/doc%2Fpulseq_shapes_and_times.pdf for details
                raster = seq.system.rf_raster_time
                raster_str = 'rf_raster_time'
            else:
                raster = seq.system.grad_raster_time
                raster_str = 'grad_raster_time'

            if hasattr(e, 'delay'):
                if e.delay < -eps:
                    error_report.append(
                        SimpleNamespace(
                            block=block_counter, event=event, field='delay', error_type='NEGATIVE_DELAY', value=e.delay
                        )
                    )

                div_check(e.delay, raster, event=event, field='delay', raster=raster_str)

            if hasattr(e, 'duration'):
                div_check(e.duration, raster, event=event, field='duration', raster=raster_str)

            if hasattr(e, 'dwell'):
                div_check(e.dwell, seq.system.adc_raster_time, event=event, field='dwell', raster='adc_raster_time')

            if hasattr(e, 'type') and e.type == 'trap':
                div_check(
                    e.rise_time, seq.system.grad_raster_time, event=event, field='rise_time', raster='grad_raster_time'
                )
                div_check(
                    e.flat_time, seq.system.grad_raster_time, event=event, field='flat_time', raster='grad_raster_time'
                )
                div_check(
                    e.fall_time, seq.system.grad_raster_time, event=event, field='fall_time', raster='grad_raster_time'
                )

        # Check RF dead times
        if block.rf is not None:
            if block.rf.delay - block.rf.dead_time < -eps:
                error_report.append(
                    SimpleNamespace(
                        block=block_counter,
                        event='rf',
                        field='delay',
                        error_type='RF_DEAD_TIME',
                        value=block.rf.delay,
                        dead_time=block.rf.dead_time,
                    )
                )

            if block.rf.delay + block.rf.t[-1] + block.rf.ringdown_time - duration > eps:
                error_report.append(
                    SimpleNamespace(
                        block=block_counter,
                        event='rf',
                        field='duration',
                        error_type='RF_RINGDOWN_TIME',
                        value=block.rf.delay + block.rf.t[-1],
                        duration=duration,
                        ringdown_time=block.rf.ringdown_time,
                    )
                )

        # Check ADC dead times, dwell times and number of samples
        if block.adc is not None:
            if block.adc.delay - seq.system.adc_dead_time < -eps:
                error_report.append(
                    SimpleNamespace(
                        block=block_counter,
                        event='adc',
                        field='delay',
                        error_type='ADC_DEAD_TIME',
                        value=block.adc.delay,
                        dead_time=seq.system.adc_dead_time,
                    )
                )

            adc_end = block.adc.delay + block.adc.num_samples * block.adc.dwell + seq.system.adc_dead_time

            if adc_end > duration + eps:
                error_report.append(
                    SimpleNamespace(
                        block=block_counter,
                        event='adc',
                        field='duration',
                        error_type='POST_ADC_DEAD_TIME',
                        value=block.adc.delay + block.adc.num_samples * block.adc.dwell,
                        duration=duration,
                        dead_time=seq.system.adc_dead_time,
                    )
                )

            if (
                abs(block.adc.dwell / seq.sys.adc_raster_time - round(block.adc.dwell / seq.sys.adc_raster_time))
                > 1e-10
            ):
                error_report.append(
                    SimpleNamespace(
                        block=block_counter,
                        event='adc',
                        field='duration',
                        error_type='ADC_DWELL_RASTER',
                        value=seq.system.adc_raster_time,
                    )
                )

            if (
                abs(
                    block.adc.num_samples / seq.sys.adc_samples_divisor
                    - round(block.adc.num_samples / seq.sys.adc_samples_divisor)
                )
                > eps
            ):
                error_report.append(
                    SimpleNamespace(
                        block=block_counter,
                        event='adc',
                        field='duration',
                        error_type='ADC_NUM_SAMPLES_DIV',
                        value=seq.system.adc_samples_divisor,
                    )
                )

        # Gradient continuity checks
        grad_book_curr = {}
        for e in block.__dict__.values():
            if hasattr(e, 'type') and e.type == 'grad':
                ch = e.channel
                if e.first != 0:
                    if ch not in grad_book or grad_book[ch] != e.first:
                        error_report.append(
                            SimpleNamespace(
                                block=block_counter,
                                event='grad',
                                channel=ch,
                                error_type='GRAD_START_NONZERO',
                                first=e.first,
                            )
                        )
                    elif getattr(e, 'delay', 0) != 0:
                        # unexpected delay
                        error_report.append(
                            SimpleNamespace(
                                block=block_counter,
                                event='grad',
                                channel=ch,
                                error_type='GRAD_START_NONZERO',
                                first=e.first,
                            )
                        )
                    grad_book[ch] = 0
                if e.last != 0:
                    if abs((getattr(e, 'delay', 0) + getattr(e, 'shape_dur', duration)) - duration) > eps:
                        error_report.append(
                            SimpleNamespace(
                                block=block_counter,
                                event='grad',
                                channel=ch,
                                error_type='GRAD_END_NONZERO_DURATION',
                                last=e.last,
                            )
                        )
                    grad_book_curr[ch] = e.last

        if duration != 0:
            for ch, val in grad_book.items():
                if val != 0:
                    error_report.append(
                        SimpleNamespace(
                            block=block_counter,
                            event='grad',
                            channel=ch,
                            error_type='GRAD_PREV_BLOCK_NOT_CONSUMED',
                            first=val,
                        )
                    )
            grad_book = grad_book_curr

        # Soft delay logic
        if hasattr(block, 'soft_delay') and block.soft_delay is not None:
            if block.soft_delay.factor == 0:
                error_report.append(
                    SimpleNamespace(
                        block=block_counter,
                        event='soft_delay',
                        error_type='SOFT_DELAY_FACTOR',
                        hint=block.soft_delay.hint,
                        numID=block.soft_delay.numID,
                    )
                )

            # Calculate default delay value based on the current block duration
            def_del = (seq.block_durations[block_counter] - block.soft_delay.offset) * block.soft_delay.factor

            if block.soft_delay.numID >= 0:
                numID = block.soft_delay.numID  # index as dict key

                if numID not in soft_delay_state:
                    soft_delay_state[numID] = SimpleNamespace(
                        def_del=def_del,
                        hint=block.soft_delay.hint,
                    )
                else:
                    prev = soft_delay_state[numID]
                    if abs(def_del - prev.def_del) > 1e-7:
                        error_report.append(
                            SimpleNamespace(
                                block=block_counter,
                                event='soft_delay',
                                error_type='SOFT_DELAY_DUR_INCONSISTENCY',
                                hint=block.soft_delay.hint,
                                numID=block.soft_delay.numID,
                                value=prev.def_del,
                            )
                        )

                    if block.soft_delay.hint != prev.hint:
                        error_report.append(
                            SimpleNamespace(
                                block=block_counter,
                                event='soft_delay',
                                error_type='SOFT_DELAY_HINT_INCONSISTENCY',
                                hint=block.soft_delay.hint,
                                numID=block.soft_delay.numID,
                                prev_hint=prev.hint,
                            )
                        )
            else:
                error_report.append(
                    SimpleNamespace(
                        block=block_counter,
                        event='soft_delay',
                        error_type='SOFT_DELAY_INVALID_NUMID',
                        hint=block.soft_delay.hint,
                        numID=block.soft_delay.num,
                    )
                )

    # Final: make sure gradients ramped down in the last block
    for ch, val in grad_book.items():
        if val != 0:
            error_report.append(
                SimpleNamespace(
                    block=block_counter,
                    event='grad',
                    channel=ch,
                    error_type='GRAD_END_NONZERO_DURATION',
                    last=val,
                )
            )

    return len(error_report) == 0, error_report


def format_string(template: str, **kwargs: Any) -> str:
    """
    Evaluate a formatted string using the f-string syntax. Similar to
    `'{x}'.format(x=1)`, but allows arbitrary computations, e.g.
    `'{x*y}'.format(x=2,y=2)`.

    Parameters
    ----------
    template : str
        Format string.
    **kwargs : Any
        Variables to use in the formatted string.

    Returns
    -------
    str
        Formatted string.
    """
    return eval(f'f"""{template}"""', kwargs)


def indent_string(x: str, n: int = 2) -> str:
    """
    Adds indentations (`n` spaces) to every line in a string
    """
    return '\n'.join(' ' * n + y for y in x.splitlines())


def print_error_report(
    seq: Sequence,
    error_report: List[SimpleNamespace],
    full_report: bool = False,
    max_errors: int = 10,
    colored: bool = True,
) -> None:
    current_block = None

    if full_report:
        max_errors = len(error_report)

    for e in error_report[:max_errors]:
        if e.block != current_block:
            print(f'Block {e.block}:')
            current_block = e.block
            trace = seq.block_trace.get(current_block, None)

            if hasattr(trace, 'block'):
                print(
                    ('\x1b[38;5;8m' if colored else '')
                    + 'Block created here:\n'
                    + format_trace(trace.block)
                    + ('\x1b[0m' if colored else '')
                )

        unit = 'us'
        multiplier = 1e6
        if e.field == 'dwell':
            unit = 'ns'
            multiplier = 1e9

        error_message = format_string(error_messages[e.error_type], **e.__dict__, unit=unit, multiplier=multiplier)
        print(
            f'- {e.event}.{e.field}: '
            + ('\x1b[38;5;9m' if colored else '')
            + error_message
            + ('\x1b[0m' if colored else '')
        )

        if hasattr(trace, e.event) and e.event != 'block':
            print(
                ('\x1b[38;5;8m' if colored else '')
                + f'  `{e.event}` created here:\n'
                + format_trace(getattr(trace, e.event), indent=2)
                + ('\x1b[0m' if colored else '')
            )

    if len(error_report) > max_errors:
        blocks = [e.block for e in error_report[max_errors:]]

        print(f'--- {len(error_report) - max_errors} more errors in blocks {min(blocks)} to {max(blocks)} hidden ---')
