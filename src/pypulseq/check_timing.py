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
}


def check_timing(seq: Sequence) -> Tuple[bool, List[SimpleNamespace]]:
    error_report: List[SimpleNamespace] = []

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

        # Check ADC dead times
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
