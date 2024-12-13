from itertools import combinations_with_replacement

import pypulseq as pp
import pytest

"""
Tests calc_duration by feeding it some sample events with known durations.
Additionally, tests the combination of any 2 and 3 of those events.
"""


def test_no_event():
    assert pp.calc_duration() == 0


known_duration_event_zoo = [
    ('trapz_amp1', pp.make_trapezoid('x', amplitude=1, duration=1), 1),
    ('trapz_amp1_delayed1', pp.make_trapezoid('x', amplitude=1, duration=1, delay=1), 2),
    ('delay1', pp.make_delay(1), 1),
    ('delay0', pp.make_delay(0), 0),
    ('rf0_block1', pp.make_block_pulse(flip_angle=0, duration=1), 1),
    ('rf10_block1', pp.make_block_pulse(flip_angle=10, duration=1), 1),
    ('rf10_block1_delay1', pp.make_block_pulse(flip_angle=10, duration=1, delay=1), 2),
    ('adc3', pp.make_adc(duration=3, num_samples=1), 3),
    ('adc3_delayed', pp.make_adc(duration=3, delay=1, num_samples=1), 4),
    ('outputOsc042', pp.make_digital_output_pulse('osc0', duration=42), 42),
    ('outputOsc142_delay3', pp.make_digital_output_pulse('osc1', duration=42, delay=1), 43),
    ('outputExt42_delay9', pp.make_digital_output_pulse('osc1', duration=42, delay=9), 51),
    ('triggerPhysio159', pp.make_trigger('physio1', duration=59), 59),
    ('triggerPhysio259_delay1', pp.make_trigger('physio2', duration=59, delay=1), 60),
    ('label0', pp.make_label(label='SLC', type='SET', value=0), 0),
]


@pytest.mark.parametrize('name,event,expected_dur', known_duration_event_zoo)
def test_single_events(name, event, expected_dur):
    assert pp.calc_duration(event) == expected_dur


def known_duration_event_zoo_combos(num_to_combine):
    for combo in combinations_with_replacement(known_duration_event_zoo, num_to_combine):
        name = ','.join(event[0] for event in combo)
        expected_dur = max(event[2] for event in combo)
        events = (event[1] for event in combo)
        yield name, tuple(events), expected_dur


@pytest.mark.parametrize('name,events,expected_total_dur', known_duration_event_zoo_combos(2))
def test_event_combinations2(name, events, expected_total_dur):
    assert pp.calc_duration(*events) == expected_total_dur


@pytest.mark.parametrize('name,events,expected_total_dur', known_duration_event_zoo_combos(3))
def test_event_combinations3(name, events, expected_total_dur):
    assert pp.calc_duration(*events) == expected_total_dur
