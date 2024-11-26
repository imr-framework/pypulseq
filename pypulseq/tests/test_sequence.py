import math
import os
from pathlib import Path
from unittest.mock import patch

import pytest

import pypulseq as pp
from pypulseq import Sequence
from pypulseq.tests.base import Approx, compare_seq_file

expected_output_path = Path(__file__).parent / 'expected_output'


# Sequence which contains only a gauss pulse
# TODO: Consider changing this to contain all types of RF pulses
def seq_make_gauss_pulse():
    seq = Sequence()
    seq.add_block(pp.make_gauss_pulse(flip_angle=1))

    return seq


# Basic sequence with gradients in all channels, some which are identical after
# rounding.
def seq1():
    seq = Sequence()
    seq.add_block(pp.make_block_pulse(math.pi / 4, duration=1e-3))
    seq.add_block(pp.make_trapezoid('x', area=1000))
    seq.add_block(pp.make_trapezoid('y', area=-500.00001))
    seq.add_block(pp.make_trapezoid('z', area=100))
    seq.add_block(pp.make_trapezoid('x', area=-1000), pp.make_trapezoid('y', area=500))
    seq.add_block(pp.make_trapezoid('y', area=-500), pp.make_trapezoid('z', area=1000))
    seq.add_block(pp.make_trapezoid('x', area=-1000), pp.make_trapezoid('z', area=1000.00001))

    return seq


# Basic spin-echo sequence structure
def seq2():
    seq = Sequence()
    seq.add_block(pp.make_block_pulse(math.pi / 2, duration=1e-3))
    seq.add_block(pp.make_trapezoid('x', area=1000))
    seq.add_block(pp.make_trapezoid('x', area=-1000))
    seq.add_block(pp.make_block_pulse(math.pi, duration=1e-3))
    seq.add_block(pp.make_trapezoid('x', area=-500))
    seq.add_block(pp.make_trapezoid('x', area=1000, duration=10e-3), pp.make_adc(num_samples=100, duration=10e-3))

    return seq


# Basic GRE sequence with INC labels
def seq3():
    seq = Sequence()

    for i in range(10):
        seq.add_block(pp.make_block_pulse(math.pi / 8, duration=1e-3))
        seq.add_block(pp.make_trapezoid('x', area=1000))
        seq.add_block(pp.make_trapezoid('y', area=-500 + i * 100))
        seq.add_block(pp.make_trapezoid('x', area=-500))
        seq.add_block(
            pp.make_trapezoid('x', area=1000, duration=10e-3),
            pp.make_adc(num_samples=100, duration=10e-3),
            pp.make_label(label='LIN', type='INC', value=1),
        )

    return seq


# Basic GRE sequence with SET labels
def seq4():
    seq = Sequence()

    for i in range(10):
        seq.add_block(pp.make_block_pulse(math.pi / 8, duration=1e-3))
        seq.add_block(pp.make_trapezoid('x', area=1000))
        seq.add_block(pp.make_trapezoid('y', area=-500 + i * 100))
        seq.add_block(pp.make_trapezoid('x', area=-500))
        seq.add_block(
            pp.make_trapezoid('x', area=1000, duration=10e-3),
            pp.make_adc(num_samples=100, duration=10e-3),
            pp.make_label(label='LIN', type='SET', value=i),
        )

    return seq


# List of all sequence functions that will be tested with the test functions
# below.
sequence_zoo = [seq_make_gauss_pulse, seq1, seq2, seq3, seq4]


# This "test" rewrites the expected .seq output files when SAVE_EXPECTED is
# set in the environment variables.
# E.g. in a unix-based system, run: SAVE_EXPECTED=1 pytest test_sequence.py
@pytest.mark.skipif(not os.environ.get('SAVE_EXPECTED'), reason='Only save sequence files when requested')
@pytest.mark.parametrize('seq_func', sequence_zoo)
def test_sequence_save_expected(seq_func):
    seq_name = str(seq_func.__name__)

    # Generate sequence and write to file
    seq = seq_func()
    seq.write(expected_output_path / (seq_name + '.seq'))


# Test whether a sequence can be plotted.
@pytest.mark.parametrize('seq_func', sequence_zoo)
@patch('matplotlib.pyplot.show')
def test_plot(mock_show, seq_func):
    seq = seq_func()

    seq.plot()
    seq.plot(show_blocks=True)


# Test whether the sequence is the approximately the same after writing a .seq
# file and reading it back in.
@pytest.mark.parametrize('seq_func', sequence_zoo)
def test_sequence_writeread(seq_func, tmp_path):
    seq_name = str(seq_func.__name__)
    output_filename = tmp_path / (seq_name + '.seq')

    # Generate sequence
    seq = seq_func()

    # Write sequence to file
    seq.write(output_filename)

    # Check if written sequence file matches expected sequence file
    compare_seq_file(output_filename, expected_output_path / (seq_name + '.seq'))

    # Read written sequence file back in
    seq2 = pp.Sequence(system=seq.system)
    seq2.read(output_filename)

    # Clean up written sequence file
    output_filename.unlink()

    # Test for approximate equality of all blocks
    assert set(seq2.block_events.keys()) == set(seq.block_events.keys())
    for block_counter in seq.block_events:
        assert seq2.get_block(block_counter) == Approx(
            seq.get_block(block_counter), abs=1e-6, rel=1e-5
        ), f'Block {block_counter} does not match'

    # Test for approximate equality of all gradient waveforms
    for a, b in zip(seq2.get_gradients(), seq.get_gradients()):
        if a is None and b is None:
            continue
        if a is None or b is None:
            raise AssertionError()

        assert a.x == Approx(b.x, abs=1e-3, rel=1e-3)
        assert a.c == Approx(b.c, abs=1e-3, rel=1e-3)

    # Test for approximate equality of kspace calculation
    assert seq2.calculate_kspace() == Approx(seq.calculate_kspace(), abs=1e-2, nan_ok=True)

    # Test whether labels are the same
    labels_seq = seq.evaluate_labels(evolution='blocks')
    labels_seq2 = seq2.evaluate_labels(evolution='blocks')

    assert labels_seq.keys() == labels_seq2.keys(), 'Sequences do not contain the same set of labels'

    for label in labels_seq:
        assert (labels_seq[label] == labels_seq2[label]).all(), f'Label {label} does not match'


# Test whether the sequence is approximately the same after recreating it by
# getting all blocks with get_block and inserting them into a new sequence
# with add_block.
@pytest.mark.parametrize('seq_func', sequence_zoo)
def test_sequence_recreate(seq_func, tmp_path):
    # Generate sequence
    seq = seq_func()

    # Insert blocks from sequence into a new sequence
    seq2 = pp.Sequence(system=seq.system)
    for b in seq.block_events:
        seq2.add_block(seq.get_block(b))

    # Test for approximate equality of all blocks
    assert set(seq2.block_events.keys()) == set(seq.block_events.keys())
    for block_counter in seq.block_events:
        assert seq2.get_block(block_counter) == Approx(
            seq.get_block(block_counter), abs=1e-6, rel=1e-5
        ), f'Block {block_counter} does not match'

    # Test for approximate equality of all gradient waveforms
    for a, b in zip(seq2.get_gradients(), seq.get_gradients()):
        if a is None and b is None:
            continue
        if a is None or b is None:
            raise AssertionError()

        assert a.x == Approx(b.x, abs=1e-4, rel=1e-4)
        assert a.c == Approx(b.c, abs=1e-4, rel=1e-4)

    # Test for approximate equality of kspace calculation
    assert seq2.calculate_kspace() == Approx(seq.calculate_kspace(), abs=1e-6, nan_ok=True)

    # Test whether labels are the same
    labels_seq = seq.evaluate_labels(evolution='blocks')
    labels_seq2 = seq2.evaluate_labels(evolution='blocks')

    assert labels_seq.keys() == labels_seq2.keys(), 'Sequences do not contain the same set of labels'

    for label in labels_seq:
        assert (labels_seq[label] == labels_seq2[label]).all(), f'Label {label} does not match'
