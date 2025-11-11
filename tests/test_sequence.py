import importlib
import math
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pypulseq as pp
import pytest
from _pytest.python_api import ApproxBase
from pypulseq import Sequence

expected_output_path = Path(__file__).parent / 'expected_output'


class Approx(ApproxBase):
    """
    Extension of pytest.approx that also handles approximate equality
    recursively within dicts, lists, tuples, and SimpleNamespace
    """

    def __repr__(self):
        return str(self.expected)

    def __eq__(self, actual):
        # if type(actual) != type(self.expected):
        #     return False
        if isinstance(self.expected, dict):
            if set(self.expected.keys()) != set(actual.keys()):
                return False

            for k in self.expected:
                if actual[k] != Approx(self.expected[k], rel=self.rel, abs=self.abs, nan_ok=self.nan_ok):
                    return False
            return True
        elif isinstance(self.expected, (list, tuple)):
            if len(self.expected) != len(actual):
                return False

            for e, a in zip(self.expected, actual, strict=False):
                if a != Approx(e, rel=self.rel, abs=self.abs, nan_ok=self.nan_ok):
                    return False
            return True
        elif isinstance(self.expected, SimpleNamespace):
            return actual.__dict__ == Approx(self.expected.__dict__, rel=self.rel, abs=self.abs, nan_ok=self.nan_ok)
        else:
            return actual == pytest.approx(self.expected, rel=self.rel, abs=self.abs, nan_ok=self.nan_ok)

    def _repr_compare(self, actual):
        # if type(actual) != type(self.expected):
        #     return [f'Actual and expected types do not match: {type(actual)} != {type(self.expected)}']
        if isinstance(self.expected, dict):
            if set(self.expected.keys()) != set(actual.keys()):
                return [f'Actual and expected keys do not match: {set(actual.keys())} != {set(self.expected.keys())}']

            r = []
            for k in self.expected:
                approx_obj = Approx(self.expected[k], rel=self.rel, abs=self.abs, nan_ok=self.nan_ok)
                if actual[k] != approx_obj:
                    r += [f'{k} does not match:']
                    r += [f'  {x}' for x in approx_obj._repr_compare(actual[k])]
            return r
        elif isinstance(self.expected, (list, tuple)):
            if len(self.expected) != len(actual):
                return [f'Actual and expected lengths do not match: {len(actual)} != {len(self.expected)}']
            r = []
            for i, (e, a) in enumerate(zip(self.expected, actual, strict=False)):
                approx_obj = Approx(e, rel=self.rel, abs=self.abs, nan_ok=self.nan_ok)
                if a != approx_obj:
                    r += [f'Index {i} does not match:']
                    r += [f'  {x}' for x in approx_obj._repr_compare(a)]
            return r
        elif isinstance(self.expected, SimpleNamespace):
            return Approx(self.expected.__dict__, rel=self.rel, abs=self.abs, nan_ok=self.nan_ok)._repr_compare(
                actual.__dict__
            )
        else:
            return pytest.approx(self.expected, rel=self.rel, abs=self.abs, nan_ok=self.nan_ok)._repr_compare(actual)


# Dummy sequence which contains only gaussian pulses with different parameters
def seq_make_gauss_pulses():
    seq = Sequence()
    seq.add_block(pp.make_gauss_pulse(flip_angle=1))
    seq.add_block(pp.make_delay(1))
    seq.add_block(pp.make_gauss_pulse(flip_angle=1, delay=1e-3))
    seq.add_block(pp.make_delay(1))
    seq.add_block(pp.make_gauss_pulse(flip_angle=math.pi / 2))
    seq.add_block(pp.make_delay(1))
    seq.add_block(pp.make_gauss_pulse(flip_angle=math.pi / 2, duration=1e-3))
    seq.add_block(pp.make_delay(1))
    seq.add_block(pp.make_gauss_pulse(flip_angle=math.pi / 2, duration=2e-3, phase_offset=math.pi / 2))
    seq.add_block(pp.make_delay(1))
    seq.add_block(pp.make_gauss_pulse(flip_angle=math.pi / 2, duration=1e-3, phase_offset=math.pi / 2, freq_offset=1e3))
    seq.add_block(pp.make_delay(1))
    seq.add_block(pp.make_gauss_pulse(flip_angle=math.pi / 2, duration=1e-3, time_bw_product=1))
    seq.add_block(pp.make_delay(1))
    seq.add_block(pp.make_gauss_pulse(flip_angle=math.pi / 2, duration=1e-3, apodization=0.1))

    return seq


# Dummy sequence which contains only sinc pulses with different parameters
def seq_make_sinc_pulses():
    seq = Sequence()
    seq.add_block(pp.make_sinc_pulse(flip_angle=1))
    seq.add_block(pp.make_delay(1))
    seq.add_block(pp.make_sinc_pulse(flip_angle=1, delay=1e-3))
    seq.add_block(pp.make_delay(1))
    seq.add_block(pp.make_sinc_pulse(flip_angle=math.pi / 2))
    seq.add_block(pp.make_delay(1))
    seq.add_block(pp.make_sinc_pulse(flip_angle=math.pi / 2, duration=1e-3))
    seq.add_block(pp.make_delay(1))
    seq.add_block(pp.make_sinc_pulse(flip_angle=math.pi / 2, duration=2e-3, phase_offset=math.pi / 2))
    seq.add_block(pp.make_delay(1))
    seq.add_block(pp.make_sinc_pulse(flip_angle=math.pi / 2, duration=1e-3, phase_offset=math.pi / 2, freq_offset=1e3))
    seq.add_block(pp.make_delay(1))
    seq.add_block(pp.make_sinc_pulse(flip_angle=math.pi / 2, duration=1e-3, time_bw_product=1))
    seq.add_block(pp.make_delay(1))
    seq.add_block(pp.make_sinc_pulse(flip_angle=math.pi / 2, duration=1e-3, apodization=0.1))

    return seq


# Dummy sequence which contains only block pulses with different parameters
def seq_make_block_pulses():
    seq = Sequence()
    seq.add_block(pp.make_block_pulse(flip_angle=1, duration=4e-3))
    seq.add_block(pp.make_delay(1))
    seq.add_block(pp.make_block_pulse(flip_angle=1, delay=1e-3, duration=4e-3))
    seq.add_block(pp.make_delay(1))
    seq.add_block(pp.make_block_pulse(flip_angle=math.pi / 2, duration=4e-3))
    seq.add_block(pp.make_delay(1))
    seq.add_block(pp.make_block_pulse(flip_angle=math.pi / 2, duration=1e-3))
    seq.add_block(pp.make_delay(1))
    seq.add_block(pp.make_block_pulse(flip_angle=math.pi / 2, duration=2e-3, phase_offset=math.pi / 2))
    seq.add_block(pp.make_delay(1))
    seq.add_block(pp.make_block_pulse(flip_angle=math.pi / 2, duration=1e-3, phase_offset=math.pi / 2, freq_offset=1e3))
    seq.add_block(pp.make_delay(1))
    seq.add_block(pp.make_block_pulse(flip_angle=math.pi / 2, duration=1e-3, time_bw_product=1))

    return seq


# Basic sequence with gradients in all channels, some which are identical after rounding.
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


# GRE sequence with preceding noise acquisition and labels
def seq5():
    sys = pp.Opts()
    seq = Sequence(sys)
    rf, gz, gzr = pp.make_sinc_pulse(flip_angle=math.pi / 8, duration=1e-3, slice_thickness=3e-3, return_gz=True)
    gx = pp.make_trapezoid(channel='x', flat_area=32 * 1 / 0.3, flat_time=32 * 1e-4, system=sys)
    adc = pp.make_adc(num_samples=32, duration=gx.flat_time, delay=gx.rise_time, system=sys)
    gx_pre = pp.make_trapezoid(channel='x', area=-gx.area / 2, duration=1e-3, system=sys)
    phase_areas = -(np.arange(32) - 32 / 2) * (1 / 0.3)

    seq.add_block(pp.make_label(label='LIN', type='SET', value=0), pp.make_label(label='SLC', type='SET', value=0))
    seq.add_block(pp.make_adc(num_samples=1000, duration=1e-3), pp.make_label(label='NOISE', type='SET', value=True))
    seq.add_block(pp.make_label(label='NOISE', type='SET', value=False))
    seq.add_block(pp.make_delay(sys.rf_dead_time))

    for pe in range(32):
        gy_pre = pp.make_trapezoid(channel='y', area=phase_areas[pe], duration=1e-3, system=sys)

        seq.add_block(rf, gz)
        seq.add_block(gx_pre, gy_pre, gzr)
        seq.add_block(gx, adc, pp.make_label(label='LIN', type='SET', value=pe))

        gy_pre.amplitude = -gy_pre.amplitude
        seq.add_block(gx_pre, gy_pre, pp.make_delay(10e-3))

    return seq


# Basic GRE sequence with Soft Delay
def seq6():
    seq = Sequence()

    for i in range(10):
        seq.add_block(pp.make_block_pulse(math.pi / 8, duration=1e-3))
        seq.add_block(pp.make_trapezoid('x', area=1000))
        seq.add_block(pp.make_trapezoid('y', area=-500 + i * 100))
        seq.add_block(pp.make_trapezoid('x', area=-500))
        seq.add_block(pp.make_soft_delay(numID=0, hint='TE', offset=1, factor=1.0, default_duration=10e-6))
        seq.add_block(
            pp.make_trapezoid('x', area=1000, duration=10e-3),
            pp.make_adc(num_samples=100, duration=10e-3),
        )

    return seq


# List of all sequence functions that will be tested with the test functions below.
sequence_zoo = [seq_make_gauss_pulses, seq_make_sinc_pulses, seq_make_block_pulses, seq1, seq2, seq3, seq4, seq5, seq6]


# List of example sequences in pypulseq/seq_examples/scripts/ to add as
# sequence tests.
seq_examples = [
    'write_gre',
    'write_gre_label',
    'write_gre_label_softdelay',
    'write_haste',
    'write_radial_gre',
    'write_tse',
    'write_epi',
    'write_epi_label',
    'write_epi_se',
    'write_epi_se_rs',
    'write_mprage',
    'write_ute',
]

# Create a seq_func for each example script and add it to the list of sequences.
# Defining a new function ensures that pytest understands the name is
# e.g. `write_gre` instead of `main`.
# Derive the relative path from the test file to the examples folder
examples_dir = Path(__file__).resolve().parents[1] / 'examples' / 'scripts'


def make_test_func(example):
    def test_func(module_name=f'examples.{example}'):
        spec = importlib.util.spec_from_file_location(module_name, examples_dir / f'{example}.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.main()

    test_func.__name__ = example

    return test_func


for example in seq_examples:
    sequence_zoo.append(make_test_func(example))


# Main Sequence test class
# Note that pytest creates a new instance of the TestSequence class for each
# test. To prevent calling seq_func() over and over, we assign it as a class
# variable, and by using parametrize with scope='class' the order of the tests
# is per seq_func instead of per test.
@pytest.mark.parametrize('seq_func', sequence_zoo, scope='class')
class TestSequence:
    # Base test that just runs the sequence function and keeps the result
    # for the next tests.
    def test_sequence(self, seq_func):
        # Reset TestSequence.seq in case seq_func throws an exception (the
        # other tests will still run, but will result in AttributeErrors)
        TestSequence.seq = None
        TestSequence.seq = seq_func()

    # This "test" rewrites the expected .seq output files when SAVE_EXPECTED is
    # set in the environment variables.
    # E.g. in a unix-based system, run: SAVE_EXPECTED=1 pytest test_sequence.py
    @pytest.mark.skipif(not os.environ.get('SAVE_EXPECTED'), reason='Only save sequence files when requested')
    def test_save_expected(self, seq_func):
        seq_name = str(seq_func.__name__)
        TestSequence.seq.write(expected_output_path / (seq_name + '.seq'))

    # Test sequence.plot() method
    def test_plot(self, seq_func):
        if seq_func.__name__ in ['seq1', 'seq2', 'seq3', 'seq4', 'seq5']:
            with patch('matplotlib.pyplot.show'):
                TestSequence.seq.plot()
                TestSequence.seq.plot(show_blocks=True)
                TestSequence.seq.plot(time_range=(0, 1e-3))
                TestSequence.seq.plot(time_disp='ms')
                TestSequence.seq.plot(grad_disp='mT/m')
                plt.close('all')

    # Test sequence.test_report() method
    def test_test_report(self, seq_func):
        if seq_func.__name__ in seq_examples or seq_func.__name__ in ['seq2', 'seq3', 'seq4', 'seq5', 'seq6']:
            report = TestSequence.seq.test_report()
            assert isinstance(report, str), 'test_report() did not return a string'
            assert len(report) > 0, 'test_report() returned an empty string'

    # Test whether the sequence is the approximately the same after writing a .seq
    # file and reading it back in.
    def test_writeread(self, seq_func, tmp_path, compare_seq_file):
        seq_name = str(seq_func.__name__)
        output_filename = tmp_path / (seq_name + '.seq')

        seq = TestSequence.seq

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
        assert list(seq2.block_events.keys()) == list(seq.block_events.keys()), 'Sequence block IDs are not identical'
        for block_counter in seq.block_events:
            block_orig = seq.get_block(block_counter)
            block_compare = seq2.get_block(block_counter)

            # if hasattr(block_orig, 'rf') and hasattr(block_orig.rf, 'use'):
            #     from copy import deepcopy

            #     block_orig = deepcopy(block_orig)
            #     block_orig.rf.use = 'undefined'

            assert block_compare == Approx(block_orig, abs=1e-5, rel=1e-5), f'Block {block_counter} does not match'

        # Test for approximate equality of all gradient waveforms
        for a, b, channel in zip(seq2.get_gradients(), seq.get_gradients(), ['x', 'y', 'z'], strict=False):
            if a is None and b is None:
                continue
            assert a is not None and b is not None

            # TODO: C[0] is slope of gradient, on the order of max_slew? So expect abs rounding errors in range of 1e2?
            assert a.x == Approx(b.x, abs=1e-5, rel=1e-5), (
                f'Time axis of gradient waveform for channel {channel} does not match'
            )
            assert a.c[0] == Approx(b.c[0], abs=1e2, rel=1e-3), (
                f'First-order coefficients of piecewise-polynomial gradient waveform for channel {channel} do not match'
            )
            assert a.c[1] == Approx(b.c[1], abs=1e-5, rel=1e-5), (
                f'Zero-order coefficients of piecewise-polynomial gradient waveform for channel {channel} do not match'
            )

        # Restore RF use for k-space calculation
        for block_counter in seq.block_events:
            block_orig = seq.get_block(block_counter)
            # if hasattr(block_orig, 'rf') and hasattr(block_orig.rf, 'use'):
            #     block_compare = seq2.get_block(block_counter)
            #     block_compare.rf.use = block_orig.rf.use

        # Test for approximate equality of kspace calculation
        assert seq2.calculate_kspace() == Approx(seq.calculate_kspace(), abs=1e-1, nan_ok=True)

        # Test whether labels are the same
        labels_seq = seq.evaluate_labels(evolution='blocks')
        labels_seq2 = seq2.evaluate_labels(evolution='blocks')

        assert labels_seq.keys() == labels_seq2.keys(), 'Sequences do not contain the same set of labels'

        for label in labels_seq:
            assert (labels_seq[label] == labels_seq2[label]).all(), f'Label {label} does not match'

    # Test whether the sequence is approximately the same after recreating it by
    # getting all blocks with get_block and inserting them into a new sequence
    # with add_block.
    # NOTE: In order to keep the order of shapes the same, sequences need to
    #       put RF events before gradient events and order arbitrary/extended
    #       gradient events in X, Y, Z order when passing them to
    #       seq.add_block(...). i.e. seq.add_block(rf, gx, gy, gz)
    def test_recreate(self, seq_func):  # noqa: ARG002
        seq = TestSequence.seq

        # Insert blocks from sequence into a new sequence
        seq2 = pp.Sequence(system=seq.system)
        for b in seq.block_events:
            seq2.add_block(seq.get_block(b))

        # Test for approximate equality of all blocks
        assert list(seq2.block_events.keys()) == list(seq.block_events.keys()), 'Sequence block IDs are not identical'
        for block_counter in seq.block_events:
            assert seq2.get_block(block_counter) == Approx(seq.get_block(block_counter), abs=1e-9, rel=1e-9), (
                f'Block {block_counter} does not match'
            )

        # Test for approximate equality of all gradient waveforms
        for a, b, channel in zip(seq2.get_gradients(), seq.get_gradients(), ['x', 'y', 'z'], strict=False):
            if a is None and b is None:
                continue
            assert a is not None and b is not None

            assert a.x == Approx(b.x, abs=1e-9, rel=1e-9), (
                f'Time axis of gradient waveform for channel {channel} does not match'
            )
            assert a.c[0] == Approx(b.c[0], abs=1e-9, rel=1e-9), (
                f'First-order coefficients of piecewise-polynomial gradient waveform for channel {channel} do not match'
            )
            assert a.c[1] == Approx(b.c[1], abs=1e-9, rel=1e-9), (
                f'Zero-order coefficients of piecewise-polynomial gradient waveform for channel {channel} do not match'
            )

        # Test for approximate equality of kspace calculation
        assert seq2.calculate_kspace() == Approx(seq.calculate_kspace(), abs=1e-6, nan_ok=True)

        # Test whether labels are the same
        labels_seq = seq.evaluate_labels(evolution='blocks')
        labels_seq2 = seq2.evaluate_labels(evolution='blocks')

        assert labels_seq.keys() == labels_seq2.keys(), 'Sequences do not contain the same set of labels'

        for label in labels_seq:
            assert (labels_seq[label] == labels_seq2[label]).all(), f'Label {label} does not match'
