import math
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import numpy.testing as npt
import pypulseq as pp
import pytest
from pypulseq import Sequence

from conftest import Approx, get_rotation_matrix

expected_output_path = Path(__file__).parent / 'expected_output'


# Basic sequence with 0, 30, 45, 60, 90 deg
def seq_make_radial():
    # init sequence
    seq = Sequence()

    # init rf pulse
    rf = pp.make_block_pulse(math.pi / 2, duration=1e-3)

    # init readout
    gread = pp.make_trapezoid('x', area=1000)

    # init angle list
    theta = np.asarray((0.0, 30.0, 45.0, 60.0, 90.0))
    theta = np.deg2rad(theta)
    rot = [get_rotation_matrix('z', th) for th in theta]

    # build sequence
    for n in range(len(theta)):
        seq.add_block(rf)

        # add readouts
        seq.add_block(gread, pp.make_rotation(rot[n]))

    # add 0 again
    seq.add_block(rf)

    # add readouts
    seq.add_block(gread, pp.make_rotation(rot[0]))

    return seq


def seq_make_radial_norotext():
    # init sequence
    seq = Sequence()

    # init rf pulse
    rf = pp.make_block_pulse(math.pi / 2, duration=1e-3)

    # init readout
    gread = pp.make_trapezoid('x', area=1000)

    # init angle list
    theta = np.asarray((0.0, 30.0, 45.0, 60.0, 90.0))
    theta = np.deg2rad(theta)

    # build sequence
    for n in range(len(theta)):
        seq.add_block(rf)

        # add readouts
        seq.add_block(*pp.rotate(gread, angle=theta[n], axis='z'))

    # add 0 again
    seq.add_block(rf)

    # add readouts
    seq.add_block(*pp.rotate(gread, angle=theta[0], axis='z'))

    return seq


# Test sequence
def test_sequence():
    seq = seq_make_radial()
    blocks = np.stack(list(seq.block_events.values()), axis=1)

    # check rf
    rf = (
        1,
        0,
        1,
        0,
        1,
        0,
        1,
        0,
        1,
        0,
        1,
        0,
    )  # alternate between no rf and gaussian pulse
    npt.assert_allclose(blocks[1], rf)

    # check gradients
    g = (
        0,
        1,
        0,
        1,
        0,
        1,
        0,
        1,
        0,
        1,
        0,
        1,
    )  # alternate between no readout and radial spoke
    npt.assert_allclose(blocks[2], g)

    # check extension (we have 0=no rot, 1=0.0deg, 2=30.0deg, 3=45.0deg, 4=60.0deg, 5=90.0deg)
    ext = (
        0,
        1,
        0,
        2,
        0,
        3,
        0,
        4,
        0,
        5,
        0,
        1,
    )  # last event re-use ROTATION[1] = 90deg rotation about z
    npt.assert_allclose(blocks[6], ext)

    # verify that the only extension is ROTATIONS (id=1, string=[ROTATIONS])
    assert len(seq.extension_numeric_idx) == 1
    assert seq.extension_numeric_idx[0] == 1
    assert len(seq.extension_string_idx) == 1
    assert seq.extension_string_idx[0] == 'ROTATIONS'

    assert len(seq.extensions_library.data) == 5
    npt.assert_allclose(seq.extensions_library.data[1], (1, 1, 0))
    npt.assert_allclose(seq.extensions_library.data[2], (1, 2, 0))
    npt.assert_allclose(seq.extensions_library.data[3], (1, 3, 0))
    npt.assert_allclose(seq.extensions_library.data[4], (1, 4, 0))
    npt.assert_allclose(seq.extensions_library.data[5], (1, 5, 0))

    # verify that rotation_events 1-5 contains the correct matrix
    for n in (1, 3, 5, 7, 9, 11):
        b = seq.get_block(n)
        assert b.rf is not None
        assert b.gx is None
        assert b.gy is None
        assert b.gz is None
        assert hasattr(b, 'rotation') is False

    for n in (2, 4, 6, 8, 10, 12):
        b = seq.get_block(n)
        assert b.rf is None
        assert b.gx is not None
        assert b.gy is None
        assert b.gz is None
        assert hasattr(b, 'rotation') is True

    npt.assert_allclose(
        seq.get_block(2).rotation.rot_quaternion.as_matrix(), get_rotation_matrix('z', np.deg2rad(0.0)), atol=1e-12
    )
    npt.assert_allclose(
        seq.get_block(4).rotation.rot_quaternion.as_matrix(), get_rotation_matrix('z', np.deg2rad(30.0)), atol=1e-12
    )
    npt.assert_allclose(
        seq.get_block(6).rotation.rot_quaternion.as_matrix(), get_rotation_matrix('z', np.deg2rad(45.0)), atol=1e-12
    )
    npt.assert_allclose(
        seq.get_block(8).rotation.rot_quaternion.as_matrix(), get_rotation_matrix('z', np.deg2rad(60.0)), atol=1e-12
    )
    npt.assert_allclose(
        seq.get_block(10).rotation.rot_quaternion.as_matrix(), get_rotation_matrix('z', np.deg2rad(90.0)), atol=1e-12
    )
    npt.assert_allclose(
        seq.get_block(12).rotation.rot_quaternion.as_matrix(), get_rotation_matrix('z', np.deg2rad(0.0)), atol=1e-12
    )


# Test again explicit gradient rotation
def test_vs_rotate():
    seq = seq_make_radial()
    seq2 = seq_make_radial_norotext()

    # test waveforms()
    waveforms = seq.waveforms()
    waveforms2 = seq2.waveforms()

    assert len(waveforms) == len(waveforms2)
    for n in range(len(waveforms)):
        npt.assert_allclose(waveforms[n], waveforms2[n])

    # test waveforms_and_times()
    waveforms_and_times = seq.waveforms_and_times()
    waveforms_and_times2 = seq2.waveforms_and_times()

    assert len(waveforms_and_times) == len(waveforms_and_times2)
    for n in range(len(waveforms_and_times)):
        assert len(waveforms_and_times[n]) == len(waveforms_and_times[n])
        for m in range(len(waveforms_and_times[n])):
            npt.assert_allclose(waveforms_and_times[n][m], waveforms_and_times2[n][m])

    # test k-space
    kspace = seq.calculate_kspace()
    kspace2 = seq2.calculate_kspace()

    for n in range(len(kspace)):
        npt.assert_allclose(kspace[n], kspace2[n])


# This "test" rewrites the expected .seq output files when SAVE_EXPECTED is
# set in the environment variables.
# E.g. in a unix-based system, run: SAVE_EXPECTED=1 pytest test_sequence.py
@pytest.mark.skipif(
    not os.environ.get('SAVE_EXPECTED'),
    reason='Only save sequence files when requested',
)
def test_sequence_save_expected():
    # Generate sequence and write to file
    seq = seq_make_radial()
    seq.write(expected_output_path / 'seq_make_radial.seq')


# Test whether a sequence can be plotted.
@patch('matplotlib.pyplot.show')
def test_plot(mock_show):
    seq = seq_make_radial()

    seq.plot()
    seq.plot(show_blocks=True)


# Test whether the sequence is the approximately the same after writing a .seq
# file and reading it back in.
def test_sequence_writeread(tmp_path, compare_seq_file):
    output_filename = tmp_path / 'seq_make_radial.seq'

    # Generate sequence
    seq = seq_make_radial()

    # Write sequence to file
    seq.write(output_filename)

    # Check if written sequence file matches expected sequence file
    compare_seq_file(output_filename, expected_output_path / 'seq_make_radial.seq')

    # Read written sequence file back in
    seq2 = pp.Sequence(system=seq.system)
    seq2.read(output_filename)

    # Clean up written sequence file
    output_filename.unlink()

    # Test for approximate equality of all blocks
    assert set(seq2.block_events.keys()) == set(seq.block_events.keys())
    for block_counter in seq.block_events:
        assert seq2.get_block(block_counter) == Approx(seq.get_block(block_counter), abs=1e-6, rel=1e-5), (
            f'Block {block_counter} does not match'
        )

    # Test for approximate equality of all gradient waveforms
    for a, b in zip(seq2.get_gradients(), seq.get_gradients()):
        if a == None and b == None:
            continue
        if a == None or b == None:
            raise AssertionError()

        if a.x != Approx(b.x, abs=1e-3, rel=1e-3):
            raise AssertionError()
        if a.c != Approx(b.c, abs=1e-3, rel=1e-3):
            raise AssertionError()

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
def test_sequence_recreate(tmp_path):
    # Generate sequence
    seq = seq_make_radial()

    # Insert blocks from sequence into a new sequence
    seq2 = pp.Sequence(system=seq.system)
    for b in seq.block_events:
        seq2.add_block(seq.get_block(b))

    # Test for approximate equality of all blocks
    assert set(seq2.block_events.keys()) == set(seq.block_events.keys())
    for block_counter in seq.block_events:
        assert seq2.get_block(block_counter) == Approx(seq.get_block(block_counter), abs=1e-6, rel=1e-5), (
            f'Block {block_counter} does not match'
        )

    # Test for approximate equality of all gradient waveforms
    for a, b in zip(seq2.get_gradients(), seq.get_gradients()):
        if a == None and b == None:
            continue
        if a == None or b == None:
            raise AssertionError()

        if a.x != Approx(b.x, abs=1e-4, rel=1e-4):
            raise AssertionError()
        if a.c != Approx(b.c, abs=1e-4, rel=1e-4):
            raise AssertionError()

    # Test for approximate equality of kspace calculation
    assert seq2.calculate_kspace() == Approx(seq.calculate_kspace(), abs=1e-6, nan_ok=True)

    # Test whether labels are the same
    labels_seq = seq.evaluate_labels(evolution='blocks')
    labels_seq2 = seq2.evaluate_labels(evolution='blocks')

    assert labels_seq.keys() == labels_seq2.keys(), 'Sequences do not contain the same set of labels'

    for label in labels_seq:
        assert (labels_seq[label] == labels_seq2[label]).all(), f'Label {label} does not match'
