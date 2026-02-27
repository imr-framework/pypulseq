import numpy as np
import pypulseq as pp
import pytest

from conftest import get_rotation_matrix

# Gradient definitions used in tests
gx_trap = pp.make_trapezoid('x', area=1000, duration=1e-3)
gx_extended = pp.make_extended_trapezoid('x', amplitudes=[0, 100000, 0], times=[0, 1e-4, 2e-4])
gx_extended_delay = pp.make_extended_trapezoid('x', amplitudes=[0, 100000, 0], times=[1e-4, 2e-4, 3e-4])
gx_endshigh = pp.make_extended_trapezoid('x', amplitudes=[0, 100000, 100000], times=[0, 1e-4, 2e-4])
gx_startshigh = pp.make_extended_trapezoid('x', amplitudes=[100000, 100000, 0], times=[0, 1e-4, 2e-4])
gx_startshigh2 = pp.make_extended_trapezoid('x', amplitudes=[200000, 100000, 0], times=[0, 1e-4, 2e-4])
gx_allhigh = pp.make_extended_trapezoid('x', amplitudes=[100000, 100000, 100000], times=[0, 1e-4, 2e-4])
delay = pp.make_delay(1e-3)

# Rotations
rotmat = pp.make_rotation(get_rotation_matrix('z', np.deg2rad(90.0)))
eye = pp.make_rotation(get_rotation_matrix('z', np.deg2rad(0.0)))

## Test gradient continuity checks in add_block


def test_gradient_continuity1():
    # Trap followed by extended gradient: No error
    seq = pp.Sequence()
    seq.add_block(gx_trap)
    seq.add_block(gx_extended)
    seq.add_block(gx_trap)


def test_gradient_continuity2():
    # Trap followed by non-zero gradient
    with pytest.raises(RuntimeError):
        seq = pp.Sequence()
        seq.add_block(gx_trap)
        seq.add_block(gx_startshigh)  # raises


def test_gradient_continuity3():
    # Gradient starts at non-zero in first block
    with pytest.raises(RuntimeError):
        seq = pp.Sequence()
        seq.add_block(gx_startshigh)  # raises


def test_gradient_continuity4():
    # Gradient starts and ends at non-zero
    with pytest.raises(RuntimeError):
        seq = pp.Sequence()
        seq.add_block(delay)
        seq.add_block(gx_allhigh)


def test_gradient_continuity5():
    # Gradient starts at zero and has a delay: No error
    seq = pp.Sequence()
    seq.add_block(gx_extended_delay)


def test_gradient_continuity6():
    # Gradient starts at non-zero in other blocks
    with pytest.raises(RuntimeError):
        seq = pp.Sequence()
        seq.add_block(delay)
        seq.add_block(gx_startshigh)  # raises


def test_gradient_continuity7():
    # Gradient ends high and is followed by empty block
    with pytest.raises(RuntimeError):
        seq = pp.Sequence()
        seq.add_block(gx_endshigh)
        seq.add_block(delay)  # raises


def test_gradient_continuity8():
    # Gradient ends high and is followed by trapezoid
    with pytest.raises(RuntimeError):
        seq = pp.Sequence()
        seq.add_block(gx_endshigh)
        seq.add_block(gx_trap)  # raises


def test_gradient_continuity9():
    # Gradient ends high and is followed by connecting gradient: No error
    seq = pp.Sequence()
    seq.add_block(gx_endshigh)
    seq.add_block(gx_startshigh)


def test_gradient_continuity10():
    # Gradient in last block ends high: No error, this is caught by seq.write()
    seq = pp.Sequence()
    seq.add_block(gx_endshigh)


def test_gradient_continuity11():
    # Non-zero, but non-connecting gradients
    with pytest.raises(RuntimeError):
        seq = pp.Sequence()
        seq.add_block(gx_endshigh)
        seq.add_block(gx_startshigh2)


## Test gradient continuity checks in set_block


def test_gradient_continuity_setblock1():
    # Use set_block to insert gradient
    with pytest.raises(RuntimeError):
        seq = pp.Sequence()
        seq.add_block(delay)
        seq.add_block(delay)
        seq.add_block(delay)

        seq.set_block(1, gx_startshigh)


def test_gradient_continuity_setblock2():
    with pytest.raises(RuntimeError):
        seq = pp.Sequence()
        seq.add_block(delay)
        seq.add_block(delay)
        seq.add_block(delay)

        seq.set_block(2, gx_startshigh)


def test_gradient_continuity_setblock3():
    with pytest.raises(RuntimeError):
        seq = pp.Sequence()
        seq.add_block(delay)
        seq.add_block(delay)
        seq.add_block(delay)

        seq.set_block(3, gx_startshigh)


def test_gradient_continuity_setblock4():
    # Overwrite valid gradient with empty block
    with pytest.raises(RuntimeError):
        seq = pp.Sequence()
        seq.add_block(gx_endshigh)
        seq.add_block(gx_allhigh)
        seq.add_block(gx_startshigh)

        seq.set_block(2, delay)


def test_gradient_continuity_setblock5():
    # Overwrite valid gradient with gradient that is valid on one side
    with pytest.raises(RuntimeError):
        seq = pp.Sequence()
        seq.add_block(gx_endshigh)
        seq.add_block(gx_allhigh)
        seq.add_block(gx_startshigh)

        seq.set_block(2, gx_startshigh)


def test_gradient_continuity_setblock6():
    # Add new gradient with non-contiguous block numbering
    with pytest.raises(RuntimeError):
        seq = pp.Sequence()
        seq.add_block(gx_endshigh)
        seq.add_block(gx_allhigh)
        seq.add_block(gx_startshigh)

        seq.set_block(6, gx_startshigh)


def test_gradient_continuity_setblock7():
    # Valid sequence with non-contiguous block numbering
    seq = pp.Sequence()
    seq.set_block(10, gx_endshigh)
    seq.set_block(5, gx_allhigh)
    seq.set_block(7, gx_startshigh)

    assert list(seq.block_events.keys()) == [10, 5, 7]


def test_gradient_continuity_setblock8():
    # Add new gradient after valid override
    seq = pp.Sequence()
    seq.add_block(gx_endshigh)
    seq.add_block(gx_allhigh)
    seq.add_block(gx_startshigh)
    seq.set_block(2, gx_allhigh)
    seq.add_block(gx_endshigh)
    seq.add_block(gx_allhigh)
    seq.add_block(gx_startshigh)


# Rotations
def test_gradient_continuity_rot1():
    # Trap followed by extended gradient: No error
    seq = pp.Sequence()
    seq.add_block(gx_trap, eye)
    seq.add_block(gx_extended, eye)
    seq.add_block(gx_trap, eye)


def test_gradient_continuity_rot2():
    # Trap followed by non-zero gradient
    with pytest.raises(RuntimeError):
        seq = pp.Sequence()
        seq.add_block(gx_trap, eye)
        seq.add_block(gx_startshigh, eye)  # raises


def test_gradient_continuity_rot3():
    # Gradient starts at non-zero in first block
    with pytest.raises(RuntimeError):
        seq = pp.Sequence()
        seq.add_block(gx_startshigh, eye)  # raises


def test_gradient_continuity_rot4():
    # Gradient starts and ends at non-zero
    with pytest.raises(RuntimeError):
        seq = pp.Sequence()
        seq.add_block(delay)
        seq.add_block(gx_allhigh, eye)


def test_gradient_continuity_rot5():
    # Gradient starts at zero and has a delay: No error
    seq = pp.Sequence()
    seq.add_block(gx_extended_delay, eye)


def test_gradient_continuity_rot6():
    # Gradient starts at non-zero in other blocks
    with pytest.raises(RuntimeError):
        seq = pp.Sequence()
        seq.add_block(delay)
        seq.add_block(gx_startshigh, eye)  # raises


def test_gradient_continuity_rot7():
    # Gradient ends high and is followed by empty block
    with pytest.raises(RuntimeError):
        seq = pp.Sequence()
        seq.add_block(gx_endshigh, eye)
        seq.add_block(delay)  # raises


def test_gradient_continuity_rot8():
    # Gradient ends high and is followed by trapezoid
    with pytest.raises(RuntimeError):
        seq = pp.Sequence()
        seq.add_block(gx_endshigh, eye)
        seq.add_block(gx_trap)  # raises


def test_gradient_continuity_rot9():
    # Gradient ends high and is followed by connecting gradient: No error
    seq = pp.Sequence()
    seq.add_block(gx_endshigh, eye)
    seq.add_block(gx_startshigh, eye)


def test_gradient_continuity_rot10():
    # Gradient in last block ends high: No error, this is caught by seq.write()
    seq = pp.Sequence()
    seq.add_block(gx_endshigh, eye)


def test_gradient_continuity_rot11():
    # Non-zero, but non-connecting gradients
    with pytest.raises(RuntimeError):
        seq = pp.Sequence()
        seq.add_block(gx_endshigh, eye)
        seq.add_block(gx_startshigh2, eye)


def test_gradient_continuity_rot12():
    # Non-zero, both grad are rotated by the same angle: No error
    seq = pp.Sequence()
    seq.add_block(gx_endshigh, rotmat)
    seq.add_block(gx_startshigh, rotmat)


def test_gradient_continuity_rot13():
    # Non-zero, new grad has different rotation from previous
    with pytest.raises(RuntimeError):
        seq = pp.Sequence()
        seq.add_block(gx_endshigh, eye)
        seq.add_block(gx_startshigh, rotmat)


def test_gradient_continuity_rot14():
    # Non-zero, new grad has different rotation from previous
    with pytest.raises(RuntimeError):
        seq = pp.Sequence()
        seq.add_block(gx_endshigh, rotmat)
        seq.add_block(gx_startshigh, eye)


## Test gradient continuity checks in set_block


def test_gradient_continuity_setblock_rot1():
    # Use set_block to insert gradient
    with pytest.raises(RuntimeError):
        seq = pp.Sequence()
        seq.add_block(delay)
        seq.add_block(delay)
        seq.add_block(delay)

        seq.set_block(1, gx_startshigh, eye)


def test_gradient_continuity_setblock_rot2():
    with pytest.raises(RuntimeError):
        seq = pp.Sequence()
        seq.add_block(delay)
        seq.add_block(delay)
        seq.add_block(delay)

        seq.set_block(2, gx_startshigh, eye)


def test_gradient_continuity_setblock_rot3():
    with pytest.raises(RuntimeError):
        seq = pp.Sequence()
        seq.add_block(delay)
        seq.add_block(delay)
        seq.add_block(delay)

        seq.set_block(3, gx_startshigh, eye)


def test_gradient_continuity_setblock_rot4():
    # Overwrite valid gradient with empty block
    with pytest.raises(RuntimeError):
        seq = pp.Sequence()
        seq.add_block(gx_endshigh, eye)
        seq.add_block(gx_allhigh, eye)
        seq.add_block(gx_startshigh, eye)

        seq.set_block(2, delay)


def test_gradient_continuity_setblock_rot5():
    # Overwrite valid gradient with gradient that is valid on one side
    with pytest.raises(RuntimeError):
        seq = pp.Sequence()
        seq.add_block(gx_endshigh, eye)
        seq.add_block(gx_allhigh, eye)
        seq.add_block(gx_startshigh, eye)

        seq.set_block(2, gx_startshigh, eye)


def test_gradient_continuity_setblock_rot6():
    # Add new gradient with non-contiguous block numbering
    with pytest.raises(RuntimeError):
        seq = pp.Sequence()
        seq.add_block(gx_endshigh, eye)
        seq.add_block(gx_allhigh, eye)
        seq.add_block(gx_startshigh, eye)

        seq.set_block(6, gx_startshigh, eye)


def test_gradient_continuity_setblock_rot7():
    # Valid sequence with non-contiguous block numbering
    seq = pp.Sequence()
    seq.set_block(10, gx_endshigh, eye)
    seq.set_block(5, gx_allhigh, eye)
    seq.set_block(7, gx_startshigh, eye)

    assert list(seq.block_events.keys()) == [10, 5, 7]


def test_gradient_continuity_setblock_rot8():
    # Overwrite valid gradient with rotated gradient
    with pytest.raises(RuntimeError):
        seq = pp.Sequence()
        seq.add_block(gx_endshigh, eye)
        seq.add_block(gx_allhigh, eye)
        seq.add_block(gx_startshigh, eye)

        seq.set_block(1, gx_endshigh, rotmat)


def test_gradient_continuity_setblock_rot9():
    # Overwrite valid gradient with rotated gradient
    with pytest.raises(RuntimeError):
        seq = pp.Sequence()
        seq.add_block(gx_endshigh, eye)
        seq.add_block(gx_allhigh, eye)
        seq.add_block(gx_startshigh, eye)

        seq.set_block(2, gx_allhigh, rotmat)


def test_gradient_continuity_setblock_rot10():
    # Overwrite valid gradient with rotated gradient
    with pytest.raises(RuntimeError):
        seq = pp.Sequence()
        seq.add_block(gx_endshigh, eye)
        seq.add_block(gx_allhigh, eye)
        seq.add_block(gx_startshigh, eye)

        seq.set_block(3, gx_startshigh, rotmat)


def test_gradient_continuity_setblock_rot11():
    # Overwrite valid non-rotated gradient with rotated gradient
    with pytest.raises(RuntimeError):
        seq = pp.Sequence()
        seq.add_block(gx_endshigh)
        seq.add_block(gx_allhigh)
        seq.add_block(gx_startshigh)

        seq.set_block(1, gx_endshigh, rotmat)


def test_gradient_continuity_setblock_rot12():
    # Overwrite valid gradient with rotated gradient
    with pytest.raises(RuntimeError):
        seq = pp.Sequence()
        seq.add_block(gx_endshigh)
        seq.add_block(gx_allhigh)
        seq.add_block(gx_startshigh)

        seq.set_block(2, gx_allhigh, rotmat)


def test_gradient_continuity_setblock_rot13():
    # Overwrite valid gradient with rotated gradient
    with pytest.raises(RuntimeError):
        seq = pp.Sequence()
        seq.add_block(gx_endshigh)
        seq.add_block(gx_allhigh)
        seq.add_block(gx_startshigh)

        seq.set_block(3, gx_startshigh, rotmat)
