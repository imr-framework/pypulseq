import numpy as np
import pypulseq as pp
import pytest


def _make_simple_gre_sequence() -> pp.Sequence:
    system = pp.Opts()
    seq = pp.Sequence(system)

    rf, gz, gzr = pp.make_sinc_pulse(
        flip_angle=np.deg2rad(10),
        slice_thickness=3e-3,
        system=system,
        return_gz=True,
        delay=system.rf_dead_time,
    )

    gx = pp.make_trapezoid(channel='x', flat_area=64 / 0.256, flat_time=3.2e-3, system=system)
    gx_pre = pp.make_trapezoid(channel='x', area=-gx.area / 2, duration=1e-3, system=system)
    adc = pp.make_adc(num_samples=64, duration=gx.flat_time, delay=gx.rise_time, system=system)

    seq.add_block(rf, gz)
    seq.add_block(gx_pre, gzr)
    seq.add_block(gx, adc)

    return seq


def test_calculate_kspace_legacy_output_shape():
    seq = _make_simple_gre_sequence()

    k_traj_adc, k_traj, t_excitation, t_refocusing, t_adc = seq.calculate_kspace()

    assert isinstance(t_excitation, list)
    assert isinstance(t_refocusing, list)
    assert isinstance(t_adc, np.ndarray)

    assert k_traj_adc.ndim == 2
    assert k_traj.ndim == 2
    assert k_traj_adc.shape[0] == k_traj.shape[0]
    assert k_traj_adc.shape[1] == t_adc.shape[0]
    assert len(t_excitation) > 0


def test_calculate_kspace_dict_output_contains_t_ktraj_and_matches_legacy():
    seq = _make_simple_gre_sequence()

    legacy = seq.calculate_kspace()
    result = seq.calculate_kspace(output_as_dict=True)

    assert set(result.keys()) == {
        'k_traj_adc',
        'k_traj',
        't_ktraj',
        't_excitation',
        't_refocusing',
        't_adc',
    }

    assert isinstance(result['t_ktraj'], np.ndarray)
    assert result['t_ktraj'].ndim == 1
    assert result['t_ktraj'].shape[0] >= result['t_adc'].shape[0]

    k_traj_adc, k_traj, t_excitation, t_refocusing, t_adc = legacy

    np.testing.assert_allclose(result['k_traj_adc'], k_traj_adc)
    np.testing.assert_allclose(result['k_traj'], k_traj)
    np.testing.assert_allclose(result['t_adc'], t_adc)
    assert result['t_excitation'] == t_excitation
    assert result['t_refocusing'] == t_refocusing


def test_calculate_kspacePP_deprecated_passthrough_output_as_dict():
    seq = _make_simple_gre_sequence()

    with pytest.warns(DeprecationWarning):
        result = seq.calculate_kspacePP(output_as_dict=True)

    assert 't_ktraj' in result
    assert isinstance(result['t_ktraj'], np.ndarray)
