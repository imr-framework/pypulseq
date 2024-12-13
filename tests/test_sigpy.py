# sms - check MB
# slr - check slice profile

import importlib.util

import numpy as np
import pypulseq as pp
import pytest
from pypulseq.opts import Opts
from pypulseq.sigpy_pulse_opts import SigpyPulseOpts


def test_sigpy_import():
    if importlib.util.find_spec('pypulseq.make_sigpy_pulse'):
        # Attempt to import and ensure no issues
        pass
    else:
        with pytest.raises(ModuleNotFoundError, match='SigPy is not installed.'):
            raise ModuleNotFoundError('SigPy is not installed.')


@pytest.mark.sigpy
def test_slr():
    import sigpy.mri.rf as rf
    from pypulseq.make_sigpy_pulse import sigpy_n_seq

    print('Testing SLR design')

    time_bw_product = 4
    slice_thickness = 3e-3  # Slice thickness
    flip_angle = np.pi / 2
    # Set system limits
    system = Opts(
        max_grad=32,
        grad_unit='mT/m',
        max_slew=130,
        slew_unit='T/m/s',
        rf_ringdown_time=30e-6,
        rf_dead_time=100e-6,
    )
    pulse_cfg = SigpyPulseOpts(
        pulse_type='slr',
        ptype='st',
        ftype='ls',
        d1=0.01,
        d2=0.01,
        cancel_alpha_phs=False,
        n_bands=3,
        band_sep=20,
        phs_0_pt='None',
    )
    rfp, gz, _, pulse = sigpy_n_seq(
        flip_angle=flip_angle,
        system=system,
        duration=3e-3,
        slice_thickness=slice_thickness,
        time_bw_product=4,
        return_gz=True,
        pulse_cfg=pulse_cfg,
        plot=False,
    )

    seq = pp.Sequence()
    seq.add_block(rfp)

    [a, b] = rf.sim.abrm(
        pulse,
        np.arange(-20 * time_bw_product, 20 * time_bw_product, 40 * time_bw_product / 2000),
        True,
    )
    mag_xy = 2 * np.multiply(np.conj(a), b)
    # pl.LinePlot(Mxy)
    # print(np.sum(np.abs(Mxy)))
    # peaks, dict = sis.find_peaks(np.abs(Mxy),threshold=0.5, plateau_size=40)
    plateau_widths = np.sum(np.abs(mag_xy) > 0.8)
    assert plateau_widths == 29


@pytest.mark.sigpy
def test_sms():
    import sigpy.mri.rf as rf
    from pypulseq.make_sigpy_pulse import sigpy_n_seq

    print('Testing SMS design')

    time_bw_product = 4
    slice_thickness = 3e-3  # Slice thickness
    flip_angle = np.pi / 2
    n_bands = 3
    # Set system limits
    system = Opts(
        max_grad=32,
        grad_unit='mT/m',
        max_slew=130,
        slew_unit='T/m/s',
        rf_ringdown_time=30e-6,
        rf_dead_time=100e-6,
    )
    pulse_cfg = SigpyPulseOpts(
        pulse_type='sms',
        ptype='st',
        ftype='ls',
        d1=0.01,
        d2=0.01,
        cancel_alpha_phs=False,
        n_bands=n_bands,
        band_sep=20,
        phs_0_pt='None',
    )
    rfp, gz, _, pulse = sigpy_n_seq(
        flip_angle=flip_angle,
        system=system,
        duration=3e-3,
        slice_thickness=slice_thickness,
        time_bw_product=4,
        return_gz=True,
        pulse_cfg=pulse_cfg,
        plot=False,
    )

    seq = pp.Sequence()
    seq.add_block(rfp)

    [a, b] = rf.sim.abrm(
        pulse,
        np.arange(-20 * time_bw_product, 20 * time_bw_product, 40 * time_bw_product / 2000),
        True,
    )
    mag_xy = 2 * np.multiply(np.conj(a), b)
    # pl.LinePlot(Mxy)
    # print(np.sum(np.abs(Mxy)))
    # peaks, dict = sis.find_peaks(np.abs(Mxy),threshold=0.5, plateau_size=40)
    plateau_widths = np.sum(np.abs(mag_xy) > 0.8)
    # if slr has 29 > 0.8, then sms with MB = n_bands
    assert (29 * n_bands) == plateau_widths
