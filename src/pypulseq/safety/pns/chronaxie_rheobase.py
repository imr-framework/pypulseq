"""Peripheral nerve stimulation (PNS) model ported from Jon-Fredrik Nielsen's MATLAB implementation.

Original implementation: https://github.com/toppeMRI/toppe/blob/main/%2Btoppe/pns.m
"""

from __future__ import annotations

import warnings

import numpy as np


def _isodd(n: int) -> bool:
    return n % 2 > 0


def pns(
    grad: np.ndarray,
    coil: str,
    *,
    print_output: bool = True,
    gdt: float = 4e-6,
    plot: bool = True,
    model: int = 1,
) -> tuple[np.ndarray, np.ndarray, float, float, float, np.ndarray | None, np.ndarray | None]:
    """Calculate PNS threshold from gradient waveforms.

    Args:
        grad: Gradient array in [T/m]. Expected shapes:
            (dimensions, points, repetitions) or (points, repetitions, dimensions).
        coil: One of {'xrm', 'xrmw', 'whole', 'zoom', 'hrmb', 'hrmw', 'magnus'}.
        print_output: Print summary values to stdout.
        gdt: Gradient update time in seconds.
        plot: Plot waveforms and thresholds.
        model: 1 for time-domain convolution, 2 for frequency-domain approach.

    Returns:
        (PThresh, pt, PTmax, gmax, smax, t, f)
    """
    grad = np.asarray(grad)

    if grad.ndim == 3:
        n1, n2, n3 = grad.shape
        if (n1 > n2) and (n1 > n3):
            grad = np.transpose(grad, (2, 0, 1))

    if np.iscomplexobj(grad):
        if grad.shape[0] != 1:
            raise ValueError('grad complex and grad.shape[0] != 1')
        grad = np.concatenate((np.real(grad), np.imag(grad)), axis=0)

    if np.isnan(grad).any():
        warnings.warn('grad contains NaN', RuntimeWarning, stacklevel=2)
    if np.isinf(grad).any():
        warnings.warn('grad contains inf', RuntimeWarning, stacklevel=2)

    coil_params = {
        'xrmw': (360e-6, 20.0, 0.324),
        'xrm': (334e-6, 23.4, 0.333),
        'whole': (370e-6, 23.7, 0.344),
        'zoom': (354e-6, 29.1, 0.309),
        'hrmb': (359e-6, 26.5, 0.370),
        'hrmw': (642.4e-6, 17.9, 0.310),
        'magnus': (611e-6, 55.2, 0.324),
    }
    try:
        chronaxie, rheobase, alpha = coil_params[coil.lower()]
    except KeyError as exc:
        raise ValueError(f'gradient coil ({coil}) unknown') from exc

    if gdt is None:
        gdt = 4e-6

    if (chronaxie > 700e-6) or (chronaxie < 200e-6):
        warnings.warn(f'chronaxie={chronaxie}; typical values in help', RuntimeWarning, stacklevel=2)
    if (rheobase > 80) or (rheobase < 17):
        warnings.warn(f'rheobase={rheobase}; typical values in help', RuntimeWarning, stacklevel=2)
    if (alpha > 0.4) or (alpha < 0.3):
        warnings.warn(f'alpha={alpha}; typical values in help', RuntimeWarning, stacklevel=2)

    if model == 2:
        if _isodd(grad.shape[1]):
            grad = np.concatenate((grad, np.zeros((grad.shape[0], 1, grad.shape[2]), dtype=grad.dtype)), axis=1)
        zf2 = int(np.ceil(7e-3 / gdt / 2) * 2)
        if grad.shape[1] < zf2:
            dozf = True
        else:
            dozf = np.any(grad[:, -zf2:, 0] != 0)
        if dozf:
            grad = np.concatenate(
                (grad, np.zeros((grad.shape[0], zf2, grad.shape[2]), dtype=grad.dtype)),
                axis=1,
            )

    if grad.ndim > 3:
        raise ValueError('grad.ndim > 3')
    if grad.ndim < 2:
        raise ValueError('grad.ndim < 2')
    if grad.ndim == 2:
        grad = grad[:, :, np.newaxis]

    n1, n2, n3 = grad.shape
    if n1 > 3:
        raise ValueError('grad.shape[0] > 3')
    if n2 == 1:
        raise ValueError('grad.shape[1] == 1')
    if n3 > n2:
        warnings.warn('grad.shape[2] > grad.shape[1]', RuntimeWarning, stacklevel=2)

    srmin = rheobase / alpha
    sr = np.concatenate((np.zeros((n1, 1, n3), dtype=float), np.diff(grad, axis=1)), axis=1) / gdt

    decay = None
    sr_fd = None
    if model == 1:
        pt = np.zeros((n1, n2, n3), dtype=float)
        scale = 100.0 * gdt * chronaxie / srmin
        for l3 in range(n3):
            for l2 in range(n2):
                idx = np.arange(l2, -1, -1)
                denom = (chronaxie + (idx + 0.5) * gdt) ** 2
                pt[:, l2, l3] = scale * np.sum(sr[:, : l2 + 1, l3] / denom[np.newaxis, :], axis=1)
    elif model == 2:
        decay = np.fft.fftshift(np.fft.fft(1.0 / (((np.arange(n2) + 0.5) * gdt + chronaxie) ** 2)))
        sr_fd = np.fft.fftshift(np.fft.fft(sr, axis=1), axes=1)
        rep_decay = np.broadcast_to(decay[np.newaxis, :, np.newaxis], (n1, n2, n3))
        pt = (100.0 * gdt * chronaxie / srmin) * np.fft.ifft(
            np.fft.ifftshift(sr_fd * rep_decay, axes=1),
            axis=1,
        )
        pt = np.real_if_close(pt, tol=1000)
    else:
        raise ValueError(f'model ({model}) unknown')

    if np.isnan(pt).any():
        warnings.warn('pt contains NaN', RuntimeWarning, stacklevel=2)
    if np.isinf(pt).any():
        warnings.warn('pt contains inf', RuntimeWarning, stacklevel=2)

    pthresh = np.sqrt(np.sum(pt**2, axis=0, keepdims=True))

    t = None
    f = None
    rsos_grad = np.sqrt(np.sum(grad[:, :, 0] ** 2, axis=0))
    rsos_sr = np.sqrt(np.sum(sr[:, :, 0] ** 2, axis=0))

    if model == 2:
        f = (np.arange(-n2 / 2, n2 / 2) / n2 / gdt) * 1e-3

    if plot:
        import matplotlib.pyplot as plt

        t = np.arange(n2) * gdt * 1e3
        fig, axs = plt.subplots(2, 2, figsize=(11, 7))

        for dim in range(n1):
            axs[0, 0].plot(t, grad[dim, :, 0] * 1e3, label=f'G{dim + 1}')
        axs[0, 0].plot(t, rsos_grad * 1e3, 'r--', label='RSS')
        axs[0, 0].set_xlabel('time [ms]')
        axs[0, 0].set_ylabel('grad [mT/m]')
        axs[0, 0].legend(loc='upper right')

        for dim in range(n1):
            axs[1, 0].plot(t, sr[dim, :, 0], label=f'S{dim + 1}')
        axs[1, 0].plot(t, rsos_sr, 'r--', label='RSS')
        axs[1, 0].set_xlabel('time [ms]')
        axs[1, 0].set_ylabel('slewrate [T/m/s]')
        axs[1, 0].legend(loc='upper right')

        for dim in range(n1):
            axs[1, 1].plot(t, pt[dim, :, 0], label=f'pt{dim + 1}')
        axs[1, 1].plot(t, pthresh[0, :, 0], 'r--', label='PThresh')
        axs[1, 1].plot([t[0], t[-1]], [100, 100], 'm:', label='+100%')
        axs[1, 1].plot([t[0], t[-1]], [80, 80], 'm:', label='+80%')
        axs[1, 1].plot([t[0], t[-1]], [-100, -100], 'm:', label='-100%')
        axs[1, 1].plot([t[0], t[-1]], [-80, -80], 'm:', label='-80%')
        ymax = 1.05 * np.max(np.concatenate((pthresh[0, :, 0], np.array([100.0]))))
        axs[1, 1].set_xlim(0.0, t[-1])
        axs[1, 1].set_ylim(-ymax, ymax)
        axs[1, 1].set_xlabel('time [ms]')
        axs[1, 1].set_ylabel('PNS threshold [%]')
        axs[1, 1].grid(True)
        axs[1, 1].legend(loc='upper right')

        if model == 2 and sr_fd is not None and decay is not None and f is not None:
            axs[0, 1].plot(
                f,
                np.abs(sr_fd[:, :, 0]).max(axis=0) / np.max(np.abs(sr_fd[:, :, 0])),
                label='|FFT(SR)| max over axes',
            )
            axs[0, 1].plot(f, np.abs(decay) / np.max(np.abs(decay)), 'r', label='|decay|')
            axs[0, 1].set_xlabel('freq [kHz]')
            axs[0, 1].set_ylabel('FFT of decay + slewrate')
            axs[0, 1].set_xlim(-5.0, 5.0)
            axs[0, 1].set_ylim(-0.05, 1.05)
            axs[0, 1].grid(True)
            axs[0, 1].legend(loc='upper right')
        else:
            axs[0, 1].axis('off')

        fig.tight_layout()
        plt.show()

    if print_output:
        print(f'\nchronaxie = {chronaxie * 1e6:g} [us]')
        print(f'rheobase  = {rheobase:g} [T/s]')
        print(f'effective coil length: alpha = {alpha * 1e2:g} [cm]')
        print(f'Stimulation slewrate:  SRmin = {srmin:g} [T/m/s]')
        print(f'Gradient update time:    gdt = {gdt * 1e6:g} [us]')
        print(f'Maximum gradient strength    = {np.max(grad) * 1e3:g} [mT/m]')
        print(f'Maximum slewrate             = {np.max(sr):g} [T/m/s]')

        if n3 == 1:
            print(f'Maximum PThresh = {np.max(np.abs(pthresh)):.4g} [%]')
        else:
            print('Maximum PThresh = ')
            for l3 in range(n3):
                print(f'\tl3={l3 + 1:g}: {np.max(np.abs(pthresh[:, :, l3])):.4g} [%]')
            print(f'\tall: {np.max(np.abs(pthresh)):.4g} [%]')

        pmax = np.max(pthresh)
        if pmax > 80:
            if pmax > 100:
                print('Warning: PThresh exceeding first controlled mode (100%)!!!')
            else:
                print('Warning: PThresh exceeding normal mode (80%)!')

    ptmax = float(np.max(pthresh))
    gmax = float(np.max(rsos_grad))
    smax = float(np.max(rsos_sr))

    return pthresh, pt, ptmax, gmax, smax, t, f


def _build_demo_gradient(points: int = 2000, gdt: float = 4e-6) -> np.ndarray:
    """Create a smooth 3-axis demo gradient waveform in [T/m]."""
    t = np.arange(points) * gdt
    grad = np.zeros((3, points, 1), dtype=float)

    env = np.sin(np.pi * np.linspace(0.0, 1.0, points)) ** 2
    grad[0, :, 0] = 0.020 * np.sin(2 * np.pi * 900 * t) * env
    grad[1, :, 0] = 0.015 * np.sin(2 * np.pi * 1200 * t + np.pi / 4) * env
    grad[2, :, 0] = 0.010 * np.sin(2 * np.pi * 700 * t + np.pi / 2) * env
    return grad


def run_demo() -> None:
    """Run a small end-to-end demo of the PNS model."""
    gdt = 4e-6
    grad = _build_demo_gradient(points=2500, gdt=gdt)
    coil = 'xrm'

    pthresh, pt, ptmax, gmax, smax, t_ms, f_khz = pns(
        grad,
        coil,
        gdt=gdt,
        print_output=True,
        plot=True,
        model=1,
    )

    print('\nDemo summary')
    print(f'coil: {coil}')
    print(f'PThresh shape: {pthresh.shape}')
    print(f'pt shape: {pt.shape}')
    print(f'PTmax: {ptmax:.4f} %')
    print(f'gmax: {gmax * 1e3:.4f} mT/m')
    print(f'smax: {smax:.4f} T/m/s')
    print(f't returned: {t_ms is not None}')
    print(f'f returned: {f_khz is not None}')


if __name__ == '__main__':
    run_demo()
