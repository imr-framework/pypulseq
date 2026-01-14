from __future__ import annotations

import typing
from math import comb
from typing import Tuple, Union

import numpy as np
from scipy.interpolate import PPoly

if typing.TYPE_CHECKING:
    from pypulseq.Sequence.sequence import Sequence


def calc_moments_btensor(
    seq: Sequence,
    calcB: bool = True,
    calcm1: bool = False,
    calcm2: bool = False,
    calcm3: bool = False,
    Ndummy: int = 0,
) -> Tuple[np.ndarray, ...]:
    """
    B-tensor calculator for Sequence object.

    Parameters
    ----------
    seq : Sequence
        The Pulseq sequence object to plot.
    calcB : bool, default=True
        Toggle B-Tensor calculation.
    calcm1 : bool, default=False
        Toggle first-order gradient moment calculation.
    calcm2 : bool, default=False
        Toggle second-order gradient moment calculation.
    calcm3: bool, default=False
        Toggle third-order gradient moment calculation.
    Ndummy : int, default=0
        Number of dummy scans to be skipped for the calculation.

    Returns
    -------
    B : np.ndarray
        The B-tensor for each of the R sequence repetitions.
        It is shaped (R, 3, 3).
    m1 : np.ndarray
        First-order gradient moment along (x, y, z) for each of
        the R sequence repetitions. It is shaped (R, 3).
    m2 : np.ndarray
        Second-order gradient moment along (x, y, z) for each of
        the R sequence repetitions. It is shaped (R, 3).
    m3 : np.ndarray
        Third-order gradient moment along (x, y, z) for each of
        the R sequence repetitions. It is shaped (R, 3).

    Notes
    -----
    - Uses 2*pi scaling like MATLAB (phase/radians convention).
    - Assumes one excitation defines each repetition/TR segment.
    - TE per repetition is computed as t_echo = 2*t_refocusing - t_excitation
      (same limitation as MATLAB: not correct for more complex refocusing schemes).
    """
    if calcB is False and calcm1 is False and calcm2 is False and calcm3 is False:
        raise ValueError('At least one output among B, m1, m2, m3 must be requested')
    if Ndummy < 0:
        raise ValueError('Ndummy must be >= 0')

    # Get RF timing
    t_excitation, _, t_refocusing, _ = seq.rf_times()
    R_all = len(t_excitation)
    if R_all == 0:
        raise ValueError('No excitations found in sequence (t_excitation is empty).')

    if len(t_refocusing) < R_all:
        raise ValueError(
            'Not enough refocusing pulses found for calc_moments_btensor. '
            f'Need at least {R_all}, found {len(t_refocusing)}.'
        )

    if Ndummy >= R_all:
        raise ValueError(f'Ndummy={Ndummy} must be < number of excitations ({R_all}).')

    # Gradients as PPoly (x,y,z)
    gw_pp = seq.get_gradients()
    if len(gw_pp) < 3:
        raise ValueError('Expected 3 gradient channels from seq.get_gradients().')

    # Build global knot grid like MATLAB unique([breaks, t_exc, t_ref, t_echo])
    grad_knots = []
    for g in gw_pp[:3]:
        if g is not None:
            grad_knots.append(np.asarray(g.x, dtype=float))
    grad_knots = np.concatenate(grad_knots) if grad_knots else np.array([], dtype=float)

    t_exc = np.asarray(t_excitation, dtype=float)
    t_ref = np.asarray(t_refocusing[:R_all], dtype=float)
    t_echo = 2.0 * t_ref - t_exc  # MATLAB TODO caveat
    tn = _unique_sorted(np.concatenate([grad_knots, t_exc, t_ref, t_echo]))

    # If gradients are empty and tn degenerate, still need a minimal grid.
    if tn.size < 2:
        raise ValueError('Unable to build a valid time grid for gradients (tn has <2 points).')

    g_lin = [_fill_pp_coefs(gw_pp[i], tn) for i in range(3)]

    # repetitions after dummy
    R = R_all - Ndummy

    B = np.zeros((R, 3, 3), dtype=float)
    m1 = np.zeros((R, 3), dtype=float)
    m2 = np.zeros((R, 3), dtype=float)
    m3 = np.zeros((R, 3), dtype=float)

    seq_end = float(tn[-1])

    for r in range(R):
        k = r + Ndummy  # absolute repetition index in t_exc/t_ref arrays

        t0 = float(t_exc[k])
        t1 = float(t_exc[k + 1]) if (k < R_all - 1) else seq_end
        te = float(t_echo[k])
        tref = float(t_ref[k])

        if te < t0 or te > t1:
            raise ValueError(f'Computed TE={te} is outside repetition segment [{t0},{t1}] for repetition {r} (k={k}).')

        # restrict gradients to repetition segment and apply refocusing sign flip
        g_tr = [_restrict_piecewise_linear(g_lin[i], t0, t1) for i in range(3)]
        g_tr = [_flip_after(g_tr[i], tref) for i in range(3)]

        # integration grid inside repetition, up to TE
        grids = []
        for i in range(3):
            grids.append(g_tr[i].x if g_tr[i] is not None else np.array([t0, te]))

        t_int = _unique_sorted(np.concatenate(([t0, te], *grids)))
        t_int = t_int[(t_int >= t0) & (t_int <= te)]
        if t_int.size < 2:
            t_int = np.array([t0, te], dtype=float)

        # sample gradients on integration grid
        g_samp = []
        for i in range(3):
            if g_tr[i] is None:
                g_samp.append(np.zeros_like(t_int))
            else:
                vals = g_tr[i](t_int)
                g_samp.append(np.nan_to_num(vals, nan=0.0))

        # q(t) = 2*pi * integral g dt
        q_samp = []
        for i in range(3):
            area = _cumtrapz_on_breakpoints(g_samp[i], t_int)
            q_samp.append(2.0 * np.pi * area)

        if calcB:
            for i in range(3):
                for j in range(3):
                    B[r, i, j] = _integrate_piecewise_linear_product(q_samp[i], q_samp[j], t_int)

        if calcm1:
            for i in range(3):
                m1[r, i] = 2.0 * np.pi * _integrate_g_tn(g_samp[i], t_int, n=1)

        if calcm2:
            for i in range(3):
                m2[r, i] = 2.0 * np.pi * _integrate_g_tn(g_samp[i], t_int, n=2)

        if calcm3:
            for i in range(3):
                m3[r, i] = 2.0 * np.pi * _integrate_g_tn(g_samp[i], t_int, n=3)

    return B, m1, m2, m3


def _unique_sorted(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return x
    return np.unique(x)


def _fill_pp_coefs(pp: Union[PPoly, None], xn: np.ndarray) -> Union[PPoly, None]:
    """
    Direct translation of MATLAB's fillPpCoefs function.
    """
    if pp is None:
        return None

    if xn.size < 2:
        return None

    x_old = pp.x
    c_old = pp.c
    order = c_old.shape[0]
    n_intervals = len(xn) - 1

    # Find indices
    idx = np.zeros(n_intervals, dtype=int)
    for i in range(n_intervals):
        matches = np.where(np.abs(x_old[:-1] - xn[i]) < 1e-14)[0]
        if len(matches) > 0:
            idx[i] = matches[0] + 1
        else:
            idx[i] = 0

    new_coefs = np.zeros((order, n_intervals))

    for i in range(n_intervals):
        if idx[i] > 0:
            # Simple copy
            new_coefs[:, i] = c_old[:, idx[i] - 1]
        elif i > 0:
            # Taylor expansion with proper binomial coefficient
            dx = xn[i] - xn[i - 1]
            for k in range(order):
                for l in range(k + 1):
                    new_coefs[order - 1 - l, i] += new_coefs[order - 1 - k, i - 1] * comb(k, l) * (dx ** (k - l))

    return PPoly(new_coefs, xn, extrapolate=False)


def _restrict_piecewise_linear(pp_lin: Union[PPoly, None], a: float, b: float) -> Union[PPoly, None]:
    if pp_lin is None:
        return None
    if b <= a:
        return None
    knots = pp_lin.x
    grid = _unique_sorted(np.concatenate(([a], knots[(knots > a) & (knots < b)], [b])))
    return _fill_pp_coefs(pp_lin, grid)  # ← Changed


def _flip_after(pp_lin: Union[PPoly, None], t_flip: float) -> Union[PPoly, None]:
    if pp_lin is None:
        return None

    x = pp_lin.x
    if t_flip <= x[0]:
        return PPoly(-pp_lin.c, x, extrapolate=False)
    if t_flip >= x[-1]:
        return pp_lin

    grid = _unique_sorted(np.concatenate((x, [t_flip])))
    pp2 = _fill_pp_coefs(pp_lin, grid)  # ← Changed

    x2 = pp2.x
    mask = x2[:-1] >= t_flip
    c2 = pp2.c.copy()
    c2[:, mask] *= -1
    return PPoly(c2, x2, extrapolate=False)


def _cumtrapz_on_breakpoints(y: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Cumulative integral assuming y is piecewise-linear between breakpoints t.
    Trapezoid on breakpoints is exact for linear segments.
    """
    dt = np.diff(t)
    inc = 0.5 * (y[:-1] + y[1:]) * dt
    return np.concatenate(([0.0], np.cumsum(inc)))


def _integrate_piecewise_linear_product(u: np.ndarray, v: np.ndarray, t: np.ndarray) -> float:
    """
    Exact integral of u(t)*v(t) dt when u and v are piecewise-linear between breakpoints t.

    On each interval, u=a_u*t+b_u, v=a_v*t+b_v => product quadratic.
    """
    dt = np.diff(t)
    if dt.size == 0:
        return 0.0

    au = np.divide(np.diff(u), dt, out=np.zeros_like(dt), where=dt != 0) # codespell:ignore
    av = np.divide(np.diff(v), dt, out=np.zeros_like(dt), where=dt != 0) # codespell:ignore
    bu = u[:-1] - au * t[:-1] # codespell:ignore
    bv = v[:-1] - av * t[:-1] # codespell:ignore

    A = au * av # codespell:ignore
    B = au * bv + av * bu # codespell:ignore
    C = bu * bv # codespell:ignore

    t0 = t[:-1]
    t1 = t[1:]
    return float(np.sum(A * (t1**3 - t0**3) / 3.0 + B * (t1**2 - t0**2) / 2.0 + C * (t1 - t0)))


def _integrate_g_tn(g: np.ndarray, t: np.ndarray, n: int) -> float:
    """
    Exact integral of g(t)*t^n dt when g is piecewise-linear between breakpoints t.
    g(t)=a*t+b => g*t^n = a*t^(n+1) + b*t^n
    """
    dt = np.diff(t)
    if dt.size == 0:
        return 0.0

    a = np.divide(np.diff(g), dt, out=np.zeros_like(dt), where=dt != 0)
    b = g[:-1] - a * t[:-1]
    t0 = t[:-1]
    t1 = t[1:]

    return float(np.sum(a * (t1 ** (n + 2) - t0 ** (n + 2)) / (n + 2) + b * (t1 ** (n + 1) - t0 ** (n + 1)) / (n + 1)))
