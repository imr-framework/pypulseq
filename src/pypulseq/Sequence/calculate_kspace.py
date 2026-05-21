from typing import Any, List, Tuple, Union

import numpy as np

import pypulseq as pp
from pypulseq import eps


def calculate_kspace(
    seq: pp.Sequence,
    trajectory_delay: Union[float, List[float], np.ndarray] = 0.0,
    gradient_offset: Union[float, List[float], np.ndarray] = 0.0,
    output_as_dict: bool = False,
) -> Union[
    Tuple[np.ndarray, np.ndarray, List[float], List[float], np.ndarray],
    dict,
]:
    """
    Calculates the k-space trajectory of the entire pulse sequence.

    Parameters
    ----------
    seq : pypulseq.Sequence
        Sequence object to calculate the k-space trajectory for.
    trajectory_delay : float or list or numpy.ndarray, default=0
        Compensation factor in seconds (s) to align ADC and gradients in the reconstruction.
        If trajectory_delay is a single value, this value will be used for all gradient channels.
        If trajectory_delay is a list or array, it is expected to have the same length as the number of gradient
        channels and the first element is applied to the first gradient channel, the second to the second, and so on.
    gradient_offset : float or list or numpy.ndarray, default=0
        Simulates background gradients (specified in Hz/m)
        If gradient_offset is a single value, this value will be used for all gradient channels.
        If gradient_offset is a list or array, it is expected to have the same length as the number of gradient
        channels and the first element is applied to the first gradient channel, the second to the second, and so on.
    output_as_dict : bool, default=False
        If True, return a dict containing all available outputs (including `t_ktraj`).
        If False, return the legacy 5-tuple output for backwards compatibility.

    Returns
    -------
    k_traj_adc : numpy.array
        K-space trajectory sampled at `t_adc` timepoints.
    k_traj : numpy.array
        K-space trajectory of the entire pulse sequence.
    t_excitation : List[float]
        Excitation timepoints.
    t_refocusing : List[float]
        Refocusing timepoints.
    t_adc : numpy.array
        Sampling timepoints.

    When `output_as_dict=True`, the returned dictionary contains the additional key `t_ktraj`.
    """
    if np.any(np.abs(trajectory_delay) > 100e-6):
        raise Warning(f'Trajectory delay of {trajectory_delay * 1e6} us is suspiciously high')

    def _build_time_axis(
        time_candidates: np.ndarray,
        t_excitation: List[float],
        t_refocusing: List[float],
        t_adc: np.ndarray,
        total_duration: float,
    ) -> Tuple[np.ndarray, float, float]:
        t_acc = 1e-10  # Temporal accuracy
        t_acc_inv = 1 / t_acc

        t_ktraj = t_acc * np.unique(
            np.round(
                t_acc_inv
                * np.array(
                    [
                        *time_candidates,
                        0,
                        *np.asarray(t_excitation) - 2 * seq.rf_raster_time,
                        *np.asarray(t_excitation) - seq.rf_raster_time,
                        *t_excitation,
                        *np.asarray(t_refocusing) - seq.rf_raster_time,
                        *t_refocusing,
                        *t_adc,
                        total_duration,
                    ]
                )
            )
        )
        return t_ktraj, t_acc, t_acc_inv

    def _sample_gradient_moments(
        grad_moments_pp: List[Union[Any, None]],
        t_ktraj: np.ndarray,
        t_acc: float,
        t_acc_inv: float,
    ) -> np.ndarray:
        k_traj = np.zeros((n_grad_channels, len(t_ktraj)))
        for i in range(n_grad_channels):
            if grad_moments_pp[i] is None:
                continue

            idx_in_support = np.where(
                np.logical_and(
                    t_ktraj >= t_acc * round(t_acc_inv * grad_moments_pp[i].x[0]),
                    t_ktraj <= t_acc * round(t_acc_inv * grad_moments_pp[i].x[-1]),
                )
            )[0]
            k_traj[i, idx_in_support] = grad_moments_pp[i](t_ktraj[idx_in_support])
            if t_ktraj[idx_in_support[-1]] < t_ktraj[-1]:
                k_traj[i, idx_in_support[-1] + 1 :] = k_traj[i, idx_in_support[-1]]

        return k_traj

    def _apply_excitation_and_refocusing_resets(
        k_traj: np.ndarray,
        t_ktraj: np.ndarray,
        t_excitation: List[float],
        t_refocusing: List[float],
        t_adc: np.ndarray,
        t_acc: float,
        t_acc_inv: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Find indices of excitation and refocusing events
        i_excitation = np.searchsorted(t_ktraj, t_acc * np.round(t_acc_inv * np.asarray(t_excitation)))
        i_refocusing = np.searchsorted(t_ktraj, t_acc * np.round(t_acc_inv * np.asarray(t_refocusing)))
        i_adc = np.searchsorted(t_ktraj, t_acc * np.round(t_acc_inv * np.asarray(t_adc)))

        i_periods = np.unique([0, *i_excitation, *i_refocusing, len(t_ktraj) - 1])
        next_excitation_idx = 0 if len(i_excitation) > 0 else -1
        next_refocusing_idx = 0 if len(i_refocusing) > 0 else -1

        # Convert gradient moments to k-space positions.
        # `k_offset` maintains k-space resets at excitation and sign inversions at refocusing.
        k_offset = -k_traj[:, 0]
        for i in range(len(i_periods) - 1):
            i_period = i_periods[i]
            i_period_end = i_periods[i + 1]
            if next_excitation_idx >= 0 and i_excitation[next_excitation_idx] == i_period:
                if abs(t_ktraj[i_period] - t_excitation[next_excitation_idx]) > t_acc:
                    raise Warning(
                        f'abs(t_ktraj[i_period]-t_excitation[next_excitation_idx]) < {t_acc} failed for next_excitation_idx={next_excitation_idx} error={t_ktraj[i_period] - t_excitation[next_excitation_idx]}'
                    )
                k_offset = -k_traj[:, i_period]
                if i_period > 0:
                    # Use nans to mark the excitation points since they interrupt the plots
                    k_traj[:, i_period - 1] = np.nan
                # -1 on len(i_excitation) for 0-based indexing
                next_excitation_idx = min(len(i_excitation) - 1, next_excitation_idx + 1)
            elif next_refocusing_idx >= 0 and i_refocusing[next_refocusing_idx] == i_period:
                k_offset = -2 * k_traj[:, i_period] - k_offset
                # -1 on len(i_excitation) for 0-based indexing
                next_refocusing_idx = min(len(i_refocusing) - 1, next_refocusing_idx + 1)

            k_traj[:, i_period:i_period_end] = k_traj[:, i_period:i_period_end] + k_offset[:, None]

        k_traj[:, i_period_end] = k_traj[:, i_period_end] + k_offset
        return k_traj, i_adc, i_excitation, i_refocusing

    total_duration = sum(seq.block_durations.values())

    # Get RF and ADC related timing information
    t_excitation, fp_excitation, t_refocusing, _ = seq.rf_times()
    t_adc, _ = seq.adc_times()

    # Convert gradient data to piecewise polynomials
    grad_waveforms_pp = seq.get_gradients(trajectory_delay, gradient_offset)
    n_grad_channels = len(grad_waveforms_pp)

    # Calculate slice positions.
    # For now we entirely rely on the excitation -- ignoring complicated interleaved refocused sequences
    if len(t_excitation) > 0:
        # Position in x, y, z
        slice_pos = np.zeros((n_grad_channels, len(t_excitation)))
        for j in range(n_grad_channels):
            if grad_waveforms_pp[j] is None:
                slice_pos[j] = np.nan
            else:
                # Estimate slice position from RF frequency offset divided by slice-select gradient amplitude.
                # Check for divisions by zero to avoid numpy warning.
                grad_at_excitation = np.array(grad_waveforms_pp[j](t_excitation))
                slice_pos[j, grad_at_excitation != 0.0] = (
                    fp_excitation[0, grad_at_excitation != 0.0] / grad_at_excitation[grad_at_excitation != 0.0]
                )
                slice_pos[j, grad_at_excitation == 0.0] = np.nan

        slice_pos[~np.isfinite(slice_pos)] = 0  # Reset undefined to 0
    else:
        slice_pos = []

    # Integrate waveforms as piecewise polynomials (pp) to produce gradient moments
    grad_moments_pp = []
    time_candidates = []
    for i in range(n_grad_channels):
        if grad_waveforms_pp[i] is None:
            grad_moments_pp.append(None)
            continue

        grad_moments_pp.append(grad_waveforms_pp[i].antiderivative())
        time_candidates.append(grad_moments_pp[i].x)
        # "Sample" ramps for display purposes.  Otherwise piecewise-linear display (plot) fails
        ramp_idx = np.flatnonzero(np.abs(grad_moments_pp[i].c[0, :]) > 1e-7 * seq.system.max_slew)

        # Do nothing if there are no ramps
        if ramp_idx.shape[0] == 0:
            continue

        starts = np.int64(np.floor((grad_moments_pp[i].x[ramp_idx] + eps) / seq.grad_raster_time))
        ends = np.int64(np.ceil((grad_moments_pp[i].x[ramp_idx + 1] - eps) / seq.grad_raster_time))

        # Create all ranges starts[0]:ends[0], starts[1]:ends[1], etc.
        lengths = ends - starts + 1
        inds = np.ones((lengths).sum())
        # Calculate output index where each range will start
        start_inds = np.cumsum(np.concatenate(([0], lengths[:-1])))
        # Create element-wise differences that will cumsum into
        # the final indices: [starts[0], 1, 1, starts[1]-starts[0]-lengths[0]+1, 1, etc.]
        inds[start_inds] = np.concatenate(([starts[0]], np.diff(starts) - lengths[:-1] + 1))

        time_candidates.append(np.cumsum(inds) * seq.grad_raster_time)
    if len(time_candidates) > 0:
        time_candidates = np.concatenate(time_candidates)
    else:
        time_candidates = np.zeros(0)

    # Create a time axis that covers all interesting event boundaries and gradient ramp transitions.
    t_ktraj, t_acc, t_acc_inv = _build_time_axis(time_candidates, t_excitation, t_refocusing, t_adc, total_duration)

    # Sample the integrated gradients (gradient moments) on the time axis to get a continuous k-space trajectory.
    k_traj = _sample_gradient_moments(grad_moments_pp, t_ktraj, t_acc, t_acc_inv)

    # Apply k-space resets at excitation and inversions at refocusing.
    k_traj, i_adc, _, _ = _apply_excitation_and_refocusing_resets(
        k_traj,
        t_ktraj,
        t_excitation,
        t_refocusing,
        t_adc,
        t_acc,
        t_acc_inv,
    )
    k_traj_adc = k_traj[:, i_adc]

    if output_as_dict:
        return {
            'k_traj_adc': k_traj_adc,
            'k_traj': k_traj,
            't_ktraj': t_ktraj,
            't_excitation': t_excitation,
            't_refocusing': t_refocusing,
            't_adc': t_adc,
        }

    return k_traj_adc, k_traj, t_excitation, t_refocusing, t_adc
