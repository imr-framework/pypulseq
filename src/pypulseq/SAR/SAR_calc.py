# Copyright of the Board of Trustees of Columbia University in the City of New York
from pathlib import Path
from typing import Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from scipy import interpolate

from pypulseq.calc_duration import calc_duration
from pypulseq.Sequence.sequence import Sequence


def _calc_SAR(Q: np.ndarray, I: np.ndarray) -> np.ndarray:
    """
    Compute the SAR output for a given Q matrix and I current values.

    Parameters
    ----------
    Q : numpy.ndarray
        Q matrix. Refer Graesslin, Ingmar, et al. "A specific absorption rate prediction concept for parallel
        transmission MR." Magnetic resonance in medicine 68.5 (2012): 1664-1674.
    I : numpy.ndarray
        I matrix, capturing the current (in Amps) on each of the transmit channels. Refer Graesslin, Ingmar, et al. "A
        specific absorption rate prediction concept for parallel transmission MR." Magnetic resonance in medicine
        68.5 (2012): 1664-1674.

    Returns
    -------
    SAR : numpy.ndarray
       Contains the SAR value for a particular Q matrix
    """
    if len(I.shape) == 1:  # Just to fit the multi-transmit case for now, TODO
        I = np.tile(I, (Q.shape[0], 1))  # Nc x Nt

    I_fact = np.divide(np.matmul(I, np.conjugate(I).T), I.shape[1])
    SAR_temp = np.multiply(Q, I_fact)
    SAR = np.abs(np.sum(SAR_temp[:]))

    return SAR


def _load_Q() -> Tuple[np.ndarray, np.ndarray]:
    """
    Load Q matrix that is precomputed based on the VHM model for 8 channels. Refer Graesslin, Ingmar, et al. "A
    specific absorption rate prediction concept for parallel transmission MR." Magnetic resonance in medicine 68.5
    (2012): 1664-1674.

    Returns
    -------
    Qtmf, Qhmf : numpy.ndarray
        Contains the Q-matrix of global SAR values for body-mass and head-mass respectively.
    """
    # Load relevant Q matrices computed from the model - this code will be integrated later - starting from E fields
    path_Q = str(Path(__file__).parent / 'QGlobal.mat')
    Q = sio.loadmat(path_Q)
    Q = Q['Q']
    val = Q[0, 0]

    Qtmf = val['Qtmf']
    Qhmf = val['Qhmf']
    return Qtmf, Qhmf


def _SAR_from_seq(seq: Sequence, Qtmf: np.ndarray, Qhmf: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute global whole body and head only SAR values for the given `seq` object.

    Parameters
    ----------
    seq : Sequence
        Sequence object to calculate for which SAR values will be calculated.
    Qtmf : numpy.ndarray
        Q-matrix of global SAR values for body-mass.
    Qhmf : numpy.ndarray
        Q-matrix of global SAR values for head-mass.

    Returns
    -------
    SAR_wbg : numpy.ndarray
        SAR values for body-mass.
    SAR_hg : numpy.ndarray
        SAR values for head-mass.
    t : numpy.ndarray
        Corresponding time points.
    """
    # Identify RF blocks and compute SAR - 10 seconds must be less than twice and 6 minutes must be less than
    # 4 (WB) and 3.2 (head-20)
    block_events = seq.block_events
    num_events = len(block_events)
    t = np.zeros(num_events)
    SAR_wbg = np.zeros(t.shape)
    SAR_hg = np.zeros(t.shape)
    t_prev = 0

    for block_counter in block_events:
        block = seq.get_block(block_counter)
        block_dur = calc_duration(block)
        t[block_counter - 1] = t_prev + block_dur
        t_prev = t[block_counter - 1]
        if hasattr(block, 'rf') and block.rf is not None:  # has rf event
            rf = block.rf
            signal = rf.signal
            # This rf could be parallel transmit as well
            SAR_wbg[block_counter] = _calc_SAR(Qtmf, signal)
            SAR_hg[block_counter] = _calc_SAR(Qhmf, signal)

    return SAR_wbg, SAR_hg, t


def _SAR_interp(SAR: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Interpolate SAR values for one second resolution.

    Parameters
    ----------
    SAR : numpy.ndarray
        SAR values
    t : numpy.ndarray
        Current time points.

    Returns
    -------
    SAR_interp : numpy.ndarray
        Interpolated values of SAR for a temporal resolution of 1 second.
    t_sec : numpy.ndarray
        Time points at 1 second resolution.
    """
    t_sec = np.arange(1, np.floor(t[-1]) + 1, 1)
    f = interpolate.interp1d(t, SAR)
    SAR_interp = f(t_sec)
    return SAR_interp, t_sec


def _SAR_lims_check(
    SARwbg_lim_s, SARhg_lim_s, tsec
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Check for SAR violations as compared to IEC 10 second and 6 minute averages;
    returns SAR values that are interpolated for the fixed IEC time intervals.

    Parameters
    ----------
    SARwbg_lim_s : numpy.ndarray
    SARhg_lim_s : numpy.ndarray
    tsec : numpy.ndarray

    Returns
    -------
    SAR_wbg_tensec : numpy.ndarray
    SAR_wbg_sixmin : numpy.ndarray
    SAR_hg_tensec : numpy.ndarray
    SAR_hg_sixmin : numpy.ndarray
    SAR_wbg_sixmin_peak : numpy.ndarray
    SAR_hg_sixmin_peak : numpy.ndarray
    SAR_wbg_tensec_peak : numpy.ndarray
    SAR_hg_tensec_peak : numpy.ndarray
    """
    if tsec[-1] > 10:
        six_min_threshold_wbg = 4
        ten_sec_threshold_wbg = 8

        six_min_threshold_hg = 3.2
        ten_sec_threshold_hg = 6.4

        SAR_wbg_lim_app = np.concatenate((np.zeros(5), SARwbg_lim_s, np.zeros(5)), axis=0)
        SAR_hg_lim_app = np.concatenate((np.zeros(5), SARhg_lim_s, np.zeros(5)), axis=0)

        SAR_wbg_tensec = _do_sw_sar(SAR_wbg_lim_app, tsec, 10)  # < 2  SARmax
        SAR_hg_tensec = _do_sw_sar(SAR_hg_lim_app, tsec, 10)  # < 2 SARmax
        SAR_wbg_tensec_peak = np.round(np.max(SAR_wbg_tensec), 2)
        SAR_hg_tensec_peak = np.round(np.max(SAR_hg_tensec), 2)

        if (np.max(SAR_wbg_tensec) > ten_sec_threshold_wbg) or (np.max(SAR_hg_tensec) > ten_sec_threshold_hg):
            print('Pulse exceeding 10 second Global SAR limits, increase TR')
        SAR_wbg_sixmin = 'NA'
        SAR_hg_sixmin = 'NA'
        SAR_wbg_sixmin_peak = 'NA'
        SAR_hg_sixmin_peak = 'NA'

        if tsec[-1] > 600:
            SAR_wbg_lim_app = np.concatenate((np.zeros(300), SARwbg_lim_s, np.zeros(300)), axis=0)
            SAR_hg_lim_app = np.concatenate((np.zeros(300), SARhg_lim_s, np.zeros(300)), axis=0)

            SAR_hg_sixmin = _do_sw_sar(SAR_hg_lim_app, tsec, 600)
            SAR_wbg_sixmin = _do_sw_sar(SAR_wbg_lim_app, tsec, 600)
            SAR_wbg_sixmin_peak = np.round(np.max(SAR_wbg_sixmin), 2)
            SAR_hg_sixmin_peak = np.round(np.max(SAR_hg_sixmin), 2)

            if (np.max(SAR_hg_sixmin) > six_min_threshold_wbg) or (np.max(SAR_hg_sixmin) > six_min_threshold_hg):
                print('Pulse exceeding 10 second Global SAR limits, increase TR')
    else:
        print('Need at least 10 seconds worth of sequence to calculate SAR')
        SAR_wbg_tensec = 'NA'
        SAR_wbg_sixmin = 'NA'
        SAR_hg_tensec = 'NA'
        SAR_hg_sixmin = 'NA'
        SAR_wbg_sixmin_peak = 'NA'
        SAR_hg_sixmin_peak = 'NA'
        SAR_wbg_tensec_peak = 'NA'
        SAR_hg_tensec_peak = 'NA'

    return (
        SAR_wbg_tensec,
        SAR_wbg_sixmin,
        SAR_hg_tensec,
        SAR_hg_sixmin,
        SAR_wbg_sixmin_peak,
        SAR_hg_sixmin_peak,
        SAR_wbg_tensec_peak,
        SAR_hg_tensec_peak,
    )


def _do_sw_sar(SAR: np.ndarray, tsec: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Compute a sliding window average of SAR values.

    Parameters
    ----------
    SAR : numpy.ndarray
        SAR values.
    tsec : numpy.ndarray
        Corresponding time points at 1 second resolution.
    t : numpy.ndarray
        Corresponding time points.

    Returns
    -------
    SAR_timeavag : numpy.ndarray
        Sliding window time average of SAR values.
    """
    SAR_time_avg = np.zeros(len(tsec) + int(t))
    for instant in range(int(t / 2), int(t / 2) + (int(tsec[-1]))):  # better to go from  -sw / 2: sw / 2
        SAR_time_avg[instant] = sum(SAR[range(instant - int(t / 2), instant + int(t / 2) - 1)]) / t
    SAR_time_avg = SAR_time_avg[int(t / 2) : int(t / 2) + (int(tsec[-1]))]
    return SAR_time_avg


def calc_SAR(file: Union[str, Path, Sequence]) -> None:
    """
    Compute Global SAR values on the `.seq` object for head and whole body over the specified time averages.

    Parameters
    ----------
    file : str, Path or Sequence
        `.seq` file for which global SAR values will be computed. Can be path to `.seq` file as `str` or `Path`, or the
        `Sequence` object itself.

    Raises
    ------
    ValueError
        If `file` is a `str` or `Path` to the `.seq` file and this file does not exist on disk.
    """
    if isinstance(file, (str, Path)):
        if isinstance(file, str):
            file = Path(file)

        if file.exists() and file.is_file():
            seq_obj = Sequence()
            seq_obj.read(str(file))
            seq_obj = seq_obj
        else:
            raise ValueError('Seq file does not exist.')
    else:
        seq_obj = file

    Q_tmf, Q_hmf = _load_Q()
    SAR_wbg, SAR_hg, t = _SAR_from_seq(seq_obj, Q_tmf, Q_hmf)
    SARwbg_lim, tsec = _SAR_interp(SAR_wbg, t)
    SARhg_lim, tsec = _SAR_interp(SAR_hg, t)
    (
        SAR_wbg_tensec,
        SAR_wbg_sixmin,
        SAR_hg_tensec,
        SAR_hg_sixmin,
        SAR_wbg_sixmin_peak,
        SAR_hg_sixmin_peak,
        SAR_wbg_tensec_peak,
        SAR_hg_tensec_peak,
    ) = _SAR_lims_check(SARwbg_lim, SARhg_lim, tsec)

    # Plot 10 sec average SAR
    if tsec[-1] > 10:
        plt.plot(tsec, SAR_wbg_tensec, 'x-', label='Whole Body: 10sec')
        plt.plot(tsec, SAR_hg_tensec, '.-', label='Head only: 10sec')

        # plt.plot(t, SARwbg, label='Whole Body - instant')
        # plt.plot(t, SARhg, label='Whole Body - instant')

        plt.xlabel('Time (s)')
        plt.ylabel('SAR (W/kg)')
        plt.title('Global SAR  - Mass Normalized -  Whole body and head only')

        plt.legend()
        plt.grid(True)
        plt.show()
