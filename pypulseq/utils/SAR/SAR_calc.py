# Copyright of the Board of Trustees of Columbia University in the City of New York

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib
import scipy.io as sio
from scipy import interpolate

from pypulseq.Sequence.sequence import Sequence
from pypulseq.calc_duration import calc_duration


def __calc_SAR(Q, I):
    """
        Compute the SAR output for a given Q matrix and I current values.

        Parameters
        ----------
        Q : numpy.ndarray
        I : numpy.ndarray

        Returns
        -------
        SAR : numpy.ndarray
           contains the SAR value for a particular Q matrix
        """

    if len(I.shape) == 1:  # Just to fit the multi-transmit case for now, ToDo
        I = np.matlib.repmat(I, Q.shape[0], 1)  # Nc x Nt

    Ifact = np.divide(np.matmul(I, np.matrix(I).getH()), I.shape[1])
    SAR_temp = np.multiply(Q, Ifact)
    SAR = np.abs(np.sum(SAR_temp[:]))
    return SAR


def __loadQ():
    """
    Load Q matrix that is precomputed based on the VHM model for 8 channels.

    Returns
    -------
    Qtmf, Qhmf : numpy.ndarray
        Contains the Q-matrix, GSAR head and body for now.
    """

    # Load relevant Q matrices computed from the model - this code will be integrated later - starting from E fields
    Qpath = str(Path(__file__).parent / 'QGlobal.mat')
    Qmat = sio.loadmat(Qpath)
    Q = Qmat['Q']
    val = Q[0, 0]

    Qtmf = val['Qtmf']
    Qhmf = val['Qhmf']
    return Qtmf, Qhmf


def __SAR_from_seq(seq, Qtmf, Qhmf):
    """
    Compute global whole body and head only SAR values.

    Parameters
    ----------
    seq : Sequence
    Qtmf : numpy.ndarray
    Qhmf : numpy.ndarray

    Returns
    -------
    SAR_wbg_vec : numpy.ndarray
    SAR_hg_vec : numpy.ndarray
    t_vec : numpy.ndarray
        Contains the Q-matrix, GSAR head and body for now.
    """

    # Identify RF blocks and compute SAR - 10 seconds must be less than twice and 6 minutes must be less than
    # 4 (WB) and 3.2 (head-20)
    block_events = seq.block_events
    num_events = len(block_events)
    t_vec = np.zeros(num_events)
    SAR_wbg_vec = np.zeros(t_vec.shape)
    SAR_hg_vec = np.zeros(t_vec.shape)
    t_prev = 0

    for iB in block_events:
        block = seq.get_block(iB)
        block_dur = calc_duration(block)
        t_vec[iB - 1] = t_prev + block_dur
        t_prev = t_vec[iB - 1]
        if hasattr(block, 'rf'):  # has rf
            rf = block.rf
            t = rf.t
            signal = rf.signal
            # This rf could be parallel transmit as well
            SAR_wbg_vec[iB] = __calc_SAR(Qtmf, signal)
            SAR_hg_vec[iB] = __calc_SAR(Qhmf, signal)

    return SAR_wbg_vec, SAR_hg_vec, t_vec


def __SAR_interp(SAR, t):
    """
    Interpolate SAR values for one second resolution.

    Parameters
    ----------
    SAR : numpy.ndarray
    t : numpy.ndarray

    Returns
    -------
    __SAR_interp : numpy.ndarray
    tsec : numpy.ndarray
        Interpolated values of SAR for a temporal resolution of 1 second
    """
    tsec = np.arange(1, np.floor(t[-1]) + 1, 1)
    f = interpolate.interp1d(t, SAR)
    SARinterp = f(tsec)
    return SARinterp, tsec


def __SAR_lims_check(SARwbg_lim_s, SARhg_lim_s, tsec):
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
        SixMinThresh_wbg = 4
        TenSecThresh_wbg = 8

        SixMinThresh_hg = 3.2
        TenSecThresh_hg = 6.4

        SARwbg_lim_app = np.concatenate((np.zeros(5), SARwbg_lim_s, np.zeros(5)), axis=0)
        SARhg_lim_app = np.concatenate((np.zeros(5), SARhg_lim_s, np.zeros(5)), axis=0)

        SAR_wbg_tensec = __do_sw_sar(SARwbg_lim_app, tsec, 10)  # < 2  SARmax
        SAR_hg_tensec = __do_sw_sar(SARhg_lim_app, tsec, 10)  # < 2 SARmax
        SAR_wbg_tensec_peak = np.round(np.max(SAR_wbg_tensec), 2)
        SAR_hg_tensec_peak = np.round(np.max(SAR_hg_tensec), 2)

        if ((np.max(SAR_wbg_tensec) > TenSecThresh_wbg) or (np.max(SAR_hg_tensec) > TenSecThresh_hg)):
            print('Pulse exceeding 10 second Global SAR limits, increase TR')
        SAR_wbg_sixmin = 'NA'
        SAR_hg_sixmin = 'NA'
        SAR_wbg_sixmin_peak = 'NA'
        SAR_hg_sixmin_peak = 'NA'

        if tsec[-1] > 600:
            SARwbg_lim_app = np.concatenate((np.zeros(300), SARwbg_lim_s, np.zeros(300)), axis=0)
            SARhg_lim_app = np.concatenate((np.zeros(300), SARhg_lim_s, np.zeros(300)), axis=0)

            SAR_hg_sixmin = __do_sw_sar(SARhg_lim_app, tsec, 600)
            SAR_wbg_sixmin = __do_sw_sar(SARwbg_lim_app, tsec, 600)
            SAR_wbg_sixmin_peak = np.round(np.max(SAR_wbg_sixmin), 2)
            SAR_hg_sixmin_peak = np.round(np.max(SAR_hg_sixmin), 2)

            if ((np.max(SAR_hg_sixmin) > SixMinThresh_wbg) or (np.max(SAR_hg_sixmin) > SixMinThresh_hg)):
                print('Pulse exceeding 10 second Global SAR limits, increase TR')
    else:
        print('Need at least 10 seconds worth of sequence to calculate SAR')
        SAR_wbg_tensec = 'NA'
        SAR_wbg_sixmin = 'NA'
        SAR_hg_tensec = "NA"
        SAR_hg_sixmin = "NA"
        SAR_wbg_sixmin_peak = 'NA'
        SAR_hg_sixmin_peak = 'NA'
        SAR_wbg_tensec_peak = 'NA'
        SAR_hg_tensec_peak = 'NA'

    return SAR_wbg_tensec, SAR_wbg_sixmin, SAR_hg_tensec, SAR_hg_sixmin, SAR_wbg_sixmin_peak, SAR_hg_sixmin_peak, SAR_wbg_tensec_peak, SAR_hg_tensec_peak


def __do_sw_sar(SAR, tsec, t):
    """
    Compute a sliding window average of SAR values.

    Parameters
    ----------
    SAR : numpy.ndarray
    tsec : numpy.ndarray
    t : numpy.ndarray

    Returns
    -------
    SAR_timeavag : numpy.ndarray
        Sliding window time average of SAR values
    """
    SAR_timeavg = np.zeros(len(tsec) + int(t))
    for instant in range(int(t / 2), int(t / 2) + (int(tsec[-1]))):  # better to go from  -sw / 2: sw / 2
        SAR_timeavg[instant] = sum(SAR[range(instant - int(t / 2), instant + int(t / 2) - 1)]) / t
    SAR_timeavg = SAR_timeavg[int(t / 2):int(t / 2) + (int(tsec[-1]))]
    return SAR_timeavg


def calc_SAR(seq):
    """
    Compute Global SAR values on the `seq` object for head and whole body over the specified time averages.

    Parameters
    ----------
    seq : Sequence
        pypulseq `Sequence` object for which global SAR values will be computed.
    """

    if isinstance(seq, str):
        seq_obj = Sequence()
        seq_obj.read(seq)
        seq = seq_obj

    Qtmf, Qhmf = __loadQ()
    SARwbg, SARhg, t_vec = __SAR_from_seq(seq, Qtmf, Qhmf)
    SARwbg_lim, tsec = __SAR_interp(SARwbg, t_vec)
    SARhg_lim, tsec = __SAR_interp(SARhg, t_vec)
    SAR_wbg_tensec, SAR_wbg_sixmin, SAR_hg_tensec, SAR_hg_sixmin, SAR_wbg_sixmin_peak, SAR_hg_sixmin_peak, SAR_wbg_tensec_peak, SAR_hg_tensec_peak = __SAR_lims_check(
        SARwbg_lim, SARhg_lim, tsec)

    # Plot 10 sec average SAR
    if (tsec[-1] > 10):
        plt.plot(tsec, SAR_wbg_tensec, label='Whole Body: 10sec')
        plt.plot(tsec, SAR_hg_tensec, label='Head only: 10sec')

        # plt.plot(t_vec, SARwbg, label='Whole Body - instant')
        # plt.plot(t_vec, SARhg, label='Whole Body - instant')

        plt.xlabel('Time (s)')
        plt.ylabel('SAR (W/kg)')
        plt.title('Global SAR  - Mass Normalized -  Whole body and head only')
        ax = plt.subplot(111)

        # ax.legend()
        plt.grid(True)
        plt.show()
