"""
This script computes the global head and body Specific Absoprtion Rate (SAR) values based on the Visible HUman Male model
This assumes an eight channel multi-transmit system with a scaled B1+
Parameters
----------
    .seq --> the sequence file object for which the SAR has to be computed
    Requires coms_server_flask to be running before the unit test is run (i.e.: run coms_server_flask.py first)

Returns
-------
    payload
     - contains the Q-matrix, GSAR head and body for now
     - will include local SAR based on disucssions related to ongoing project

Performs
--------
    IEC checks on SAR resulting from a given sequence file


Author: Sairam Geethanath
Date: 05/07/2019
Version 0.0
Copyright of the Board of Trustees of  Columbia University in the City of New York
"""

from pypulseq.Sequence.sequence import Sequence
from pypulseq.calc_duration import calc_duration
import scipy.io as sio
from scipy import interpolate
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import time


def calc_SAR(Q, I):
    if len(I.shape) == 1:  # Just to fit the multi-transmit case for now, ToDo
        I = np.matlib.repmat(I, Q.shape[0], 1)  # Nc x Nt

    Ifact = np.divide(np.matmul(I, np.matrix(I).getH()), I.shape[1])
    SAR_temp = np.multiply(Q, Ifact)
    SAR = np.abs(np.sum(SAR_temp[:]))
    return SAR


def loadQ():
    # Load relevant Q matrices computed from the model - this code will be integrated later - starting from E fields
    Qmat = sio.loadmat(
        './src/server/rf/tx/SAR_calc/QGlobal.mat')  # Hardcoded for ever, will introduce methods to compute as well but really slow at the moment ToDO
    Q = Qmat['Q']
    val = Q[0, 0]

    Qtmf = val['Qtmf']
    Qhmf = val['Qhmf']
    return Qtmf, Qhmf


def SARfromseq(fname, Qtmf, Qhmf):
    # Read sequence from file path supplied to the method
    obj = Sequence()
    obj.read('./src/server/rf/tx/SAR_calc/' + fname)  # replaced by

    # Identify rf blocks and compute SAR - 10 seconds must be less than twice and 6 minutes must be less than 4 (WB) and 3.2 (head-20)
    blockEvents = obj.block_events
    numEvents = len(blockEvents)
    t_vec = np.zeros(numEvents)
    SARwbg_vec = np.zeros(t_vec.shape)
    SARhg_vec = np.zeros(t_vec.shape)
    t_prev = 0

    for iB in blockEvents:
        block = obj.get_block(iB)
        block_dur = calc_duration(block)
        t_vec[iB - 1] = t_prev + block_dur
        t_prev = t_vec[iB - 1]
        if ('rf' in block):  # has rf
            rf = block['rf']
            t = rf.t
            signal = rf.signal
            # This rf could be parallel transmit as well
            SARwbg_vec[iB] = calc_SAR(Qtmf, signal)
            SARhg_vec[iB] = calc_SAR(Qhmf, signal)

    return SARwbg_vec, SARhg_vec, t_vec


def SARinterp(SAR, t):
    tsec = np.arange(1, np.floor(t[-1]) + 1, 1)
    f = interpolate.interp1d(t, SAR)
    SARinterp = f(tsec)
    return SARinterp, tsec


def SARlimscheck(SARwbg_lim_s, SARhg_lim_s, tsec):
    # Declare constants for checks - IEC 60601-2 - W/kg for six minute and ten seconds for whole body and head

    if (tsec[-1] > 10):

        SixMinThresh_wbg = 4
        TenSecThresh_wbg = 8

        SixMinThresh_hg = 3.2
        TenSecThresh_hg = 6.4

        SARwbg_lim_app = np.concatenate((np.zeros(5), SARwbg_lim_s, np.zeros(5)),axis=0)
        SARhg_lim_app = np.concatenate((np.zeros(5), SARhg_lim_s, np.zeros(5)),axis=0)

        SAR_wbg_tensec = do_sw_sar(SARwbg_lim_app, tsec, 10)  # < 2  SARmax
        SAR_hg_tensec = do_sw_sar(SARhg_lim_app, tsec, 10)  # < 2 SARmax
        SAR_wbg_tensec_peak = np.round(np.max(SAR_wbg_tensec),2)
        SAR_hg_tensec_peak = np.round(np.max(SAR_hg_tensec),2)

        if ((np.max(SAR_wbg_tensec) > TenSecThresh_wbg) or (np.max(SAR_hg_tensec) > TenSecThresh_hg)):
            print('Pulse exceeding 10 second Global SAR limits, increase TR')
        SAR_wbg_sixmin = 'NA'
        SAR_hg_sixmin = 'NA'
        SAR_wbg_sixmin_peak = 'NA'
        SAR_hg_sixmin_peak = 'NA'

        if (tsec[-1] > 600):

            SARwbg_lim_app = np.concatenate((np.zeros(300), SARwbg_lim_s, np.zeros(300)),axis=0)
            SARhg_lim_app = np.concatenate((np.zeros(300), SARhg_lim_s, np.zeros(300)),axis=0)

            SAR_hg_sixmin = do_sw_sar(SARhg_lim_app, tsec, 600)
            SAR_wbg_sixmin = do_sw_sar(SARwbg_lim_app, tsec, 600)
            SAR_wbg_sixmin_peak = np.round(np.max(SAR_wbg_sixmin),2)
            SAR_hg_sixmin_peak = np.round(np.max(SAR_hg_sixmin),2)

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


def do_sw_sar(SAR, tsec, t):
    SAR_timeavg = np.zeros(len(tsec)+int(t))
    for instant in range(int(t/2),int(t/2)+ (int(tsec[-1]))):  # better to go from  -sw / 2: sw / 2
        SAR_timeavg[instant] = sum(SAR[range(instant - int(t/2), instant + int(t/2) - 1)]) / t
    SAR_timeavg = SAR_timeavg[int(t/2):int(t/2)+ (int(tsec[-1]))]
    return SAR_timeavg


def payload_process(fname='rad2D.seq'):
    Qtmf, Qhmf = loadQ()
    SARwbg, SARhg, t_vec = SARfromseq(fname, Qtmf, Qhmf)
    SARwbg_lim, tsec = SARinterp(SARwbg, t_vec)
    SARhg_lim, tsec = SARinterp(SARhg, t_vec)
    SAR_wbg_tensec, SAR_wbg_sixmin, SAR_hg_tensec, SAR_hg_sixmin, SAR_wbg_sixmin_peak, SAR_hg_sixmin_peak, SAR_wbg_tensec_peak, SAR_hg_tensec_peak = SARlimscheck(
        SARwbg_lim, SARhg_lim, tsec)

    imgpath = './src/coms/coms_ui/static/rf/tx/SAR/'
    timestamp = time.strftime("%Y%m%d%H%M%S")
    fname_store = timestamp + "_SAR1.png"
    payload = {
        "filename": fname_store,
        "SAR_wbg_tensec_peak": SAR_wbg_tensec_peak,
        "SAR_wbg_sixmin_peak": SAR_wbg_sixmin_peak,
        "SAR_hg_tensec_peak": SAR_hg_tensec_peak,
        "SAR_hg_sixmin_peak": SAR_hg_sixmin_peak,  # random.randint(4, 100),
    }
    #
    # print(payload)
    # Display and save figures in hardcoded paths for now

    # Plot 10 sec average SAR
    if (tsec[-1] > 10):
        print('Display starts now..')
        plt.figure
        plt.plot(tsec, SAR_wbg_tensec, label='Whole Body:10sec')
        plt.plot(tsec, SAR_hg_tensec, label='Head only:10sec')

        # plt.plot(t_vec, SARwbg, label='Whole Body - instant')
        # plt.plot(t_vec, SARhg, label='Whole Body - instant')

        plt.xlabel('time (s)')
        plt.ylabel('SAR (W/kg)')
        plt.title('Global SAR  - Mass Normalized -  Whole body and head only')
        ax = plt.subplot(111)

        ax.legend()
        plt.grid(True)
        plt.savefig(imgpath + fname_store, bbox_inches='tight', pad_inches=0)
        #plt.show()  # Uncomment for local display - will hinder return function is persistent
    print('SAR computation performed')
    return payload

# payload_process('rad2D.seq')  # uncomment if you want to run this script directly
