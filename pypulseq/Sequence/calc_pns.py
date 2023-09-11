from types import SimpleNamespace
from typing import Tuple
import matplotlib.pyplot as plt
import pypulseq as pp
import numpy as np

from pypulseq import Sequence
from pypulseq.utils.safe_pns_prediction import safe_gwf_to_pns, safe_plot

from pypulseq.utils.siemens.readasc import readasc
from pypulseq.utils.siemens.asc_to_hw import asc_to_hw


def calc_pns(
        obj : Sequence, hardware : SimpleNamespace, do_plots: bool = True
        ) -> Tuple[bool, np.array, np.ndarray, np.array]:
    """
    Calculate PNS using safe model implementation by Szczepankiewicz and Witzel
    See http://github.com/filip-szczepankiewicz/safe_pns_prediction
    
    Returns pns levels due to respective axes (normalized to 1 and not to 100#)
    
    Parameters
    ----------
    hardware : SimpleNamespace
        Hardware specifications. See safe_example_hw() from
        the safe_pns_prediction package. Alternatively a text file
        in the .asc format (Siemens) can be passed, e.g. for Prisma
        it is MP_GPA_K2309_2250V_951A_AS82.asc (we leave it as an
        exercise to the interested user to find were these files
        can be acquired from)
    do_plots : bool, optional
        Plot the results from the PNS calculations. The default is True.

    Returns
    -------
    ok : bool
        Boolean flag indicating whether peak PNS is within acceptable limits
    pns_norm : numpy.array [N]
        PNS norm over all gradient channels, normalized to 1
    pns_components : numpy.array [Nx3]
        PNS levels per gradient channel
    t_pns : np.array [N]
        Time axis for the pns_norm and pns_components arrays
    """
    
    # acquire the entire gradient wave form
    gw = obj.waveforms_and_times()[0]
    if do_plots:
        plt.figure()
        plt.plot(gw[0][0], gw[0][1], gw[1][0], gw[1][1], gw[2][0], gw[2][1]) # plot the entire gradient shape
        plt.title('gradient wave form, in T/m')
    
    # find beginning and end times and resample GWs to a regular sampling raster
    tf = []
    tl = []
    for i in range(3):
        if gw[i].shape[1] > 0:
            tf.append(gw[i][0,0])
            tl.append(gw[i][0,-1])

    nt_min = np.floor(min(tf) / obj.grad_raster_time + pp.eps) 
    nt_max = np.ceil(max(tl) / obj.grad_raster_time - pp.eps)
    
    # shift raster positions to the centers of the raster periods
    nt_min = nt_min + 0.5
    nt_max = nt_max - 0.5
    if nt_min < 0.5:
        nt_min = 0.5

    t_axis = (np.arange(0,np.floor(nt_max-nt_min) + 1) + nt_min) * obj.grad_raster_time

    gwr = np.zeros((t_axis.shape[0],3))
    for i in range(3):
        if gw[i].shape[1] > 0:
            gwr[:,i] = np.interp(t_axis, gw[i][0], gw[i][1])

    if type(hardware) == str:
        # this loads the parameters from the provided text file
        asc, _ = readasc(hardware)
        hardware = asc_to_hw(asc)

    # use the Szczepankiewicz' and Witzel's implementation
    [pns_comp,res] = safe_gwf_to_pns(gwr/obj.system.gamma, np.nan*np.ones(t_axis.shape[0]), obj.grad_raster_time, hardware) # the RF vector is unused in the code inside but it is zeropaded and exported ... 
    
    # use the exported RF vector to detect and undo zero-padding
    pns_comp = 0.01 * pns_comp[~np.isfinite(res.rf[1:]),:]
    
    # calc pns_norm and the final ok/not_ok
    pns_norm = np.sqrt((pns_comp**2).sum(axis=1))
    ok = all(pns_norm<1)
    
    # ready
    if do_plots:
        # plot results
        plt.figure()
        safe_plot(pns_comp*100, obj.grad_raster_time)

    return ok, pns_norm, pns_comp, t_axis