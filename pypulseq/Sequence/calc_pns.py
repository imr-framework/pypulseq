import math
from types import SimpleNamespace
from typing import Tuple, List

import matplotlib.pyplot as plt
import pypulseq as pp
import numpy as np

from pypulseq import Sequence
from pypulseq.utils.safe_pns_prediction import safe_gwf_to_pns, safe_plot

from pypulseq.utils.siemens.readasc import readasc
from pypulseq.utils.siemens.asc_to_hw import asc_to_hw


def calc_pns(
        obj: Sequence,
        hardware: SimpleNamespace,
        time_range: List[float] = None,
        do_plots: bool = True
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
    
    dt = obj.grad_raster_time
    # Get gradients as piecewise-polynomials
    gw_pp = obj.get_gradients(time_range=time_range)
    ng = len(gw_pp)
    max_t = max(g.x[-1] for g in gw_pp if g != None) - 1e-10
    
    # Determine sampling points
    if time_range == None:
        nt = int(np.ceil(max_t/dt))
        t = (np.arange(nt) + 0.5)*dt
    else:
        tmax = min(time_range[1], max_t) - max(time_range[0], 0)
        nt = int(np.ceil(tmax/dt))
        t = max(time_range[0], 0) + (np.arange(nt) + 0.5)*dt
    
    # Sample gradients
    gw = np.zeros((t.shape[0], ng))
    for i in range(ng):
        if gw_pp[i] != None:
            gw[:,i] = gw_pp[i](t)
            
    
    if do_plots:
        plt.figure()
        for i in range(ng):
            if gw_pp[i] != None:
                plt.plot(gw_pp[i].x[1:-1], gw_pp[i].c[1,:-1])
        plt.title('gradient wave form, in Hz/m')
    
    if type(hardware) == str:
        # this loads the parameters from the provided text file
        asc, _ = readasc(hardware)
        hardware = asc_to_hw(asc)

    # use the Szczepankiewicz' and Witzel's implementation
    [pns_comp,res] = safe_gwf_to_pns(gw/obj.system.gamma, np.nan*np.ones(t.shape[0]), obj.grad_raster_time, hardware) # the RF vector is unused in the code inside but it is zeropaded and exported ... 
    
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

    return ok, pns_norm, pns_comp, t