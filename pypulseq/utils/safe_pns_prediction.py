# This code is a direct Python translation of the relevant functions in
# https://github.com/filip-szczepankiewicz/safe_pns_prediction/ to perform
# PNS calculations with pypulseq
#
# A small modification was made to safe_plot to plot long sequences better


# BSD 3-Clause License

# Copyright (c) 2018, Filip Szczepankiewicz and Thomas Witzel
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from types import SimpleNamespace

import numpy as np
import matplotlib.pyplot as plt


def safe_example_hw():
    # function hw = safe_example_hw()
    #
    # SAFE model parameters for EXAMPLE scanner hardware (not a real scanner).
    # See comments for units.
    
    hw = SimpleNamespace()
    hw.name          = 'MP_GPA_EXAMPLE'
    hw.checksum      = '1234567890'
    hw.dependency    = ''
    
    hw.x = SimpleNamespace()
    hw.x.tau1        =  0.20  # ms
    hw.x.tau2        =  0.03  # ms
    hw.x.tau3        =  3.00  # ms
    hw.x.a1          =  0.40
    hw.x.a2          =  0.10
    hw.x.a3          =  0.50
    hw.x.stim_limit  = 30.0   # T/m/s
    hw.x.stim_thresh = 24.0   # T/m/s
    hw.x.g_scale     = 0.35   # 1
    
    hw.y = SimpleNamespace()
    hw.y.tau1        =  1.50  # ms
    hw.y.tau2        =  2.50  # ms
    hw.y.tau3        =  0.15  # ms
    hw.y.a1          =  0.55
    hw.y.a2          =  0.15
    hw.y.a3          =  0.30
    hw.y.stim_limit  = 15.0   # T/m/s
    hw.y.stim_thresh = 12.0   # T/m/s
    hw.y.g_scale     = 0.31   # 1
    
    hw.z = SimpleNamespace()
    hw.z.tau1        =  2.00  # ms
    hw.z.tau2        =  0.12  # ms
    hw.z.tau3        =  1.00  # ms
    hw.z.a1          =  0.42
    hw.z.a2          =  0.40
    hw.z.a3          =  0.18
    hw.z.stim_limit  = 25.0   # T/m/s
    hw.z.stim_thresh = 20.0   # T/m/s
    hw.z.g_scale     = 0.25   # 1
    return hw


def safe_example_gwf():
    # function function [gwf, rf, dt] = safe_example_gwf()
    # Waveform with some frequency matching by Filip Szczepankiewicz.
    #
    # Waveform was optimized in the NOW framework by Jens Sjölund et al.
    # https://github.com/jsjol/NOW
    #
    # Optimization was Maxwell-compensated to remove effects of concomitant
    # gradients.
    # https://arxiv.org/ftp/arxiv/papers/1903/1903.03357.pdf
    
    ## STE
    dt  = 1e-3 # ms
    
    # T/m
    gwf = 0.08 * np.array([
        [0,         0,         0],
        [-0.2005,    0.9334,    0.3029],
        [-0.2050,    0.9324,    0.3031],
        [-0.2146,    0.9302,    0.3032],
        [-0.2313,    0.9263,    0.3030],
        [-0.2589,    0.9193,    0.3019],
        [-0.3059,    0.9060,    0.2980],
        [-0.3892,    0.8767,    0.2883],
        [-0.3850,    0.7147,    0.3234],
        [-0.3687,    0.5255,    0.3653],
        [-0.3509,    0.3241,    0.4070],
        [-0.3323,    0.1166,    0.4457],
        [-0.3136,   -0.0906,    0.4783],
        [-0.2956,   -0.2913,    0.5019],
        [-0.2790,   -0.4793,    0.5139],
        [-0.2642,   -0.6491,    0.5118],
        [-0.2518,   -0.7957,    0.4939],
        [-0.2350,   -0.8722,    0.4329],
        [-0.2187,   -0.9111,    0.3541],
        [-0.2063,   -0.9409,    0.2747],
        [-0.1977,   -0.9627,    0.1933],
        [-0.1938,   -0.9768,    0.1080],
        [-0.1967,   -0.9820,    0.0159],
        [-0.2114,   -0.9751,   -0.0883],
        [-0.2292,   -0.9219,   -0.2150],
        [-0.2299,   -0.8091,   -0.3561],
        [-0.2290,   -0.6748,   -0.5011],
        [-0.2253,   -0.5239,   -0.6460],
        [-0.2178,   -0.3620,   -0.7868],
        [-0.2056,   -0.1948,   -0.9194],
        [-0.1391,   -0.0473,   -0.9908],
        [-0.0476,    0.0607,   -0.9987],
        [ 0.0215,    0.1452,   -0.9909],
        [ 0.0725,    0.2136,   -0.9759],
        [ 0.1114,    0.2709,   -0.9579],
        [ 0.1426,    0.3204,   -0.9383],
        [ 0.1690,    0.3641,   -0.9177],
        [ 0,         0,         0],
        [ 0,         0,         0],
        [ 0,         0,         0],
        [ 0,         0,         0],
        [ 0,         0,         0],
        [ 0,         0,         0],
        [ 0,         0,         0],
        [-0.3734,   -0.1768,    0.9125],
        [-0.3825,   -0.2310,    0.8965],
        [-0.3919,   -0.2895,    0.8752],
        [-0.4015,   -0.3543,    0.8465],
        [-0.4108,   -0.4290,    0.8065],
        [-0.4182,   -0.5202,    0.7469],
        [-0.4178,   -0.6423,    0.6451],
        [-0.3855,   -0.8173,    0.4321],
        [-0.3110,   -0.9418,    0.1401],
        [-0.2526,   -0.9669,   -0.0674],
        [-0.2100,   -0.9541,   -0.2213],
        [-0.1766,   -0.9227,   -0.3474],
        [-0.1491,   -0.8788,   -0.4570],
        [-0.1258,   -0.8239,   -0.5555],
        [-0.1056,   -0.7583,   -0.6459],
        [-0.0882,   -0.6809,   -0.7293],
        [-0.0734,   -0.5900,   -0.8061],
        [-0.0615,   -0.4830,   -0.8753],
        [-0.0533,   -0.3556,   -0.9349],
        [-0.0506,   -0.2005,   -0.9801],
        [-0.0575,   -0.0019,   -1.0000],
        [-0.0909,    0.2976,   -0.9521],
        [-0.3027,    0.9509,   -0.0860],
        [-0.2737,    0.9610,   -0.0692],
        [-0.2524,    0.9675,   -0.0596],
        [-0.2364,    0.9719,   -0.0533],
        [-0.2245,    0.9749,   -0.0490],
        [-0.2158,    0.9770,   -0.0459],
        [-0.2097,    0.9785,   -0.0439],
        [-0.2058,    0.9794,   -0.0426],
        [-0.2039,    0.9798,   -0.0420],
        [ 0,         0,         0]
        ])
    
    rf = np.ones(gwf.shape[0])
    rf[40:] = -1
    
    return gwf, rf, dt


def safe_hw_check(hw):
    # function safe_hw_check(hw)
    #
    # Make sure that all is well with the hardware configuration.
    
    if abs(hw.x.a1 + hw.x.a2 + hw.x.a3 - 1) > 0.001 or \
       abs(hw.y.a1 + hw.y.a2 + hw.y.a3 - 1) > 0.001 or \
       abs(hw.z.a1 + hw.z.a2 + hw.z.a3 - 1) > 0.001:
        raise ValueError('Hardware specification a1+a2+a3 must be equal to 1!')
    
    axl = ['x', 'y', 'z']
    fnl = ['stim_limit', 'stim_thresh', 'tau1', 'tau2', 'tau3', 'a1', 'a2', 'a3', 'g_scale']
    
    for axn in axl:
        if not hasattr(hw, axn):
            raise ValueError(f"'{axn}' missing in hardware specification")
        
        hw_ax = getattr(hw, axn)
        for par in fnl:
            if not hasattr(hw_ax, par):
                raise ValueError(f"'{axn}.{par}' missing in hardware specification")


def safe_longest_time_const(hw):
    # function ltau = safe_longest_time_const(hw)
    # Get the longest time constant. Can be used to estimate the size of zero
    # padding.

    return max([hw.x.tau1, hw.x.tau2, hw.x.tau3,
                hw.y.tau1, hw.y.tau2, hw.y.tau3,
                hw.z.tau1, hw.z.tau2, hw.z.tau3])


def safe_pns_model(dgdt, dt, hw):
    # function stim = safe_pns_model(dgdt, dt, hw)
    #
    # dgdt (nx3) is in T/m/s
    # dt   (1x1) is in s
    # All time coefficients (a1 and tau1 etc.) are in ms.
    #
    # This PNS model is based on the SAFE-abstract
    # SAFE-Model - A New Method for Predicting Peripheral Nerve Stimulations in MRI
    # by Franz X. Herbank and Matthias Gebhardt. Abstract No 2007. 
    # Proc. Intl. Soc. Mag. Res. Med. 8, 2000, Denver, Colorado, USA
    # https://cds.ismrm.org/ismrm-2000/PDF7/2007.PDF
    # 
    # The main SAFE-model was coded by Thomas Witzel @ Martinos Center,
    # MGH, HMS, Boston, MA, USA.
    # 
    # The code was adapted/expanded/corrected by Filip Szczepankiewicz @ LMI
    # BWH, HMS, Boston, MA, USA, and Lund University, Sweden.
    
    stim1 = hw.a1 * abs( safe_tau_lowpass(dgdt     , hw.tau1, dt * 1000) )
    stim2 = hw.a2 *      safe_tau_lowpass(abs(dgdt), hw.tau2, dt * 1000)  
    stim3 = hw.a3 * abs( safe_tau_lowpass(dgdt     , hw.tau3, dt * 1000) )
    
    stim = (stim1 + stim2 + stim3) / hw.stim_limit * hw.g_scale * 100
    
    return stim
    
    # Not sure where something goes awry, probably in the lowpass filter, but
    # compared to the Siemens simulator we are exactly a factor of pi off, so
    # I'm dividing the final result by pi.
    # Note also that the final result is essentially some kind of arbitrary
    # unit. - TW
    
    # UPDATE 210720 - The pi factor was not quite correct. Instead, the correct
    # factor was determined by the gradient scale factor (hw.g_scale, defined 
    # in the .asc file). Thanks to Maxim Zaitsev for supporting this buggfix and 
    # validating that the updated code is accurate. - FSz


def safe_tau_lowpass(dgdt, tau, dt, eps=1e-16):
    # function fw = safe_tau_lowpass(dgdt, tau, dt)
    #
    # Apply a RC lowpass filter with time constant tau = RC to data with sampling
    # interval dt. NOTE tau and dt need to be in the same unit (i.e. s or ms)
    # The SAFE model abstract by Hebrank et.al. just says "Lowpass with time-constant tau",
    # so I decided to make the most simple filter possible here.
    # The RC lowpass is also appealing because its something Siemens could have
    # easily implemented on their hardware stimulation monitors, so I'm probably
    # pretty close. - TW
    #
    # UPDATE 230206 - There was a factor alpha missing on the first sample it
    # has now been corrected. Thanks to Oliver Schad for finding this error.
    # - FSz
    
    alpha = dt / (tau + dt)
    
    # Calculate number of elements in filter to reach desired accuracy (eps)
    n = min(round(np.log(eps) / np.log(1-alpha)), dgdt.shape[0])
    filt = (1-alpha)**np.arange(n)

    # Implements lowpass filter using convolution to get rid of for loop in original code
    return alpha * np.convolve(dgdt, filt)[:dgdt.shape[0]]


def safe_gwf_to_pns(gwf, rf, dt, hw, do_padding=True):
    # function [pns, res] = safe_gwf_to_pns(gwf, rf, dt, hw, doPadding)
    # 
    # gwf (nx3) in T/m
    # dt  (1x1) in s
    # hw  (struct) is structure that describes the hardware configuration and PNS
    # response. Example: hw = safe_example_hw().
    # doPadding adds zeropadding based on the decay time.
    #
    # This PNS model is based on the SAFE-abstract
    # SAFE-Model - A New Method for Predicting Peripheral Nerve Stimulations in MRI
    # by Franz X. Herbank and Matthias Gebhardt. Abstract No 2007. 
    # Proc. Intl. Soc. Mag. Res. Med. 8, 2000, Denver, Colorado, USA
    # https://cds.ismrm.org/ismrm-2000/PDF7/2007.PDF
    # 
    # The main SAFE-model was coded by Thomas Witzel @ Martinos Center,
    # MGH, HMS, Boston, MA, USA.
    # 
    # The code was adapted/expanded by Filip Szczepankiewicz @ LMI
    # BWH, HMS, Boston, MA, USA.

    if do_padding:
        zpt = safe_longest_time_const(hw) * 4 / 1000 # s
        pad1 = round(zpt/4/dt)
        pad2 = round(zpt/1/dt)

        gwf = np.pad(gwf, ((pad1, pad2), (0,0)))
        rf = np.pad(rf, (pad1, pad2))

    safe_hw_check(hw)
    
    dgdt = np.diff(gwf, axis=0) / dt
    pns = np.zeros(dgdt.shape)
    
    pns[:,0] = safe_pns_model(dgdt[:,0], dt, hw.x)
    pns[:,1] = safe_pns_model(dgdt[:,1], dt, hw.y)
    pns[:,2] = safe_pns_model(dgdt[:,2], dt, hw.z)
    
    # Export relevant paramters
    res = SimpleNamespace()
    res.pns  = pns
    res.gwf  = gwf
    res.rf   = rf
    res.dgdt = dgdt
    res.dt   = dt
    res.hw   = hw
    
    return pns, res

def safe_plot(pns, dt=None, envelope=True, envelope_points=500):
    # function h = safe_plot(pns, dt)
    # pns is relative PNS waveform (nx3)
    # dt is time step size in seconds.
        
    pnsnorm = np.sqrt((pns**2).sum(axis=1))
    
    # FZ: Added option to plot the moving maximum of pns and pnsnorm to keep
    #     plots for long sequences intelligible
    if envelope and pns.shape[0] > envelope_points:
        N = int(np.ceil(pns.shape[0] / envelope_points))
        if dt != None:
            dt *= N
        
        if pns.shape[0] % N != 0:
            pns = np.concatenate((pns, np.zeros((N - pns.shape[0] % N, pns.shape[1]))))
            pnsnorm = np.concatenate((pnsnorm, np.zeros((N - pnsnorm.shape[0] % N))))

        pns = pns.reshape(pns.shape[0]//N, N, pns.shape[1])
        pns = pns.max(axis=1)
        pnsnorm = pnsnorm.reshape(pnsnorm.shape[0]//N, N)
        pnsnorm = pnsnorm.max(axis=1)
        
    if dt == None:
        ttot    = 1 # au
        xlabstr = 'Time [a.u.]'
    else:
        ttot = pns.shape[0] * dt * 1000 # ms
        xlabstr = 'Time [ms]'

    
    t = np.linspace(0, ttot, pns.shape[0])
        
    plt.plot(t, pns[:,0], 'r-',
             t, pns[:,1], 'g-',
             t, pns[:,2], 'b-',
             t, pnsnorm , 'k-')
        
    plt.ylim([0, 120])
    plt.xlim([min(t), max(t)])
    
    plt.title(f'Predicted PNS ({max(pnsnorm):0.0f}%)')
    
    plt.xlabel(xlabstr)
    plt.ylabel('Relative stimulation [%]')
    
    plt.plot([0, max(t)], [max(pnsnorm), max(pnsnorm)], 'k:')

    plt.legend([f'X ({max(pns[:,0]):0.0f}%)',
                f'Y ({max(pns[:,1]):0.0f}%)',
                f'Z ({max(pns[:,2]):0.0f}%)',
                f'nrm ({max(pnsnorm):0.0f}%)'], loc='best')


def safe_example():
    # Load an exampe gradient waveform
    [gwf, rf, dt] = safe_example_gwf()
    
    # Load reponse parameters for example hardware
    hw = safe_example_hw()
    
    # Check if hardware parameters are consistent
    safe_hw_check(hw)
    
    # Check if this hw is part of the library (validate hw)
    # safe_hw_verify(hw)
    
    # Predict PNS levels
    pns, res = safe_gwf_to_pns(gwf, rf, dt, hw, 1)
    
    # Plot some results
    safe_plot(pns, dt)


if __name__ == '__main__':
    safe_example()
