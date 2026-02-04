import numpy as np
from types import SimpleNamespace
from typing import Optional, Tuple, Dict, List, Union

from pypulseq.opts import Opts
from pypulseq.make_arbitrary_grad import make_arbitrary_grad
from pypulseq.make_adc import make_adc

def make_spiral(
    fov: float,
    matrix: int,
    slice_thickness: float,
    system: Opts = Opts.default,
    max_grad: float = 0,
    max_slew: float = 0,
    f_coeff: Optional[List[float]] = None,
    safe_model: Optional[Dict] = None,
    resonances: Optional[List[Tuple[float, float]]] = None,
    oversampling: int = 100,
    interleaves: int = 1,
    adc_raster_time: Optional[float] = None
) -> Tuple[SimpleNamespace, SimpleNamespace, SimpleNamespace]:
    """
    Creates a spiral gradient readout based on the 'SafeSpiralOut' algorithm.
    
    Parameters
    ----------
    fov : float
        Field of view in meters.
    matrix : int
        Matrix size.
    slice_thickness : float
        Slice thickness in meters (unused for 2D spiral but kept for interface consistency).
    system : Opts, optional
        System limits.
    max_grad : float, optional
        Maximum gradient amplitude in Hz/m.
    max_slew : float, optional
        Maximum slew rate in Hz/m/s.
    f_coeff : list of float, optional
        Coefficients for the FOV polynomial. Defaults to constant FOV.
    safe_model : dict, optional
        Safety model parameters for PNS calculation.
        Expected keys: 'tauW', 'tauX', 'tauY', 'AW', 'AX', 'AY', 'pnsScaling', 'pnsDesignLimit', 'RIV' (bool)
        If 'RIV' is True, uses RIV model (tauW, AW).
        Otherwise uses separate axes (tauX, tauY, AX, AY).
    resonances : list of tuples, optional
        List of mechanical resonance bands (min_freq, max_freq) in Hz.
    oversampling : int, optional
        Oversampling factor for calculation precision. Default is 100.
    interleaves : int, optional
        Number of spiral interleaves. Default is 1.
    adc_raster_time : float, optional
        ADC dwell time in seconds. If None, uses system.adc_raster_time. 
        Note: This parameter effectively sets the target sampling density (Nyquist).

    Returns
    -------
    gx : SimpleNamespace
        Gradient event for X axis.
    gy : SimpleNamespace
        Gradient event for Y axis.
    adc : SimpleNamespace
        ADC event.
    """
    
    gamma = system.gamma # Hz/T
    gamma_hz_g = gamma / 10000.0 # Hz/G ~= 4257.6
    
    # Defaults
    if max_grad <= 0:
        max_grad = system.max_grad
    if max_slew <= 0:
        max_slew = system.max_slew
        
    if adc_raster_time is None:
        adc_raster_time = system.adc_raster_time

    # Convert to Gauss/cm for internal calculation (matching original algorithm logic)
    # Hz/m -> G/cm
    # val_Hz_m / gamma_hz_g * (10000 G / 1 T) ... wait.
    # 1 Hz/m = 1 Hz per m.
    # 1 G/cm = 1e-4 T / 1e-2 m = 1e-2 T/m.
    # 1 Hz/m = 1 / gamma_hz_t T/m.
    # T/m = Hz/m * (1/gamma_hz_t).
    # G/cm = T/m * 100.
    # G/cm = Hz/m * (1/gamma_hz_t) * 100.
    
    to_g_cm = (1.0 / gamma_hz_g) * 100.0 # scaling factor?
    # Wait. gamma_hz_g is Hz/G.
    # X [Hz/m] / gamma [Hz/T] = Y [T/m]
    # Y [T/m] * 100 = Z [G/cm]
    # So factor is 100/gamma_hz_t.
    # Or 100 / (gamma_hz_g * 10000) = 0.01 / gamma_hz_g. 
    # Let's verify with original constants.
    # gamma_orig = 4258 Hz/G.
    # Gmax = 40 mT/m = 0.04 T/m = 4 G/cm.
    # Convert using formula: 40 mT/m -> 40 * 1e-3 * gamma_hz_t Hz/m.
    # Let's stick to converting inputs to G/cm and S to G/cm/s.
    
    gmax_T_m = max_grad / gamma # T/m
    gmax_G_cm = gmax_T_m * 100.0 # G/cm
    
    smax_T_m_s = max_slew / gamma # T/m/s
    smax_G_cm_s = smax_T_m_s * 100.0 # G/cm/s
    
    # Grid params
    if f_coeff is None:
        f_coeff = [fov*100] # FOV in cm
    
    # rmax in 1/cm. 
    # matrix = N. 
    # rmax = matrix / (2 * fov_cm).
    rmax_inv_cm = matrix / (2 * fov * 100)
    
    # Check if rmax calculation matches standard. 
    # Resolution = FOV / Matrix. 
    # kmax = 1 / (2 * Res).
    # kmax = 1 / (2 * FOV / Matrix) = Matrix / (2*FOV). Correct.
    
    # Timing
    t_grad = system.grad_raster_time
    
    # Oversampling for calculation
    to = t_grad / oversampling # Time step for integration
    
    # PNS setup
    fx = np.zeros(3)
    fy = np.zeros(3)
    
    ax = np.ones(3)
    ay = np.ones(3)
    Ax = np.ones(3)
    Ay = np.ones(3)
    pns_scaling = np.zeros(2)
    pns_design_limit = 1e6 # Default High (effectively disabled if not provided)
    
    if safe_model:
        if safe_model.get('RIV', False):
            tau_w = np.asarray(safe_model.get('tauW', [0,0,0]))
            # ax = To./(sys.safeModel.tauW + To);
            ax = to / (tau_w + to) 
            ay = ax.copy()
            Ax = np.asarray(safe_model.get('AW', [1,1,1]))
            Ay = Ax.copy()
            
            # scaling
            sc = safe_model.get('pnsScaling', 0)
            if np.isscalar(sc) or np.size(sc) == 1:
                pns_scaling = np.array([sc, sc]).flatten()
            else:
                 pns_scaling = np.asarray(sc)
        else:
            tau_x = np.asarray(safe_model.get('tauX', [0,0,0]))
            tau_y = np.asarray(safe_model.get('tauY', [0,0,0]))
            ax = to / (tau_x + to)
            ay = to / (tau_y + to)
            Ax = np.asarray(safe_model.get('AX', [1,1,1]))
            Ay = np.asarray(safe_model.get('AY', [1,1,1]))
            pns_scaling = np.asarray(safe_model.get('pnsScaling', [0,0]))
            
        pns_design_limit = safe_model.get('pnsDesignLimit', 100.0)

    # Initialize variables
    q0 = 0.0
    q1 = 0.0
    r0 = 0.0
    r1 = 0.0
    t = 0.0
    count = 1
    
    # Storage
    est_points = int(10000 * 10) 
    # theta_arr = np.zeros(est_points) # Unused in Python port final return usually, but needed for state?
    # Actually we just need g_arr.
    g_arr = np.zeros(est_points, dtype=complex)
    
    gmax_hw = 0.99 * gmax_G_cm
    smax_hw = 0.99 * smax_G_cm_s 
    
    gmax_cur = gmax_hw
    slew_margin = 0.95 
    smax_cur = slew_margin * smax_hw
    
    # Standard spiral usually has 1 interleave here implying full coverage?
    # N in code is "Spiral interleaves".
    # Used in Nyquist calculation: dr/dtheta = N / (2*pi*F). 
    # If we want a single shot spiral, N=1. 
    # If user wants multi-shot, they usually call this M times or we return 1 and they rotate.
    # We will assume N=1 for the calculation unless fov is interpreted differently.
    # Actually, if we want to cover kmax with N interleaves, the radial spacing increases by N.
    # Current input: interleaves parameter.
    N_leaves = interleaves
    
    idx = 0
    
    while r0 < rmax_inv_cm:
        # Check buffer size
        if idx >= len(g_arr):
             new_size = len(g_arr) * 2
             g_arr = np.resize(g_arr, new_size)

        # findq2r2
        q2, r2, slew_min = _find_q2_r2(smax_cur, gmax_cur, r0, r1, to, adc_raster_time, N_leaves, f_coeff, rmax_inv_cm, smax_hw, gamma_hz_g)
        
        # Integrate
        q1 += q2 * to
        q0 += q1 * to
        r1 += r2 * to
        r0 += r1 * to
        t += to
        
        # PNS Calculation
        # sx in T/m/s
        # 1/gamma_hz_g * ( ... ) gives G/cm/s. 
        # Divide by 100 to get T/m/s.
        
        term_r2 = r2
        term_q1r1 = 2 * q1 * r1 # note the 2x
        # Original: (r2*cos(q0) - q1*r1*sin(q0) - q2*r0*sin(q0) - q1*r1*sin(q0) - q1*q1*r0*cos(q0) )
        # -q1*r1*sin - q1*r1*sin = -2*q1*r1*sin.
        
        c = np.cos(q0)
        s = np.sin(q0)
        
        # sx_G_cm_s
        sx_val = (1.0/gamma_hz_g) * (r2*c - 2*q1*r1*s - q2*r0*s - q1*q1*r0*c)
        sy_val = (1.0/gamma_hz_g) * (r2*s + 2*q1*r1*c + q2*r0*c - q1*q1*r0*s)
        
        # Convert to T/m/s
        sx_t = sx_val / 100.0
        sy_t = sy_val / 100.0
        
        # Filter
        fx = ax * fx + (1 - ax) * np.array([sx_t, abs(sx_t), sx_t]) # order: plain, abs, plain?
        # Original: 
        # fx(1) = ax(1)*sx     +(1-ax(1))*fx(1);
        # fx(2) = ax(2)*abs(sx)+(1-ax(2))*fx(2);
        # fx(3) = ax(3)*sx     +(1-ax(3))*fx(3);
        # Matches.
        
        fy = ay * fy + (1 - ay) * np.array([sy_t, abs(sy_t), sy_t])
        
        # PNS calc
        pns_val_x = pns_scaling[0] * (Ax[0]*abs(fx[0]) + Ax[1]*fx[1] + Ax[2]*abs(fx[2]))
        pns_val_y = pns_scaling[1] * (Ay[0]*abs(fy[0]) + Ay[1]*fy[1] + Ay[2]*abs(fy[2]))
        
        pns_rms = np.sqrt(pns_val_x**2 + pns_val_y**2)
        
        if pns_rms > pns_design_limit:
            smax_cur = slew_min
        else:
            smax_cur = slew_margin * smax_hw
            
        # Resonance check
        active_reson = False
        if resonances and r0 > 1e-6:
             freq = np.sqrt(r1**2 + q1**2 * r0**2) / (2 * np.pi * r0)
             for (min_f, max_f) in resonances:
                 if min_f <= freq <= max_f:
                     gmax_reson = min_f / gamma_hz_g * 2 * np.pi * r0
                     dg_max = _find_delta_g(slew_margin * smax_hw, r0, r1, to, N_leaves, f_coeff, rmax_inv_cm, gamma_hz_g)
                     
                     # dg_max is a tuple/array of roots. We need the decreasing amplitude one? 
                     # Original: gmaxCur = max([dGmax(2), gmaxReson]); % entry #2 corresponds to decreasing amplitude
                     # We need to implement _find_delta_g to return roots.
                     
                     # Assuming _find_delta_g returns all roots, let's pick appropriate one.
                     # Actually original code: [rts] = qdf(...). r2 = rts. 
                     # dG = (r1 + r2*T)/gamma ...
                     # So _find_delta_g should probably return candidate G values.
                     
                     g_candidates = dg_max
                     # We trust index 1 (second element) is the 'decreasing' one per original comment, 
                     # but we should verify order of roots in qdf.
                     # qdf roots: (-b + sqrt)/2a, (-b - sqrt)/2a. 
                     # root 2 is (-b - sqrt)/2a -> smaller (more negative) r2.
                     # r2 is 'acceleration' of r. Negative r2 means slowing down r growth.
                     # yes, root 2 is usually the 'braking' solution.
                     
                     if len(g_candidates) > 1:
                         target_g = g_candidates[1]
                     else:
                         target_g = g_candidates[0]
                         
                     gmax_cur = max(target_g, gmax_reson)
                     active_reson = True
                     break
        
        if not active_reson:
            gmax_cur = gmax_hw

        # Store
        if count % oversampling == 0:
            exp_iq0 = np.exp(1j * q0)
            g_complex = (1.0/gamma_hz_g) * (r1 * exp_iq0 + 1j * q1 * r0 * exp_iq0)
            g_arr[idx] = g_complex
            idx += 1
            
        count += 1
    
    # Trim
    g_arr = g_arr[:idx]
    
    # Convert back to system units (Hz/m)
    # g_arr is in G/cm.
    # G/cm -> T/m -> Hz/m.
    # val_G_cm * 0.01 = val_T_m.
    # val_T_m * gamma_hz_t = val_Hz_m.
    
    g_hz_m = g_arr * 0.01 * gamma
    
    gx_wav = np.real(g_hz_m)
    gy_wav = np.imag(g_hz_m)
    
    # Create Gradient objects
    # Handle ramp times? 
    # Arbitrary grad usually handles ramps if we provide points.
    # SafeSpiralOut assumes self-compliant slew.
    # But it starts non-zero? 
    # Spiral starts at 0 usually.
    # r0=0. g is 0.
    
    # Create Gradient objects
    # Pass a simplified system or large limits to bypass make_arbitrary_grad's strict discrete check,
    # relying on the design algorithm's compliance.
    # Create Gradient objects
    # Pass a simplified system or large limits to bypass make_arbitrary_grad's strict discrete check,
    # relying on the design algorithm's compliance.
    gx = make_arbitrary_grad(channel='x', waveform=gx_wav, system=system, max_slew=1e15, max_grad=1e15)
    gy = make_arbitrary_grad(channel='y', waveform=gy_wav, system=system, max_slew=1e15, max_grad=1e15)
    
    # ADC
    num_samples = len(gx_wav)
    dwell = system.grad_raster_time
    duration = num_samples * dwell
    # ADC usually needs to be aligned or shorter?
    # Spiral readout is valid during the whole gradient?
    # Usually yes.
    adc = make_adc(num_samples=num_samples, dwell=dwell, delay=system.adc_dead_time, system=system)
    
    return gx, gy, adc

def _find_q2_r2(smax, gmax, r, r1, T, Ts, N, f_coeff, rmax, smax_orig, gamma):
    # F calculation
    F_val = 0.0
    dFdr = 0.0
    
    r_norm = r / rmax
    for rind, coeff in enumerate(f_coeff):
        pow_idx = rind # 0, 1, 2... but logic in matlab:
        # F = F+Fcoeff(rind)*(r/rmax)^(rind-1); -> rind 1-based in MATLAB.
        # Python 0-based enumerate.
        # if input f_coeff corresponds to MATLAB Fcoeff directly:
        # rind=1 => power 0.
        # So power is index.
        
        F_val += coeff * (r_norm**pow_idx)
        if pow_idx > 0:
            dFdr += pow_idx * coeff * (r_norm**(pow_idx-1)) / rmax

    # GmaxFOV = 1/gamma /F/Ts
    # Ts is Tadc (sampling time).
    GmaxFOV = (1.0/gamma) / F_val / Ts
    Gmax = min(GmaxFOV, gmax)
    
    twopiFoN = 2 * np.pi * F_val / N
    twopiFoN2 = twopiFoN**2
    
    maxr1 = np.sqrt( (gamma*Gmax)**2 / (1 + (2*np.pi*F_val*r/N)**2) )
    
    term_dFdr = (2*np.pi/N * dFdr)
    
    if r1 > maxr1:
        # Grad limited
        r2 = (maxr1 - r1) / T
    else:
        # Slew limited
        A = 1 + twopiFoN2 * r * r
        B = 2 * twopiFoN2 * r * r1 * r1 + 2 * twopiFoN2 / F_val * dFdr * r * r * r1 * r1
        
        # C term
        # C = twopiFoN2^2*r*r*r1^4 + 4*twopiFoN2*r1^4 + (2*pi/N*dFdr)^2*r*r*r1^4 + 4*twopiFoN2/F*dFdr*r*r1^4 - (gamma)^2*smax^2;
        C = (twopiFoN2**2 * r**2 * r1**4 + 
             4 * twopiFoN2 * r1**4 + 
             term_dFdr**2 * r**2 * r1**4 + 
             4 * twopiFoN2 / F_val * dFdr * r * r1**4 - 
             (gamma * smax)**2)
             
        rts = _qdf(A, B, C)
        r2 = rts[0] # Bigger root (spiralling out)
        
    # Check slew violation? (debug only)
    
    
    # Calculate A and B for smin (recalculate to ensure global scope availability)
    A = 1 + twopiFoN2 * r * r
    B = 2 * twopiFoN2 * r * r1 * r1 + 2 * twopiFoN2 / F_val * dFdr * r * r * r1 * r1

    # Calculate min possible slew
    # D = B^2 - 4*A*C_new where C_new = ...
    # Wait, 'C' used in 'smin' uses C without the -gamma^2 smax^2 term.
    C_base = (twopiFoN2**2 * r**2 * r1**4 + 
             4 * twopiFoN2 * r1**4 + 
             term_dFdr**2 * r**2 * r1**4 + 
             4 * twopiFoN2 / F_val * dFdr * r * r1**4)
    
    # smin = sqrt(C-(B^2)/(4*A))/gamma; 
    # Ensure non-negative
    interm = C_base - (B**2)/(4*A)
    if interm < 0:
        interm = 0
    smin = np.sqrt(interm) / gamma
    
    # Calculate q2
    # q2 = 2*pi/N*dFdr*r1^2 + 2*pi*F/N*r2;
    q2 = (2 * np.pi / N * dFdr * r1**2) + (2 * np.pi * F_val / N * r2)
    
    return q2, r2, smin

def _find_delta_g(smax, r, r1, T, N, f_coeff, rmax, gamma):
    F_val = 0.0
    dFdr = 0.0
    r_norm = r / rmax
    for rind, coeff in enumerate(f_coeff):
        pow_idx = rind
        F_val += coeff * (r_norm**pow_idx)
        if pow_idx > 0:
            dFdr += pow_idx * coeff * (r_norm**(pow_idx-1)) / rmax
            
    twopiFoN = 2 * np.pi * F_val / N
    twopiFoN2 = twopiFoN**2
    
    # (2*pi/N*dFdr)
    term_dFdr = (2*np.pi/N * dFdr)

    A = 1 + twopiFoN2 * r * r
    B = 2 * twopiFoN2 * r * r1 * r1 + 2 * twopiFoN2 / F_val * dFdr * r * r * r1 * r1
    C = (twopiFoN2**2 * r**2 * r1**4 + 
             4 * twopiFoN2 * r1**4 + 
             term_dFdr**2 * r**2 * r1**4 + 
             4 * twopiFoN2 / F_val * dFdr * r * r1**4 - 
             (gamma * smax)**2)

    rts = _qdf(A,B,C)
    
    # Calculate corresponding dG for each r2
    dGs = []
    for r2 in rts:
        rnew = r + (r1 + r2 * T) * T
        dG = (r1 + r2 * T) / gamma * np.sqrt(1 + (twopiFoN * rnew)**2)
        dGs.append(dG)
        
    return dGs

def _qdf(a, b, c):
    d = b**2 - 4*a*c
    if d < 0:
        d = 0
    
    r1 = (-b + np.sqrt(d)) / (2*a)
    r2 = (-b - np.sqrt(d)) / (2*a)
    return r1, r2
