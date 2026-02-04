import numpy as np
from pypulseq.opts import Opts
from pypulseq.make_spiral import make_spiral

def test_make_spiral_basic_fast():
    """Test basic spiral generation with small matrix for speed."""
    # Use small matrix to reduce computation steps
    # Note: max_grad=32mT/m, max_slew=130T/m/s
    system = Opts(max_grad=32, grad_unit='mT/m', max_slew=130, slew_unit='T/m/s', grad_raster_time=10e-6, adc_raster_time=1e-6)
    fov = 0.20
    matrix = 32 # Small matrix for fast test
    
    # oversampling=1 is unsafe for accuracy but fast for "it runs" check. 
    # oversampling=5 is a compromise.
    gx, gy, adc = make_spiral(fov=fov, matrix=matrix, system=system, oversampling=5)
    
    assert gx is not None
    assert gy is not None
    assert adc is not None
    
    # Check if we got something
    assert len(gx.waveform) > 100
    assert len(gx.waveform) == len(gy.waveform)
    
    # Check limits
    assert np.max(np.abs(gx.waveform)) <= system.max_grad * 1.01 # allow tiny float error
    assert np.max(np.abs(gy.waveform)) <= system.max_grad * 1.01

def test_make_spiral_pns_fast():
    """Test spiral generation with PNS model (fast)."""
    system = Opts(max_grad=32, grad_unit='mT/m', max_slew=130, slew_unit='T/m/s', grad_raster_time=10e-6)
    fov = 0.20
    matrix = 32
    
    safe_model = {
        'RIV': False,
        'tauX': [1.5e-3, 1.5e-3, 1.5e-3], 
        'tauY': [1.5e-3, 1.5e-3, 1.5e-3],
        'AX': [0.5, 0.2, 0.2],
        'AY': [0.5, 0.2, 0.2],
        'pnsScaling': [1.0, 1.0],
        'pnsDesignLimit': 0.6 # Low limit to force PNS active
    }
    
    gx, gy, adc = make_spiral(fov=fov, matrix=matrix, system=system, safe_model=safe_model, oversampling=5)
    
    assert gx is not None
    assert len(gx.waveform) > 0

def test_make_spiral_resonance_fast():
    """Test spiral generation with resonance avoidance (fast)."""
    system = Opts(max_grad=32, grad_unit='mT/m', max_slew=130, slew_unit='T/m/s')
    fov = 0.20
    matrix = 32 
    
    # Add a resonance band likely to be hit
    # Spiral frequency starts low and increases.
    # f = gamma * G / (2pi k). 
    # With small matrix, we reach high k quickly.
    resonances = [
        (500, 1500) # Broad band
    ]
    
    gx, gy, adc = make_spiral(fov=fov, matrix=matrix, system=system, resonances=resonances, oversampling=5)
    assert gx is not None
    assert len(gx.waveform) > 0

def test_make_spiral_interleaves():
    """Test spiral generation with multiple interleaves."""
    system = Opts(max_grad=32, grad_unit='mT/m', max_slew=130, slew_unit='T/m/s')
    fov = 0.20
    matrix = 32
    
    # 1 interleave
    gx1, _, _ = make_spiral(fov=fov, matrix=matrix, system=system, interleaves=1, oversampling=5)
    
    # 2 interleaves (should be shorter or same duration but sparser? 
    # N interleaves means each arm covers 1/N of k-space density radially?
    # dr/dtheta = N / (2*pi*F). 
    # Larger N -> larger dr/dtheta -> radius grows faster -> reaches kmax faster -> shorter duration.
    gx2, _, _ = make_spiral(fov=fov, matrix=matrix, system=system, interleaves=2, oversampling=5)
    
    assert len(gx2.waveform) < len(gx1.waveform)

