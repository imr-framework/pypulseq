from pathlib import Path

import numpy as np

path_version = Path(__file__).parent.parent / 'VERSION'
with open(str(path_version), 'r') as version_file:
    major, minor, revision = version_file.read().strip().split('.')
    major = int(major)
    minor = int(minor)


# =========
# BANKER'S ROUNDING FIX
# =========
def round_half_up(n, decimals=0):
    """
    Avoid banker's rounding inconsistencies; from https://realpython.com/python-rounding/#rounding-half-up
    """
    multiplier = 10 ** decimals
    return np.floor(np.abs(n) * multiplier + 0.5) / multiplier


# =========
# PACKAGE-LEVEL IMPORTS
# =========
from pypulseq.SAR.SAR_calc import calc_SAR
from pypulseq.Sequence.sequence import Sequence
from pypulseq.add_gradients import add_gradients
from pypulseq.align import align
from pypulseq.calc_duration import calc_duration
from pypulseq.calc_ramp import calc_ramp
from pypulseq.calc_rf_center import calc_rf_center
from pypulseq.make_adc import make_adc
from pypulseq.make_arbitrary_rf import make_arbitrary_rf
from pypulseq.make_block_pulse import make_block_pulse
from pypulseq.sigpy2pulseq import *
from pypulseq.make_delay import make_delay
from pypulseq.make_digital_output_pulse import make_digital_output_pulse
from pypulseq.make_extended_trapezoid import make_extended_trapezoid
from pypulseq.make_extended_trapezoid_area import make_extended_trapezoid_area
from pypulseq.make_gauss_pulse import make_gauss_pulse
from pypulseq.make_label import make_label
from pypulseq.make_sinc_pulse import make_sinc_pulse
from pypulseq.make_trap_pulse import make_trapezoid
from pypulseq.make_trigger import make_trigger
from pypulseq.opts import Opts
from pypulseq.points_to_waveform import points_to_waveform
from pypulseq.split_gradient import split_gradient
from pypulseq.split_gradient_at import split_gradient_at
from pypulseq.supported_labels import get_supported_labels
from pypulseq.traj_to_grad import traj_to_grad
