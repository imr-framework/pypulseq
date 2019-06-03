import numpy as np

from pypulseq.make_extended_trapezoid import make_extended_trapezoid
from pypulseq.Sequence.sequence import Sequence

# gs2amp = np.load('/Users/sravan953/Documents/CU/Projects/PyPulseq/GS2amp.npy').squeeze()
# gs2times = np.load('/Users/sravan953/Documents/CU/Projects/PyPulseq/GS2times.npy').squeeze()
# gs2 = make_extended_trapezoid(channel='z', times=gs2times, amplitudes=gs2amp)
seq = Sequence()
seq.read('/Users/sravan953/Documents/CU/Projects/PyPulseq/gre_matlab.seq')
