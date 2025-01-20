import numpy as np
from matplotlib import pyplot as plt

import pypulseq as pp

"""
Read a sequence into MATLAB. The `Sequence` class provides an implementation of the _open file format_ for MR sequences
described here: http://pulseq.github.io/specification.pdf. This example demonstrates parsing an MRI sequence stored in
this format, accessing sequence parameters and visualizing the sequence.
"""

# Read a sequence file - a sequence can be loaded from the open MR file format using the `read` method.
seq_name = 'epi_rs.seq'

system = pp.Opts(B0=2.89)  # Need system here if we want 'detectRFuse' to detect fat-sat pulses
seq = pp.Sequence(system)
seq.read(seq_name, detect_rf_use=True)

# Sanity check to see if the reading and writing are consistent
seq.write('read_test.seq')
# os_system(f'diff -s -u {seq_name} read_test.seq -echo')  # Linux only

"""
Access sequence parameters and blocks. Parameters defined with in the `[DEFINITIONS]` section of the sequence file
are accessed with the `get_definition()` method. These are user-specified definitions and do not effect the execution of
the sequence.
"""
seq_name = seq.get_definition('Name')

# Calculate and display real TE, TR as well as slew rates and gradient amplitudes
test_report = seq.test_report()
print(test_report)

# Sequence blocks are accessed with the `get_block()` method. As shown in the output the first block is a selective
# excitation block and contains an RF pulse and gradient and on the z-channel.
b1 = seq.get_block(1)

# Further information about each event can be obtained by accessing the appropriate fields of the block struct. In
# particular, the complex RF signal is stored in the field `signal`.
rf = b1.rf

plt.subplot(211)
plt.plot(rf.t, np.abs(rf.signal))
plt.ylabel('RF magnitude')

plt.subplot(212)
plt.plot(1e3 * rf.t, np.angle(rf.signal))
plt.xlabel('t (ms)')
plt.ylabel('RF phase')

# The next three blocks contain: three gradient events; a delay; and readout gradient with ADC event, each with
# corresponding fields defining the details of the events.
b2 = seq.get_block(2)
b3 = seq.get_block(3)
b4 = seq.get_block(4)

# Plot the sequence. Visualize the sequence using the `plot()` method of the class. This creates a new figure and shows
# ADC, RF and gradient events. The axes are linked so zooming is consistent. In this example, a simple gradient echo
# sequence for MRI is displayed.
# seq.plot()

"""
The details of individual pulses are not well-represented when the entire sequence is visualized. Interactive zooming
is helpful here. Alternatively, a time range can be specified. An additional parameter also allows the display units to
be changed for easy reading. Further, the handle of the created figure can be returned if required.
"""
# seq.plot(time_range=[0, 16e-3], time_disp='ms')

"""
Modifying sequence blocks. In addition to loading a sequence and accessing sequence blocks, blocks # can be modified.
In this example, a Hamming window is applied to the # first RF pulse of the sequence and the flip angle is changed to
45 degrees. The remaining RF pulses are unchanged.
"""
rf2 = rf
duration = rf2.t[-1]
t = rf2.t - duration / 2  # Center time about 0
alpha = 0.5
BW = 4 / duration  # Time bandwidth product = 4
window = 1.0 - alpha + alpha * np.cos(2 * np.pi * t / duration)  # Hamming window
signal = window * np.sinc(BW * t)

# Normalize area to achieve 2*pi rotation
signal = signal / (seq.rf_raster_time * np.sum(np.real(signal)))

# Scale to 45 degree flip angle
rf2.signal = signal * 45 / 360

b1.rf = rf2
seq.set_block(1, b1)

# Second check to see what has changed
seq.write('read_test2.seq')
# os_system(f'diff -s -u {seq_name} read_test2.seq -echo')  # Linux only

# The amplitude of the first rf pulse is reduced due to the reduced flip-angle. Notice the reduction is not exactly a
# factor of two due to the windowing function.
amp1_in_Hz = max(abs(seq.get_block(1).rf.signal))
amp2_in_Hz = max(abs(seq.get_block(6).rf.signal))
