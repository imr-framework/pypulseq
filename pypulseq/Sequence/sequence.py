import numpy as np
from matplotlib import pyplot as plt

from pypulseq.Sequence import block
from pypulseq.Sequence.read_seq import read
from pypulseq.Sequence.write_seq import write
from pypulseq.calc_duration import calc_duration
from pypulseq.event_lib import EventLibrary
from pypulseq.opts import Opts


class Sequence:
    def __init__(self, system=Opts()):
        self.system = system
        # EventLibrary.data is a dict
        self.shape_library = EventLibrary()
        self.rf_library = EventLibrary()
        self.grad_library = EventLibrary()
        self.adc_library = EventLibrary()
        self.delay_library = EventLibrary()
        self.block_events = {}
        self.rf_raster_time = self.system.rf_raster_time
        self.grad_raster_time = self.system.grad_raster_time

    def __str__(self):
        s = "Sequence:"
        s += "\nshape_library: " + str(self.shape_library)
        s += "\nrf_library: " + str(self.rf_library)
        s += "\ngrad_library: " + str(self.grad_library)
        s += "\nadc_library: " + str(self.adc_library)
        s += "\ndelay_library: " + str(self.delay_library)
        s += "\nrf_raster_time: " + str(self.rf_raster_time)
        s += "\ngrad_raster_time: " + str(self.grad_raster_time)
        s += "\nblock_events: " + str(len(self.block_events))
        return s

    def duration(self):
        num_blocks = len(self.block_events)
        event_count = 0
        duration = 0
        for i in range(1, num_blocks + 1):
            block = self.get_block(i)
            duration += calc_duration(block)

        return duration, num_blocks, event_count

    def add_block(self, *args):
        """
        Adds supplied list of Holder objects as a Block.

        Parameters
        ----------
        args : list
            List of Holder objects to be added as a Block.
        """

        block.add_block(self, len(self.block_events) + 1, *args)

    def get_block(self, block_index):
        """
        Returns Block at block_index.

        Parameters
        ----------
        block_index : int
            Index of required block.
        """

        return block.get_block(self, block_index)

    def read(self, file_path):
        read(self, file_path)

    def write(self, name):
        write(self, name)

    def plot(self, time_range=(0, np.inf)):
        """
        Show Matplotlib plot of all Events in the Sequence object.

        Parameters
        ----------
        time_range : List
            Time range (x-axis limits) for plot to be shown. Default is 0 to infinity (entire plot shown).
        """

        fig1, fig2 = plt.figure(1), plt.figure(2)
        f11, f12, f13 = fig1.add_subplot(311), fig1.add_subplot(312), fig1.add_subplot(313)
        f2 = [fig2.add_subplot(311), fig2.add_subplot(312), fig2.add_subplot(313)]
        t0 = 0
        for iB in range(1, len(self.block_events) + 1):
            block = self.get_block(iB)
            is_valid = time_range[0] <= t0 <= time_range[1]
            if is_valid:
                if block is not None:
                    if 'adc' in block:
                        adc = block['adc']
                        t = adc.delay + [(x * adc.dwell) for x in range(0, int(adc.num_samples))]
                        f11.plot((t0 + t), np.zeros(len(t)))
                    if 'rf' in block:
                        rf = block['rf']
                        t = rf.t
                        f12.plot(np.squeeze(t0 + t), abs(rf.signal))
                        f13.plot(np.squeeze(t0 + t), np.angle(rf.signal))
                    grad_channels = ['gx', 'gy', 'gz']
                    for x in range(0, len(grad_channels)):
                        if grad_channels[x] in block:
                            grad = block[grad_channels[x]]
                            if grad.type == 'grad':
                                t = grad.t
                                waveform = 1e-3 * grad.waveform
                            else:
                                t = np.cumsum([0, grad.rise_time, grad.flat_time, grad.fall_time])
                                waveform = [1e-3 * grad.amplitude * x for x in [0, 1, 1, 0]]
                            f2[x].plot(np.squeeze(t0 + t), waveform)
            t0 += calc_duration(block)

        f11.set_ylabel('adc')
        f12.set_ylabel('rf mag hz')
        f13.set_ylabel('rf phase rad')
        [f2[x].set_ylabel(grad_channels[x]) for x in range(3)]
        # Setting display limits
        disp_range = [time_range[0], min(t0, time_range[1])]
        f11.set_xlim(disp_range)
        f12.set_xlim(disp_range)
        f13.set_xlim(disp_range)
        [x.set_xlim(disp_range) for x in f2]

        plt.show()
