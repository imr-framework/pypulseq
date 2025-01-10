"""Test function to test calc_adc_segments() and compare its output to the pulseq MATLAB function calcAdcSeg()"""

import math
import os

import numpy as np
from pypulseq import Opts, calc_adc_segments

system = Opts.default
system.adc_raster_time = 1e-7
system.grad_raster_time = 1e-5

dirpath = os.path.dirname(__file__)  # noqa: PTH120 (temporary solution)

data = np.genfromtxt(
    dirpath + '/expected_output/pulseq_calcAdcSeg.txt',
    dtype=[
        ('dwell', float),
        ('num_samples', int),
        ('adc_limit', int),
        ('adc_divisor', int),
        ('mode', int),
        ('res_num_seg', int),
        ('res_num_samples_seg', int),
    ],
)


def test_calc_adc_segments():
    # Accessing structured data
    for i, row in enumerate(data):
        dwell = row['dwell']
        num_samples = row['num_samples']
        adc_limit = row['adc_limit']
        adc_divisor = row['adc_divisor']
        res_num_seg = row['res_num_seg']
        res_num_samples_seg = row['res_num_samples_seg']
        n_mode = row['mode']
        mode = 'shorten' if n_mode == 1 else 'lengthen'

        # Similar processing as before...

        system.adc_samples_limit = adc_limit
        system.adc_samples_divisor = adc_divisor

        num_seg, num_samples_seg = calc_adc_segments(num_samples=num_samples, dwell=dwell, system=system, mode=mode)

        # Check if output is identical to matlab pulseq
        assert num_seg == res_num_seg
        assert num_samples_seg == res_num_samples_seg

        # Check if segment samples are below sample limit
        assert num_samples_seg <= adc_limit

        seg_duration = num_samples_seg * dwell
        adc_duration = seg_duration * num_seg

        # Check if each segment is on grad raster time
        assert math.isclose(round(seg_duration / system.grad_raster_time), seg_duration / system.grad_raster_time)
        # Check if total adc is on grad raster time
        assert math.isclose(round(adc_duration / system.grad_raster_time), adc_duration / system.grad_raster_time)

        # Print progress
        if i % 1000 == 0 or i == (data.shape[0] - 1):
            print(round(i / data.shape[0] * 100, 2), '%')


# Can be run with "pytest <file>" or "python <file>"
if __name__ == '__main__':
    test_calc_adc_segments()
