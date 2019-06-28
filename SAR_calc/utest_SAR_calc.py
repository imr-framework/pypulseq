# Copyright of the Board of Trustees of Columbia University in the City of New York
# Unit test Script to check for registration functions independently
"""
    1. This script unit tests the SAR calculation

    Parameters
    ----------

        .seq file path : str, optional
        Default is 'rad2D.seq'

    Returns
    -------
        status: int
        0: tests passed
        1: fail

"""

from virtualscanner.utils import constants
import virtualscanner.server.rf.tx.SAR_calc.SAR_calc_main as SAR

SAR_PATH = constants.SAR_PATH

fname = 'rad2D.seq'
payload = SAR.payload_process()