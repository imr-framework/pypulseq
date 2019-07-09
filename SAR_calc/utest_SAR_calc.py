# Copyright of the Board of Trustees of Columbia University in the City of New York
"""
    This script unit tests the SAR calculation

    Parameters
    ----------
    .seq : str, optional
        Default is 'rad2D.seq'

    Returns
    -------
    status: int
      0: tests passed
      1: fail

"""

import virtualscanner.server.rf.tx.SAR_calc.SAR_calc_main as SAR
from virtualscanner.utils import constants

SAR_PATH = constants.RF_SAR_PATH

if __name__ == '__main__':
    fname = 'rad2D.seq'
    payload = SAR.payload_process()
