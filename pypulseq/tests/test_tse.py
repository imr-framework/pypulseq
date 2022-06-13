import unittest

from pypulseq.seq_examples.scripts import write_tse
from pypulseq.tests import base


class TestTSE(unittest.TestCase):
    def test_write_epi(self):
        matlab_seq_filename = "tse_matlab.seq"
        pypulseq_seq_filename = "tse_pypulseq.seq"
        base.main(
            script=write_tse,
            matlab_seq_filename=matlab_seq_filename,
            pypulseq_seq_filename=pypulseq_seq_filename,
        )


if __name__ == "__main__":
    unittest.main()
