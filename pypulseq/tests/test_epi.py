import unittest

from pypulseq.seq_examples.scripts import write_epi
from pypulseq.tests import base


class TestEPI(unittest.TestCase):
    def test_write_epi(self):
        matlab_seq_filename = "epi_matlab.seq"
        pypulseq_seq_filename = "epi_pypulseq.seq"
        base.main(
            script=write_epi,
            matlab_seq_filename=matlab_seq_filename,
            pypulseq_seq_filename=pypulseq_seq_filename,
        )


if __name__ == "__main__":
    unittest.main()
