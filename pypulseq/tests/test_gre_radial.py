import unittest

from pypulseq.seq_examples.scripts import write_radial_gre
from pypulseq.tests import base


class TestEPISpinEchoRS(unittest.TestCase):
    def test_write_epi(self):
        matlab_seq_filename = "gre_radial_matlab.seq"
        pypulseq_seq_filename = "gre_radial_pypulseq.seq"
        base.main(
            script=write_radial_gre,
            matlab_seq_filename=matlab_seq_filename,
            pypulseq_seq_filename=pypulseq_seq_filename,
        )


if __name__ == "__main__":
    unittest.main()
