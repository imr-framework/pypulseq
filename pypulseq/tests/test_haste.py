import unittest

from pypulseq.seq_examples.scripts import write_haste
from pypulseq.tests import base


class TestHASTE(unittest.TestCase):
    def test_write_epi(self):
        matlab_seq_filename = "haste_matlab.seq"
        pypulseq_seq_filename = "haste_pypulseq.seq"
        base.main(
            script=write_haste,
            matlab_seq_filename=matlab_seq_filename,
            pypulseq_seq_filename=pypulseq_seq_filename,
        )


if __name__ == "__main__":
    unittest.main()
