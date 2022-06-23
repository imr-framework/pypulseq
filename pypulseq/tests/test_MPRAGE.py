import unittest

from pypulseq.seq_examples.scripts import write_MPRAGE
from pypulseq.tests import base


class TestMPRAGE(unittest.TestCase):
    def test_write_epi(self):
        matlab_seq_filename = "mprage_matlab.seq"
        pypulseq_seq_filename = "mprage_pypulseq.seq"
        base.main(
            script=write_MPRAGE,
            matlab_seq_filename=matlab_seq_filename,
            pypulseq_seq_filename=pypulseq_seq_filename,
        )


if __name__ == "__main__":
    unittest.main()
