import unittest

from pypulseq.seq_examples.scripts import write_ute
from pypulseq.tests import base


class TestUTE(unittest.TestCase):
    def test_write_epi(self):
        matlab_seq_filename = "ute_matlab.seq"
        pypulseq_seq_filename = "ute_pypulseq.seq"
        base.main(
            script=write_ute,
            matlab_seq_filename=matlab_seq_filename,
            pypulseq_seq_filename=pypulseq_seq_filename,
        )


if __name__ == "__main__":
    unittest.main()
