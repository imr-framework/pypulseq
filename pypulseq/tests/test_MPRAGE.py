import unittest

import pytest

from pypulseq.seq_examples.scripts import write_mprage
from pypulseq.tests import base


@pytest.mark.matlab_seq_comp
class TestMPRAGE(unittest.TestCase):
    def test_write_epi(self):
        matlab_seq_filename = 'mprage_matlab.seq'
        pypulseq_seq_filename = 'mprage_pypulseq.seq'
        base.main(
            script=write_mprage,
            matlab_seq_filename=matlab_seq_filename,
            pypulseq_seq_filename=pypulseq_seq_filename,
        )


if __name__ == '__main__':
    unittest.main()
