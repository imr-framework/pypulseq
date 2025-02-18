# import the libraries
import numpy as np
import pypulseq as pp
from seq2Philips import PhilipsTranslator

# ======
# INITIATE THE PHILIPS INTERPRETER OBJECT
# ======
# Read the sequence file
seq_file = './gre_pypulseq.seq'
Philips_seq_file = './gre_pypulseq.Philips-seq'
ffe = PhilipsTranslator(seq_file)
ffe.view_base_TR(mode='blocks')
ffe.write_philips_seq(Philips_seq_file)