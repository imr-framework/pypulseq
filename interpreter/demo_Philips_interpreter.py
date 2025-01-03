# import the libraries
import numpy as np
import pypulseq as pp
from seq2Philips import PhilipsInterpreter

# ======
# INITIATE THE PHILIPS INTERPRETER OBJECT
# ======
# Read the sequence file
seq_file = './tse_pypulseq.seq'
ffe = PhilipsInterpreter(seq_file)
ffe.view_base_TR()