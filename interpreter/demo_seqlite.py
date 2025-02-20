# import the libraries
import numpy as np
import pypulseq as pp
from seq2seqlite import seqlite

# ======
# INITIATE THE PHILIPS INTERPRETER OBJECT
# ======
# Read the sequence file
seq_file = './gre_pypulseq.seq'
Philips_seq_file = './gre_pypulseq.seq-lite'
ffe = seqlite(seq_file)
ffe.view_base_TR(mode='blocks')
ffe.write_seqlite(Philips_seq_file)