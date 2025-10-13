# import the libraries
import numpy as np
import pypulseq as pp
from seq2seqlite_adaptive import seqlite

# ======
# INITIATE THE PHILIPS INTERPRETER OBJECT
# ======
# Read the sequence file
# seq_file = './tse_pypulseq.seq' # gradient with no amplitude 
# seq_file = './epi_se_pypulseq.seq'
seq_file = './gre_pypulseq_64_200.seq'
Seq_lite_file = './gre_pypulseq_64_200.seqlite'
seq_file_new = './gre_pypulseq_64_200_new.seq'
new_seq = seqlite(seq_file)

# ffe.view_base_TR(mode='blocks')
new_seq.write(Seq_lite_file)
new_seq.visualize = True  # Set to True for visualization
new_seq.convert_seqlite_to_seq(Seq_lite_file, seq_file_new)

