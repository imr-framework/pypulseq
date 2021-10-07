from pypulseq.Sequence import sequence
#seq=sequence.sequence() #Create a new sequence object
fov=250e-3 #Define FOV
Nx=256 #Define resolution
alpha=10 #flip angle
sliceThickness=3e-3 # slice
TE=8e-3 # TE; give a vector here to have multiple TEs (e.g. for field mapping)
TR=100e-3 # only a single value for now
Nr=128 #number of radial spokes
Ndummy=20 # number of dummy scans
delta=pi / Nr 