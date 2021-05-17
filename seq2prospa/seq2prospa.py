from pypulseq.Sequence.sequence import Sequence
from pypulseq.seq2prospa import convert_seq
from pypulseq.seq2prospa import make_se, make_gre


def main(seq: Sequence):
    prospa = convert_seq.main(seq)
    return prospa


if __name__ == '__main__':
    # seq = make_gre.main()
    seq = make_se.main()
    output = main(seq)

    print(output)