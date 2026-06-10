import math
import tempfile

import pypulseq as pp


def test_binary_signature_roundtrip_multiple_sequences(tmp_path):
    seqs = make_test_sequences()

    for index, seq in enumerate(seqs):
        path_bin = tmp_path / f'signature_{index + 1}.bseq'

        seq.write_binary(path_bin)

        seq_loaded = pp.Sequence()
        seq_loaded.read_binary(path_bin)

        signature_valid, stored_signature, computed_signature = pp.verify_file_signature(path_bin)

        assert seq_loaded.signature_type.lower() == 'md5'
        assert seq_loaded.signature_file.lower() == 'bin'
        assert signature_valid
        assert stored_signature == computed_signature

    print('Test function test_binary_signature_roundtrip_multiple_sequences completed successfully')


def make_test_sequences():
    seq1 = pp.Sequence()
    gx = pp.make_trapezoid('x', area=1000, duration=1e-3)
    adc = pp.make_adc(128, duration=1e-3)
    seq1.add_block(gx, adc)
    seq1.add_block(pp.make_delay(2e-3))

    seq2 = pp.Sequence()
    rf, gz, _ = pp.make_sinc_pulse(
        math.pi / 2,
        duration=2e-3,
        slice_thickness=5e-3,
        apodization=0.5,
        time_bw_product=4,
        use='excitation',
        return_gz=True,
    )
    gz_reph = pp.make_trapezoid('z', area=-gz.area / 2, duration=1e-3)
    seq2.add_block(rf, gz)
    seq2.add_block(gz_reph)
    seq2.add_block(pp.make_delay(1e-3))

    return [seq1, seq2]


if __name__ == '__main__':
    with tempfile.TemporaryDirectory() as tmp_dir:
        test_binary_signature_roundtrip_multiple_sequences(Path(tmp_dir))
