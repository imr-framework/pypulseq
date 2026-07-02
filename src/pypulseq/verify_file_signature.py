from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Tuple, Union


def verify_file_signature(pulseq_file: Union[str, Path]) -> Tuple[bool, str, str]:
    """
    Verify the MD5 signature stored in a Pulseq text or binary file.

    Returns
    -------
    result, stored_signature, computed_signature
        ``result`` is True when the stored and recomputed signatures match.
    """
    path = Path(pulseq_file)
    data = path.read_bytes()
    if len(data) < 8:
        raise ValueError(f'File is too short to contain a Pulseq signature: {path}')

    from pypulseq.Sequence.sequence import Sequence

    codes = Sequence.get_binary_codes()
    magic_num = int.from_bytes(data[:8], byteorder='little', signed=True)
    if magic_num == codes['fileHeader']:
        stored_signature, signed_len = _read_binary_signature(data, codes)
    else:
        stored_signature, signed_len = _read_text_signature(data)

    computed_signature = hashlib.md5(data[:signed_len]).hexdigest()
    return computed_signature == stored_signature, stored_signature, computed_signature


def _read_binary_signature(data: bytes, codes) -> Tuple[str, int]:
    if len(data) < 16:
        raise ValueError('Binary Pulseq file is too short to contain a signature')

    signed_len = int.from_bytes(data[-8:], byteorder='little', signed=True)
    if signed_len < 0 or signed_len + 8 > len(data):
        raise ValueError('Invalid binary Pulseq signature length')

    pos = signed_len
    section = int.from_bytes(data[pos : pos + 8], byteorder='little', signed=True)
    pos += 8
    if section != codes['section']['signature']:
        raise ValueError('Binary Pulseq signature section not found')

    type_len = int.from_bytes(data[pos : pos + 4], byteorder='little', signed=True)
    pos += 4
    sig_type = data[pos : pos + type_len].decode('ascii')
    pos += type_len
    if sig_type.lower() != 'md5':
        raise ValueError(f'Unsupported signature type: {sig_type}')

    hash_len = int.from_bytes(data[pos : pos + 4], byteorder='little', signed=True)
    pos += 4
    return data[pos : pos + hash_len].hex(), signed_len


def _read_text_signature(data: bytes) -> Tuple[str, int]:
    marker_pos = data.find(b'\n[SIGNATURE]\n')
    marker_len = len(b'\n[SIGNATURE]\n')
    if marker_pos < 0:
        marker_pos = data.find(b'\r\n[SIGNATURE]\r\n')
        marker_len = len(b'\r\n[SIGNATURE]\r\n')
    if marker_pos < 0:
        raise ValueError('Text Pulseq signature section not found')

    sig_type = ''
    sig_hash = ''
    for raw_line in data[marker_pos + marker_len :].splitlines():
        line = raw_line.decode('ascii', errors='ignore').strip()
        if not line or line.startswith('#'):
            continue
        if line.startswith('Type '):
            sig_type = line[5:].strip()
        elif line.startswith('Hash '):
            sig_hash = line[5:].strip().lower()

    if sig_type.lower() != 'md5' or not sig_hash:
        raise ValueError('Failed to read text Pulseq MD5 signature')

    return sig_hash, marker_pos
