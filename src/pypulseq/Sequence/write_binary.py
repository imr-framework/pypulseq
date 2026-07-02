from __future__ import annotations

import hashlib
from typing import Any, Union

import numpy as np


def write_binary(self, filename: str, create_signature: bool = True) -> Union[str, None]:
    codes = self.get_binary_codes()
    with open(filename, 'wb') as fid:
        _write_int64(fid, codes['fileHeader'])
        _write_int64(fid, int(self.version_major))
        _write_int64(fid, int(self.version_minor))
        _write_int64(fid, int(self.version_revision))

        if self.definitions:
            _write_int64(fid, codes['section']['definitions'])
            _write_int64(fid, len(self.definitions))
            for key, val in self.definitions.items():
                key_bytes = key.encode('ascii')
                _write_int32(fid, len(key_bytes))
                fid.write(key_bytes)
                if isinstance(val, str):
                    data = val.encode('ascii')
                    _write_int32(fid, len(data))
                    fid.write(b'c')
                    fid.write(data)
                else:
                    arr = np.asarray(val)
                    _write_int32(fid, arr.size)
                    if np.issubdtype(arr.dtype, np.integer):
                        fid.write(b'i')
                        fid.write(arr.astype(np.int32).reshape(-1).tobytes())
                    elif np.issubdtype(arr.dtype, np.floating):
                        fid.write(b'f')
                        fid.write(arr.astype(np.float64).reshape(-1).tobytes())
                    else:
                        raise TypeError(f'unknown type of the value type for {key}')

        _write_int64(fid, codes['section']['blocks'])
        _write_int64(fid, len(self.block_events))
        for block_id, block_event in self.block_events.items():
            block_event = np.asarray(block_event, dtype=np.int32).reshape(-1)
            block_duration = self.block_durations[block_id] / self.block_duration_raster
            block_duration_raster = round(block_duration)
            if abs(block_duration_raster - block_duration) >= 1e-6:
                raise AssertionError('Block duration is not aligned to block_duration_raster')
            _write_int64(fid, block_duration_raster)
            fid.write(block_event[1:7].astype(np.int32).tobytes())

        if self.rf_library.data:
            _write_int64(fid, codes['section']['rf'])
            keys = list(self.rf_library.data.keys())
            _write_int64(fid, len(keys))
            for k in keys:
                data = np.asarray(self.rf_library.data[k], dtype=float)
                _write_int32(fid, k)
                _write_float64(fid, data[0])
                fid.write(data[1:4].astype(np.int32).tobytes())
                _write_int64(fid, round(data[4] * 1e12))
                _write_int64(fid, round(data[5] * 1e12))
                fid.write(data[6:10].astype(np.float64).tobytes())
                fid.write(str(self.rf_library.type.get(k, 'u'))[:1].encode('ascii'))

        arb_keys = [k for k, t in self.grad_library.type.items() if t == 'g']
        trap_keys = [k for k, t in self.grad_library.type.items() if t == 't']

        if arb_keys:
            _write_int64(fid, codes['section']['gradients'])
            _write_int64(fid, len(arb_keys))
            for k in arb_keys:
                data = np.asarray(self.grad_library.data[k], dtype=float)
                _write_int32(fid, k)
                fid.write(data[0:3].astype(np.float64).tobytes())
                fid.write(data[3:5].astype(np.int32).tobytes())
                _write_int64(fid, round(data[5] * 1e12))

        if trap_keys:
            _write_int64(fid, codes['section']['trapezoids'])
            _write_int64(fid, len(trap_keys))
            for k in trap_keys:
                data = np.asarray(self.grad_library.data[k], dtype=float)
                _write_int32(fid, k)
                _write_float64(fid, data[0])
                fid.write(np.round(data[1:5] * 1e12).astype(np.int64).tobytes())

        if self.adc_library.data:
            _write_int64(fid, codes['section']['adc'])
            keys = list(self.adc_library.data.keys())
            _write_int64(fid, len(keys))
            for k in keys:
                data = np.asarray(self.adc_library.data[k], dtype=float)
                _write_int32(fid, k)
                _write_int64(fid, round(data[0]))
                _write_int64(fid, round(data[1] * 1e12))
                _write_int64(fid, round(data[2] * 1e12))
                fid.write(data[3:7].astype(np.float64).tobytes())
                _write_int32(fid, round(data[7]))

        if self.shape_library.data:
            _write_int64(fid, codes['section']['shapes'])
            keys = list(self.shape_library.data.keys())
            _write_int64(fid, len(keys))
            for k in keys:
                shape = np.asarray(self.shape_library.data[k], dtype=float)
                num_samples = int(shape[0])
                data = shape[1:]
                _write_int32(fid, k)
                _write_int64(fid, num_samples)
                _write_int64(fid, len(data))
                fid.write(data.astype(np.float32).tobytes())

        if self.extensions_library.data:
            _write_int64(fid, codes['section']['extensions'])
            keys = list(self.extensions_library.data.keys())
            _write_int64(fid, len(keys))
            for k in keys:
                _write_int32(fid, k)
                fid.write(np.asarray(self.extensions_library.data[k], dtype=np.int32).reshape(-1).tobytes())

        if self.trigger_library.data:
            _write_int64(fid, codes['section']['triggers'])
            _write_int32(fid, self.get_extension_type_ID('TRIGGERS'))
            keys = list(self.trigger_library.data.keys())
            _write_int64(fid, len(keys))
            for k in keys:
                data = np.asarray(self.trigger_library.data[k], dtype=float)
                _write_int32(fid, k)
                fid.write(data[0:2].astype(np.int32).tobytes())
                fid.write(np.round(data[2:4] * 1e12).astype(np.int64).tobytes())

        if self.label_set_library.data:
            _write_int64(fid, codes['section']['labelset'])
            _write_int32(fid, self.get_extension_type_ID('LABELSET'))
            _write_int64(fid, len(self.label_set_library.data))
            for k, data in self.label_set_library.data.items():
                _write_int32(fid, k)
                fid.write(np.asarray(data, dtype=np.int32).reshape(-1).tobytes())

        if self.label_inc_library.data:
            _write_int64(fid, codes['section']['labelinc'])
            _write_int32(fid, self.get_extension_type_ID('LABELINC'))
            _write_int64(fid, len(self.label_inc_library.data))
            for k, data in self.label_inc_library.data.items():
                _write_int32(fid, k)
                fid.write(np.asarray(data, dtype=np.int32).reshape(-1).tobytes())

        if self.soft_delay_library.data:
            _write_int64(fid, codes['section']['softdelays'])
            _write_int32(fid, self.get_extension_type_ID('DELAYS'))
            _write_int64(fid, len(self.soft_delay_library.data))
            for k, data in self.soft_delay_library.data.items():
                data = np.asarray(data, dtype=float)
                hint = self.soft_delay_hints2[int(data[3]) - 1]
                _write_int32(fid, k)
                _write_int32(fid, round(data[0]))
                _write_int64(fid, round(data[1] * 1e12))
                _write_float64(fid, data[2])
                _write_int32(fid, len(hint))
                fid.write(hint.encode('ascii'))

    if create_signature:
        with open(filename, 'rb') as fid:
            payload = fid.read()
        md5_hash = hashlib.md5(payload).hexdigest()

        self.signature_type = 'md5'
        self.signature_file = 'bin'
        self.signature_value = md5_hash

        with open(filename, 'ab') as fid:
            signed_len = fid.tell()
            _write_int64(fid, codes['section']['signature'])
            sig_type = self.signature_type.encode('ascii')
            _write_int32(fid, len(sig_type))
            fid.write(sig_type)
            sig_bytes = bytes.fromhex(md5_hash)
            _write_int32(fid, len(sig_bytes))
            fid.write(sig_bytes)
            _write_int64(fid, signed_len)

        return md5_hash

    return None


def _write_int8(fid: Any, value: int) -> None:
    fid.write(np.asarray(value, dtype=np.int8).tobytes())


def _write_int32(fid: Any, value: int) -> None:
    fid.write(np.asarray(value, dtype=np.int32).tobytes())


def _write_int64(fid: Any, value: int) -> None:
    fid.write(np.asarray(value, dtype=np.int64).tobytes())


def _write_float64(fid: Any, value: float) -> None:
    fid.write(np.asarray(value, dtype=np.float64).tobytes())
