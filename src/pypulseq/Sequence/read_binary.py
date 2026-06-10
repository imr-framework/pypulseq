from __future__ import annotations

from collections import OrderedDict

import numpy as np

from pypulseq.event_lib import EventLibrary


def read_binary(self, filename: str) -> None:
    codes = self.get_binary_codes()
    with open(filename, 'rb') as fid:
        magic_num = _read_scalar(fid, np.int64)
        if magic_num != codes['fileHeader']:
            raise ValueError('Not a Pulseq binary file')
        version_major = _read_scalar(fid, np.int64)
        version_minor = _read_scalar(fid, np.int64)
        version_revision = _read_scalar(fid, np.int64)
        if version_major != int(self.version_major):
            raise ValueError(f'Unsupported version_major {version_major}')
        if version_minor != int(self.version_minor):
            raise ValueError(f'Unsupported version_minor {version_minor}')
        if version_revision != int(self.version_revision):
            raise ValueError(f'Unsupported version_revision {version_revision}')

        self.version_major = int(version_major)
        self.version_minor = int(version_minor)
        self.version_revision = int(version_revision)

        self.block_events = OrderedDict()
        self.block_durations = {}
        self.definitions = {}
        self.grad_library = EventLibrary(numpy_data=True)
        self.shape_library = EventLibrary(numpy_data=True)
        self.rf_library = EventLibrary(numpy_data=True)
        self.adc_library = EventLibrary(numpy_data=True)
        self.delay_library = EventLibrary(numpy_data=True)
        self.trigger_library = EventLibrary(numpy_data=True)
        self.label_set_library = EventLibrary(numpy_data=True)
        self.label_inc_library = EventLibrary(numpy_data=True)
        self.extensions_library = EventLibrary(numpy_data=True)
        self.soft_delay_library = EventLibrary(numpy_data=True)
        self.soft_delay_hints = {}
        self.soft_delay_hint_ids = {}
        self.soft_delay_hints2 = []
        self.extension_string_idx = []
        self.extension_numeric_idx = []
        self.signature_type = ''
        self.signature_file = ''
        self.signature_value = ''
        self.block_cache.clear()

        while True:
            section_raw = fid.read(8)
            if not section_raw:
                break
            section = int(np.frombuffer(section_raw, dtype=np.int64)[0])
            if section == codes['section']['definitions']:
                self.definitions = _read_definitions(fid)
                _sync_rasters_from_definitions(self)
            elif section == codes['section']['blocks']:
                self.block_events, self.block_durations = _read_blocks(fid, self.block_duration_raster)
            elif section == codes['section']['rf']:
                _read_rf(fid, self.rf_library)
            elif section == codes['section']['gradients']:
                _read_gradients(fid, self.grad_library)
            elif section == codes['section']['trapezoids']:
                _read_trapezoids(fid, self.grad_library)
            elif section == codes['section']['adc']:
                _read_adc(fid, self.adc_library)
            elif section == codes['section']['delays']:
                _read_legacy_delays(fid)
            elif section == codes['section']['shapes']:
                self.shape_library = _read_shapes(fid)
            elif section == codes['section']['extensions']:
                _read_extensions(fid, self.extensions_library)
            elif section == codes['section']['triggers']:
                self.set_extension_string_ID('TRIGGERS', int(_read_scalar(fid, np.int32)))
                _read_triggers(fid, self.trigger_library)
            elif section == codes['section']['labelset']:
                self.set_extension_string_ID('LABELSET', int(_read_scalar(fid, np.int32)))
                _read_labels(fid, self.label_set_library)
            elif section == codes['section']['labelinc']:
                self.set_extension_string_ID('LABELINC', int(_read_scalar(fid, np.int32)))
                _read_labels(fid, self.label_inc_library)
            elif section == codes['section']['softdelays']:
                self.set_extension_string_ID('DELAYS', int(_read_scalar(fid, np.int32)))
                _read_soft_delays(fid, self)
            elif section == codes['section']['signature']:
                sig_type, sig_value = _read_signature(fid)
                self.signature_type = sig_type
                self.signature_file = 'bin'
                self.signature_value = sig_value
            else:
                raise ValueError(f'Unknown section code: {section:x}')


def _sync_rasters_from_definitions(self) -> None:
    if 'GradientRasterTime' in self.definitions:
        self.grad_raster_time = float(np.asarray(self.definitions['GradientRasterTime']).reshape(-1)[0])
    if 'RadiofrequencyRasterTime' in self.definitions:
        self.rf_raster_time = float(np.asarray(self.definitions['RadiofrequencyRasterTime']).reshape(-1)[0])
    if 'AdcRasterTime' in self.definitions:
        self.adc_raster_time = float(np.asarray(self.definitions['AdcRasterTime']).reshape(-1)[0])
    if 'BlockDurationRaster' in self.definitions:
        self.block_duration_raster = float(np.asarray(self.definitions['BlockDurationRaster']).reshape(-1)[0])


def _read_scalar(fid, dtype):
    return np.frombuffer(fid.read(np.dtype(dtype).itemsize), dtype=dtype)[0]


def _read_definitions(fid):
    defs = {}
    num_defs = int(_read_scalar(fid, np.int64))
    for _ in range(num_defs):
        key, legacy_format = _read_definition_key(fid)
        if legacy_format:
            count = int(np.frombuffer(fid.read(1), dtype=np.int8)[0])
        else:
            count = int(_read_scalar(fid, np.int32))
        value_type = fid.read(1).decode('ascii')
        if value_type == 'f':
            values = np.frombuffer(fid.read(8 * count), dtype=np.float64).copy()
        elif value_type == 'i':
            values = np.frombuffer(fid.read(4 * count), dtype=np.int32).copy()
        elif value_type == 'c':
            values = fid.read(count).decode('ascii')
            if values.endswith('\x00'):
                values = values[:-1]
        else:
            raise ValueError(f'Unknown definition type: {value_type}')
        defs[key] = values
    return defs


def _read_definition_key(fid):
    pos = fid.tell()
    raw_len = fid.read(4)
    if len(raw_len) != 4:
        raise EOFError('Unexpected end of file while reading definition key length')
    key_len = int(np.frombuffer(raw_len, dtype=np.int32)[0])
    if 0 < key_len < 4096:
        key_bytes = fid.read(key_len)
        try:
            return key_bytes.decode('ascii'), False
        except UnicodeDecodeError:
            pass

    fid.seek(pos)
    key_bytes = bytearray()
    while True:
        c = fid.read(1)
        if c == b'':
            raise EOFError('Unexpected end of file while reading legacy definition key')
        if c == b'\x00':
            break
        key_bytes.extend(c)
    return key_bytes.decode('ascii'), True


def _read_signature(fid):
    type_len = int(_read_scalar(fid, np.int32))
    sig_type = fid.read(type_len).decode('ascii')
    hash_len = int(_read_scalar(fid, np.int32))
    sig_value = fid.read(hash_len).hex()
    _read_scalar(fid, np.int64)
    return sig_type, sig_value


def _read_blocks(fid, block_duration_raster):
    num_blocks = int(_read_scalar(fid, np.int64))
    events = OrderedDict()
    durations = {}
    for ii in range(num_blocks):
        duration_raster = float(_read_scalar(fid, np.int64))
        event_ids = np.frombuffer(fid.read(4 * 6), dtype=np.int32).copy()
        events[ii + 1] = np.concatenate(([0], event_ids)).astype(np.int32)
        durations[ii + 1] = duration_raster * block_duration_raster
    return events, durations


def _read_rf(fid, library):
    num_events = int(_read_scalar(fid, np.int64))
    for _ in range(num_events):
        event_id = int(_read_scalar(fid, np.int32))
        amp = float(_read_scalar(fid, np.float64))
        shape_ids = np.frombuffer(fid.read(4 * 3), dtype=np.int32).astype(float)
        center = float(_read_scalar(fid, np.int64)) * 1e-12
        delay = float(_read_scalar(fid, np.int64)) * 1e-12
        offsets = np.frombuffer(fid.read(8 * 4), dtype=np.float64).copy()
        use = fid.read(1).decode('ascii')
        library.insert(event_id, np.concatenate(([amp], shape_ids, [center, delay], offsets)), use)


def _read_gradients(fid, library):
    num_events = int(_read_scalar(fid, np.int64))
    for _ in range(num_events):
        event_id = int(_read_scalar(fid, np.int32))
        amp_first_last = np.frombuffer(fid.read(8 * 3), dtype=np.float64).copy()
        shape_ids = np.frombuffer(fid.read(4 * 2), dtype=np.int32).astype(float)
        delay = float(_read_scalar(fid, np.int64)) * 1e-12
        library.insert(event_id, np.concatenate((amp_first_last, shape_ids, [delay])), 'g')


def _read_trapezoids(fid, library):
    num_events = int(_read_scalar(fid, np.int64))
    for _ in range(num_events):
        event_id = int(_read_scalar(fid, np.int32))
        amp = float(_read_scalar(fid, np.float64))
        times = np.frombuffer(fid.read(8 * 4), dtype=np.int64).astype(float) * 1e-12
        library.insert(event_id, np.concatenate(([amp], times)), 't')


def _read_adc(fid, library):
    num_events = int(_read_scalar(fid, np.int64))
    for _ in range(num_events):
        event_id = int(_read_scalar(fid, np.int32))
        num = float(_read_scalar(fid, np.int64))
        dwell = float(_read_scalar(fid, np.int64)) * 1e-12
        delay = float(_read_scalar(fid, np.int64)) * 1e-12
        offsets = np.frombuffer(fid.read(8 * 4), dtype=np.float64).copy()
        phase_id = float(_read_scalar(fid, np.int32))
        library.insert(event_id, np.concatenate(([num, dwell, delay], offsets, [phase_id])))


def _read_legacy_delays(fid):
    num_events = int(_read_scalar(fid, np.int64))
    fid.read(num_events * (4 + 8))


def _read_shapes(fid):
    shape_library = EventLibrary(numpy_data=True)
    num_shapes = int(_read_scalar(fid, np.int64))
    for _ in range(num_shapes):
        shape_id = int(_read_scalar(fid, np.int32))
        num_uncompressed = int(_read_scalar(fid, np.int64))
        num_compressed = int(_read_scalar(fid, np.int64))
        data = np.frombuffer(fid.read(4 * num_compressed), dtype=np.float32).astype(float)
        shape_library.insert(shape_id, np.concatenate(([num_uncompressed], data)))
    return shape_library


def _read_extensions(fid, library):
    num_events = int(_read_scalar(fid, np.int64))
    for _ in range(num_events):
        event_id = int(_read_scalar(fid, np.int32))
        data = np.frombuffer(fid.read(4 * 3), dtype=np.int32).copy()
        library.insert(event_id, data)


def _read_triggers(fid, library):
    num_events = int(_read_scalar(fid, np.int64))
    for _ in range(num_events):
        event_id = int(_read_scalar(fid, np.int32))
        type_channel = np.frombuffer(fid.read(4 * 2), dtype=np.int32).astype(float)
        delay_duration = np.frombuffer(fid.read(8 * 2), dtype=np.int64).astype(float) * 1e-12
        library.insert(event_id, np.concatenate((type_channel, delay_duration)))


def _read_labels(fid, library):
    num_events = int(_read_scalar(fid, np.int64))
    for _ in range(num_events):
        event_id = int(_read_scalar(fid, np.int32))
        data = np.frombuffer(fid.read(4 * 2), dtype=np.int32).copy()
        library.insert(event_id, data)


def _read_soft_delays(fid, seq):
    num_events = int(_read_scalar(fid, np.int64))
    for _ in range(num_events):
        event_id = int(_read_scalar(fid, np.int32))
        num = float(_read_scalar(fid, np.int32))
        offset = float(_read_scalar(fid, np.int64)) * 1e-12
        factor = float(_read_scalar(fid, np.float64))
        hint_len = int(_read_scalar(fid, np.int32))
        hint = fid.read(hint_len).decode('ascii')
        if hint in seq.soft_delay_hint_ids:
            hint_id = seq.soft_delay_hint_ids[hint]
        else:
            hint_id = len(seq.soft_delay_hints2) + 1
            seq.soft_delay_hint_ids[hint] = hint_id
            seq.soft_delay_hints2.append(hint)
        seq.soft_delay_hints[hint] = int(num)
        seq.soft_delay_library.insert(event_id, np.array([num, offset, factor, hint_id], dtype=float))
