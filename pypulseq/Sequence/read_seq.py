import re
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from pypulseq.calc_duration import calc_duration
from pypulseq.event_lib import EventLibrary
from pypulseq.supported_labels import get_supported_labels


def read(self, path: str, detect_rf_use: bool = False) -> None:
    """
    Reads a `.seq` file from `path`.

    Parameters
    ----------
    path : Path
        Path of .seq file to be read.
    detect_rf_use : bool, default=False

    Raises
    ------
    ValueError

    RuntimeError
    """

    input_file = open(path, 'r')
    self.shape_library = EventLibrary()
    self.adc_library = EventLibrary()
    self.delay_library = EventLibrary()
    self.grad_library = EventLibrary()
    self.grad_raster_time = self.system.grad_raster_time
    self.rf_library = EventLibrary()
    self.rf_raster_time = self.system.rf_raster_time
    self.label_inc_library = EventLibrary()
    self.label_set_library = EventLibrary()
    self.trigger_library = EventLibrary()

    self.dict_block_events = {}
    self.dict_definitions = {}

    jemris_generated = False

    while True:
        section = __skip_comments(input_file)
        if section == -1:
            break
        if section == '[DEFINITIONS]':
            self.dict_definitions = __read_definitions(input_file)
        elif section == '[JEMRIS]':
            jemris_generated = True
        elif section == '[VERSION]':
            version_major, version_minor, version_revision = __read_version(input_file)

            if version_major != self.version_major:
                raise RuntimeError(f'Unsupported version_major: {version_major}. Expected: {self.version_major}')

            if version_major == 1 and version_minor == 2 and self.version_major == 1 and self.version_minor == 3:
                compatibility_mode_12x_13x = True
            else:
                compatibility_mode_12x_13x = False

                if version_minor != self.version_minor:
                    raise RuntimeError(f'Unsupported version_minor: {version_minor}. Expected: {self.version_minor}')

                if version_revision > self.version_revision:
                    raise RuntimeError(
                        f'Unsupported version_revision: {version_revision}. Expected: {self.version_revision}')

            if not compatibility_mode_12x_13x:
                self.version_major = version_major
                self.version_minor = version_minor
                self.version_revision = version_revision

        elif section == '[BLOCKS]':
            self.dict_block_events = __read_blocks(input_file, compatibility_mode_12x_13x)
        elif section == '[RF]':
            if jemris_generated:
                self.rf_library = __read_events(input_file, (1, 1, 1, 1, 1), event_library=self.rf_library)
            else:
                self.rf_library = __read_events(input_file, (1, 1, 1, 1e-6, 1, 1), event_library=self.rf_library)
        elif section == '[GRADIENTS]':
            self.grad_library = __read_events(input_file, (1, 1, 1e-6), 'g', self.grad_library)
        elif section == '[TRAP]':
            if jemris_generated:
                self.grad_library = __read_events(input_file, (1, 1e-6, 1e-6, 1e-6), 't', self.grad_library)
            else:
                self.grad_library = __read_events(input_file, (1, 1e-6, 1e-6, 1e-6, 1e-6), 't', self.grad_library)
        elif section == '[ADC]':
            self.adc_library = __read_events(input_file, (1, 1e-9, 1e-6, 1, 1), event_library=self.adc_library)
        elif section == '[DELAYS]':
            self.delay_library = __read_events(input_file, (1e-6,), event_library=self.delay_library)
        elif section == '[SHAPES]':
            self.shape_library = __read_shapes(input_file)
        elif section == '[EXTENSIONS]':
            self.extensions_library = __read_events(input_file)
        elif section[:18] == 'extension TRIGGERS':
            extension_id = int(section[18:])
            self.set_extension_string_ID('TRIGGERS', extension_id)
            self.trigger_library = __read_events(input_file, (1, 1, 1e-6, 1e-6), event_library=self.trigger_library)
        elif section[:18] == 'extension LABELSET':
            extension_id = int(section[18:])
            self.set_extension_string_ID('LABELSET', extension_id)
            l1 = lambda s: int(s)
            l2 = lambda s: get_supported_labels().index(s) + 1
            self.label_set_library = __read_and_parse_events(input_file, l1, l2)
        elif section[:18] == 'extension LABELINC':
            extension_id = int(section[18:])
            self.set_extension_string_ID('LABELINC', extension_id)
            l1 = lambda s: int(s)
            l2 = lambda s: get_supported_labels().index(s) + 1
            self.label_inc_library = __read_and_parse_events(input_file, l1, l2)
        else:
            raise ValueError(f'Unknown section code: {section}')

    self.arr_block_durations = np.zeros(len(self.dict_block_events))
    grad_channels = ['gx', 'gy', 'gz']
    grad_prev_last = np.zeros(len(grad_channels))
    for block_counter in range(len(self.dict_block_events)):
        block = self.get_block(block_counter + 1)
        block_duration = calc_duration(block)
        self.arr_block_durations[block_counter] = block_duration
        # We also need to keep track of the event IDs because some PyPulseq files written by external software may contain
        # repeated entries so searching by content will fail
        event_idx = self.dict_block_events[block_counter + 1]
        # Update the objects by filling in the fields not contained in the PyPulseq file
        for j in range(len(grad_channels)):
            if hasattr(block, grad_channels[j]):
                grad = getattr(block, grad_channels[j])
            else:
                grad_prev_last[j] = 0
                continue

            if grad.type == 'grad':
                if grad.delay > 0:
                    grad_prev_last[j] = 0

                if hasattr(grad, 'first'):
                    continue

                grad.first = grad_prev_last[j]
                # Restore samples on the edges of the gradient raster intervals for that we need the first sample
                odd_step1 = [grad.first, *2 * grad.waveform]
                odd_step2 = odd_step1 * (np.mod(range(len(odd_step1)), 2) * 2 - 1)
                waveform_odd_rest = np.cumsum(odd_step2) * (np.mod(len(odd_step2), 2) * 2 - 1)
                grad.lsat = waveform_odd_rest[-1]
                grad_prev_last[j] = grad.last

                eps = np.finfo(np.float).eps
                if grad.delay + len(grad.waveform) * self.grad_raster_time + eps < block_duration:
                    grad_prev_last[j] = 0

                amplitude = np.max(np.abs(grad.waveform))
                old_data = [amplitude, grad.shape_id, grad.delay]
                new_data = [amplitude, grad.shape_id, grad.delay, grad.first, grad.last]
                event_id = event_idx[j + 2]
                # update_data()
            else:
                grad_prev_last[j] = 0

    if detect_rf_use:
        for k in self.rf_library.keys():
            lib_data = self.rf_library.data[k]
            rf = self.rf_from_lib_data(lib_data)
            flip_deg = np.abs(np.sum(rf.signal)) * rf.t[0] * 360
            if len(lib_data) < 9:
                if flip_deg < 90.01:
                    lib_data[8] = 0
                else:
                    lib_data[8] = 2
                self.rf_library.data[k] = lib_data


def __read_definitions(input_file) -> Dict[str, str]:
    """
    Read dict_definitions from .seq file.

    Parameters
    ----------
    input_file : file object
        .seq file

    Returns
    -------
    dict_definitions : dict
        Dict object containing key value pairs of dict_definitions.
    """
    definitions = dict()
    line = __strip_line(input_file)
    while line != '' and line[0] != '#':
        tok = line.split(' ')
        try:  # Try converting every element into a float
            [float(x) for x in tok[1:]]
            definitions[tok[0]] = np.array(tok[1:], dtype=float)
        except ValueError:  # Try clause did not work!
            definitions[tok[0]] = tok[1:]
        line = __strip_line(input_file)

    return definitions


def __read_version(input_file) -> Tuple[int, int, int]:
    """
    Read version from .seq file.

    Parameters
    ----------
    input_file : file object
        .seq file

    Returns
    -------
    tuple
        Tuple of major, minor and revision number.
    """
    line = __strip_line(input_file)
    major, minor, revision = 0, 0, 0
    while line != '' and line[0] != '#':
        tok = line.split(' ')
        if tok[0] == 'major':
            major = int(tok[1])
        elif tok[0] == 'minor':
            minor = int(tok[1])
        elif tok[0] == 'revision':
            revision = tok[1]
        else:
            raise RuntimeError(f'Incompatible version. Expected: {major}{minor}{revision}')
        line = __strip_line(input_file)

    return major, minor, revision


def __read_blocks(input_file, compatibility_mode_12x_13x: bool) -> dict:
    """
    Read Pulseq blocks from .seq file.

    Parameters
    ----------
    input_file : file
        .seq file
    compatibility_mode_12x_13x : bool

    Returns
    -------
    event_table : dict
        Dict object containing key value pairs of Pulseq block ID and block definition.
    """

    line = __strip_line(input_file)

    event_table = dict()
    while line != '' and line != '#':
        block_events = np.fromstring(line, dtype=int, sep=' ')

        if compatibility_mode_12x_13x:
            event_table[block_events[0]] = np.array([*block_events[1:], 0])
        else:
            event_table[block_events[0]] = block_events[1:]

        line = __strip_line(input_file)

    return event_table


def __read_events(input_file, scale: list = (1,), event_type: str = str(),
                  event_library: EventLibrary = EventLibrary()) -> EventLibrary:
    """
    Read Pulseq events from .seq file.

    Parameters
    ----------
    input_file : file object
        .seq file
    scale : list, default=(1,)
        Scaling factor.
    event_type : str
        Type of Pulseq event.
    event_library : EventLibrary, default=EventLibrary()
        EventLibrary

    Returns
    -------
    dict_definitions : dict
        `EventLibrary` object containing Pulseq event dict_definitions.
    """
    line = __strip_line(input_file)

    while line != '' and line != '#':
        data = np.fromstring(line, dtype=float, sep=' ')
        event_id = data[0]
        data = data[1:] * scale
        if event_type == '':
            event_library.insert(key_id=event_id, new_data=data)
        else:
            event_library.insert(key_id=event_id, new_data=data, data_type=event_type)
        line = __strip_line(input_file)

    return event_library


def __read_and_parse_events(input_file, *args) -> EventLibrary:
    event_library = EventLibrary()
    line = __strip_line(input_file)

    while line != '' and line != '#':
        datas = re.split('(\s+)', line)
        datas = [d for d in datas if d != ' ']
        data = np.zeros(len(datas) - 1, dtype=np.int)
        event_id = int(datas[0])
        for i in range(1, len(datas)):
            if i > len(args):
                data[i - 1] = int(datas[i])
            else:
                data[i - 1] = args[i - 1](datas[i])
        event_library.insert(key_id=event_id, new_data=data)
        line = __strip_line(input_file)

    return event_library


def __read_shapes(input_file) -> EventLibrary:
    """
    Read Pulseq shapes from .seq file.

    Parameters
    ----------
    input_file : file
        .seq file

    Returns
    -------
    shape_library : EventLibrary
        `EventLibrary` object containing shape dict_definitions.
    """
    shape_library = EventLibrary()

    line = __skip_comments(input_file)

    while line != -1 and (line != '' or line[0:8] == 'shape_id'):
        tok = line.split(' ')
        id = int(tok[1])
        line = __skip_comments(input_file)
        tok = line.split(' ')
        num_samples = int(tok[1])
        data = []
        line = __skip_comments(input_file)
        while line != '' and line != '#':
            data.append(float(line))
            line = __strip_line(input_file)
        line = __skip_comments(input_file)
        data.insert(0, num_samples)
        data = np.asarray(data)
        shape_library.insert(key_id=id, new_data=data)
    return shape_library


def __skip_comments(input_file) -> str:
    """
    Skip one '#' comment in .seq file.

    Parameters
    ----------
    input_file : file
        .seq file

    Returns
    -------
    line : str
        First line in `input_file` after skipping one '#' comment block. Note: File pointer is remembered, so successive calls work as expected.
    """

    line = __strip_line(input_file)

    while line != -1 and (line == '' or line[0] == '#'):
        line = __strip_line(input_file)

    return line


def __strip_line(input_file) -> str:
    """
    Removes spaces and newline whitespaces.

    Parameters
    ----------
    input_file : file
        .seq file

    Returns
    -------
    line : str
        First line in input_file after removing spaces and newline whitespaces. Note: File pointer is remembered,
        so successive calls work as expected. Returns -1 for eof.
    """
    line = input_file.readline()  # If line is an empty string, end of the file has been reached
    return line.strip() if line != '' else -1
