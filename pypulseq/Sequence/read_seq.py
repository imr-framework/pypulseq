import numpy as np

from pypulseq.event_lib import EventLibrary


def read(self, path):
    """
    Reads .seq file from path, and constructs a Sequence object from the file.

    Parameters
    ----------
    path : str
        Path of .seq file to be read.
    """

    input_file = open(path, 'r')
    self.shape_library = EventLibrary()
    self.rf_library = EventLibrary()
    self.grad_library = EventLibrary()
    self.adc_library = EventLibrary()
    self.delay_library = EventLibrary()
    self.block_events = {}
    self.rf_raster_time = self.system.rf_raster_time
    self.grad_raster_time = self.system.grad_raster_time

    while True:
        section = skip_comments(input_file)
        if section == -1:
            break
        if section == '[BLOCKS]':
            self.block_events = read_blocks(input_file)
        elif section == '[RF]':
            self.rf_library = read_events(input_file, 1, None, None)
        elif section == '[GRAD]':
            self.grad_library = read_events(input_file, 1, 'g', self.grad_library)
        elif section == '[TRAP]':
            self.grad_library = read_events(input_file, [1, 1e-6, 1e-6, 1e-6], 't', self.grad_library)
        elif section == '[ADC]':
            self.adc_library = read_events(input_file, [1, 1e-9, 1e-6, 1, 1], None, None)
        elif section == '[DELAYS]':
            self.delay_library = read_events(input_file, 1e-6, None, None)
        elif section == '[SHAPES]':
            self.shape_library = read_shapes(input_file)


def read_blocks(input_file):
    """
    Read Blocks from .seq file. Blocks are single lines under the '[BLOCKS]' header in the .seq file.

    Parameters
    ----------
    input_file : file
        .seq file to be read.

    Returns
    -------
    block_events : dict
        Key-value mapping of Block ID and Event ID.
    """

    line = strip_line(input_file)
    for x in range(len(line)):
        line[x] = float(line[x])

    event_table = []
    while not (line == '\n' or line[0] == '#'):
        event_row = []
        for c in line[1:]:
            event_row.append(float(c))
        event_table.append(event_row)

        line = strip_line(input_file)
        # Break here to avoid crash when the while loop condition is evaluated for line != '\n'
        # Crash occurs because spaces have been eliminated
        if len(line) == 0:
            break

    block_events = {}
    for x in range(len(event_table)):
        block_events[x + 1] = np.array(event_table[x])

    return block_events


def read_events(input_file, scale, type, event_lib):
    scale = 1 if scale is None else scale
    event_library = event_lib if event_lib is not None else EventLibrary()

    line = strip_line(input_file)
    for x in range(len(line)):
        line[x] = float(line[x])

    while not (line == '\n' or line[0] == '#'):
        event_id = line[0]
        data = np.multiply(line[1:], scale)
        event_library.insert(event_id, data, type)

        line = strip_line(input_file)
        if not line:
            break

        for x in range(len(line)):
            line[x] = float(line[x])

    return event_library


def read_shapes(input_file):
    shape_library = EventLibrary()

    strip_line(input_file)
    line = strip_line(input_file)

    while not (line == -1 or len(line) == 0 or line[0] != 'shape_id'):
        id = int(line[1])
        line = skip_comments(input_file)
        num_samples = int(line.split(' ')[1])
        data = []
        line = skip_comments(input_file)
        line = line.split(' ')
        while not (len(line) == 0 or line[0] == '#'):
            data.append(float(line[0]))
            line = strip_line(input_file)
        line = skip_comments(input_file)
        # line could be -1 since -1 is EOF marker, returned from skipComments(inputFile)
        line = line.split(' ') if line != -1 else line
        data.insert(0, num_samples)
        data = np.reshape(data, [1, len(data)])
        shape_library.insert(id, data, None)

    return shape_library


def skip_comments(input_file):
    """
    Skip one '#' comment in .seq file.

    Parameters
    ----------
    input_file : file
        .seq file to be read.

    Returns
    -------
    line : str
        First line in input_file after skipping one '#' comment block.
        Note: File pointer is remembered, so successive calls work as expected.
    """

    line = input_file.readline()
    if line == '':
        return -1
    while line == '\n' or line[0] == '#':
        line = input_file.readline()
        if line == '':
            return -1
    line = line.strip()
    return line


def strip_line(input_file):
    """
    Remove spaces, newline whitespace and return line.

    Parameters
    ----------
    input_file : file
        .seq file to be read.

    Returns
    -------
    line : str
        First line in input_file after removing spaces and newline whitespaces.
        Note: File pointer is remembered, so successive calls work as expected.
    """
    line = input_file.readline()
    line = line.strip()
    line = line.split(' ')
    while '' in line:
        line.remove('')
    return line
