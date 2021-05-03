import numpy as np

from pypulseq.supported_labels import get_supported_labels


def write(self, file_name: str) -> None:
    """
    Writes the calling `Sequence` object as a `.seq` file with filename `file_name`.

    Parameters
    ----------
    file_name : str
        File name of `.seq` file to be written to disk.

    Raises
    ------
    RuntimeError
        If an unsupported definition is encountered.
    """
    # `>.0f` is used when only decimals have to be displayed.
    # `>g` is used when insignificant zeros have to be truncated.
    file_name += '.seq' if file_name[-4:] != '.seq' not in file_name else ''
    output_file = open(file_name, 'w')
    output_file.write('# Pulseq sequence file\n')
    output_file.write('# Created by PyPulseq\n\n')

    output_file.write('[VERSION]\n')
    output_file.write(f'major {self.version_major}\n')
    output_file.write(f'minor {self.version_minor}\n')
    output_file.write(f'revision {self.version_revision}\n')
    output_file.write('\n')

    if len(self.dict_definitions) != 0:
        output_file.write('[DEFINITIONS]\n')
        keys = list(self.dict_definitions.keys())
        values = list(self.dict_definitions.values())
        for block_counter in range(len(keys)):
            output_file.write(f'{keys[block_counter]} ')
            if isinstance(values[block_counter], str):
                output_file.write(values[block_counter] + ' ')
            elif isinstance(values[block_counter], (int, float)):
                output_file.write(f'{values[block_counter]:0.9g} ')
            elif isinstance(values[block_counter], (list, tuple, np.ndarray)):  # For example, [FOV, FOV, FOV]
                for i in range(len(values[block_counter])):
                    if isinstance(values[block_counter][i], (int, float)):
                        output_file.write(f'{values[block_counter][i]:0.9g} ')
                    else:
                        output_file.write(f'{values[block_counter][i]} ')
            else:
                raise RuntimeError('Unsupported definition')
            output_file.write('\n')
        output_file.write('\n')

    output_file.write('# Format of blocks:\n')
    output_file.write('#  #  D RF  GX  GY  GZ ADC EXT\n')
    output_file.write('[BLOCKS]\n')
    id_format_width = '{:' + str(len(str(len(self.dict_block_events)))) + 'd}'
    id_format_str = id_format_width + ' ' + '{:2d} {:2d} {:3d} {:3d} {:3d} {:2d} {:2d}\n'
    for block_counter in range(len(self.dict_block_events)):
        s = id_format_str.format(*(block_counter + 1, *self.dict_block_events[block_counter + 1]))
        output_file.write(s)
    output_file.write('\n')

    if len(self.rf_library.keys) != 0:
        output_file.write('# Format of RF events:\n')
        output_file.write('# id amplitude mag_id phase_id delay freq phase\n')
        output_file.write('# ..        Hz   ....     ....    us   Hz   rad\n')
        output_file.write('[RF]\n')
        rf_lib_keys = self.rf_library.keys
        # See comment at the beginning of this method definition
        id_format_str = '{:.0f} {:12g} {:.0f} {:.0f} {:g} {:g} {:g}\n'
        for k in rf_lib_keys.keys():
            lib_data1 = self.rf_library.data[k][0:3]
            lib_data2 = self.rf_library.data[k][4:6]
            delay = np.round(self.rf_library.data[k][3] * 1e6)
            s = id_format_str.format(k, *lib_data1, delay, *lib_data2)
            output_file.write(s)
        output_file.write('\n')

    grad_lib_values = np.array(list(self.grad_library.type.values()))
    arb_grad_mask = grad_lib_values == 'g'
    trap_grad_mask = grad_lib_values == 't'

    if any(arb_grad_mask):
        output_file.write('# Format of arbitrary gradients:\n')
        output_file.write('# id amplitude shape_id delay\n')
        output_file.write('# ..      Hz/m     ....    us\n')
        output_file.write('[GRADIENTS]\n')
        id_format_str = '{:.0f} {:12g} {:.0f} {:.0f}\n'  # See comment at the beginning of this method definition
        keys = np.array(list(self.grad_library.keys.keys()))
        for k in keys[arb_grad_mask]:
            s = id_format_str.format(k, *self.grad_library.data[k][:2], np.round(self.grad_library.data[k][2] * 1e6))
            output_file.write(s)
        output_file.write('\n')

    if any(trap_grad_mask):
        output_file.write('# Format of trapezoid gradients:\n')
        output_file.write('# id amplitude rise flat fall delay\n')
        output_file.write('# ..      Hz/m   us   us   us    us\n')
        output_file.write('[TRAP]\n')
        keys = np.array(list(self.grad_library.keys.keys()))
        id_format_str = '{:2g} {:12g} {:3g} {:4g} {:3g} {:3g}\n'
        for k in keys[trap_grad_mask]:
            data = np.copy(self.grad_library.data[k])  # Make a copy to leave the original untouched
            data[1:] = np.round(1e6 * data[1:])
            """
            Python always rounds to nearest even value, this can cause inconsistencies with MATLAB Pulseq's .seq files.
            Read more - https://stackoverflow.com/questions/29671945/format-string-rounding-inconsistent
            Numpy too - https://stackoverflow.com/questions/50374779/how-to-avoid-incorrect-rounding-with-numpy-round
            """
            s = id_format_str.format(k, *data)
            output_file.write(s)
        output_file.write('\n')

    if len(self.adc_library.keys) != 0:
        output_file.write('# Format of ADC events:\n')
        output_file.write('# id num dwell delay freq phase\n')
        output_file.write('# ..  ..    ns    us   Hz   rad\n')
        output_file.write('[ADC]\n')
        keys = self.adc_library.keys
        # See comment at the beginning of this method definition
        id_format_str = '{:.0f} {:.0f} {:.0f} {:.0f} {:g} {:g}\n'
        for k in keys.values():
            data = np.multiply(self.adc_library.data[k][0:5], [1, 1e9, 1e6, 1, 1])
            s = id_format_str.format(k, *data)
            output_file.write(s)
        output_file.write('\n')

    if len(self.delay_library.keys) != 0:
        output_file.write('# Format of delays:\n')
        output_file.write('# id delay (us)\n')
        output_file.write('[DELAYS]\n')
        keys = self.delay_library.keys
        id_format_str = '{:.0f} {:.0f}\n'  # See if-block for self.rf.library.keys
        for k in keys.values():
            s = id_format_str.format(k, *np.round(1e6 * self.delay_library.data[k]))
            output_file.write(s)
        output_file.write('\n')

    if len(self.extensions_library.keys) != 0:
        output_file.write('# Format of extension lists:\n')
        output_file.write('# id type ref next_id\n')
        output_file.write('# next_id of 0 terminates the list\n')
        output_file.write('# Extension list is followed by extension specifications\n')
        output_file.write('[EXTENSIONS]\n')
        keys = self.extensions_library.keys
        id_format_str = '{:.0f} {:.0f} {:.0f} {:.0f}\n'  # See comment at the beginning of this method definition
        for k in keys.values():
            s = id_format_str.format(k, *np.round(self.extensions_library.data[k]))
            output_file.write(s)
        output_file.write('\n')

    if len(self.trigger_library.keys) != 0:
        output_file.write('# Extension specification for digital output and input triggers:\n')
        output_file.write('# id type channel delay (us) duration (us)\n')
        output_file.write(f'extension TRIGGERS {self.get_extension_type_ID("TRIGGERS")}\n')
        keys = self.trigger_library.keys
        id_format_str = '{:.0f} {:.0f} {:.0f} {:.0f} {:.0f}\n'  # See comment at the beginning of this method definition
        for k in keys.values():
            s = id_format_str.format(k, *np.round(self.trigger_library.data[k] * [1, 1, 1e6, 1e6]))
            output_file.write(s)
        output_file.write('\n')

    if len(self.label_set_library.keys) != 0:
        lbls = get_supported_labels()

        output_file.write('# Extension specification for setting labels:\n')
        output_file.write('# id set labelstring\n')
        tid = self.get_extension_type_ID('LABELSET')
        output_file.write(f'extension LABELSET {tid}\n')
        keys = self.label_set_library.keys
        id_format_str = '{:.0f} {:.0f} {}\n'  # See comment at the beginning of this method definition
        for k in keys.values():
            s = id_format_str.format(k, self.label_set_library.data[k][0], lbls[self.label_set_library.data[k][1] - 1])
            output_file.write(s)
        output_file.write('\n')

        output_file.write('# Extension specification for setting labels:\n')
        output_file.write('# id set labelstring\n')
        tid = self.get_extension_type_ID('LABELINC')
        output_file.write(f'extension LABELINC {tid}\n')
        keys = self.label_inc_library.keys
        id_format_str = '{:.0f} {:.0f} {}\n'  # See comment at the beginning of this method definition
        for k in keys.values():
            s = id_format_str.format(k, self.label_inc_library.data[k][0], lbls[self.label_inc_library.data[k][1] - 1])
            output_file.write(s)
        output_file.write('\n')

    if len(self.shape_library.keys) != 0:
        output_file.write('# Sequence Shapes\n')
        output_file.write('[SHAPES]\n\n')
        keys = self.shape_library.keys
        for k in keys.values():
            shape_data = self.shape_library.data[k]
            s = 'shape_id {:.0f}\n'
            s = s.format(k)
            output_file.write(s)
            s = 'num_samples {:.0f}\n'
            s = s.format(shape_data[0])
            output_file.write(s)
            s = '{:.9g}\n' * len(shape_data[1:])
            s = s.format(*shape_data[1:])
            output_file.write(s)
            output_file.write('\n')
