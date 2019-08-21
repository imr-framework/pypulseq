import numpy as np


def write(self, file_name):
    """
    Writes the calling `Sequence` object as a `.seq` file with filename `file_name`.

    Parameters
    ----------
    file_name : str
        File name of `.seq` file to be written to disk.
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

    if len(self.definitions) != 0:
        output_file.write('[DEFINITIONS]\n')
        keys = list(self.definitions.keys())
        values = list(self.definitions.values())
        for i in range(len(keys)):
            output_file.write(f'{keys[i]} ')
            if isinstance(values[i], str):
                output_file.write(f'{values[i]} ')
            else:
                for t in range(len(values[i])):
                    output_file.write(f'{values[i][t]:.6g} ')
            output_file.write('\n')
        output_file.write('\n')

    output_file.write('# Format of blocks:\n')
    output_file.write('#  #  D RF  GX  GY  GZ ADC\n')
    output_file.write('[BLOCKS]\n')
    id_format_width = '{:' + str(len(str(len(self.block_events)))) + 'd} '
    id_format_str = id_format_width + '{:2d} {:2d} {:3d} {:3d} {:3d} {:2d}\n'
    for i in range(len(self.block_events)):
        s = id_format_str.format(*np.insert(self.block_events[i + 1], 0, (i + 1)))
        output_file.write(s)
    output_file.write('\n')

    if len(self.rf_library.keys) != 0:
        output_file.write('# Format of RF events:\n')
        output_file.write('# id amplitude mag_id phase_id delay freq phase\n')
        output_file.write('# ..        Hz   ....     ....    us   Hz   rad\n')
        output_file.write('[RF]\n')
        rf_lib_keys = self.rf_library.keys
        # id_format_str = '{:>1.0f} {:>12g} {:>1.0f} {:>1.0f} {:>g} {:>g}\n'
        id_format_str = '{:.0f} {:12g} {:.0f} {:.0f} {:g} {:g} {:g}\n'
        for k in rf_lib_keys.keys():
            lib_data1 = self.rf_library.data[k][0:3]
            lib_data2 = self.rf_library.data[k][4:7]
            delay = round(self.rf_library.data[k][3] * 1e6)
            s = id_format_str.format(k, *lib_data1, delay, *lib_data2)
            output_file.write(s)
        output_file.write('\n')

    grad_lib_values = np.array(list(self.grad_library.type.values()))
    arb_grad_mask = np.where(grad_lib_values == 'g')[0]
    trap_grad_mask = np.where(grad_lib_values == 't')[0]

    if len(arb_grad_mask) != 0:
        output_file.write('# Format of arbitrary gradients:\n')
        output_file.write('# id amplitude shape_id delay\n')
        output_file.write('# ..      Hz/m     ....    us\n')
        output_file.write('[GRADIENTS]\n')
        id_format_str = '{:.0f} {:12g} {:.0f} {:.0f}\n'
        keys = np.array(list(self.grad_library.keys.keys()))
        for k in keys[arb_grad_mask]:
            s = id_format_str.format(k, *self.grad_library.data[k][:2], round(self.grad_library.data[k][2] * 1e6))
            output_file.write(s)
        output_file.write('\n')

    if len(trap_grad_mask) != 0:
        output_file.write('# Format of trapezoid gradients:\n')
        output_file.write('# id amplitude rise flat fall delay\n')
        output_file.write('# ..      Hz/m   us   us   us    us\n')
        output_file.write('[TRAP]\n')
        keys = np.array(list(self.grad_library.keys.keys()))
        id_format_str = '{:2g} {:12g} {:3g} {:4g} {:3g} {:3g}\n'
        for k in keys[trap_grad_mask]:
            data = self.grad_library.data[k]
            data[1:] = np.round(1e6 * data[1:])
            s = id_format_str.format(k, *data)
            output_file.write(s)
        output_file.write('\n')

    if len(self.adc_library.keys) != 0:
        output_file.write('# Format of ADC events:\n')
        output_file.write('# id num dwell delay freq phase\n')
        output_file.write('# ..  ..    ns    us   Hz   rad\n')
        output_file.write('[ADC]\n')
        keys = self.adc_library.keys
        id_format_str = '{:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.6g}\n'
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
        id_format_str = '{:.0f} {:.0f}\n'
        for k in keys.values():
            s = id_format_str.format(k, *np.round(1e6 * self.delay_library.data[k]))
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
