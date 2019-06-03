import numpy as np


def write(self, file_name):
    """
    Writes a .seq file from the Sequence object calling the method.
    >.0f is used when only decimals have to be displayed.
    >g is used when insignificant zeros have to be truncated.

    Parameters
    ----------
    file_name : str
        File name of .seq file to be written.
    """
    file_name += '.seq' if file_name[-4:] != '.seq' not in file_name else ''
    output_file = open(file_name, 'w')
    output_file.write('# Pulseq sequence file\n')
    output_file.write('# Created by PyPulseq\n\n')

    if len(self.definitions) != 0:
        output_file.write('[DEFINITIONS]\n')
        keys = self.definitions.keys()
        values = self.definitions.values()
        for i in len(keys):
            output_file.write(keys[i])
            if values[i].isalpha():
                output_file.write(values[i])
            else:
                output_file.write(f'{values[i]:.g}')

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
        id_format_str = '{:d} {:12g} {:0g} {:0g} {:g} {:g} {:g}\n'
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

    if any(arb_grad_mask):
        output_file.write('# Format of arbitrary gradients:\n')
        output_file.write('# id amplitude shape_id delay\n')
        output_file.write('# ..      Hz/m     ....    us\n')
        output_file.write('[GRADIENTS]\n')
        keys = np.array(list(self.grad_library.keys.keys()))
        id_format_str = '{:0g} {:12g} {:0g} {:0g}\n'
        for k in keys[arb_grad_mask]:
            s = id_format_str.format(k, *self.grad_library.data[k][:2], round(self.grad_library[k][4] * 1e6))
            output_file.write(s)
        output_file.write('\n')

    if any(trap_grad_mask):
        output_file.write('# Format of trapezoid gradients:\n')
        output_file.write('# id amplitude rise flat fall delay\n')
        output_file.write('# ..      Hz/m   us   us   us    us\n')
        output_file.write('[TRAP]\n')
        keys = np.array(list(self.grad_library.keys.keys()))
        for k in keys[trap_grad_mask]:
            id_format_str = '{:2g} {:12g} {:3g} {:4g} {:3g} {:3g}\n'
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
        id_format_str = '{:0g} {:0g} {:.0f} {:.0f} {:0g} {:0g}\n'
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
        id_format_str = '{:0g} {:0g}\n'
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
            s = 'shape_id {:0g}\n'
            s = s.format(k)
            output_file.write(s)
            s = 'num_samples {:0g}\n'
            s = s.format(shape_data[0])
            output_file.write(s)
            #
            # str_shape_data = []
            # for val in shape_data:
            #     str_shape_data.append(str(val))
            # str_shape_data = ' '.join(str_shape_data)
            # shape_data = np.fromstring(str_shape_data, sep=' ')
            #
            s = '{:.9g}\n' * len(shape_data[1:])
            s = s.format(*shape_data[1:])
            output_file.write(s)
            output_file.write('\n')
