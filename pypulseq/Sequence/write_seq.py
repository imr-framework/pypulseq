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
    output_file.write("# Pulseq sequence file\n")
    output_file.write("# Created by Python/GPI Lab\n\n")
    output_file.write("# Format of blocks:\n")
    output_file.write("#  #  D RF  GX  GY  GZ ADC\n")
    output_file.write("[BLOCKS]\n")
    id_format_width = len(str(len(self.block_events)))
    id_format_str = '{:>' + str(id_format_width) + '}'
    id_format_str += ' {:>2.0f} {:>2.0f} {:>3.0f} {:>3.0f} {:>3.0f} {:>2.0f}\n'
    for i in range(0, len(self.block_events)):
        s = id_format_str.format(*np.insert(self.block_events[i + 1].astype(int), 0, (i + 1)))
        output_file.write(s)
    output_file.write('\n')

    if len(self.rf_library.keys) != 0:
        output_file.write('# Format of RF events:\n')
        output_file.write('# id amplitude mag_id phase_id freq phase\n')
        output_file.write('# ..        Hz   ....     ....   Hz   rad\n')
        output_file.write('[RF]\n')
        rf_lib_keys = self.rf_library.keys
        id_format_str = '{:>1.0f} {:>12g} {:>1.0f} {:>1.0f} {:>g} {:>g}\n'
        for k in rf_lib_keys.keys():
            lib_data = self.rf_library.data[k][0:5]
            s = id_format_str.format(*np.insert(lib_data, 0, k))
            output_file.write(s)
        output_file.write('\n')

    grad_lib_values = np.array(list(self.grad_library.type.values()))
    arb_grad_mask = np.where(grad_lib_values == 'g')[0]
    trap_grad_mask = np.where(grad_lib_values == 't')[0]

    if any(arb_grad_mask):
        output_file.write('# Format of arbitrary gradients:\n')
        output_file.write('# id amplitude shape_id\n')
        output_file.write('# ..      Hz/m     ....\n')
        output_file.write('[GRADIENTS]\n')
        keys = np.array(list(self.grad_library.keys.keys()))
        id_format_str = '{:>1.0f} {:>12g} {:>1.0f} \n'
        for k in keys[arb_grad_mask]:
            s = id_format_str.format(*np.insert(self.grad_library.data[k], 0, k))
            output_file.write(s)
        output_file.write('\n')

    if any(trap_grad_mask):
        output_file.write('# Format of trapezoid gradients:\n')
        output_file.write('# id amplitude rise flat fall\n')
        output_file.write('# ..      Hz/m   us   us   us\n')
        output_file.write('[TRAP]\n')
        keys = np.array(list(self.grad_library.keys.keys()))
        for k in keys[trap_grad_mask]:
            id_format_str = '{:>2.0f} {:>12g} {:>3.0f} {:>4.0f} {:>3.0f}\n'
            data = self.grad_library.data[k]
            data = np.reshape(data, (1, data.shape[0]))
            data[0][1:] = np.round(1e6 * data[0][1:], decimals=2)
            s = id_format_str.format(*np.insert(data, 0, k))
            output_file.write(s)
        output_file.write('\n')

    if len(self.adc_library.keys) != 0:
        output_file.write('# Format of ADC events:\n')
        output_file.write('# id num dwell delay freq phase\n')
        output_file.write('# ..  ..    ns    us   Hz   rad\n')
        output_file.write('[ADC]\n')
        keys = self.adc_library.keys
        id_format_str = '{:>2.0f} {:>3.0f} {:>6.0f} {:>3.0f} {:>g} {:>g}\n'
        for k in keys.values():
            data = np.multiply(self.adc_library.data[k][0:5], [1, 1e9, 1e6, 1, 1])
            s = id_format_str.format(*np.insert(data, 0, k))
            output_file.write(s)
    output_file.write('\n')

    if len(self.delay_library.keys) != 0:
        output_file.write('# Format of delays:\n')
        output_file.write('# id delay (us)\n')
        output_file.write('[DELAYS]\n')
        keys = self.delay_library.keys
        id_format_str = '{:>.0f} {:>.0f}\n'
        for k in keys.values():
            data = np.round(1e6 * self.delay_library.data[k])
            s = id_format_str.format(*np.insert(data, 0, k))
            output_file.write(s)
        output_file.write('\n')

    if len(self.shape_library.keys) != 0:
        output_file.write('# Sequence Shapes\n')
        output_file.write('[SHAPES]\n\n')
        keys = self.shape_library.keys
        for k in keys.values():
            shape_data = self.shape_library.data[k]
            s = 'shape_id {:>.0f}\n'
            s = s.format(k)
            output_file.write(s)
            s = 'num_samples {:>g}\n'
            s = s.format(shape_data[0][0])
            output_file.write(s)
            s = '{:g}\n'
            for x in shape_data[0][1:]:
                s1 = s.format(x)
                output_file.write(s1)
            output_file.write('\n')
