import numpy as np


class EventLibrary:
    def __init__(self):
        # size of data is 0x0 because of range_len in find()
        self.keys, self.data, self.lengths, self.type, self.keymap = {}, {}, {}, {}, {}

    def __str__(self):
        s = "EventLibrary:"
        s += "\nkeys: " + str(len(self.keys))
        s += "\ndata: " + str(len(self.data))
        s += "\nlengths: " + str(len(self.lengths))
        s += "\ntype: " + str(len(self.type))
        return s

    def find(self, new_data):
        data_string = np.array2string(new_data, formatter={'float_kind': lambda x: "%.6g" % x})
        data_string = data_string.replace('[', '')
        data_string = data_string.replace(']', '')
        try:
            key_id = self.keymap[data_string]
            found = True
        except:
            key_id = 1 if len(self.keys) == 0 else max(self.keys) + 1
            found = False

        return key_id, found

    def insert(self, key_id, new_data, data_type):
        if not isinstance(new_data, np.ndarray):
            new_data = np.array(new_data)
        self.keys[key_id] = key_id
        self.data[key_id] = new_data
        self.lengths[key_id] = max(self.data[key_id].shape)
        data_string = np.array2string(new_data, formatter={'float_kind': lambda x: "%.6g" % x})
        data_string = data_string.replace('[', '')
        data_string = data_string.replace(']', '')
        self.keymap[data_string] = key_id
        self.type[key_id] = data_type
