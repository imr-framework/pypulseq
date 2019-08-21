import numpy as np


class EventLibrary:
    """
    Defines an event library. Provides methods to insert new data and find existing data.

    Attributes
    ----------
    keys : dict
        Key-value pairs of event keys and corresponding... event keys.
    data : dict
        Key-value pairs of event keys and corresponding data.
    lengths : dict
        Key-value pairs of event keys and corresponding length of data values in `self.data`.
    type : dict
        Key-value pairs of event keys and corresponding event types.
    keymap : dict
        Key-value pairs of data values and corresponding event keys.
    """

    def __init__(self):
        self.keys, self.data, self.lengths, self.type, self.keymap = dict(), dict(), dict(), dict(), dict()

    def __str__(self):
        s = "EventLibrary:"
        s += "\nkeys: " + str(len(self.keys))
        s += "\ndata: " + str(len(self.data))
        s += "\nlengths: " + str(len(self.lengths))
        s += "\ntype: " + str(len(self.type))
        return s

    def find(self, new_data: np.ndarray):
        """
        Finds data `new_data` in event library.

        Parameters
        ----------
        new_data : np.ndarray
            Data to be found in event library.

        Returns
        -------
        key_id : int
            Key of `new_data` in event library, if found.
        found : bool
            If `new_data` was found in the event library or not.
        """
        data_string = np.array2string(new_data, formatter={'float': lambda x: f'{x:.6g}'})
        data_string = data_string.replace('[', '')
        data_string = data_string.replace(']', '')
        try:
            key_id = self.keymap[data_string]
            found = True
        except:
            key_id = 1 if len(self.keys) == 0 else max(self.keys) + 1
            found = False

        return key_id, found

    def insert(self, key_id: int, new_data: np.ndarray, data_type: str = None):
        """
        Inserts `new_data` of data type `data_type` into the event library with key `key_id`.

        Parameters
        ----------
        key_id : int
            Key of `new_data`.
        new_data : np.ndarray
            Data to be inserted into event library.
        data_type : str
            Data type of `new_data`.
        """
        self.keys[key_id] = key_id
        self.data[key_id] = new_data
        self.lengths[key_id] = max(new_data.shape)
        data_string = np.array2string(new_data, formatter={'float_kind': lambda x: "%.6g" % x})
        data_string = data_string.replace('[', '')
        data_string = data_string.replace(']', '')
        self.keymap[data_string] = key_id
        if data_type is not None:
            self.type[key_id] = data_type

    def get(self, key_id: int):
        return {'key': self.keys[key_id], 'data': self.data[key_id], 'length': self.lengths[key_id],
                'type': self.type[key_id]}
