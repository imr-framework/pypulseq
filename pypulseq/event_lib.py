from types import SimpleNamespace
from typing import Tuple

import numpy as np


class EventLibrary:
    """
    Defines an event library ot maintain a list of events. Provides methods to insert new data and find existing data.

    Sequence Properties:
    - keys - A list of event IDs
    - data - A struct array with field 'array' to store data of varying lengths, remaining compatible with codegen.
    - lengths - Corresponding lengths of the data arrays
    - type - Type to distinguish events in the same class (e.g. trapezoids and arbitrary gradients)

    Sequence Methods:
    - find - Find an event in the library
    - insert - Add a new event to the library

    See also `Sequence.py`.

    Attributes
    ----------
    keys : dict{str, int}
        Key-value pairs of event keys and corresponding... event keys.
    data : dict{str: numpy.array}
        Key-value pairs of event keys and corresponding data.
    lengths : dict{str, int}
        Key-value pairs of event keys and corresponding length of data values in `self.data`.
    type : dict{str, str}
        Key-value pairs of event keys and corresponding event types.
    keymap : dict{str, int}
        Key-value pairs of data values and corresponding event keys.
    """

    def __init__(self):
        self.keys = dict()
        self.data = dict()
        self.lengths = dict()
        self.type = dict()
        self.keymap = dict()
        self.next_free_ID = 1

    def __str__(self) -> str:
        s = "EventLibrary:"
        s += "\nkeys: " + str(len(self.keys))
        s += "\ndata: " + str(len(self.data))
        s += "\nlengths: " + str(len(self.lengths))
        s += "\ntype: " + str(len(self.type))
        return s

    def find(self, new_data: np.ndarray) -> Tuple[int, bool]:
        """
        Finds data `new_data` in event library.

        Parameters
        ----------
        new_data : numpy.ndarray
            Data to be found in event library.

        Returns
        -------
        key_id : int
            Key of `new_data` in event library, if found.
        found : bool
            If `new_data` was found in the event library or not.
        """
        new_data = np.array(new_data)
        data_string = np.array2string(
            new_data, formatter={"float": lambda x: f"{x:.6g}"}
        )
        data_string = data_string.replace("[", "")
        data_string = data_string.replace("]", "")
        try:
            key_id = self.keymap[data_string]
            found = True
        except KeyError:
            key_id = 1 if len(self.keys) == 0 else max(self.keys) + 1
            found = False

        return key_id, found

    def find_or_insert(
        self, new_data: np.ndarray, data_type: str = str()
    ) -> Tuple[int, bool]:
        """
        Lookup a data structure in the given library and return the index of the data in the library. If the data does
        not exist in the library it is inserted right away. The data is a 1xN array with event-specific data.

        See also  insert `pypulseq.Sequence.sequence.Sequence.add_block()`.

        Parameters
        ----------
        new_data : numpy.ndarray
            Data to be found (or added, if not found) in event library.
        data_type : str, default=str()
            Type of data.

        Returns
        -------
        key_id : int
            Key of `new_data` in event library, if found.
        found : bool
            If `new_data` was found in the event library or not.
        """
        if not isinstance(new_data, np.ndarray):
            new_data = np.array(new_data)
        data_string = new_data.tobytes()

        if data_string in self.keymap:
            key_id = self.keymap[data_string]
            found = True
        else:
            key_id = self.next_free_ID
            found = False

            # Insert
            self.keys[key_id] = key_id
            self.data[key_id] = new_data
            self.lengths[key_id] = np.max(new_data.shape)

            if data_type != str():
                self.type[key_id] = data_type

            self.keymap[data_string] = key_id
            self.next_free_ID = key_id + 1  # Update next_free_id

        return key_id, found

    def insert(self, key_id: int, new_data: np.ndarray, data_type: str = str()) -> int:
        """
        Add event to library.

        See also `pypulseq.event_library.EventLibrary.find()`.

        Parameters
        ----------
        key_id : int
            Key of `new_data`.
        new_data : numpy.ndarray
            Data to be inserted into event library.
        data_type : str, default=str()
            Data type of `new_data`.

        Returns
        -------
        key_id : int
            Key ID of inserted event.
        """
        if isinstance(key_id, float):
            key_id = int(key_id)

        if key_id == 0:
            key_id = self.next_free_ID

        new_data = np.array(new_data)
        self.keys[key_id] = key_id
        self.data[key_id] = new_data
        self.lengths[key_id] = max(new_data.shape)
        if data_type != str():
            self.type[key_id] = data_type

        data_string = np.array2string(
            new_data, formatter={"float_kind": lambda x: "%.6g" % x}
        )
        data_string = data_string.replace("[", "")
        data_string = data_string.replace("]", "")
        self.keymap[data_string] = key_id

        if key_id >= self.next_free_ID:
            self.next_free_ID += 1  # Update next_free_id

        return key_id

    def get(self, key_id: int) -> dict:
        """

        Parameters
        ----------
        key_id : int

        Returns
        -------
        dict
        """
        return {
            "key": self.keys[key_id],
            "data": self.data[key_id],
            "length": self.lengths[key_id],
            "type": self.type[key_id],
        }

    def out(self, key_id: int) -> SimpleNamespace:
        """
        Get element from library by key.

        See also `pypulseq.event_library.EventLibrary.find()`.

        Parameters
        ----------
        key_id : int

        Returns
        -------
        out : SimpleNamespace
        """
        out = SimpleNamespace()
        out.key = self.keys[key_id]
        out.data = self.data[key_id]
        out.length = self.lengths[key_id]
        out.type = self.type[key_id]

        return out

    def update(
        self,
        key_id: int,
        old_data: np.ndarray,
        new_data: np.ndarray,
        data_type: str = str(),
    ):
        """
        Parameters
        ----------
        key_id : int
        old_data : numpy.ndarray
        new_data : numpy.ndarray
        data_type : str, default=str()
        """
        if len(self.keys) >= key_id:
            data_string = np.array2string(
                old_data, formatter={"float_kind": lambda x: "%.6g" % x}
            )
            data_string = data_string.replace("[", "")
            data_string = data_string.replace("]", "")
            del self.keymap[data_string]

        self.insert(key_id, new_data, data_type)

    def update_data(
        self,
        key_id: int,
        old_data: np.ndarray,
        new_data: np.ndarray,
        data_type: str = str(),
    ):
        """
        Parameters
        ----------
        key_id : int
        old_data : np.ndarray
        new_data : np.ndarray
        data_type : str
        """
        self.update(key_id, old_data, new_data, data_type)
