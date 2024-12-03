from types import SimpleNamespace
from typing import Tuple, Union

try:
    from typing import Self
except ImportError:
    from typing import TypeVar

    Self = TypeVar('Self', bound='EventLibrary')

import math

import numpy as np


class EventLibrary:
    """
    Defines an event library to maintain a list of events. Provides methods to insert new data and find existing data.

    Sequence Properties:
    - data - A struct array with field 'array' to store data of varying lengths, remaining compatible with codegen.
    - type - Type to distinguish events in the same class (e.g. trapezoids and arbitrary gradients)

    Sequence Methods:
    - find - Find an event in the library
    - insert - Add a new event to the library

    See also `Sequence.py`.

    Attributes
    ----------
    data : dict{str: numpy.array}
        Key-value pairs of event keys and corresponding data.
    type : dict{str, str}
        Key-value pairs of event keys and corresponding event types.
    keymap : dict{str, int}
        Key-value pairs of data values and corresponding event keys.
    """

    def __init__(self, numpy_data=False):
        self.data = {}
        self.type = {}
        self.keymap = {}
        self.next_free_ID = 1
        self.numpy_data = numpy_data

    def __str__(self) -> str:
        s = 'EventLibrary:'
        s += '\ndata: ' + str(len(self.data))
        s += '\ntype: ' + str(len(self.type))
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
        if self.numpy_data:
            new_data = np.asarray(new_data)
            key = new_data.tobytes()
        else:
            key = tuple(new_data)

        if key in self.keymap:
            key_id = self.keymap[key]
            found = True
        else:
            key_id = self.next_free_ID
            found = False

        return key_id, found

    def find_or_insert(self, new_data: np.ndarray, data_type: str = str()) -> Tuple[int, bool]:
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
        if self.numpy_data:
            new_data = np.asarray(new_data)
            new_data.flags.writeable = False
            key = new_data.tobytes()
        else:
            key = tuple(new_data)

        if key in self.keymap:
            key_id = self.keymap[key]
            found = True
        else:
            key_id = self.next_free_ID
            found = False

            # Insert
            self.data[key_id] = new_data

            if data_type != str():
                self.type[key_id] = data_type

            self.keymap[key] = key_id
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

        if self.numpy_data:
            new_data = np.asarray(new_data)
            new_data.flags.writeable = False
            key = new_data.tobytes()
        else:
            key = tuple(new_data)

        self.data[key_id] = new_data
        if data_type != str():
            self.type[key_id] = data_type

        self.keymap[key] = key_id

        if key_id >= self.next_free_ID:
            self.next_free_ID = key_id + 1  # Update next_free_id

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
            'key': key_id,
            'data': self.data[key_id],
            'type': self.type[key_id],
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
        out.key = key_id
        out.data = self.data[key_id]
        out.type = self.type[key_id]

        return out

    def update(
        self,
        key_id: int,
        old_data: Union[np.ndarray, None],  # noqa: ARG002
        new_data: np.ndarray,
        data_type: str = str(),
    ):
        """
        Parameters
        ----------
        key_id : int
        old_data : numpy.ndarray (Ignored!)
        new_data : numpy.ndarray
        data_type : str, default=str()
        """
        if key_id in self.data and self.data[key_id] in self.keymap:
            del self.keymap[self.data[key_id]]

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
        old_data : np.ndarray (Ignored!)
        new_data : np.ndarray
        data_type : str
        """
        self.update(key_id, old_data, new_data, data_type)

    def remove_duplicates(self, digits: Union[int, Tuple[int]]) -> Tuple[Self, dict]:
        """
        Remove duplicate events from this event library by rounding the data
        according to the significant `digits` specification, and then removing
        duplicate events.
        Returns a new event library, leaving the current one intact.

        Parameters
        ----------
        digits : Union[int, List[int]]
            For libraries with `numpy_data == True`:
                A single number specifying the number of significant digits
                after rounding.
            Otherwise:
                A tuple of numbers specifying the number of significant digits
                after rounding for each entry in the event data tuple.

        Returns
        -------
        new_library : EventLibrary
            Event library with the duplicate events removed
        mapping : dict
            Dictionary containing a mapping of IDs in the old library to IDs
            in the new library.
        """

        def round_data(data: Tuple[float], digits: Tuple[int]) -> Tuple[float]:
            """
            Round the data tuple to a specified number of significant digits,
            specified by `digits`. Rounding behavior is similar to the {.Ng}
            format specifier if N > 0, and similar to {.0f} otherwise.
            """
            return tuple(
                round(d, dig - int(math.ceil(math.log10(abs(d) + 1e-12))) if dig > 0 else -dig)
                for d, dig in zip(data, digits)
            )

        def round_data_numpy(data: np.ndarray, digits: int) -> np.ndarray:
            """
            Round the data array to a specified number of significant digits,
            specified by `digits`. Rounding behavior is similar to the {.Ng}
            format specifier if N > 0, and similar to {.0f} otherwise.
            """
            mags = 10 ** (digits - (np.ceil(np.log10(abs(data) + 1e-12))) if digits > 0 else -digits)
            result = np.round(data * mags) / mags
            result.flags.writeable = False
            return result

        # Round library data based on `digits` specification
        if self.numpy_data:
            rounded_data = {x: round_data_numpy(self.data[x], digits) for x in self.data}
        else:
            rounded_data = {x: round_data(self.data[x], digits) for x in self.data}

        # Initialize filtered library
        new_library = EventLibrary(numpy_data=self.numpy_data)

        # Initialize ID mapping. Always include 0:0 to allow the mapping dict
        # to be used for mapping block_events (which can contain 0, i.e. no
        # event)
        mapping = {0: 0}

        # Recreate library using rounded values
        for k, v in sorted(rounded_data.items()):
            mapping[k], _ = new_library.find_or_insert(v, self.type[k] if k in self.type else str())

        return new_library, mapping
