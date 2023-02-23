from copy import deepcopy
from types import SimpleNamespace
from typing import List, Union

import numpy as np

from pypulseq.calc_duration import calc_duration


def align(
    **kwargs: Union[SimpleNamespace, List[SimpleNamespace]]
) -> List[SimpleNamespace]:
    """
    Sets delays of the objects within the block to achieve the desired alignment of the objects in the block. Aligns
    objects as per specified alignment options by setting delays of the pulse sequence events within the block. All
    previously configured delays within objects are taken into account during calculating of the block duration but
    then reset according to the selected alignment. Possible values for align_spec are 'left', 'center', 'right'.

    Parameters
    ----------
    args : dict{str, [SimpleNamespace, ...]}
        Dictionary mapping of alignment options and `SimpleNamespace` objects.
        Format: alignment_spec1=SimpleNamespace, alignment_spec2=[SimpleNamespace, ...], ...
        Alignment spec must be one of `left`, `center` or `right`.

    Returns
    -------
    objects : [SimpleNamespace, ...]
        List of aligned `SimpleNamespace` objects.

    Raises
    ------
    ValueError
        If first parameter is not of type `str`.
        If invalid alignment spec is passed. Must be one of `left`, `center` or `right`.

    Examples
    --------
    al_grad1, al_grad2, al_grad3 = align(right=[grad1, grad2, grad3])
    """
    alignment_specs = list(kwargs.keys())
    if not isinstance(alignment_specs[0], str):
        raise ValueError(
            f"First parameter must be of type str. Passed: {type(alignment_specs[0])}"
        )

    alignment_options = ["left", "center", "right"]
    if np.any([align_opt not in alignment_options for align_opt in alignment_specs]):
        raise ValueError("Invalid alignment spec.")

    alignments = []
    objects = []
    for curr_align in alignment_specs:
        objects_to_align = kwargs[curr_align]
        curr_align = alignment_options.index(curr_align)
        if isinstance(objects_to_align, (list, np.ndarray, tuple)):
            alignments.extend([curr_align] * len(objects_to_align))
            objects.extend(objects_to_align)
        elif isinstance(objects_to_align, SimpleNamespace):
            alignments.extend([curr_align])
            objects.append(objects_to_align)

    dur = calc_duration(*objects)

    # copy() to emulate pass-by-value; otherwise passed events are modified
    objects = deepcopy(objects)

    # Set new delays
    for i in range(len(objects)):
        if alignments[i] == 0:
            objects[i].delay = 0
        elif alignments[i] == 1:
            objects[i].delay = (dur - calc_duration(objects[i])) / 2
        elif alignments[i] == 2:
            objects[i].delay = dur - calc_duration(objects[i]) + objects[i].delay
            if objects[i].delay < 0:
                raise ValueError(
                    "align() attempts to set a negative delay, probably some RF pulses ignore rf_ringdown_time"
                )

    return objects
