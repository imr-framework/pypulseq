from types import SimpleNamespace
from typing import List, Union

import numpy as np

from pypulseq.calc_duration import calc_duration


def align(**kwargs: Union[SimpleNamespace, List[SimpleNamespace]]) -> List[SimpleNamespace]:
    """
    Aligns `SimpleNamespace` objects as per specified alignment options by setting delays of the pulse sequence events
    within the block. All previously configured delays within objects are taken into account during calculating of the
    block duration but then reset according to the selected alignment. Possible values for align_spec are 'left',
    'center', 'right'.

    Parameters
    ----------
    args : dict[str, list[SimpleNamespace]
        Dictionary mapping of alignment options and `SimpleNamespace` objects.
        Template: alignment_spec1=SimpleNamespace, alignment_spec2=[SimpleNamespace, ...], ...
        Alignment spec must be one of `left`, `center` or `right`.

    Returns
    -------
    objects : list
        List of aligned `SimpleNamespace` objects.

    Raises
    ------
    ValueError
        If first parameter is not of type `str`.
        If invalid alignment spec is passed. Must be one of `left`, `center` or `right`.
    """
    alignment_specs = list(kwargs.keys())
    if not isinstance(alignment_specs[0], str):
        raise ValueError(f'First parameter must be of type str. Passed: {type(alignment_specs[0])}')

    alignment_options = ['left', 'center', 'right']
    if np.any([align_opt not in alignment_options for align_opt in alignment_specs]):
        raise ValueError('Invalid alignment spec.')

    alignments = []
    objects = []
    for a in alignment_specs:
        objects_to_align = kwargs[a]
        a = alignment_options.index(a)
        if isinstance(objects_to_align, (list, np.ndarray, tuple)):
            alignments.extend([a] * len(objects_to_align))
            objects.extend(objects_to_align)
        elif isinstance(objects_to_align, SimpleNamespace):
            alignments.extend([a])
            objects.append(objects_to_align)

    dur = calc_duration(*objects)

    for i in range(len(objects)):
        if alignments[i] == 0:
            objects[i].delay = 0
        elif alignments[i] == 1:
            objects[i].delay = (dur - calc_duration(objects[i])) / 2
        elif alignments[i] == 2:
            objects[i].delay = dur - calc_duration(objects[i]) + objects[i].delay

    return objects
