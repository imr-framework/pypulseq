import numpy as np

from pypulseq.calc_duration import calc_duration


def align(*args) -> list:
    """
    Aligns `SimpleNamespace` blocks as per specified alignment options by setting delays of the pulse sequence events
    within the block. All previously configured delays within objects are taken into account during calculating of the
    block duration but then reset according to the selected alignment. Possible values for align_spec are 'left',
    'center', 'right'.

    Parameters
    ----------
    args : list
        List of alignment options and `SimpleNamespace` blocks.
        Template: [alignment_spec, 'SimpleNamespace` block, [alignment_spec, `SimpleNamespace` block, ...]].
        Alignment spec can be one of `left`, `center` or `right`.

    Returns
    -------
    objects : list
        List of aligned `SimpleNamespace` blocks.
    """
    alignment_options = ['left', 'center', 'right']
    if not isinstance(args[0], str):
        raise ValueError('First parameter must be of type str.')

    curr_align = alignment_options.index(args[0]) if args[0] in alignment_options else None

    i_objects = []
    alignments = []
    for i in range(1, len(args)):
        if curr_align is None:
            raise ValueError('Invalid alignment spec.')
        if isinstance(args[i], str):
            curr_align = alignment_options.index(args[i]) if args[i] in alignment_options else None
            continue
        i_objects.append(i)
        alignments.append(curr_align)

    args = np.array(args)
    objects = args[i_objects]
    dur = calc_duration(*objects)

    for i in range(len(objects)):
        if alignments[i] == 0:
            objects[i].delay = 0
        elif alignments[i] == 1:
            objects[i].delay = (dur - calc_duration(objects[i])) / 2
        elif alignments[i] == 2:
            objects[i].delay = dur - calc_duration(objects[i]) + objects[i].delay

    return objects
