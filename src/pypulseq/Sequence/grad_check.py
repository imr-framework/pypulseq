from pypulseq import eps


def grad_check(self, block_index, check_g, duration):
    """
    Check if connection to the previous block is correct.

    Parameters
    ----------
    block_index : int
        Current block index.
    check_g : SimpleNamespace
        Structure containing current gradient start and end (t, g) values for each
        axis.
    duration : float
        Current block duration.

    Raises
    ------
    RuntimeError
        If either 1) initial block start with non-zero amplitude;
        2) gradients starting with non-zero amplitude have a delay;
        3) gradients starting with non-zero amplitude have different initial
           amplitude value than the previous block at connecting point;
        4) gradients ending with non-zero amplitude are not aligned with block raster;
        4) gradients ending with non-zero amplitude are not aligned have different initial
           amplitude value than the next block at connecting point.

    """
    max_slew_dt = self.system.max_slew * self.system.grad_raster_time

    # If block has duration 0, nothing to do
    if duration > 0:
        for ax in range(3):
            grad_to_check = check_g[ax]

            # If gradient event starts with non-zero amplitude and 1) is in first block or 2) has a delay, error
            # no need to compare against previous or following
            if abs(grad_to_check.start[1]) > max_slew_dt:
                if block_index == 1:
                    raise RuntimeError('First gradient in the the first block has to start at 0.')
                elif grad_to_check.start[0] > eps:
                    raise RuntimeError('No delay allowed for gradients which start with a non-zero amplitude')

            # If gradient event ends with non-zero amplitude and is not alined with block boundary, error
            # no need to compare against previous or following
            if abs(grad_to_check.stop[1]) > max_slew_dt and abs(grad_to_check.stop[0] - duration) > 1e-7:
                raise RuntimeError('A gradient that does not end at zero needs to be aligned to the block boundary.')

        # Now we know that gradient itself is ok. If this is the first block in the sequence
        # it is over.
        check_next = _get_neighboring_blocks(self, block_index)

        # We now loop over axes and check connection point with previous
        # and, if required, with next gradient
        if self.next_free_block_ID > 1:
            # TODO: add rotation of grad_check_data_prev / grad_check_data_next
            # to logical axis according to current block rot_quaternion (v1.5.1)
            for ax in range(3):
                grad_to_check = check_g[ax]

                # Check whether the difference between the last gradient value and
                # the first value of the new gradient is achievable with the
                # specified slew rate.
                if abs(self.grad_check_data_prev.last_grad_vals[ax] - grad_to_check.start[1]) > max_slew_dt:
                    raise RuntimeError(
                        'Two consecutive gradients need to have the same amplitude at the connection point'
                    )

                # Check whether the difference between the first gradient value
                # in the next block and the last value of the new gradient is
                # achievable with the specified slew rate.
                if (
                    check_next
                    and abs(self.grad_check_data_next.first_grad_vals[ax] - grad_to_check.stop[1]) > max_slew_dt
                ):
                    raise RuntimeError(
                        'Two consecutive gradients need to have the same amplitude at the connection point'
                    )

                # TODO: add rotation to physical axis according to current block rot_quaternion
                # before caching (v1.5.1)

                # Now cache current block.
                self.grad_check_data_prev.last_grad_vals[ax] = grad_to_check.stop[1]
                self.grad_check_data_next.first_grad_vals[ax] = grad_to_check.start[1]
        else:
            # Just cache
            for ax in range(3):
                grad_to_check = check_g[ax]

                # TODO: add rotation to physical axis according to current block rot_quaternion
                # before caching (v1.5.1)

                # Now cache current block.
                self.grad_check_data_prev.last_grad_vals[ax] = grad_to_check.stop[1]
                self.grad_check_data_next.first_grad_vals[ax] = grad_to_check.start[1]

        self.grad_check_data_prev.valid_for_block_num = block_index
        self.grad_check_data_next.valid_for_block_num = block_index - 1


def _get_neighboring_blocks(self, block_index):
    check_next = False
    blocks = None
    if self.next_free_block_ID > 1:
        # First handle previous block.
        # If cached block is already the one before the new one
        # use it. Otherwise, read it.
        if self.grad_check_data_prev.valid_for_block_num != block_index - 1:
            # Initialize to empty / trapezoid gradient
            self.grad_check_data_prev.valid_for_block_num = block_index - 1
            self.grad_check_data_prev.last_grad_vals = [0.0, 0.0, 0.0]

            # Get previous block ID
            if block_index == self.next_free_block_ID:
                # New block inserted
                prev_block_index = next(reversed(self.block_events))
            else:
                blocks = list(self.block_events) if blocks is None else blocks
                try:
                    # Existing block overwritten
                    idx = blocks.index(block_index)
                    prev_block_index = blocks[idx - 1] if idx > 0 else None
                except ValueError:
                    # Inserting a new block with non-contiguous numbering
                    prev_block_index = next(reversed(self.block_events))

            # Fill previous block gradient data.
            # If empty or trapezoid, implicitly leave it to 0
            for ax in range(3):
                if prev_block_index is not None:
                    prev_id = self.block_events[prev_block_index][ax + 2]
                    if prev_id != 0:
                        prev_lib = self.grad_library.get(prev_id)
                        prev_type = prev_lib['type']
                        prev_dat = prev_lib['data']

                        if prev_type == 'g':
                            # TODO: change prev_dat[5] to prev_dat[2] in v1.5.x
                            self.grad_check_data_prev.last_grad_vals[ax] = prev_dat[5]

        # Now handle next block.
        if block_index < self.next_free_block_ID and self.grad_check_data_next.valid_for_block_num != block_index + 1:
            # Initialize to empty / trapezoid gradient
            self.grad_check_data_next.valid_for_block_num = block_index + 1
            self.grad_check_data_next.first_grad_vals = [0.0, 0.0, 0.0]

            # Get next block ID
            blocks = list(self.block_events) if blocks is None else blocks
            try:
                # Existing block overwritten
                idx = blocks.index(block_index)
                next_block_index = blocks[idx + 1] if idx < len(blocks) - 1 else None
                check_next = True
            except ValueError:
                # Inserting a new block with non-contiguous numbering
                next_block_index = None

            # Fill next block gradient data.
            # If empty or trapezoid, implicitly leave it to 0
            for ax in range(3):
                if next_block_index is not None:
                    next_id = self.block_events[next_block_index][ax + 2]
                    if next_id != 0:
                        next_lib = self.grad_library.get(next_id)
                        next_type = next_lib['type']
                        next_dat = next_lib['data']

                        if next_type == 'g':
                            # TODO: change next_dat[4] to next_dat[1] in v1.5.x
                            self.grad_check_data_next.first_grad_vals[ax] = next_dat[4]

    return check_next
