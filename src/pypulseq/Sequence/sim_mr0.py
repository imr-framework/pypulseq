import shutil
import tempfile
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np

from pypulseq import Sequence


def sim_seq_mr0(seq: Sequence, work_path: Optional[str] = '', base_name: str = 'mr0sim', no_clean_up: bool = False):
    # Import inside function to avoid circular import
    import MRzeroCore as mr0

    # Setup working directory
    if not work_path:
        shm_path = Path('/dev/shm')
        if shm_path.is_dir():
            work_path = shm_path / 'mr0sim'
        else:
            work_path = Path(tempfile.gettempdir()) / 'mr0sim'
    else:
        work_path = Path(work_path)

    work_path.mkdir(parents=True, exist_ok=True)

    seq_path = work_path / f'{base_name}.seq'
    phantom_path = work_path / 'numerical_brain_cropped.mat'

    # Download phantom if needed
    if not phantom_path.exists():
        url = (
            'https://github.com/MRsources/MRzero-Core/raw/main/documentation/playground_mr0/numerical_brain_cropped.mat'
        )
        try:
            urllib.request.urlretrieve(url, phantom_path)
        except Exception as e:
            print(f'WARNING: Failed to download numerical phantom from {url}')
            print('The simulation with MR0 will probably fail.')
            print(f'Error: {e}')

    # Write sequence to disk
    seq.write(str(seq_path))

    # Set up simulation parameters
    dB0 = 0
    sz = (128, 128)

    # Load and manipulate phantom
    obj_p = mr0.VoxelGridPhantom.load_mat(str(phantom_path))
    obj_p = obj_p.interpolate(sz[0], sz[1], 1)
    obj_p.T2dash[:] = 30e-3
    obj_p.B0 += dB0
    obj_p = obj_p.build()

    # Load sequence
    seq0 = mr0.Sequence.import_file(str(seq_path))

    # Simulate
    graph = mr0.compute_graph(seq0, obj_p, 200, 1e-3)
    signal = mr0.execute_graph(graph, seq0, obj_p, print_progress=False)
    np.save(work_path / f'{base_name}_signal.npy', signal)

    # Optionally clean up
    if not no_clean_up and work_path.exists():
        shutil.rmtree(work_path)

    return signal
