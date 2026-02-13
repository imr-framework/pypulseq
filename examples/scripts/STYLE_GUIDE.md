# Example Script Style Guide

This document describes the conventions and coding style for PyPulseq example scripts.
It serves as a reference for contributors writing new examples or refactoring existing ones.

For a minimal reference implementation, see `write_gre.py`.
For a more advanced example, see `write_gre_label_softdelay.py`.

---

## General Rules

- Follow **PEP 8** naming conventions: all variable names must be `lowercase_with_underscores`.
- Use **descriptive variable names** that convey meaning (e.g., `readout_duration` instead of `ro_dur`).
- Use the **`n_` prefix** for all variables describing a count or number of something
  (e.g., `n_x`, `n_y`, `n_slices`, `n_echo`, `n_spokes`).
- Variables derived from or related to another variable should **start with the referring
  variable name** (e.g., `te_delay`, `te_min`, `tr_delay`, `adc_duration`, `gx_flat_time`).
- Use `np.deg2rad()` for flip angle conversions, never manual `* np.pi / 180`.
- Import PyPulseq consistently as `import pypulseq as pp`.

---

## Script Structure

Every example script should follow this structure:

### 1. Imports

```python
import numpy as np

import pypulseq as pp
```

Third-party imports first, then PyPulseq — separated by blank lines.
If additional standard library imports are needed (e.g., `warnings`), place them
before the third-party imports.

### 2. `main` Function

All sequence logic lives inside a `main()` function. The function signature uses:

- **Positional arguments** for control flags (`plot`, `test_report`, `write_seq`, `seq_filename`).
- A **bare `*`** separator to force all sequence parameters to be keyword-only.
- **Type hints** on all parameters.

```python
def main(
    plot: bool = False,
    test_report: bool = False,
    write_seq: bool = False,
    seq_filename: str = 'gre_pypulseq.seq',
    *,
    fov: float = 256e-3,
    n_x: int = 64,
    n_y: int = 64,
    flip_angle_deg: float = 10,
    slice_thickness: float = 3e-3,
    tr: float = 12e-3,
    te: float = 5e-3,
):
```

### 3. Numpy-Style Docstring

Every `main` function must have a numpy-style docstring documenting all parameters and
the return value:

```python
    """Create a basic gradient echo (GRE) sequence.

    Parameters
    ----------
    plot : bool, optional
        Plot the sequence diagram. Default is False.
    test_report : bool, optional
        Print a test report. Default is False.
    write_seq : bool, optional
        Write the sequence to a .seq file. Default is False.
    seq_filename : str, optional
        Output filename for the .seq file. Default is 'gre_pypulseq.seq'.
    fov : float, optional
        Field of view in meters. Default is 256e-3.
    n_x : int, optional
        Number of readout samples. Default is 64.
    ...

    Returns
    -------
    seq : pypulseq.Sequence
        The GRE sequence object.
    """
```

### 4. Function Body Order

Inside `main`, the code should follow this general order:

1. **System limits** — create `system = pp.Opts(...)`.
2. **Sequence object** — create `seq = pp.Sequence(system)`.
3. **Create events** — RF pulses, gradients, ADC, delays.
4. **Calculate timing** — compute delays, round to raster time.
5. **Construct sequence** — loop over slices/phase encodes, add blocks.
6. **Check timing** — call `seq.check_timing()` and print result.
7. **Test report** — conditionally call `seq.test_report()`.
8. **Visualization** — conditionally call `seq.plot()`.
9. **Set definitions** — call `seq.set_definition()` (always, not only when writing).
10. **Write .seq file** — conditionally call `seq.write()`.
11. **Return** — return `seq`.

### 5. `if __name__` Block

```python
if __name__ == '__main__':
    main(plot=True, write_seq=True)
```

---

## Naming Conventions

### Function Parameters

| Convention | Example | Avoid |
|---|---|---|
| Count/number prefix `n_` | `n_x`, `n_y`, `n_slices`, `n_echo` | `Nx`, `Ny`, `nSlices` |
| Descriptive names | `flip_angle_deg`, `readout_duration` | `alpha`, `ro_dur` |
| Lowercase with underscores | `slice_thickness`, `fov` | `SliceThickness`, `FOV` |
| Flip angle parameter in degrees | `flip_angle_deg: float = 10` | `alpha`, `fa` |

### Common Variable Names

| Variable | Description |
|---|---|
| `seq` | Sequence object (`pp.Sequence`) |
| `system` | System limits object (`pp.Opts`) |
| `rf` | RF excitation pulse |
| `rf_ref` | RF refocusing pulse |
| `rf_prep` | RF preparation/inversion pulse |
| `gz` | Slice selection gradient |
| `gz_reph` | Slice rephasing gradient |
| `gx` | Readout gradient |
| `gx_pre` | Readout prephaser gradient |
| `gx_spoil` | Readout spoiler gradient |
| `gz_spoil` | Slice spoiler gradient |
| `adc` | ADC readout event |
| `delta_k` | k-space step size (`1 / fov`) |
| `phase_areas` | Array of phase encoding areas |

### Derived / Timing Variables

Variables related to a timing parameter should start with that parameter's name:

| Variable | Description |
|---|---|
| `te_delay` | Delay to achieve desired TE |
| `te_min` | Minimum achievable TE |
| `tr_delay` | Delay to achieve desired TR |
| `tr_min` | Minimum achievable TR |
| `ti_delay` | Delay to achieve desired TI |
| `adc_duration` | Duration of the ADC readout |
| `readout_duration` | Duration of the readout flat-top |

### Loop Variables

| Variable | Description |
|---|---|
| `i_phase` | Phase encoding loop index |
| `i_slice` | Slice loop index |
| `i_echo` | Echo loop index |
| `i_excitation` | Excitation loop index |
| `i_partition` | Partition (3D) loop index |

---

## Timing Calculations

When computing durations that need rounding to the raster time, **always split the
calculation and rounding into two separate steps**:

```python
# Good: two separate steps
te_delay = te - pp.calc_duration(gx_pre) - gz.fall_time - gz.flat_time / 2 - pp.calc_duration(gx) / 2
te_delay = np.ceil(te_delay / seq.grad_raster_time) * seq.grad_raster_time

# Avoid: combined in one expression
te_delay = np.ceil(
    (te - pp.calc_duration(gx_pre) - gz.fall_time - gz.flat_time / 2 - pp.calc_duration(gx) / 2)
    / seq.grad_raster_time
) * seq.grad_raster_time
```

---

## Things to Avoid

- **No section banner comments** like `# ====== SETUP ======`. Use short inline comments
  where needed.
- **No single-letter variable names** (except standard loop counters like `i`, `j` — but
  prefer descriptive loop variables like `i_phase`, `i_slice`).
- **No uppercase variable names** for sequence parameters (e.g., use `te`, not `TE`).
- **No `import math`** — use `numpy` equivalents (`np.ceil`, `np.floor`, `np.sqrt`,
  `np.pi`, etc.) instead of `math` functions.
- **No `from pypulseq import ...`** — always use `import pypulseq as pp`.
- **No conditional `set_definition`** — always set FOV and Name definitions, not only
  inside `if write_seq`.
