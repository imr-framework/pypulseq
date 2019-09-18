<p align="center">
<img src="logo.png"/>
</p>

# PyPulseq

Pulse sequence design is a significant component of MRI research. However, multi-vendor studies require researchers to
be acquainted with each hardware platform's programming environment.

`PyPulseq` enables vendor-neutral pulse sequence design in Python [[1]](#references). The pulse sequences can be
exported as a `.seq` file to be run on  Siemens/[GE]/[Bruker] hardware by leveraging their respective
Pulseq interpreters. This tool is targeted at MRI pulse sequence designers, researchers, students and other interested
users. It is a translation of the Pulseq framework originally written in Matlab [[2]](#references).

It is strongly recommended to first read the [Pulseq specification]  before proceeding. The specification
document defines the concepts required for pulse sequence design using `PyPulseq`. API docs can be found [here][api-docs].

## Dependencies
1. numpy>=1.16.3
2. matplotlib>=3.0.3

## 1 minute demo
1. Clone this repository.
2. `cd` into this repository (or set as working directory in an IDE).
3. Install [dependencies](#dependencies).
3. Run any of the example scripts on Python 3.6 or above.
4. Inspect plots!
?. Get in touch regarding running the `.seq` files on your Siemens/[GE]/[Bruker] scanner.

## Custom pulse sequences
Getting started with pulse sequence design using `PyPulseq` is simple:
1. `pip install pypulseq` in your virtual environment (>=Python 3.6).
2. First, define system limits in `Opts` and then create a `Sequence` object with it:
    ```python
    from pypulseq.opts import Opts
    from pypulseq.Sequence.sequence import Sequence

    system = Opts(max_grad=32, grad_unit='mT/m', max_slew=130, slew_unit='mT/m/s')
    seq = Sequence(system=system)
    ```
3. Then, design gradient, RF or ADC pulse sequence events:
    ```python
    from pypulseq.make_sinc_pulse import make_sinc_pulse

    Nx, Ny = 256, 256 # matrix size
    fov = 220e-3 # field of view
    delta_k = fov / Nx

    # RF sinc pulse with a 90 degree flip angle
    rf90, _, _ = make_sinc_pulse(flip_angle=90, system=system, slice_thickness=5e-3, apodization=0.5, time_bw_product=4)

    # Frequency encode, trapezoidal event
    gx = make_trapezoid(channel='x', flat_area=Nx * delta_k, flat_time=6.4e-3, system=system)

    # ADC readout
    adc = make_adc(num_samples=Nx, duration=gx.flat_time, delay=gx.rise_time, system=system)
    ```
4. Add these pulse sequence events to the `Sequence` object from step 2. One or more events can be executed
simultaneously, simply pass them all to the `add_block()` method. For example, the `gx` and `adc` pulse sequence events
need to be executed simultaneously:
    ```python
    seq.add_block(rf90)
    seq.add_block(gx, adc)
    ```
5. Visualize plots:
    ```python
    seq.plot()
    ```
6. Generate a `.seq` file to be executed on a real MR scanner:
    ```python
    seq.write('demo.seq')
    ```

## Contributing and Community guidelines
`PyPulseq` adheres to a code of conduct adapted from the [Contributor Covenant] code of conduct.
Contributing guidelines can be found [here][contrib-guidelines].

---
## References
1. Ravi, Keerthi Sravan, et al. "Pulseq-Graphical Programming Interface: Open source visual environment for prototyping
pulse sequences and integrated magnetic resonance imaging algorithm development." Magnetic resonance imaging 52 (2018):
9-15.
2. Layton, Kelvin J., et al. "Pulseq: a rapid and hardware‐independent pulse sequence prototyping framework." Magnetic
resonance in medicine 77.4 (2017): 1544-1552.

[api-docs]: https://pypulseq.readthedocs.io/en/latest
[Bruker]: https://github.com/pulseq/bruker_interpreter
[Contributor Covenant]: http://contributor-covenant.org
[contrib-guidelines]: https://github.com/imr-framework/pypulseq/blob/master/CONTRIBUTING.md
[GE]: https://toppemri.github.io
[Pulseq specification]: https://pulseq.github.io/specification.pdf
