<p align="center">

![PyPulseq](logo.png)

</p>

<p align = "left">

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python](https://img.shields.io/badge/python-3.7--3.13-blue)](https://pypi.org/project/pypulseq/)
[![FAIR checklist badge](https://fairsoftwarechecklist.net/badge.svg)](https://fairsoftwarechecklist.net/v0.2?f=31&a=32113&i=32322&r=133)

</p>

# PyPulseq: A Python Package for MRI Pulse Sequence Design

## PyPulseq (v1.5.0) is compatible with all Pulseq interpreter sequences with version >= 1.5.0. The new features introduced with Pulseq 1.5.1 are not yet supported, but will be added in the near future. 

## Table of contents 🧾

1. [📃 General Information][section-general-info]
2. [🔨 Installation][section-installation]
3. [⚡ Lightning-start - PyPulseq in your browser!][section-lightning-start]
4. [🏃‍♂ Quickstart - example scripts][section-quickstart-examples]
5. [🤿 Deep dive - custom pulse sequences][section-deep-dive]
6. [👥 Contributing and Community guidelines][section-contributing]
7. [📖 References][section-references]

---

## 1. General Information

Pulse sequence design is a significant component of MRI research. However, multi-vendor studies require researchers to
be acquainted with each hardware platform's programming environment.

PyPulseq enables vendor-neutral pulse sequence design in Python [[1,2]][section-references]. The pulse sequences can be
exported as a `.seq` file to be run on Siemens, [GE], [Bruker] and now also Philips hardware by leveraging their respective Pulseq interpreters. This tool is targeted at MRI pulse sequence designers, researchers, students and other interested
users. It is a translation of the Pulseq framework originally written in Matlab [[3]][section-references].

👉 Currently, PyPulseq is compatible with Pulseq >= 1.5.0. The new features introduced with Pulseq 1.5.1 are not yet supported, but will be added in the near future. 👈

It is strongly recommended to first read the [Pulseq specification]  before proceeding. The specification
document defines the concepts required for pulse sequence design using PyPulseq.

If you use PyPulseq in your work, please cite the publications listed under [References][section-references].

---

## 2. 🔨 Installation

PyPulseq is available on the python Package Index [PyPi](https://pypi.org/project/pypulseq/) and can be installed using the command

```bash
pip install pypulseq
```

To use the [sigpy](https://sigpy.readthedocs.io/en/latest/) functionality of `make_sigpy_pulse.py` run `pip install pypulseq[sigpy]` to install the required dependencies and enable this functionality.

The latest features and minor bug fixes might not be included in the latest release version. If you want to use the bleeding edge version of PyPulseq, you can install it directly from the development branch of this repository using the command

```bash
pip install git+https://github.com/imr-framework/pypulseq@master
```

👉 PyPulseq is **now available on conda**. It can be installed using the command

```bash
conda install conda-forge::pypulseq
```

---

## 3. ⚡ Lightning-start - PyPulseq in your browser

1. Create a new notebook on [Google Colab][google-colab]
2. Install PyPulseq using `pip install pypulseq`
3. Get going!

---

## 4. 🏃‍♂ Example scripts

The PyPulseq repository contains several example sequences in the [examples](/examples/) folder. Every example script or example notebook creates a pulse sequence, plots the pulse timing diagram and finally saves the sequence as a `.seq` file to disk.

---

## 5. 🤿 Deep dive - custom pulse sequences

Getting started with pulse sequence design using `PyPulseq` is simple:

1. First, define system limits in `Opts` and then create a `Sequence` object with it:

    ```python
    import pypulseq as pp

    system = pp.Opts(max_grad=32, grad_unit='mT/m', max_slew=130, slew_unit='mT/m/ms')
    seq = pp.Sequence(system=system)
    ```

2. Then, design gradient, RF or ADC pulse sequence events:

    ```python
    Nx, Ny = 256, 256 # matrix size
    fov = 220e-3 # field of view
    delta_k = fov / Nx

    # RF sinc pulse with a 90 degree flip angle
    rf90 = pp.make_sinc_pulse(flip_angle=90, duration=2e-3, system=system, slice_thickness=5e-3, apodization=0.5,time_bw_product=4, use='excitation')

    # Frequency encode, trapezoidal event
    gx = pp.make_trapezoid(channel='x', flat_area=Nx * delta_k, flat_time=6.4e-3, system=system)

    # ADC readout
    adc = pp.make_adc(num_samples=Nx, duration=gx.flat_time, delay=gx.rise_time, system=system)
    ```

3. Add these pulse sequence events to the `Sequence` object. One or more events can be executed simultaneously, simply pass them all to the `add_block()` method. For example, the `gx` and `adc` pulse sequence events need to be executed simultaneously:

    ```python
    seq.add_block(rf90)
    seq.add_block(gx, adc)
    ```

4. Visualize plots:

    ```python
    seq.plot()
    ```

5. Generate a `.seq` file to be executed on a real MR scanner:

    ```python
    seq.write('demo.seq')
    ```

---

## 6. 👥 Contributing and Community guidelines

`PyPulseq` adheres to a code of conduct adapted from the [Contributor Covenant] code of conduct.
Contributing guidelines can be found [here][contrib-guidelines].

---

## 7. 📖 References

1. Ravi, Keerthi, Sairam Geethanath, and John Vaughan. "PyPulseq: A Python Package for MRI Pulse Sequence Design."
Journal of Open Source Software 4.42 (2019): 1725.
2. Ravi, Keerthi Sravan, et al. "Pulseq-Graphical Programming Interface: Open source visual environment for prototyping
pulse sequences and integrated magnetic resonance imaging algorithm development." Magnetic resonance imaging 52 (2018):
9-15.
3. Layton, Kelvin J., et al. "Pulseq: a rapid and hardware‐independent pulse sequence prototyping framework." Magnetic
resonance in medicine 77.4 (2017): 1544-1552.

[Bruker]: https://github.com/pulseq/bruker_interpreter
[Contributor Covenant]: http://contributor-covenant.org
[GE]: https://toppemri.github.io
[Pulseq specification]: https://pulseq.github.io/specification.pdf
[contrib-guidelines]: https://github.com/imr-framework/pypulseq/blob/master/CONTRIBUTING.md
[google-colab]: https://colab.research.google.com/
[section-general-info]: #1-general-information
[section-contributing]: #7--contributing-and-community-guidelines
[section-deep-dive]: #6--deep-dive---custom-pulse-sequences
[section-installation]: #3--installation
[section-lightning-start]: #4--lightning-start---pypulseq-in-your-browser
[section-quickstart-examples]: #5--quickstart---example-scripts
[section-references]: #8--references
