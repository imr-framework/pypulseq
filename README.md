<p align="center">
<img src="logo.png"/>
</p>
# pypulseq
Pulseq in Python

Pulse sequence design is a significant component of MRI research. However, multi-vendor studies require researchers to 
be acquainted with each hardware platform's programming environment.

`pypulseq` enables vendor-neutral pulse sequence design in Python [[1]](#references). The pulse sequences can be 
exported as a `.seq` file to be run on  Siemens/[GE](https://toppemri.github.io)/
[Bruker](https://github.com/pulseq/bruker_interpreter) hardware by leveraging their respective Pulseq interpreters. 
This tool is targeted at MR pulse sequence designers, MRI researchers and other interested users. It is a translation 
of the Pulseq framework originally written in Matlab [[2]](#references).

It is strongly recommended to first read the [Pulseq specification](https://pulseq.github.io/specification.pdf) 
before proceeding. The specification document defines the concepts required for pulse sequence design using `pypulseq`.   

## 1 minute demo
1. Clone this repository.
2. Run any of the example scripts in your favourite IDE on Python 3.6 or above.
3. Inspect plots!
4. Get in touch regarding running the `.seq` files on your Siemens/GE/Bruker scanner.

## Custom pulse sequences
Getting started with pulse sequence design using `pypulseq` is simple:
1. `pip install pypulseq` on Python 3.6 or above.
2. First, define system limits in `Opts` and then create a `Sequence` object with it:
    ```python
    from pypulseq.opts import Opts
    from pypulseq.Sequence.sequence import Sequence
    
    system = Opts(max_grad=32, grad_unit='mT/m', max_slew=130, slew_unit='mT/m/s')
    seq = Sequence(system=system)
    ```
3. Then, design gradient, RF or ADC pulse sequence events. An example RF sinc pulse with a 90 degree flip angle:
    ```python
    from pypulseq.make_sinc_pulse import make_sinc_pulse
    rf90, _, _ = make_sinc_pulse(flip_angle=90, system=system, slice_thickness=5e-3, apodization=0.5, time_bw_product=4)
    ```

### References
1. Ravi, Keerthi Sravan, et al. "Pulseq-Graphical Programming Interface: Open source visual environment for prototyping 
pulse sequences and integrated magnetic resonance imaging algorithm development." Magnetic resonance imaging 52 (2018): 
9-15.
2. Layton, Kelvin J., et al. "Pulseq: a rapid and hardware‐independent pulse sequence prototyping framework." Magnetic 
resonance in medicine 77.4 (2017): 1544-1552.
