---
title: 'PyPulseq: A Python Package for MRI Pulse Sequence Design'
tags:
  - Python
  - MRI
  - pulse sequence design
  - vendor neutral
authors:
  - name: Keerthi Sravan Ravi
    orcid: 0000-0001-6886-0101
    affiliation: 1
  - name: Sairam Geethanath
    orcid: 0000-0002-3776-4114
    affiliation: 1
  - name: John Thomas Vaughan Jr.
    orcid: 0000-0002-6933-3757
    affiliation: 1  
affiliations:
 - name: Columbia Magnetic Resonance Research Center, Columbia University in the City of New York, USA
   index: 1
date: 21 August 2019
bibliography: paper.bib
---

# Summary

Magnetic Resonance Imaging (MRI) is a critical component of healthcare. MRI data is acquired by playing a series of 
radio-frequency and magnetic field gradient pulses. Designing these pulse sequences requires knowledge of specific 
programming environments depending on the vendor hardware (generations) and software (revisions) intended for 
implementation. This impedes the pace of prototyping. Pulseq [@layton2017pulseq] introduced an open source file 
standard for pulse sequences that can be deployed on Siemens/GE via [TOPPE](https://toppemri.github.io) 
[@nielsen2018toppe]/[Bruker](https://github.com/pulseq/bruker_interpreter) platforms. In this work, we introduce 
`PyPulseq`, which enables pulse sequence programming in Python. Its advantages are zero licensing fees and easy 
integration with deep learning methods developed in Python. `PyPulseq` is aimed at MRI researchers, faculty, students, 
and other allied field researchers such as those in neuroscience. We have leveraged this tool for several published 
research works [@poojar2019rapid; @gehua2019ismrm; @ravi2018amri; @ravi2019accessible-amri; @ravi2018imrframework; 
@ravi2019selfadmin].

# Statement of need

MRI is a non-invasive diagnostic imaging tool. It is a critical component of healthcare and has a significant impact on 
diagnosis and treatment assessment. Structural, functional and metabolic MRI generate valuable information that aid in 
the accurate diagnosis of a wide range of pathologies. A unique strength of MRI is the ability to visualise diverse 
pathologies achieved by the flexibility in designing tailored pulse sequences. MRI pulse sequences are a collection of 
radio-frequency and gradient waveforms that are executed on the scanner hardware to acquire raw data. 

Research efforts on pulse sequence design are directed at achieving faster scan times, improving tissue contrast and 
increasing Signal-to-Noise Ratio (SNR). However, designing pulse sequences requires knowledge of specific programming 
environments depending on the vendor hardware (generations) and software (revisions) intended for implementation. 
Typically, MRI researchers program and simulate the pulse sequences on computers and execute them on MRI scanners. This 
typically involves considerable effort, impeding the pace of prototyping and therefore research and development. This 
also hampers multi-site multi-vendor studies as it requires researchers to be acquainted with each vendor's programming 
environment. Furthermore, harmonizing acquisition across MRI vendors will enable reproducible research. This work 
introduces an open source tool that enables pulse sequence programming for Siemens/GE/Bruker platforms in Python, based 
on the Pulseq standard [@layton2017pulseq].

# Introduction to the Pulseq file format: `.seq`

The `.seq` file format introduced in Pulseq [@layton2017pulseq] is a novel way to capture a pulse sequence as plain 
text. The file format was designed with the following design criteria in mind: human-readable, easily parsable, vendor 
independent, compact and low-level [@layton2017pulseq]. A pulse sequence comprises of radiofrequency pulses, magnetic 
field gradient waveforms, delays or analog-to-digital converter (ADC) readout *events*. A *block* comprises of one or 
more *events* occurring simultaneously. *Event* envelopes are defined by *shapes*, which are run-length encoded and 
stored in the `.seq` file. In a `.seq` file, each *event* and *shape* is identified uniquely by an integer. *Blocks* 
are constructed by assembling the uniquely referenced *events*. Therefore, any custom pulse sequence can be synthesised 
by concatenating *blocks*.

# About `PyPulseq`

The `PyPulseq` package presented in this work is an open source vendor-neutral MRI pulse sequence design tool. It 
enables researchers and users to program pulse sequences in Python, and export them as a `.seq` file. These `.seq` files 
can be executed on the three MRI vendors by leveraging vendor-specific interpreters. The MRI methods have been reported 
previously [@ravi2018pulseq-gpi]. The `PyPulseq` package allows for both representing and deploying custom sequences. 
This work focuses on the software aspect of the tool. `PyPulseq` was entirely developed in Python, and this has multiple 
advantages. Firstly, unlike existing C++ frameworks such as ODIN [@jochimsen2004odin] and SequenceTree [@magland2016pulse],
`PyPulseq` does not require any compilation of the pulse sequence scripts. Secondly, it does not involve any licensing 
fees that are otherwise associated with other scientific research platforms such as MATLAB. Thirdly, there has been a 
proliferation of deep learning projects developed in Python in recent years. These advantages allow `PyPulseq` to be 
integrated with projects related to various stages of the MRI pipeline. For example - deep learning techniques for 
acquisition (intelligent slice planning in @ravi2018amri) and related downstream reconstruction. Finally, the 
standard Python package manager - PyPI - enables convenient installs on multiple OS platforms. These Python-derived 
benefits ensure that `PyPulseq` can reach a wider audience.

We have leveraged the `PyPulseq` library to implement acquisition oriented components of the Autonomous MRI (AMRI) 
package [@ravi2018amri; @ravi2019accessible-amri; @ravi2019selfadmin], Virtual Scanner [@gehua2019ismrm], and the 
non-Cartesian acquisition library [@ravi2018imrframework]. Also, the [`PyPulseq-gpi`](https://github.com/imr-framework/pypulseq/tree/pypulseq-gpi) branch 
integrates a previous version of `PyPulseq` with [GPI](http://gpilab.com/) to enable GUI-based pulse sequence design. This work has 
been previously reported [@ravi2018pulseq-gpi] and is not within the scope of this JOSS submission. Currently, 
`PyPulseq` does not support external triggers and interactive slice planning. Raw data acquired with pulse sequences 
designed with `PyPulseq` cannot be reconstructed vendor-supplied tools. `PyPulseq` is a translation of Pulseq from 
MATLAB [@layton2017pulseq].

# Target audience

`PyPulseq` is aimed at MRI researchers focusing on pulse sequence design, image reconstruction, and MRI physics. We also 
envisage PyPulseq to be utilized for replicability and reproducibility studies such as those for functional MRI 
(multi-site, multi-vendor). The package could also serve as a hands-on teaching aid for MRI faculty and students. 
Beginners can get started with the bundled example pulse sequences. More familiar users can import the appropriate 
packages to construct and deploy custom pulse sequences.

# Acknowledgements

This study was funded (in part) by the 'MR Technology Development Grant' and the 'Seed Grant Program for MR Studies' 
of the Zuckerman Mind Brain Behavior Institute at Columbia University (PI: Geethanath).

# References
