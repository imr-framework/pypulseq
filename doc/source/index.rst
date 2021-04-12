pypulseq
====================================
.. image:: ../../logo.png
   :align: center

Introduction
============
`pypulseq <https://github.com/imr-framework/pypulseq>`_ enables vendor-neutral pulse sequence design in Python [1]_ [2]_. The pulse sequences can be exported as a `.seq` file to be run on  Siemens/`GE <https://toppemri.github.io>`_/`Bruker <https://github.com/pulseq/bruker_interpreter>`_ hardware by leveraging their respective Pulseq interpreters. This tool is targeted at MR pulse sequence designers, MRI researchers and other interested users. It is a translation of the Pulseq framework originally written in Matlab [3]_.

It is strongly recommended to first read the `Pulseq specification <https://pulseq.github.io/specification.pdf>`_ before proceeding. The specification document defines the concepts required for pulse sequence design using `pypulseq`.

.. [1] Ravi, Keerthi, Sairam Geethanath, and John Vaughan. "PyPulseq: A Python Package for MRI Pulse Sequence Design." Journal of Open Source Software 4.42 (2019): 1725.

.. [2] Ravi, Keerthi Sravan, et al. "Pulseq-Graphical Programming Interface: Open source visual environment for prototyping pulse sequences and integrated magnetic resonance imaging algorithm development." Magnetic resonance imaging 52 (2018): 9-15.

.. [3] Layton, Kelvin J., et al. "Pulseq: a rapid and hardware‐independent pulse sequence prototyping framework." Magnetic resonance in medicine 77.4 (2017): 1544-1552.


API documentation
==================

.. toctree::
    :maxdepth: 2

    modules
