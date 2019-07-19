---
title: 'pypulseq'
tags:
  - Python
  - MRI
  - pulse sequence design
  - vendor neutral
authors:
  - name: Keerthi Sravan Ravi
    orcid: 0000-0001-6886-0101
    affiliation: 1
affiliations:
 - name: Columbia Magnetic Resonance Research Center, Columbia University in the City of New York, USA
   index: 1
date: 17 July 2019
bibliography: paper.bib
---

# Summary

Magnetic Resonance Imaging (MRI) is a non-invasive diagnostic imaging tool. It is a critical component of healthcare 
and has a significant impact on diagnosis and treatment assessment. Structural, functional and metabolic MRI generate 
valuable information that aid in the accurate diagnosis of a wide range of pathologies.

A key aspect of MRI research is pulse sequence design. Pulse sequences directly impact the time required for the MRI 
scan and the value of the information obtained. Medical researchers program the pulse sequences on computers and 
execute them on MR scanners. However, the programming language differs across vendors. This hampers multi-site 
multi-vendor studies as it requires researchers to be acquainted with each vendor's programming environment.  

The `pypulseq` package presented in this work is an open-source vendor-neutral MRI pulse sequence design tool. It 
enables researchers and users to program pulse sequences in Python and export as a `.seq` file. These `.seq` files can 
be executed on Siemens/GE/Bruker MR hardware by leveraging vendor-specific interpreters.

The `pypulseq-gpi` branch on the Github repository includes GPI 'Nodes' that enable GUI-based pulse sequence design. 
This work has been previously reported [@pulseq-gpi] and is not within the scope of this JOSS submission. 

# Acknowledgements

This study was funded (in part) by the Seed Grant Program for MR Studies of the Zuckerman Mind Brain Behavior Institute 
at Columbia University (PI: Geethanath).

# References