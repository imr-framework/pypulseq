import setuptools

import pypulseq

try:  # Unicode decode error on Windows
    with open("README.md", "r") as fh:
        long_description = fh.read()
except:
    long_description = 'Pulseq on Python'

setuptools.setup(
    name="pypulseq",
    version=pypulseq.__version__,
    author="Keerthi Sravan Ravi",
    author_email="ks3621@columbia.edu",
    description="Pulseq in Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/imr-framework/pypulseq",
    packages=setuptools.find_packages(exclude=['pypulseq.utils.SAR', 'pypulseq.seq_examples']),
    install_requires=['numpy>=1.16.3', 'matplotlib>=3.0.3'],
    license='License :: OSI Approved :: GNU Affero General Public License v3',
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: GNU Affero General Public License v3",
        "Operating System :: OS Independent",
    ],
)
